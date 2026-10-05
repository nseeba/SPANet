from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import shutil
import json

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE, WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar
)

from spanet import JetReconstructionModel, Options


class AssignmentDiagnosticsCallback(pl.Callback):
    """
    Log lightweight validation diagnostics to TensorBoard.

    The callback intentionally evaluates only a configurable number of validation
    batches so that it can run during normal training without dominating epoch
    time on large samples.
    """

    def __init__(self, max_batches: int = 10):
        super().__init__()
        self.max_batches = max_batches

    @staticmethod
    def _has_tensorboard_experiment(logger) -> bool:
        return (
            logger is not None and
            hasattr(logger, "experiment") and
            hasattr(logger.experiment, "add_histogram")
        )

    @staticmethod
    def _roc_figure(targets: np.ndarray, scores: np.ndarray, title: str):
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import auc, roc_curve
        except Exception:
            return None, None

        if len(np.unique(targets)) < 2:
            return None, None

        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, roc_auc

    @staticmethod
    def _assignment_scores(outputs, jet_predictions, model) -> list[np.ndarray]:
        scores = []
        batch_indices = None

        for assignment_logits, prediction, symmetries in zip(
            outputs.assignments,
            jet_predictions,
            model.event_info.product_symbolic_groups.values(),
        ):
            prediction = torch.as_tensor(prediction, device=assignment_logits.device, dtype=torch.long)
            if batch_indices is None:
                batch_indices = torch.arange(prediction.shape[0], device=assignment_logits.device)

            index = (batch_indices, *prediction.T)
            probability = torch.exp(assignment_logits[index]) * symmetries.order()
            scores.append(probability.detach().cpu().numpy())

        return scores

    @staticmethod
    def _classification_confidences(classifications: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        output = {}
        for name, logits in classifications.items():
            probabilities = torch.softmax(logits, dim=1)
            output[name] = probabilities.max(dim=1).values.detach().cpu().numpy()
        return output

    @staticmethod
    def _pair_correct(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.all(np.sort(prediction, axis=1) == np.sort(target, axis=1), axis=1)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.max_batches <= 0 or not self._has_tensorboard_experiment(trainer.logger):
            return

        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, (list, tuple)):
            val_loader = val_loader[0]
        if val_loader is None:
            return

        device = pl_module.device
        product_names = list(pl_module.event_info.product_particles)
        detection_scores = {name: [] for name in product_names}
        detection_targets = {name: [] for name in product_names}
        assignment_scores = {name: [] for name in product_names}
        assignment_correct = {name: [] for name in product_names}
        classification_confidences = {}

        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= self.max_batches:
                    break

                batch = pl_module.transfer_batch_to_device(batch, device, 0)
                sources, _, targets, _, _ = batch
                outputs = pl_module.forward(sources)
                jet_predictions, particle_scores, _, _ = pl_module.predict(sources)
                prediction_scores = self._assignment_scores(outputs, jet_predictions, pl_module)

                for i, name in enumerate(product_names):
                    target, mask, _ = targets[i]
                    target_np = target.detach().cpu().numpy()
                    mask_np = mask.detach().cpu().numpy().astype(bool)
                    prediction_np = jet_predictions[i]

                    detection_scores[name].append(particle_scores[i])
                    detection_targets[name].append(mask_np)

                    assignment_scores[name].append(prediction_scores[i][mask_np])
                    if mask_np.any():
                        assignment_correct[name].append(
                            self._pair_correct(prediction_np[mask_np], target_np[mask_np])
                        )

                for name, confidence in self._classification_confidences(outputs.classifications).items():
                    classification_confidences.setdefault(name, []).append(confidence)

        if was_training:
            pl_module.train()

        writer = trainer.logger.experiment
        step = trainer.current_epoch + 1

        for name in product_names:
            det_score = np.concatenate(detection_scores[name]) if detection_scores[name] else np.array([])
            det_target = np.concatenate(detection_targets[name]) if detection_targets[name] else np.array([])
            if det_score.size:
                writer.add_histogram(f"Diagnostics/{name}/detection_score", det_score, step)
                figure, roc_auc = self._roc_figure(det_target.astype(int), det_score, f"{name} detection ROC")
                if figure is not None:
                    writer.add_figure(f"Diagnostics/{name}/detection_roc", figure, step, close=True)
                    writer.add_scalar(f"Diagnostics/{name}/detection_auc", roc_auc, step)

            assign_score = np.concatenate(assignment_scores[name]) if assignment_scores[name] else np.array([])
            assign_correct = np.concatenate(assignment_correct[name]) if assignment_correct[name] else np.array([])
            if assign_score.size:
                writer.add_histogram(f"Diagnostics/{name}/assignment_score", assign_score, step)
                if assign_correct.size:
                    writer.add_histogram(
                        f"Diagnostics/{name}/assignment_score_correct",
                        assign_score[assign_correct],
                        step,
                    )
                    writer.add_histogram(
                        f"Diagnostics/{name}/assignment_score_wrong",
                        assign_score[~assign_correct],
                        step,
                    )
                    figure, roc_auc = self._roc_figure(
                        assign_correct.astype(int),
                        assign_score,
                        f"{name} assignment-score ROC",
                    )
                    if figure is not None:
                        writer.add_figure(f"Diagnostics/{name}/assignment_score_roc", figure, step, close=True)
                        writer.add_scalar(f"Diagnostics/{name}/assignment_score_auc", roc_auc, step)

        for name, confidences in classification_confidences.items():
            confidence = np.concatenate(confidences)
            writer.add_histogram(f"Diagnostics/classification/{name}_confidence", confidence, step)


class EpochSummaryCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        keys = (
            "loss/total_loss_epoch",
            "validation_average_jet_accuracy",
            "validation_accuracy",
        )
        values = []
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                values.append(f"{key}={value:.5g}")

        if values:
            print(f"Epoch {trainer.current_epoch + 1}: " + ", ".join(values), flush=True)


def main(
        event_file: str,
        training_file: str,
        validation_file: str,
        options_file: Optional[str],
        checkpoint: Optional[str],
        state_dict: Optional[str],
        freeze_state_dict: bool,

        log_dir: str,
        name: str,

        torch_script: bool,
        fp16: bool,
        verbose: bool,
        full_events: bool,

        profile: bool,
        gpus: Optional[int],
        epochs: Optional[int],
        time_limit: Optional[str],
        batch_size: Optional[int],
        limit_dataset: Optional[float],
        diagnostics_batches: int,
        validate_every_n_epochs: int,
        log_every_n_steps: int,
        device_stats: bool,
        random_seed: int,
    ):

    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in environ:
        master = False

    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs

    if random_seed > 0:
        options.dataset_randomization = random_seed

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    if master:
        options.display()

    # -------------------------------------------------------------------------------------------------------
    # Begin the training loop
    # -------------------------------------------------------------------------------------------------------

    # Create the initial model on the CPU
    model = JetReconstructionModel(options, torch_script)

    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")

        state_dict = torch.load(state_dict, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if master:
            print(f"Missing Keys: {missing_keys}")
            print(f"Unexpected Keys: {unexpected_keys}")

        if freeze_state_dict:
            for pname, parameter in model.named_parameters():
                if pname in state_dict:
                    parameter.requires_grad_(False)

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    log_dir = getcwd() if log_dir is None else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=name)
    # logger = (
    #     WandbLogger(name=name, save_dir=log_dir)
    #     if _WANDB_AVAILABLE else
    #     TensorBoardLogger(save_dir=log_dir, name=name)
    # )

    # Create the checkpoint for this training run. We will save the best validation networks based on 'accuracy'
    callbacks = [
        ModelCheckpoint(
            verbose=options.verbose_output,
            filename='{epoch}-{step}-{validation_average_jet_accuracy:.3f}',
            monitor='validation_average_jet_accuracy',
            save_top_k=1,
            mode='max',
            save_last=True
        ),
        LearningRateMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1),
        EpochSummaryCallback(),
    ]
    if device_stats:
        callbacks.append(DeviceStatsMonitor())
    if diagnostics_batches > 0:
        callbacks.append(AssignmentDiagnosticsCallback(max_batches=diagnostics_batches))

    epochs = options.epochs
    profiler = None
    if profile:
        epochs = 1
        profiler = PyTorchProfiler(emit_nvtx=True)

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(
        accelerator="gpu" if options.num_gpu > 0 else "auto",
        devices=options.num_gpu if options.num_gpu > 0 else "auto",
        strategy="ddp" if options.num_gpu > 1 else "auto",
        precision="16-mixed" if fp16 else "32-true",

        # Manual PCGrad performs clipping after the projected gradients are
        # merged; Lightning's automatic clipping is incompatible with manual
        # optimization.
        gradient_clip_val=(
            None if options.pcgrad
            else options.gradient_clip if options.gradient_clip > 0 else None
        ),
        max_epochs=epochs,
        max_time=time_limit,
        check_val_every_n_epoch=validate_every_n_epochs,
        log_every_n_steps=log_every_n_steps,
        num_sanity_val_steps=0,

        logger=logger,
        profiler=profiler,
        callbacks=callbacks
    )

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)

        with open(f"{trainer.logger.log_dir}/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

        shutil.copy2(options.event_info_file, f"{trainer.logger.log_dir}/event.yaml")

    trainer.fit(model, ckpt_path=checkpoint)
    # -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Input file containing event symmetry information.")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="Input file containing training data.")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="Input file containing Validation data. If not provided, will use training data split.")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load the training state from. "
                             "Fully restores model weights and optimizer state.")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load from checkpoint but only the model weights. "
                             "Can be partial as the weights don't have to match one-to-one.")

    parser.add_argument("-fsf", "--freeze_state_dict", action='store_true',
                        help="Freeze any weights that were loaded from the state dict. "
                             "Used for finetuning new layers.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run and an identifier for WANDB.")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs to train for")
    
    parser.add_argument("-t", "--time_limit", type=str, default=None,
                        help="Time limit for training, in the format DD:HH:MM:SS.")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("--diagnostics-batches", type=int, default=0,
                        help="Log TensorBoard score histograms and ROC figures from this many validation batches. "
                             "Set to 0 to disable.")

    parser.add_argument("--validate-every-n-epochs", type=int, default=1,
                        help="Run validation every N training epochs.")

    parser.add_argument("--log-every-n-steps", type=int, default=50,
                        help="TensorBoard scalar logging interval in training steps.")

    parser.add_argument("--device-stats", action="store_true",
                        help="Enable Lightning DeviceStatsMonitor. Useful for debugging, but verbose and slower.")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Torch AMP for training.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-r", "--random_seed", type=int, default=0,
                        help="Set random seed for cross-validation.")

    parser.add_argument("-ts", "--torch_script", action='store_true',
                        help="Compile the neural network using torchscript.")

    parser.add_argument("--profile", action='store_true',
                        help="Profile network for a single training epoch.")

    main(**parser.parse_args().__dict__)
