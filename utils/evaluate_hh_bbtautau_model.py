#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


def add_repo_to_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


SPANET_DIR = add_repo_to_path()

from spanet import JetReconstructionModel, Options  # noqa: E402
from spanet.dataset.types import Source  # noqa: E402
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions  # noqa: E402


def checkpoint_score(path: Path) -> float:
    match = re.search(r"validation_average_jet_accuracy=([0-9]+(?:\\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else -np.inf


def best_checkpoint_from_last(run_dir: Path) -> Path | None:
    last = run_dir / "checkpoints" / "last.ckpt"
    if not last.exists():
        return None
    payload = torch.load(last, map_location="cpu")
    for callback_state in payload.get("callbacks", {}).values():
        best_path = callback_state.get("best_model_path")
        if best_path and Path(best_path).exists():
            return Path(best_path)
    return None


def best_checkpoint(run_dir: Path) -> Path:
    best = best_checkpoint_from_last(run_dir)
    if best is not None:
        return best
    checkpoints = list((run_dir / "checkpoints").glob("epoch*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {run_dir / 'checkpoints'}")
    return max(checkpoints, key=checkpoint_score)


def resolve_spanet_path(path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    return str(path if path.is_absolute() else SPANET_DIR / path)


def load_model(
    run_dir: Path,
    checkpoint: Path | None,
    batch_size: int,
    use_gpu: bool,
    test_file: Path | None,
    event_file: Path | None,
) -> JetReconstructionModel:
    checkpoint = best_checkpoint(run_dir) if checkpoint is None else checkpoint
    options = Options.load(str(run_dir / "options.json"))
    options.batch_size = batch_size
    if event_file is not None:
        options.event_info_file = str(event_file)
    options.event_info_file = resolve_spanet_path(options.event_info_file)
    options.training_file = resolve_spanet_path(options.training_file)
    options.validation_file = resolve_spanet_path(options.validation_file)
    options.testing_file = resolve_spanet_path(options.testing_file)
    if test_file is not None:
        options.testing_file = str(test_file)

    model = JetReconstructionModel(options)
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if use_gpu:
        model = model.cuda()
    print(f"checkpoint: {checkpoint}")
    return model


def assignment_scores(outputs, predictions, model):
    scores = []
    for assignment_logits, prediction, symmetries in zip(
        outputs.assignments,
        predictions,
        model.event_info.product_symbolic_groups.values(),
    ):
        prediction = torch.as_tensor(prediction, device=assignment_logits.device, dtype=torch.long)
        batch_indices = torch.arange(prediction.shape[0], device=assignment_logits.device)
        probability = torch.exp(assignment_logits[(batch_indices, *prediction.T)]) * symmetries.order()
        scores.append(probability.detach().cpu().numpy())
    return scores


def normalize_symmetric_targets(model, predictions, targets):
    predictions = [p.copy() for p in predictions]
    targets = [t.copy() for t in targets]
    for target, prediction, decoder in zip(targets, predictions, model.branch_decoders):
        for indices in decoder.permutation_indices:
            if len(indices) > 1:
                prediction[:, indices] = np.sort(prediction[:, indices], axis=1)
                target[:, indices] = np.sort(target[:, indices], axis=1)
    return predictions, targets


def pair_correct(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.all(np.sort(prediction, axis=1) == np.sort(target, axis=1), axis=1)


def evaluate(model: JetReconstructionModel, split: str, max_batches: int | None, fp16: bool):
    dataset = {
        "train": model.training_dataset,
        "validation": model.validation_dataset,
        "test": model.testing_dataset,
    }[split]
    if dataset is None:
        raise ValueError(f"Split {split} is not available")

    loader = DataLoader(
        dataset,
        batch_size=model.options.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    product_names = list(model.event_info.product_particles)
    output = {name: defaultdict(list) for name in product_names}
    output["num_jets"] = []

    device = next(model.parameters()).device
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch_idx % 50 == 0:
            print(f"batch {batch_idx}")

        sources, num_jets, targets, _, _ = batch
        sources = tuple(Source(x[0].to(device), x[1].to(device)) for x in sources)
        output["num_jets"].append(num_jets.cpu().numpy())

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=fp16):
            network_outputs = model.forward(sources)

        raw_predictions = extract_predictions([
            np.nan_to_num(assignment.detach().cpu().numpy(), -np.inf)
            for assignment in network_outputs.assignments
        ])
        detection_probabilities = np.stack([
            torch.sigmoid(detection).detach().cpu().numpy()
            for detection in network_outputs.detections
        ])
        prediction_scores = assignment_scores(network_outputs, raw_predictions, model)

        target_arrays = [target.detach().cpu().numpy() for target, _, _ in targets]
        mask_arrays = [mask.detach().cpu().numpy().astype(bool) for _, mask, _ in targets]
        predictions, target_arrays = normalize_symmetric_targets(model, raw_predictions, target_arrays)

        for i, name in enumerate(product_names):
            valid = mask_arrays[i]
            correct = np.zeros_like(valid, dtype=bool)
            if valid.any():
                correct[valid] = pair_correct(predictions[i][valid], target_arrays[i][valid])

            output[name]["prediction"].append(predictions[i])
            output[name]["target"].append(target_arrays[i])
            output[name]["mask"].append(valid)
            output[name]["correct"].append(correct)
            output[name]["assignment_probability"].append(prediction_scores[i])
            output[name]["detection_probability"].append(detection_probabilities[i])

    merged = {"num_jets": np.concatenate(output["num_jets"])}
    for name in product_names:
        merged[name] = {key: np.concatenate(values) for key, values in output[name].items()}
    return merged


def write_output(output_file: Path, results: dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as f:
        f.create_dataset("num_jets", data=results["num_jets"])
        for target, values in results.items():
            if target == "num_jets":
                continue
            group = f.create_group(target)
            for key, value in values.items():
                group.create_dataset(key, data=value)
    print(f"wrote {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--event-file", type=Path, help="Override event YAML. Useful for evaluating older checkpoints after changing the default event file.")
    parser.add_argument("--test-file", type=Path, help="Explicit HDF5 file to use when --split test is selected.")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    use_gpu = args.gpu and torch.cuda.is_available()
    model = load_model(args.run_dir, args.checkpoint, args.batch_size, use_gpu, args.test_file, args.event_file)
    results = evaluate(model, split=args.split, max_batches=args.max_batches, fp16=args.fp16 and use_gpu)
    write_output(args.output, results)


if __name__ == "__main__":
    main()
