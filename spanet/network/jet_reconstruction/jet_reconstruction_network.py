import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.types import Tuple, Outputs, Source, Predictions

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder

from spanet.network.prediction_selection import extract_predictions
from spanet.network.utilities.gradient_scale import scale_gradient
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase

TArray = np.ndarray


def detection_weighted_offsets(outputs: Outputs, branch_offsets) -> np.ndarray:
    """ Per-event decoding offsets: log detection probability plus a fixed per-branch offset.

    Returns a [B, NUM_BRANCHES] float32 array for extract_predictions.
    """
    offsets = torch.stack([
        F.logsigmoid(detection.float()) + float(branch_offset)
        for detection, branch_offset in zip(outputs.detections, branch_offsets)
    ], dim=1)
    return np.ascontiguousarray(offsets.detach().cpu().numpy(), dtype=np.float32)


def default_assignment_fn(outputs: Outputs, decoding_order=None, branch_offsets=None):
    """ Decode assignments. With branch_offsets (one value per branch), greedy decoding is
    detection-weighted: see Options.detection_weighted_decoding.
    """
    offsets = None if branch_offsets is None else detection_weighted_offsets(outputs, branch_offsets)
    return extract_predictions([
        np.nan_to_num(assignment.detach().cpu().numpy(), nan=-np.inf, neginf=-np.inf)
        for assignment in outputs.assignments
    ], decoding_order, offsets)


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options, torch_script: bool = False):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim

        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))

        self.encoder = compile_module(JetEncoder(
            options,
        ))

        self.branch_decoders = nn.ModuleList([
            BranchDecoder(
                options,
                event_particle_name,
                self.event_info.product_particles[event_particle_name].names,
                product_symmetry,
                self.enable_softmax
            )
            for event_particle_name, product_symmetry
            in self.event_info.product_symmetries.items()
        ])

        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

        self.classification_decoder = compile_module(ClassificationDecoder(
            options,
            self.training_dataset
        ))

        branch_names = list(self.event_info.product_symmetries.keys())
        for option_name in ("branch_shared_gradient_scale", "branch_extra_encoder_layers"):
            unknown = set(getattr(options, option_name)) - set(branch_names)
            if unknown:
                raise ValueError(f"{option_name} refers to unknown branches {sorted(unknown)}, "
                                 f"available branches are {branch_names}")

        self.branch_gradient_scales = [
            float(options.branch_shared_gradient_scale.get(name, 1.0))
            for name in branch_names
        ]

        # Branch indices in the order in which they choose jets. None means greedy decoding.
        self.decoding_order = None
        if options.assignment_decoding_order:
            if sorted(options.assignment_decoding_order) != sorted(branch_names):
                raise ValueError(f"assignment_decoding_order must list every branch exactly once, "
                                 f"got {options.assignment_decoding_order}, branches are {branch_names}")
            self.decoding_order = [branch_names.index(name) for name in options.assignment_decoding_order]

        # Per-branch offsets for detection-weighted greedy decoding. None means plain greedy.
        unknown = set(options.decoding_branch_offsets) - set(branch_names)
        if unknown:
            raise ValueError(f"decoding_branch_offsets refers to unknown branches {sorted(unknown)}, "
                             f"available branches are {branch_names}")
        self.decoding_branch_offsets = None
        if options.detection_weighted_decoding:
            if self.decoding_order is not None:
                raise ValueError("detection_weighted_decoding cannot be combined with assignment_decoding_order.")
            self.decoding_branch_offsets = [
                float(options.decoding_branch_offsets.get(name, 0.0)) for name in branch_names
            ]

        # An example input for generating the network's graph, batch size of 2
        # self.example_input_array = tuple(x.contiguous() for x in self.training_dataset[:2][0])

    @property
    def enable_softmax(self):
        return True

    def forward(self, sources: Tuple[Source, ...]) -> Outputs:
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Create output lists for each particle in event.
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }

        # Pass the shared hidden state to every decoder branch. A branch with a gradient
        # scale below 1.0 sees the same hidden state but trains the shared layers less, or
        # not at all for a scale of 0.0.
        for decoder, gradient_scale in zip(self.branch_decoders, self.branch_gradient_scales):
            branch_hidden = scale_gradient(hidden, gradient_scale) if self.training else hidden
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(branch_hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarising vectors to their correct structure.
            encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
            for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict the valid regressions for any real values associated with the event.
        regressions = self.regression_decoder(encoded_vectors)

        # Predict additional classification targets for any branch of the event.
        classifications = self.classification_decoder(encoded_vectors)

        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications
        )

    def decode_assignments(self, outputs: Outputs):
        return default_assignment_fn(outputs, self.decoding_order, self.decoding_branch_offsets)

    def predict(self, sources: Tuple[Source, ...], assignment_fn=None) -> Predictions:
        with torch.no_grad():
            outputs = self.forward(sources)

            # Extract assignment probabilities and find the least conflicting assignment.
            # Without an explicit assignment_fn, decode in the configured branch order.
            assignments = self.decode_assignments(outputs) if assignment_fn is None else assignment_fn(outputs)

            # Convert detection logits into probabilities and move to CPU.
            detections = np.stack([
                torch.sigmoid(detection).cpu().numpy()
                for detection in outputs.detections
            ])

            # Move regressions to CPU and away from torch.
            regressions = {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }

            classifications = {
                key: value.cpu().argmax(1).numpy()
                for key, value in outputs.classifications.items()
            }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        # Run the base prediction step
        with torch.no_grad():
            assignments = [
                np.nan_to_num(assignment.detach().cpu().numpy(), nan=-np.inf, neginf=-np.inf)
                for assignment in self.forward(sources).assignments
            ]

        # Find the optimal selection of jets from the output distributions.
        if self.decoding_branch_offsets is not None:
            raise NotImplementedError("predict_assignments does not support detection-weighted decoding, "
                                      "use predict instead.")
        return extract_predictions(assignments, self.decoding_order)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5
