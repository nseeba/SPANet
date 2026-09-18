#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from types import MethodType
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from torch import nn
from torch.nn import functional as F


def add_repo_to_path() -> Path:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


SPANET_DIR = add_repo_to_path()

from spanet import JetReconstructionModel, Options  # noqa: E402
from spanet.dataset.types import Source  # noqa: E402
from spanet.network.layers.branch_decoder import BranchDecoder  # noqa: E402
from spanet.network.layers.transformer.gated_transformer import GTrXL  # noqa: E402


class ExportableSelfAttention(nn.Module):
    """
    Inference-only replacement for SPANet's GTrXL ``nn.MultiheadAttention``.

    Recent torch versions can introduce data-dependent guards while exporting
    ``nn.MultiheadAttention``.  SPANet uses it here only as self-attention with
    packed QKV weights, so this module keeps the checkpoint parameters and
    spells the same computation with export-friendly tensor ops.
    """

    def __init__(self, attention: nn.MultiheadAttention):
        super().__init__()

        if attention.batch_first:
            raise ValueError("ExportableSelfAttention expects sequence-first MultiheadAttention.")
        if attention.kdim != attention.embed_dim or attention.vdim != attention.embed_dim:
            raise ValueError("ExportableSelfAttention expects packed QKV projection weights.")

        self.embed_dim = attention.embed_dim
        self.num_heads = attention.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = attention.dropout

        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        **kwargs,
    ):
        seq_len, batch_size, _ = query.shape

        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        output = output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.embed_dim)

        return self.out_proj(output), None


def checkpoint_score(path: Path) -> float:
    match = re.search(r"validation_average_jet_accuracy=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else -float("inf")


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


def load_model(run_dir: Path, checkpoint: Path | None, event_file: Path | None) -> JetReconstructionModel:
    checkpoint = best_checkpoint(run_dir) if checkpoint is None else checkpoint
    options = Options.load(str(run_dir / "options.json"))
    if event_file is not None:
        options.event_info_file = str(event_file)
    options.event_info_file = resolve_spanet_path(options.event_info_file)
    options.training_file = resolve_spanet_path(options.training_file)
    options.validation_file = resolve_spanet_path(options.validation_file)
    options.testing_file = resolve_spanet_path(options.testing_file)

    model = JetReconstructionModel(options)
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"checkpoint: {checkpoint}")
    return model


def make_exportable_output_mask(max_jets: int):
    def create_output_mask(self: BranchDecoder, output: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        batch_sequence_mask = sequence_mask.transpose(0, 1).contiguous()

        padding_mask_operands = [batch_sequence_mask.squeeze(-1) * 1] * self.degree
        padding_mask = torch.einsum(self.padding_mask_operation, *padding_mask_operands).bool()

        if self.degree != 2:
            raise NotImplementedError("Exportable BranchDecoder output mask currently supports two-daughter branches.")

        indices = torch.arange(max_jets, device=output.device)
        diagonal_mask = (indices[:, None] != indices[None, :]).unsqueeze(0)

        return (padding_mask & diagonal_mask).bool()

    return create_output_mask


def make_model_exportable(model: nn.Module, max_jets: int) -> None:
    for module in model.modules():
        if isinstance(module, GTrXL) and isinstance(module.attention, nn.MultiheadAttention):
            module.attention = ExportableSelfAttention(module.attention)
        elif isinstance(module, BranchDecoder):
            module.create_output_mask = MethodType(make_exportable_output_mask(max_jets), module)


class SPANetHHBBTauTauExport(nn.Module):
    """
    Thin inference wrapper with a simple ColumnFlow-facing input contract.

    Inputs:
        features: float tensor with shape (batch, max_jets, n_features)
        mask: bool tensor with shape (batch, max_jets)

    Outputs:
        tuple(assignments, detections), where each item is a tuple ordered as
        defined in the event file, currently (higgs_bb, vbf).
    """

    def __init__(self, model: JetReconstructionModel):
        super().__init__()
        self.model = model
        self.log_features = tuple(bool(x) for x in model.event_info.log_features("Source"))

    def _apply_input_log_transform(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformed = []
        for i, use_log in enumerate(self.log_features):
            column = features[:, :, i]
            if use_log:
                column = torch.log(column + 1.0) * mask
            transformed.append(column.unsqueeze(-1))
        return torch.cat(transformed, dim=-1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        features = self._apply_input_log_transform(features, mask)
        outputs = self.model.forward((Source(features, mask),))
        assignments = tuple(outputs.assignments)
        detections = tuple(torch.sigmoid(detection) for detection in outputs.detections)
        return assignments, detections


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="SPANet training run directory containing options.json.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .pt2 file.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to export. Defaults to best checkpoint.")
    parser.add_argument("--event-file", type=Path, help="Override event YAML. Useful when exporting older checkpoints.")
    parser.add_argument("--max-jets", type=int, default=16)
    parser.add_argument("--num-features", type=int, help="Number of input features. Defaults to the number implied by the event file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.run_dir, args.checkpoint, args.event_file)
    make_model_exportable(model, args.max_jets)
    wrapper = SPANetHHBBTauTauExport(model).eval()
    num_features = model.event_info.num_features("Source")
    if args.num_features is not None and args.num_features != num_features:
        raise ValueError(
            f"--num-features={args.num_features} does not match event file features={num_features} for Source."
        )
    print(f"export input features: {num_features}")

    features = torch.zeros((2, args.max_jets, num_features), dtype=torch.float32)
    mask = torch.zeros((2, args.max_jets), dtype=torch.bool)
    mask[:, :4] = True

    batch_dim = torch.export.Dim("batch", min=1)
    exported = torch.export.export(
        wrapper,
        (features, mask),
        dynamic_shapes=(
            {0: batch_dim},
            {0: batch_dim},
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, args.output)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
