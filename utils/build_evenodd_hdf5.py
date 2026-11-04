#!/usr/bin/env python3
"""Build even/odd SPANet training files from per-sample HDF5 files written by convert_hh_bbtautau_parquet.py.

Two models are trained, one per event-number parity, and each is evaluated on the events it never saw:

    even model: train/val = even event numbers, test = all odd event numbers
    odd model:  train/val = odd event numbers,  test = all even event numbers

The inputs are read one file at a time and streamed into the six output files, so memory use stays at the
size of one input file. Every input must contain METADATA/event and CLASSIFICATIONS/EVENT/process_type.

Example:
    python utils/build_evenodd_hdf5.py /local/.../v2_evenodd/parts --output-dir /local/.../v2_evenodd --prefix vbf_ggf_
    -> vbf_ggf_even_train.h5, vbf_ggf_even_val.h5, vbf_ggf_even_test.h5, vbf_ggf_odd_train.h5, ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


MODELS = ("even", "odd")
SPLITS = ("train", "val", "test")
OUTPUTS = tuple((model, split) for model in MODELS for split in SPLITS)
PROCESS_NAMES = {0: "ggf", 1: "vbf", -1: "background"}


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        paths.extend(sorted(item.rglob("*.h5")) if item.is_dir() else [item])
    if not paths:
        raise FileNotFoundError("No input HDF5 files found.")
    return paths


def event_datasets(group: h5py.Group, prefix: str = "") -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """All datasets below group, as path -> (shape without the event axis, dtype)."""
    out = {}
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            out.update(event_datasets(value, path))
        else:
            out[path] = (value.shape[1:], value.dtype)
    return out


def parse_keep_fractions(items: list[str]) -> dict[int, float]:
    fractions = {}
    for item in items:
        name, value = item.split("=")
        codes = [code for code, process in PROCESS_NAMES.items() if process == name]
        if not codes:
            raise ValueError(f"Unknown process {name!r} in --keep-fraction, use one of {sorted(PROCESS_NAMES.values())}")
        fractions[codes[0]] = float(value)
    return fractions


def assign(path: Path, rng: np.random.RandomState, val_fraction: float, keep: dict[int, float]) -> dict:
    """Event indices of one input file for each of the six outputs."""
    with h5py.File(path, "r") as f:
        event = f["METADATA/event"][:]
        process = f["CLASSIFICATIONS/EVENT/process_type"][:]

    kept = np.ones(len(event), dtype=bool)
    for code, fraction in keep.items():
        selected = process == code
        kept[selected] = rng.uniform(size=int(selected.sum())) < fraction

    is_even = (event % 2) == 0
    is_val = rng.uniform(size=len(event)) < val_fraction
    indices = {}
    for model in MODELS:
        own = is_even if model == "even" else ~is_even
        indices[(model, "train")] = np.nonzero(kept & own & ~is_val)[0]
        indices[(model, "val")] = np.nonzero(kept & own & is_val)[0]
        indices[(model, "test")] = np.nonzero(kept & ~own)[0]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Per-sample HDF5 files or directories (searched recursively).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="", help="Output name prefix, e.g. vbf_ggf_ -> vbf_ggf_even_train.h5")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of each model's own-parity events used for validation.")
    parser.add_argument("--keep-fraction", nargs="*", default=[], help="Random subsampling per process, e.g. ggf=0.5")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--compression", default="lzf", choices=("gzip", "lzf", "none"))
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    paths = expand_inputs(args.inputs)
    keep = parse_keep_fractions(args.keep_fraction)
    rng = np.random.RandomState(args.seed)

    # pass 1: check that all files share one structure and assign every event
    with h5py.File(paths[0], "r") as f:
        structure = event_datasets(f)
    for required in ("METADATA/event", "CLASSIFICATIONS/EVENT/process_type", "INPUTS/Source/MASK"):
        if required not in structure:
            raise KeyError(f"{required} missing in {paths[0]}; recreate the per-sample files with the current converter.")

    assignments = []
    for path in paths:
        with h5py.File(path, "r") as f:
            if event_datasets(f) != structure:
                raise ValueError(f"{path} has a different structure than {paths[0]} (features, max jets or dtypes differ).")
        assignments.append(assign(path, rng, args.val_fraction, keep))
    sizes = {out: sum(len(a[out]) for a in assignments) for out in OUTPUTS}

    # create the six outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        for model, split in OUTPUTS:
            out_path = args.output_dir / f"{args.prefix}{model}_{split}.h5"
            out = h5py.File(out_path, "w")
            out.attrs["model_parity"] = model
            out.attrs["split"] = split
            out.attrs["description"] = (
                f"{model}-parity model, {split}: "
                + ("events with the other parity" if split == "test" else f"{model} event numbers")
            )
            for name, (shape, dtype) in structure.items():
                out.create_dataset(name, shape=(sizes[(model, split)], *shape), dtype=dtype, compression=compression)
            files[(model, split)] = out
            print(f"{out_path}: {sizes[(model, split)]} events", flush=True)

        # pass 2: stream every input file into the outputs
        offsets = {out: 0 for out in OUTPUTS}
        for path, indices in zip(paths, assignments):
            with h5py.File(path, "r") as f:
                counts = {out: len(indices[out]) for out in OUTPUTS}
                for name in structure:
                    data = f[name][:]
                    for out in OUTPUTS:
                        if counts[out]:
                            files[out][name][offsets[out]:offsets[out] + counts[out]] = data[indices[out]]
            for out in OUTPUTS:
                offsets[out] += counts[out]
            print(f"  {path}: " + ", ".join(f"{m}_{s} {counts[(m, s)]}" for m, s in OUTPUTS), flush=True)
    finally:
        for out in files.values():
            out.close()

    # summary
    for model, split in OUTPUTS:
        with h5py.File(args.output_dir / f"{args.prefix}{model}_{split}.h5", "r") as f:
            event = f["METADATA/event"][:]
            process = f["CLASSIFICATIONS/EVENT/process_type"][:]
            b1, q1 = f["TARGETS/higgs_bb/b1"][:], f["TARGETS/vbf/q1"][:]
            parities = sorted({"even" if e % 2 == 0 else "odd" for e in np.unique(event % 2)})
            per_process = {PROCESS_NAMES.get(int(c), str(c)): int(n) for c, n in zip(*np.unique(process, return_counts=True))}
            print(f"{model}_{split}: {len(event)} events, parities {parities}, per process {per_process}, "
                  f"HH truth {np.mean(b1 >= 0):.3f}, VBF truth {np.mean(q1 >= 0):.3f}")


if __name__ == "__main__":
    main()
