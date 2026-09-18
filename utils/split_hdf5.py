#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


SPLITS = ("train", "val", "test")


def find_num_events(source: h5py.File) -> int:
    return int(source["INPUTS"]["Source"]["MASK"].shape[0])


def make_split_assignment(
    num_events: int,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")

    sizes = {
        "train": int(round(train * num_events)),
        "val": int(round(val * num_events)),
    }
    sizes["test"] = num_events - sizes["train"] - sizes["val"]

    assignment = np.empty(num_events, dtype=np.uint8)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_events)

    start = 0
    for split_id, split_name in enumerate(SPLITS):
        stop = start + sizes[split_name]
        assignment[indices[start:stop]] = split_id
        start = stop

    return assignment, sizes


def copy_attrs(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def dataset_kwargs(compression: str | None) -> dict:
    if compression is None:
        return {}
    return {"compression": compression, "chunks": True}


def is_event_dataset(dataset: h5py.Dataset, event_count: int) -> bool:
    return bool(dataset.shape) and dataset.shape[0] == event_count


def create_output_dataset(
    source_dataset: h5py.Dataset,
    target_group: h5py.Group,
    name: str,
    shape: tuple[int, ...],
    compression: str | None,
) -> h5py.Dataset:
    target_dataset = target_group.create_dataset(
        name,
        shape=shape,
        dtype=source_dataset.dtype,
        **dataset_kwargs(compression),
    )
    copy_attrs(source_dataset, target_dataset)
    return target_dataset


def copy_non_event_dataset(
    source_dataset: h5py.Dataset,
    target_group: h5py.Group,
    name: str,
    compression: str | None,
) -> None:
    data = source_dataset[()]
    kwargs = dataset_kwargs(compression) if getattr(data, "shape", ()) else {}
    target_dataset = target_group.create_dataset(name, data=data, **kwargs)
    copy_attrs(source_dataset, target_dataset)


def prepare_outputs(
    source_group: h5py.Group,
    target_groups: dict[str, h5py.Group],
    sizes: dict[str, int],
    event_count: int,
    compression: str | None,
    path: str = "",
) -> dict[str, dict[str, h5py.Dataset]]:
    event_datasets: dict[str, dict[str, h5py.Dataset]] = {}
    copy_targets = target_groups.values()
    for target_group in copy_targets:
        copy_attrs(source_group, target_group)

    for key, value in source_group.items():
        item_path = f"{path}/{key}" if path else key
        if isinstance(value, h5py.Group):
            child_targets = {
                split_name: target_groups[split_name].create_group(key)
                for split_name in SPLITS
            }
            event_datasets.update(
                prepare_outputs(value, child_targets, sizes, event_count, compression, item_path)
            )
            continue

        if not is_event_dataset(value, event_count):
            print(f"Preparing non-event dataset {item_path} shape={value.shape}", flush=True)
            for split_name in SPLITS:
                copy_non_event_dataset(value, target_groups[split_name], key, compression)
            continue

        print(f"Preparing event dataset {item_path} source_shape={value.shape}", flush=True)
        targets = {}
        for split_name in SPLITS:
            split_shape = (sizes[split_name], *value.shape[1:])
            targets[split_name] = create_output_dataset(
                value,
                target_groups[split_name],
                key,
                split_shape,
                compression,
            )
        event_datasets[item_path] = {"source": value, **targets}

    return event_datasets


def stream_dataset(
    path: str,
    datasets: dict[str, h5py.Dataset],
    assignment: np.ndarray,
    chunk_size: int,
) -> None:
    source = datasets["source"]
    offsets = {split_name: 0 for split_name in SPLITS}
    event_count = source.shape[0]
    print(f"Streaming {path} shape={source.shape}", flush=True)

    for start in range(0, event_count, chunk_size):
        stop = min(start + chunk_size, event_count)
        data = source[start:stop]
        labels = assignment[start:stop]

        for split_id, split_name in enumerate(SPLITS):
            local_indices = np.nonzero(labels == split_id)[0]
            if local_indices.size == 0:
                continue

            offset = offsets[split_name]
            next_offset = offset + local_indices.size
            datasets[split_name][offset:next_offset] = data[local_indices]
            offsets[split_name] = next_offset

        if start == 0 or stop == event_count or (start // chunk_size) % 10 == 0:
            print(f"  {path}: {stop}/{event_count}", flush=True)


def split_file(
    input_file: Path,
    output_dir: Path,
    prefix: str,
    assignment: np.ndarray,
    sizes: dict[str, int],
    compression: str | None,
    chunk_size: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split_name: output_dir / f"{prefix}{split_name}.h5"
        for split_name in SPLITS
    }

    with h5py.File(input_file, "r") as source:
        event_count = find_num_events(source)
        targets = {
            split_name: h5py.File(output_paths[split_name], "w")
            for split_name in SPLITS
        }
        try:
            for split_name, target in targets.items():
                copy_attrs(source, target)
                target.attrs["source_file"] = str(input_file)
                target.attrs["num_source_events"] = event_count
                target.attrs["num_split_events"] = sizes[split_name]
                target.attrs["split_name"] = split_name

            event_datasets = prepare_outputs(source, targets, sizes, event_count, compression)
            for path, datasets in event_datasets.items():
                stream_dataset(path, datasets, assignment, chunk_size)
        finally:
            for target in targets.values():
                target.close()

    return output_paths


def print_summary(path: Path) -> None:
    with h5py.File(path, "r") as file:
        mask = file["INPUTS"]["Source"]["MASK"]
        b1 = file["TARGETS"]["higgs_bb"]["b1"][:]
        b2 = file["TARGETS"]["higgs_bb"]["b2"][:]
        q1 = file["TARGETS"]["vbf"]["q1"][:]
        q2 = file["TARGETS"]["vbf"]["q2"][:]
        print(path)
        print(f"  events: {mask.shape[0]}")
        print(f"  max_jets: {mask.shape[1]}")
        print(f"  higgs_bb complete: {np.sum((b1 >= 0) & (b2 >= 0))}")
        print(f"  vbf complete: {np.sum((q1 >= 0) & (q2 >= 0))}")
        print(f"  both complete: {np.sum((b1 >= 0) & (b2 >= 0) & (q1 >= 0) & (q2 >= 0))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a SPANet HDF5 file into train/val/test files.")
    parser.add_argument("input", type=Path, help="Input SPANet HDF5 file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train.h5, val.h5, test.h5.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix, e.g. vbf_signal_ -> vbf_signal_train.h5.")
    parser.add_argument("--train", type=float, default=0.8, help="Training fraction.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=12345, help="Random split seed.")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Number of events to stream per chunk.")
    parser.add_argument(
        "--compression",
        default="lzf",
        choices=("gzip", "lzf", "none"),
        help="Output compression. lzf is much faster than gzip; none is fastest but larger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with h5py.File(args.input, "r") as source:
        num_events = find_num_events(source)

    print(f"Input: {args.input}", flush=True)
    print(f"Events: {num_events}", flush=True)
    print(f"Split fractions: train={args.train}, val={args.val}, test={args.test}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Chunk size: {args.chunk_size}", flush=True)
    print(f"Compression: {compression or 'none'}", flush=True)

    assignment, sizes = make_split_assignment(num_events, args.train, args.val, args.test, args.seed)
    for split_name in SPLITS:
        print(f"{split_name}: {sizes[split_name]} events", flush=True)

    output_paths = split_file(
        input_file=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        assignment=assignment,
        sizes=sizes,
        compression=compression,
        chunk_size=args.chunk_size,
    )
    for split_name in SPLITS:
        print_summary(output_paths[split_name])


if __name__ == "__main__":
    main()
