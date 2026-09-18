#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from convert_hh_bbtautau_parquet import (
    EMPTY_TARGET,
    build_dataset,
    expand_inputs,
    load_chunks,
    resolve_source_collection,
    source_jets,
)


def fraction(mask: np.ndarray) -> str:
    total = len(mask)
    passed = int(np.sum(mask))
    value = 0.0 if total == 0 else 100.0 * passed / total
    return f"{passed}/{total} ({value:.2f}%)"


def summarize(dataset: dict) -> None:
    source = dataset["INPUTS"]["Source"]
    targets = dataset["TARGETS"]
    process_type = dataset["CLASSIFICATIONS"]["EVENT"]["process_type"]

    mask = source["MASK"]
    n_jets = np.sum(mask, axis=1)
    b1 = targets["higgs_bb"]["b1"]
    b2 = targets["higgs_bb"]["b2"]
    q1 = targets["vbf"]["q1"]
    q2 = targets["vbf"]["q2"]

    hhb_complete = (b1 >= 0) & (b2 >= 0)
    vbf_complete = (q1 >= 0) & (q2 >= 0)
    all_complete = hhb_complete & vbf_complete
    all_distinct = all_complete & np.array([
        len({int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3])}) == 4
        for indices in np.stack([b1, b2, q1, q2], axis=1)
    ])

    print("Dataset summary")
    print(f"  events: {mask.shape[0]}")
    print(f"  max_jets: {mask.shape[1]}")
    print(f"  mean_jets: {np.mean(n_jets):.2f}")
    print(f"  min/max jets: {int(np.min(n_jets))}/{int(np.max(n_jets))}")
    print(f"  higgs_bb complete: {fraction(hhb_complete)}")
    print(f"  vbf complete: {fraction(vbf_complete)}")
    print(f"  both complete: {fraction(all_complete)}")
    print(f"  all four targets distinct: {fraction(all_distinct)}")

    print("Process types")
    for value, count in zip(*np.unique(process_type, return_counts=True)):
        print(f"  {int(value)}: {int(count)}")

    print("Feature checks")
    for name, values in source.items():
        if name == "MASK":
            continue
        finite = np.isfinite(values[mask])
        print(f"  {name}: finite {fraction(finite)}")


def print_event_dump(chunks: list[tuple[Path, ak.Array]], dataset: dict, n_events: int, source_collection: str) -> None:
    if n_events <= 0:
        return

    offset = 0
    b1 = dataset["TARGETS"]["higgs_bb"]["b1"]
    b2 = dataset["TARGETS"]["higgs_bb"]["b2"]
    q1 = dataset["TARGETS"]["vbf"]["q1"]
    q2 = dataset["TARGETS"]["vbf"]["q2"]

    print("Event dumps")
    printed = 0
    for path, events in chunks:
        collection = resolve_source_collection(events, source_collection)
        jets = source_jets(events, collection)
        for local_idx in range(len(events)):
            global_idx = offset + local_idx
            if printed >= n_events:
                return

            labels = {
                int(b1[global_idx]): "b1",
                int(b2[global_idx]): "b2",
                int(q1[global_idx]): "q1",
                int(q2[global_idx]): "q2",
            }
            labels.pop(EMPTY_TARGET, None)

            print(f"  event {global_idx} from {path.name} ({collection})")
            print(f"    targets: b1={b1[global_idx]}, b2={b2[global_idx]}, q1={q1[global_idx]}, q2={q2[global_idx]}")
            for jet_idx, jet in enumerate(ak.to_list(jets[local_idx])):
                label = labels.get(jet_idx, "")
                btag = jet.get("btagDeepFlavB", float("nan"))
                print(
                    "    "
                    f"{jet_idx:02d} {label:>2} "
                    f"pt={jet['pt']:8.2f} eta={jet['eta']:7.3f} phi={jet['phi']:7.3f} "
                    f"mass={jet['mass']:7.2f} btag={btag:8.4f}"
                )
            printed += 1
        offset += len(events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HH->bbtautau SPANet truth labels from parquet files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input parquet files or directories.")
    parser.add_argument("--source-collection", default="auto", help="Defaults to SPANetJet when available.")
    parser.add_argument("--max-events", type=int, default=5000, help="Limit events for a fast validation pass.")
    parser.add_argument("--max-jets", type=int, default=None, help="Optional source jet padding/clipping.")
    parser.add_argument("--hhb-dr", type=float, default=0.4)
    parser.add_argument("--vbf-truth", choices=("auto", "genjet", "isvbf"), default="genjet")
    parser.add_argument("--lhe-to-gen-dr", type=float, default=0.4)
    parser.add_argument("--gen-to-reco-dr", type=float, default=0.3)
    parser.add_argument("--no-vbf-overlap-cleaning", action="store_true")
    parser.add_argument("--dump-events", type=int, default=3, help="Print this many event-level jet tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    chunks = load_chunks(paths, args.max_events)
    dataset = build_dataset(
        chunks,
        args.max_jets,
        args.source_collection,
        args.hhb_dr,
        args.vbf_truth,
        args.lhe_to_gen_dr,
        args.gen_to_reco_dr,
        not args.no_vbf_overlap_cleaning,
    )
    summarize(dataset)
    print_event_dump(chunks, dataset, args.dump_events, args.source_collection)


if __name__ == "__main__":
    main()
