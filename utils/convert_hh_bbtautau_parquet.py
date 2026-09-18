#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import pyarrow.parquet as pq


FEATURES = (
    "mass",
    "pt",
    "eta",
    "phi",
    "btagDeepFlavB",
    "btagDeepFlavCvB",
    "btagDeepFlavCvL",
    "btagDeepFlavQG",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetCvNotB",
    "btagPNetQvG",
    "btagPNetTauVJet",
    "btagUParTAK4B",
    "btagUParTAK4CvB",
    "btagUParTAK4CvL",
    "btagUParTAK4CvNotB",
    "btagUParTAK4QvG",
    "btagUParTAK4TauVJet",
)
OPTIONAL_SOURCE_COLUMNS = ("vbfjtag", "hhbtag")
EMPTY_TARGET = -1
DEFAULT_COLUMNS = [
    "SPANetJet.mass",
    "SPANetJet.pt",
    "SPANetJet.eta",
    "SPANetJet.phi",
    "SPANetJet.btagDeepFlavB",
    "SPANetJet.btagDeepFlavCvB",
    "SPANetJet.btagDeepFlavCvL",
    "SPANetJet.btagDeepFlavQG",
    "SPANetJet.btagPNetB",
    "SPANetJet.btagPNetCvB",
    "SPANetJet.btagPNetCvL",
    "SPANetJet.btagPNetCvNotB",
    "SPANetJet.btagPNetQvG",
    "SPANetJet.btagPNetTauVJet",
    "SPANetJet.btagUParTAK4B",
    "SPANetJet.btagUParTAK4CvB",
    "SPANetJet.btagUParTAK4CvL",
    "SPANetJet.btagUParTAK4CvNotB",
    "SPANetJet.btagUParTAK4QvG",
    "SPANetJet.btagUParTAK4TauVJet",
    "SPANetJet.vbfjtag",
    "SPANetJet.hhbtag",
    "GenPart.pt",
    "GenPart.eta",
    "GenPart.phi",
    "GenPart.mass",
    "GenPart.pdgId",
    "GenPart.status",
    "GenPart.genPartIdxMother",
    "GenJet.pt",
    "GenJet.eta",
    "GenJet.phi",
    "GenJet.mass",
    "LHEPart.status",
    "LHEPart.pdgId",
    "LHEPart.pt",
    "LHEPart.eta",
    "LHEPart.phi",
    "LHEPart.mass",
    "process_id",
]


def delta_r(a_eta, a_phi, b_eta, b_phi):
    dphi = (a_phi - b_phi + np.pi) % (2 * np.pi) - np.pi
    deta = a_eta - b_eta
    return np.sqrt(deta**2 + dphi**2)


def read_events(path: Path) -> ak.Array:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [
        column
        for column in DEFAULT_COLUMNS
        if column in available or column.split(".", 1)[0] in available
    ]
    return ak.from_parquet(path, columns=columns)


def padded_feature(jets: ak.Array, name: str, max_jets: int) -> np.ndarray:
    values = ak.pad_none(jets[name], max_jets, axis=1, clip=True)
    return ak.to_numpy(ak.fill_none(values, 0.0)).astype(np.float32)


def source_mask(jets: ak.Array, max_jets: int) -> np.ndarray:
    counts = ak.num(jets, axis=1)
    positions = ak.local_index(ak.pad_none(jets.pt, max_jets, axis=1, clip=True), axis=1)
    return ak.to_numpy(positions < counts[:, None]).astype(bool)


def first_two_mask_indices(mask: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    positions = ak.local_index(mask, axis=1)
    matched = positions[mask]
    padded = ak.pad_none(matched, 2, axis=1, clip=True)
    indices = ak.to_numpy(ak.fill_none(padded, EMPTY_TARGET)).astype(np.int64)
    complete = ak.to_numpy(ak.num(matched, axis=1) >= 2)
    indices[~complete] = EMPTY_TARGET
    return indices[:, 0], indices[:, 1]


def nearest_object_match_indices(targets: ak.Array, refs: ak.Array) -> tuple[ak.Array, ak.Array]:
    dr = delta_r(
        targets.eta[:, :, None],
        targets.phi[:, :, None],
        refs.eta[:, None, :],
        refs.phi[:, None, :],
    )
    idx = ak.fill_none(ak.argmin(dr, axis=2, mask_identity=True), EMPTY_TARGET)
    min_dr = ak.fill_none(ak.min(dr, axis=2, mask_identity=True), 999.0)
    return idx, min_dr


def sourcejet_not_in_hhb_mask(reco: ak.Array, b1: np.ndarray, b2: np.ndarray) -> ak.Array:
    positions = ak.local_index(reco, axis=1)
    return (positions != b1[:, None]) & (positions != b2[:, None])


def has_truth2_inputs(events: ak.Array) -> bool:
    return all(field in events.fields for field in ("LHEPart", "GenJet"))


def resolve_source_collection(events: ak.Array, requested: str) -> str:
    if requested != "auto":
        if requested not in events.fields:
            raise ValueError(f"Requested source collection {requested!r}, but it is missing in the input parquet.")
        return requested

    if "SPANetJet" in events.fields:
        return "SPANetJet"

    raise ValueError("Input parquet must contain SPANetJet.")


def source_jets(events: ak.Array, source_collection: str) -> ak.Array:
    return events[source_collection]


def vbf_targets_from_genjets(
    events: ak.Array,
    source_collection: str,
    hhb_indices: tuple[np.ndarray, np.ndarray],
    lhe_to_gen_dr: float,
    gen_to_reco_dr: float,
    use_overlap_cleaning: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reco = source_jets(events, source_collection)
    gen = events.GenJet
    lhe = events.LHEPart

    lhe_vbf = lhe[(lhe.status == 1) & (abs(lhe.pdgId) <= 6) & (abs(lhe.pdgId) != 5)]
    has_two_lhe = ak.num(lhe_vbf, axis=1) >= 2

    reco_mask = (reco.pt > 20.0) & (abs(reco.eta) < 4.7)
    if use_overlap_cleaning:
        reco_mask = reco_mask & sourcejet_not_in_hhb_mask(reco, *hhb_indices)
    reco_clean = reco[reco_mask]
    reco_source_pos = ak.local_index(reco, axis=1)[reco_mask]

    lhe_to_gen_idx, lhe_to_gen_min_dr = nearest_object_match_indices(lhe_vbf, gen)
    ok_lhe_to_gen = (
        has_two_lhe
        & (lhe_to_gen_min_dr[:, 0] < lhe_to_gen_dr)
        & (lhe_to_gen_min_dr[:, 1] < lhe_to_gen_dr)
        & (lhe_to_gen_idx[:, 0] != lhe_to_gen_idx[:, 1])
    )

    gen_pos = ak.local_index(gen)
    matched_gen0 = ak.firsts(gen[gen_pos == lhe_to_gen_idx[:, 0]])
    matched_gen1 = ak.firsts(gen[gen_pos == lhe_to_gen_idx[:, 1]])
    matched_gen = ak.concatenate([ak.singletons(matched_gen0), ak.singletons(matched_gen1)], axis=1)

    gen_to_reco_idx, gen_to_reco_min_dr = nearest_object_match_indices(matched_gen, reco_clean)
    ok_gen_to_reco = (
        ok_lhe_to_gen
        & (gen_to_reco_min_dr[:, 0] < gen_to_reco_dr)
        & (gen_to_reco_min_dr[:, 1] < gen_to_reco_dr)
        & (gen_to_reco_idx[:, 0] != gen_to_reco_idx[:, 1])
    )

    reco_clean_pos = ak.local_index(reco_clean, axis=1)
    matched_source_idx0 = ak.fill_none(
        ak.firsts(reco_source_pos[reco_clean_pos == gen_to_reco_idx[:, 0]]),
        EMPTY_TARGET,
    )
    matched_source_idx1 = ak.fill_none(
        ak.firsts(reco_source_pos[reco_clean_pos == gen_to_reco_idx[:, 1]]),
        EMPTY_TARGET,
    )

    indices = np.column_stack([
        ak.to_numpy(matched_source_idx0).astype(np.int64),
        ak.to_numpy(matched_source_idx1).astype(np.int64),
    ])
    valid = ak.to_numpy(ok_gen_to_reco)
    indices[~valid] = EMPTY_TARGET

    return indices[:, 0], indices[:, 1]


def vbf_targets(
    events: ak.Array,
    source_collection: str,
    truth_mode: str,
    hhb_indices: tuple[np.ndarray, np.ndarray],
    lhe_to_gen_dr: float,
    gen_to_reco_dr: float,
    use_overlap_cleaning: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if has_truth2_inputs(events):
        return vbf_targets_from_genjets(
            events,
            source_collection,
            hhb_indices,
            lhe_to_gen_dr,
            gen_to_reco_dr,
            use_overlap_cleaning,
        )

    raise ValueError("VBF genjet truth requires LHEPart and GenJet in the input parquet.")


def higgs_bb_truth(events: ak.Array) -> ak.Array:
    if "GenPart" not in events.fields:
        raise ValueError("Higgs bb truth requires GenPart in the input parquet.")

    genpart = events.GenPart
    parent_idx = genpart.genPartIdxMother
    valid_parent = parent_idx >= 0
    safe_parent_idx = ak.where(valid_parent, parent_idx, 0)
    parent_pdgid = ak.where(valid_parent, genpart.pdgId[safe_parent_idx], 0)
    direct_higgs_b = genpart[(abs(genpart.pdgId) == 5) & (abs(parent_pdgid) == 25)]
    status23_higgs_b = direct_higgs_b[direct_higgs_b.status == 23]
    use_status23 = ak.num(status23_higgs_b, axis=1) >= 2
    b_children = ak.where(use_status23, status23_higgs_b[:, :2], direct_higgs_b[:, :2])
    return b_children[:, :2]


def matched_truth_indices(truth: ak.Array, reco: ak.Array, dr_max: float) -> tuple[np.ndarray, np.ndarray]:
    n_events = len(reco)
    missing = np.full((n_events, 2), EMPTY_TARGET, dtype=np.int64)

    has_two_truth = ak.num(truth, axis=1) >= 2
    has_two_reco = ak.num(reco, axis=1) >= 2
    if not bool(ak.any(has_two_truth & has_two_reco)):
        return missing[:, 0], missing[:, 1]

    truth = ak.pad_none(truth, 2, axis=1, clip=True)
    dr = delta_r(
        truth.eta[:, :, None],
        truth.phi[:, :, None],
        reco.eta[:, None, :],
        reco.phi[:, None, :],
    )
    idx = ak.fill_none(ak.argmin(dr, axis=2, mask_identity=True), EMPTY_TARGET)
    min_dr = ak.fill_none(ak.min(dr, axis=2, mask_identity=True), 999.0)

    idx_np = ak.to_numpy(idx).astype(np.int64)
    min_dr_np = ak.to_numpy(min_dr)
    valid = (
        ak.to_numpy(has_two_truth & has_two_reco)
        & (min_dr_np[:, 0] < dr_max)
        & (min_dr_np[:, 1] < dr_max)
        & (idx_np[:, 0] != idx_np[:, 1])
    )
    idx_np[~valid] = EMPTY_TARGET
    return idx_np[:, 0], idx_np[:, 1]


def infer_process_type(path: Path, events: ak.Array) -> np.ndarray:
    text = str(path).lower()
    if "hh_vbf" in text or "vbf" in text:
        value = 1
    elif "hh_ggf" in text or "gghh" in text or "ggf" in text:
        value = 0
    else:
        value = -1
    return np.full(len(events), value, dtype=np.int64)


def load_chunks(paths: list[Path], max_events: int | None) -> list[tuple[Path, ak.Array]]:
    chunks: list[tuple[Path, ak.Array]] = []
    remaining = max_events

    for path in paths:
        events = read_events(path)
        if remaining is not None:
            if remaining <= 0:
                break
            events = events[:remaining]
            remaining -= len(events)
        if len(events):
            chunks.append((path, events))

    return chunks


def build_dataset(
    chunks: list[tuple[Path, ak.Array]],
    max_jets: int | None,
    source_collection: str,
    extra_features: tuple[str, ...],
    hhb_dr: float,
    vbf_truth: str,
    lhe_to_gen_dr: float,
    gen_to_reco_dr: float,
    use_overlap_cleaning: bool,
) -> dict:
    if not chunks:
        raise ValueError("No events were loaded from the requested parquet files.")

    source_collections = [resolve_source_collection(events, source_collection) for _, events in chunks]
    source_features = tuple(dict.fromkeys((*FEATURES, *extra_features)))
    inferred_max_jets = max(
        int(ak.max(ak.num(source_jets(events, collection), axis=1)))
        for (_, events), collection in zip(chunks, source_collections)
    )
    max_jets = max_jets or inferred_max_jets

    arrays: dict[str, list[np.ndarray]] = {
        "MASK": [],
        **{feature: [] for feature in source_features},
        "b1": [],
        "b2": [],
        "q1": [],
        "q2": [],
        "process_type": [],
    }

    for (path, events), collection in zip(chunks, source_collections):
        jets = source_jets(events, collection)
        arrays["MASK"].append(source_mask(jets, max_jets))
        missing = [feature for feature in source_features if feature not in jets.fields]
        if missing:
            raise ValueError(
                f"Requested source feature(s) {missing} missing from {collection} in {path}. "
                f"Available fields: {jets.fields}"
            )
        for feature in source_features:
            arrays[feature].append(padded_feature(jets, feature, max_jets))

        b1, b2 = matched_truth_indices(higgs_bb_truth(events), jets, hhb_dr)
        q1, q2 = vbf_targets(
            events,
            collection,
            vbf_truth,
            (b1, b2),
            lhe_to_gen_dr,
            gen_to_reco_dr,
            use_overlap_cleaning,
        )
        arrays["b1"].append(b1)
        arrays["b2"].append(b2)
        arrays["q1"].append(q1)
        arrays["q2"].append(q2)
        arrays["process_type"].append(infer_process_type(path, events))

    return {
        "INPUTS": {
            "Source": {
                "MASK": np.concatenate(arrays["MASK"], axis=0),
                **{feature: np.concatenate(arrays[feature], axis=0) for feature in source_features},
            },
        },
        "TARGETS": {
            "higgs_bb": {
                "b1": np.concatenate(arrays["b1"], axis=0),
                "b2": np.concatenate(arrays["b2"], axis=0),
            },
            "vbf": {
                "q1": np.concatenate(arrays["q1"], axis=0),
                "q2": np.concatenate(arrays["q2"], axis=0),
            },
        },
        "CLASSIFICATIONS": {
            "EVENT": {
                "process_type": np.concatenate(arrays["process_type"], axis=0),
            },
        },
    }


def write_group(group: h5py.Group, data: dict, compression: str | None) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            write_group(group.create_group(key), value, compression)
        else:
            group.create_dataset(key, data=value, compression=compression)


def print_summary(dataset: dict) -> None:
    mask = dataset["INPUTS"]["Source"]["MASK"]
    b1 = dataset["TARGETS"]["higgs_bb"]["b1"]
    b2 = dataset["TARGETS"]["higgs_bb"]["b2"]
    q1 = dataset["TARGETS"]["vbf"]["q1"]
    q2 = dataset["TARGETS"]["vbf"]["q2"]
    print(f"events: {mask.shape[0]}")
    print(f"max_jets: {mask.shape[1]}")
    print(f"higgs_bb complete: {np.sum((b1 >= 0) & (b2 >= 0))}")
    print(f"vbf complete: {np.sum((q1 >= 0) & (q2 >= 0))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HH->bbtautau parquet files to SPANet HDF5.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input parquet files or directories.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output HDF5 file.")
    parser.add_argument(
        "--source-collection",
        default="auto",
        help="Reco jet collection to use as SPANet Source. Defaults to SPANetJet.",
    )
    parser.add_argument("--max-jets", type=int, default=None, help="Pad or clip the source jet collection to this length.")
    parser.add_argument(
        "--extra-features",
        nargs="*",
        default=(),
        choices=OPTIONAL_SOURCE_COLUMNS,
        help="Additional source jet fields to copy into INPUTS/Source for diagnostics/baselines. "
             "They do not affect training unless they are also added to the SPANet event YAML.",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional event limit for tests.")
    parser.add_argument("--hhb-dr", type=float, default=0.4, help="Maximum dR for Higgs b -> source jet matching.")
    parser.add_argument(
        "--vbf-truth",
        choices=("auto", "genjet"),
        default="genjet",
        help="VBF target definition. Uses LHEPart->GenJet->source jet truth2.",
    )
    parser.add_argument("--lhe-to-gen-dr", type=float, default=0.4, help="Maximum dR for LHE VBF parton -> GenJet matching.")
    parser.add_argument("--gen-to-reco-dr", type=float, default=0.3, help="Maximum dR for GenJet -> source jet matching.")
    parser.add_argument(
        "--no-vbf-overlap-cleaning",
        action="store_true",
        help="Do not remove truth-matched Higgs bb source jets before GenJet -> source jet matching.",
    )
    parser.add_argument("--compression", default="gzip", choices=("gzip", "lzf", "none"), help="HDF5 compression.")
    return parser.parse_args()


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.parquet")))
        else:
            paths.append(item)
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise FileNotFoundError("No input parquet files found.")
    return paths


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    chunks = load_chunks(paths, args.max_events)
    dataset = build_dataset(
        chunks,
        args.max_jets,
        args.source_collection,
        tuple(args.extra_features),
        args.hhb_dr,
        args.vbf_truth,
        args.lhe_to_gen_dr,
        args.gen_to_reco_dr,
        not args.no_vbf_overlap_cleaning,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    with h5py.File(args.output, "w") as output:
        write_group(output, dataset, compression)

    print_summary(dataset)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
