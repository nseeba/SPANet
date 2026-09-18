#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np

from convert_hh_bbtautau_parquet import (
    EMPTY_TARGET,
    delta_r,
    expand_inputs,
    higgs_bb_truth,
    load_chunks,
    matched_truth_indices,
    nearest_object_match_indices,
    resolve_source_collection,
    source_jets,
    sourcejet_not_in_hhb_mask,
)


FEATURES = ("pt", "eta", "phi", "mass")


def to_numpy(values) -> np.ndarray:
    return ak.to_numpy(ak.fill_none(values, np.nan))


def object_at_indices(objects: ak.Array, indices: np.ndarray) -> ak.Array:
    positions = ak.local_index(objects, axis=1)
    return ak.firsts(objects[positions == indices[:, None]])


def append_object(out: dict[str, list[np.ndarray]], prefix: str, objects: ak.Array, valid: np.ndarray) -> None:
    for feature in FEATURES:
        if feature in objects.fields:
            out[f"{prefix}_{feature}"].append(to_numpy(objects[feature])[valid])
        else:
            out[f"{prefix}_{feature}"].append(np.full(np.sum(valid), np.nan))
    if "btagDeepFlavB" in objects.fields:
        out[f"{prefix}_btagDeepFlavB"].append(to_numpy(objects.btagDeepFlavB)[valid])
    if "pdgId" in objects.fields:
        out[f"{prefix}_pdgId"].append(to_numpy(objects.pdgId)[valid])
    if "status" in objects.fields:
        out[f"{prefix}_status"].append(to_numpy(objects.status)[valid])


def append_delta(out: dict[str, list[np.ndarray]], name: str, first: ak.Array, second: ak.Array, valid: np.ndarray) -> None:
    out[name].append(to_numpy(delta_r(first.eta, first.phi, second.eta, second.phi))[valid])


def collect_hhb(
    events: ak.Array,
    source_collection: str,
    hhb_dr: float,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    reco = source_jets(events, source_collection)
    truth = ak.pad_none(higgs_bb_truth(events), 2, axis=1, clip=True)
    b1, b2 = matched_truth_indices(truth, reco, hhb_dr)

    out: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "gen_pt", "gen_eta", "gen_phi", "gen_mass", "gen_pdgId", "gen_status",
            "reco_pt", "reco_eta", "reco_phi", "reco_mass", "reco_btagDeepFlavB",
            "gen_reco_dr", "reco_index",
        )
    }

    for slot, indices in enumerate((b1, b2)):
        truth_obj = truth[:, slot]
        reco_obj = object_at_indices(reco, indices)
        valid = indices >= 0
        append_object(out, "gen", truth_obj, valid)
        append_object(out, "reco", reco_obj, valid)
        append_delta(out, "gen_reco_dr", truth_obj, reco_obj, valid)
        out["reco_index"].append(indices[valid])

    return {key: np.concatenate(values) if values else np.array([]) for key, values in out.items()}, (b1, b2)


def collect_vbf(
    events: ak.Array,
    source_collection: str,
    hhb_indices: tuple[np.ndarray, np.ndarray],
    lhe_to_gen_dr: float,
    gen_to_reco_dr: float,
    use_overlap_cleaning: bool,
) -> dict[str, np.ndarray]:
    reco = source_jets(events, source_collection)
    gen = events.GenJet
    lhe = events.LHEPart

    lhe_vbf = lhe[(lhe.status == 1) & (abs(lhe.pdgId) <= 6) & (abs(lhe.pdgId) != 5)]
    lhe_vbf = ak.pad_none(lhe_vbf, 2, axis=1, clip=True)
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

    gen_pos = ak.local_index(gen, axis=1)
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
    matched_source_idx = (
        ak.to_numpy(matched_source_idx0).astype(np.int64),
        ak.to_numpy(matched_source_idx1).astype(np.int64),
    )

    out: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "lhe_pt", "lhe_eta", "lhe_phi", "lhe_mass", "lhe_pdgId", "lhe_status",
            "gen_pt", "gen_eta", "gen_phi", "gen_mass",
            "reco_pt", "reco_eta", "reco_phi", "reco_mass", "reco_btagDeepFlavB",
            "lhe_gen_dr", "gen_reco_dr", "reco_index",
        )
    }

    for slot, source_indices in enumerate(matched_source_idx):
        lhe_obj = lhe_vbf[:, slot]
        gen_obj = matched_gen[:, slot]
        reco_obj = object_at_indices(reco, source_indices)
        valid = ak.to_numpy(ok_gen_to_reco)

        append_object(out, "lhe", lhe_obj, valid)
        append_object(out, "gen", gen_obj, valid)
        append_object(out, "reco", reco_obj, valid)
        append_delta(out, "lhe_gen_dr", lhe_obj, gen_obj, valid)
        append_delta(out, "gen_reco_dr", gen_obj, reco_obj, valid)
        out["reco_index"].append(source_indices[valid])

    return {key: np.concatenate(values) if values else np.array([]) for key, values in out.items()}


def collect_matching(
    chunks: list[tuple[Path, ak.Array]],
    source_collection: str,
    hhb_dr: float,
    lhe_to_gen_dr: float,
    gen_to_reco_dr: float,
    use_overlap_cleaning: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    hhb_parts = []
    vbf_parts = []

    for _, events in chunks:
        collection = resolve_source_collection(events, source_collection)
        hhb, hhb_indices = collect_hhb(events, collection, hhb_dr)
        vbf = collect_vbf(events, collection, hhb_indices, lhe_to_gen_dr, gen_to_reco_dr, use_overlap_cleaning)
        hhb_parts.append(hhb)
        vbf_parts.append(vbf)

    def merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate([part[key] for part in parts])
            for key in parts[0]
        }

    return merge(hhb_parts), merge(vbf_parts)


def summarize(hhb: dict[str, np.ndarray], vbf: dict[str, np.ndarray]) -> None:
    print("Matched object summary")
    print(f"  HH b matched objects: {len(hhb['reco_pt'])}")
    print(f"  VBF matched objects: {len(vbf['reco_pt'])}")
    for label, data in (("HH b", hhb), ("VBF", vbf)):
        print(label)
        for key in sorted(k for k in data if k.endswith("_dr")):
            values = data[key]
            print(f"  {key}: mean={np.mean(values):.4f}, median={np.median(values):.4f}, p95={np.percentile(values, 95):.4f}")


def plot_hist(ax, arrays: list[tuple[str, np.ndarray]], bins=60, density=True, log=False) -> None:
    for label, values in arrays:
        values = values[np.isfinite(values)]
        if len(values):
            ax.hist(values, bins=bins, histtype="step", linewidth=1.8, density=density, label=label)
    if log:
        ax.set_yscale("log")
    ax.legend()


def make_plots(hhb: dict[str, np.ndarray], vbf: dict[str, np.ndarray], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = (
        (
            "vbf",
            "VBF truth matching: LHEPart -> GenJet -> SPANetJet",
            vbf,
            (
                ("VBF LHEPart light quark", "lhe"),
                ("VBF matched GenJet", "gen"),
                ("VBF matched SPANetJet", "reco"),
            ),
        ),
        (
            "hhb",
            "HH b truth matching: Higgs GenPart b -> SPANetJet",
            hhb,
            (
                ("HH b GenPart from Higgs", "gen"),
                ("HH b matched SPANetJet", "reco"),
            ),
        ),
    )

    for label, title, data, entries in plot_specs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(title)
        for ax, feature, xlabel in (
            (axes[0, 0], "pt", "pT [GeV]"),
            (axes[0, 1], "eta", "eta"),
            (axes[1, 0], "phi", "phi"),
            (axes[1, 1], "mass", "mass [GeV]"),
        ):
            plot_hist(ax, [(entry_label, data[f"{prefix}_{feature}"]) for entry_label, prefix in entries])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Normalized matched objects")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / f"{label}_matched_object_kinematics.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Angular distances used in truth matching")
    plot_hist(axes[0], [("HH b GenPart -> SPANetJet", hhb["gen_reco_dr"])], bins=50, density=False, log=True)
    axes[0].set_xlabel("DeltaR")
    axes[0].set_ylabel("Matched objects")
    plot_hist(
        axes[1],
        [
            ("VBF LHEPart -> GenJet", vbf["lhe_gen_dr"]),
            ("VBF GenJet -> SPANetJet", vbf["gen_reco_dr"]),
        ],
        bins=50,
        density=False,
        log=True,
    )
    axes[1].set_xlabel("DeltaR")
    axes[1].set_ylabel("Matched objects")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "matching_delta_r.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Matched-object response and eta residuals")
    plot_hist(
        axes[0],
        [
            ("HH b SPANetJet pT / GenPart pT", hhb["reco_pt"] / hhb["gen_pt"]),
            ("VBF GenJet pT / LHEPart pT", vbf["gen_pt"] / vbf["lhe_pt"]),
            ("VBF SPANetJet pT / GenJet pT", vbf["reco_pt"] / vbf["gen_pt"]),
        ],
        bins=np.linspace(0, 2.5, 80),
    )
    axes[0].set_xlabel("pT response")
    axes[0].set_ylabel("Normalized matched objects")
    plot_hist(
        axes[1],
        [
            ("HH b SPANetJet eta - GenPart eta", hhb["reco_eta"] - hhb["gen_eta"]),
            ("VBF GenJet eta - LHEPart eta", vbf["gen_eta"] - vbf["lhe_eta"]),
            ("VBF SPANetJet eta - GenJet eta", vbf["reco_eta"] - vbf["gen_eta"]),
        ],
        bins=np.linspace(-0.5, 0.5, 80),
    )
    axes[1].set_xlabel("eta residual")
    axes[1].set_ylabel("Normalized matched objects")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "matching_response_residuals.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("b-tag scores of matched reconstructed jets")
    plot_hist(
        axes[0],
        [("HH b matched SPANetJet", hhb["reco_btagDeepFlavB"])],
        bins=np.linspace(0, 1, 60),
        density=False,
        log=True,
    )
    axes[0].set_xlabel("btagDeepFlavB")
    axes[0].set_ylabel("Matched objects")
    plot_hist(
        axes[1],
        [("VBF matched SPANetJet", vbf["reco_btagDeepFlavB"])],
        bins=np.linspace(0, 1, 60),
        density=False,
        log=True,
    )
    axes[1].set_xlabel("btagDeepFlavB")
    axes[1].set_ylabel("Matched objects")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "matched_reco_btag.png")
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HH bb and VBF truth matching diagnostics from reduced parquets.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input parquet files or directories.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output plot directory.")
    parser.add_argument("--source-collection", default="SPANetJet")
    parser.add_argument("--max-events", type=int, default=100000, help="Event limit for diagnostics.")
    parser.add_argument("--hhb-dr", type=float, default=0.4)
    parser.add_argument("--lhe-to-gen-dr", type=float, default=0.4)
    parser.add_argument("--gen-to-reco-dr", type=float, default=0.3)
    parser.add_argument("--no-vbf-overlap-cleaning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    chunks = load_chunks(paths, args.max_events)
    hhb, vbf = collect_matching(
        chunks,
        args.source_collection,
        args.hhb_dr,
        args.lhe_to_gen_dr,
        args.gen_to_reco_dr,
        not args.no_vbf_overlap_cleaning,
    )
    summarize(hhb, vbf)
    make_plots(hhb, vbf, args.output_dir)
    print(f"wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
