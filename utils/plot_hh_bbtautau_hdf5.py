#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


EMPTY_TARGET = -1
TARGETS = ("b1", "b2", "q1", "q2")


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


def load_hdf5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {
            "mask": h5["INPUTS/Source/MASK"][:].astype(bool),
            "pt": h5["INPUTS/Source/pt"][:],
            "eta": h5["INPUTS/Source/eta"][:],
            "phi": h5["INPUTS/Source/phi"][:],
            "mass": h5["INPUTS/Source/mass"][:],
            "btag": h5["INPUTS/Source/btagDeepFlavB"][:],
            "b1": h5["TARGETS/higgs_bb/b1"][:],
            "b2": h5["TARGETS/higgs_bb/b2"][:],
            "q1": h5["TARGETS/vbf/q1"][:],
            "q2": h5["TARGETS/vbf/q2"][:],
            "process_type": h5["CLASSIFICATIONS/EVENT/process_type"][:],
        }


def valid_pair(data: dict[str, np.ndarray], first: str, second: str) -> np.ndarray:
    return (data[first] >= 0) & (data[second] >= 0) & (data[first] != data[second])


def target_values(data: dict[str, np.ndarray], feature: str, target: str) -> np.ndarray:
    indices = data[target]
    valid = indices >= 0
    rows = np.nonzero(valid)[0]
    return data[feature][rows, indices[valid]]


def pair_kinematics(data: dict[str, np.ndarray], first: str, second: str) -> dict[str, np.ndarray]:
    valid = valid_pair(data, first, second)
    rows = np.nonzero(valid)[0]
    i1 = data[first][valid]
    i2 = data[second][valid]

    pt1 = data["pt"][rows, i1]
    pt2 = data["pt"][rows, i2]
    eta1 = data["eta"][rows, i1]
    eta2 = data["eta"][rows, i2]
    phi1 = data["phi"][rows, i1]
    phi2 = data["phi"][rows, i2]
    mass1 = data["mass"][rows, i1]
    mass2 = data["mass"][rows, i2]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(np.maximum(px1**2 + py1**2 + pz1**2 + mass1**2, 0.0))

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(np.maximum(px2**2 + py2**2 + pz2**2 + mass2**2, 0.0))

    mass2_pair = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)

    return {
        "mass": np.sqrt(np.maximum(mass2_pair, 0.0)),
        "deta": deta,
        "abs_deta": np.abs(deta),
        "dphi": dphi,
        "dr": np.sqrt(deta**2 + dphi**2),
        "pt1": pt1,
        "pt2": pt2,
        "eta1": eta1,
        "eta2": eta2,
    }


def summary(data: dict[str, np.ndarray]) -> dict[str, int | float]:
    hhb = valid_pair(data, "b1", "b2")
    vbf = valid_pair(data, "q1", "q2")
    all_complete = hhb & vbf
    all_distinct = all_complete & np.array([
        len({int(data[t][i]) for t in TARGETS}) == 4
        for i in range(len(data["mask"]))
    ])
    n_jets = np.sum(data["mask"], axis=1)
    return {
        "events": len(data["mask"]),
        "max_jets": data["mask"].shape[1],
        "mean_jets": float(np.mean(n_jets)),
        "higgs_bb_complete": int(np.sum(hhb)),
        "vbf_complete": int(np.sum(vbf)),
        "both_complete": int(np.sum(all_complete)),
        "all_four_distinct": int(np.sum(all_distinct)),
    }


def print_summary(data: dict[str, np.ndarray]) -> None:
    info = summary(data)
    events = info["events"]
    print("HDF5 summary")
    for key, value in info.items():
        if key.endswith("complete") or key == "all_four_distinct":
            print(f"  {key}: {value}/{events} ({100.0 * value / events:.2f}%)")
        elif isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def make_plots(data: dict[str, np.ndarray], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    n_jets = np.sum(data["mask"], axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(n_jets, bins=np.arange(0, data["mask"].shape[1] + 2) - 0.5, histtype="step", linewidth=2)
    ax.set_xlabel("Number of SPANetJet objects")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "n_spanetjets.png")
    plt.close(fig)

    counts = [
        np.sum(valid_pair(data, "b1", "b2")),
        np.sum(valid_pair(data, "q1", "q2")),
        np.sum(valid_pair(data, "b1", "b2") & valid_pair(data, "q1", "q2")),
    ]
    labels = ["HH bb", "VBF", "Both"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, np.asarray(counts) / len(data["mask"]))
    ax.set_ylabel("Fraction of events")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "target_completeness.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, target in zip(axes.ravel(), TARGETS):
        values = data[target][data[target] >= 0]
        ax.hist(values, bins=np.arange(-0.5, data["mask"].shape[1] + 0.5), histtype="step", linewidth=2)
        ax.set_title(target)
        ax.set_xlabel("Target jet index")
        ax.set_ylabel("Targets")
    fig.tight_layout()
    fig.savefig(output_dir / "target_indices.png")
    plt.close(fig)

    for feature, xlabel in (("pt", "Jet pT"), ("eta", "Jet eta"), ("btag", "Jet btagDeepFlavB")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for target in TARGETS:
            values = target_values(data, feature, target)
            if len(values):
                ax.hist(values, bins=50, histtype="step", linewidth=1.8, density=True, label=target)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized targets")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"target_{feature}.png")
        plt.close(fig)

    for name, first, second in (("higgs_bb", "b1", "b2"), ("vbf", "q1", "q2")):
        kin = pair_kinematics(data, first, second)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        for ax, key, xlabel in (
            (axes[0], "mass", "Pair mass"),
            (axes[1], "abs_deta", "|Delta eta|"),
            (axes[2], "dr", "Delta R"),
            (axes[3], "pt1", "Leading target pT"),
        ):
            ax.hist(kin[key], bins=60, histtype="step", linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Pairs")
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_pair_kinematics.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and summarize HH bbtautau SPANet HDF5 labels.")
    parser.add_argument("input", type=Path, help="SPANet HDF5 file.")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for plots.")
    parser.add_argument("--no-plots", action="store_true", help="Only print the summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_hdf5(args.input)
    print_summary(data)
    if not args.no_plots:
        make_plots(data, args.output_dir)
        print(f"wrote plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
