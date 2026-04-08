import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SCENARIOS: Tuple[Tuple[str, str], ...] = (
    ("caseA1", "dataset/caseA1.npz"),
    ("v2g", "dataset/real_v2g.npz"),
    ("g2v", "dataset/real_g2v.npz"),
    ("g2b", "dataset/real_g2b.npz"),
    ("b2g", "dataset/real_b2g.npz"),
)
HARMONIC_ORDERS: Tuple[str, ...] = ("1", "3", "5", "7")


def load_stats(base_dir: str, scenarios: List[Tuple[str, str]]) -> Dict:
    names: List[str] = []
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    counts: List[int] = []
    for name, rel_path in scenarios:
        npz_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing npz for scenario {name}: {npz_path}")
        data = np.load(npz_path)
        if "labels" not in data:
            raise KeyError(f"labels not found in {npz_path}")
        labels = data["labels"].astype(np.float64)
        if labels.ndim != 2 or labels.shape[1] != 4:
            raise ValueError(f"labels must be [N,4], got {labels.shape} for {npz_path}")
        names.append(name)
        means.append(labels.mean(axis=0))
        stds.append(labels.std(axis=0))
        counts.append(int(labels.shape[0]))
    return {
        "names": names,
        "means": np.vstack(means),
        "stds": np.vstack(stds),
        "counts": counts,
    }


def make_grouped_bar(
    names: List[str],
    means: np.ndarray,
    stds: np.ndarray,
    out_png: str,
    yscale: str,
    show_std: bool,
):
    n_s = len(names)
    x = np.arange(len(HARMONIC_ORDERS), dtype=np.float64)
    total_w = 0.84
    bar_w = total_w / max(n_s, 1)
    offsets = (np.arange(n_s, dtype=np.float64) - (n_s - 1) / 2.0) * bar_w
    colors = list(plt.cm.tab10.colors)

    plt.figure(figsize=(10, 5))
    for i, name in enumerate(names):
        y = means[i]
        yerr = stds[i] if show_std else None
        if yscale == "log":
            y = np.maximum(y, 1e-8)
            if yerr is not None:
                yerr = np.minimum(yerr, y * 0.95)
        plt.bar(
            x + offsets[i],
            y,
            width=bar_w * 0.92,
            label=name,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.25,
            yerr=yerr,
            capsize=2.0 if yerr is not None else 0.0,
        )

    if yscale == "log":
        plt.yscale("log")
    plt.xticks(x, HARMONIC_ORDERS)
    plt.xlabel("Harmonic Order")
    plt.ylabel("Amplitude")
    plt.title(f"Harmonic Amplitude by Scenario ({yscale})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=min(5, n_s), fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_stats_csv(names: List[str], means: np.ndarray, stds: np.ndarray, counts: List[int], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "samples",
                "mean_A1",
                "mean_A3",
                "mean_A5",
                "mean_A7",
                "std_A1",
                "std_A3",
                "std_A5",
                "std_A7",
            ]
        )
        for i, name in enumerate(names):
            writer.writerow(
                [
                    name,
                    counts[i],
                    float(means[i, 0]),
                    float(means[i, 1]),
                    float(means[i, 2]),
                    float(means[i, 3]),
                    float(stds[i, 0]),
                    float(stds[i, 1]),
                    float(stds[i, 2]),
                    float(stds[i, 3]),
                ]
            )


def save_stats_json(names: List[str], means: np.ndarray, stds: np.ndarray, counts: List[int], out_json: str):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "scenario": name,
                "samples": int(counts[i]),
                "mean": {
                    "A1": float(means[i, 0]),
                    "A3": float(means[i, 1]),
                    "A5": float(means[i, 2]),
                    "A7": float(means[i, 3]),
                },
                "std": {
                    "A1": float(stds[i, 0]),
                    "A3": float(stds[i, 1]),
                    "A5": float(stds[i, 2]),
                    "A7": float(stds[i, 3]),
                },
            }
        )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Draw grouped harmonic amplitude bars for 5 scenarios (1/3/5/7)."
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="output_compare_sameinit_pinv_analytic_vs_bp")
    parser.add_argument("--yscale", type=str, default="both", choices=["linear", "log", "both"])
    parser.add_argument("--show_std", action="store_true", help="Show std error bars.")
    args = parser.parse_args()

    scenarios = list(DEFAULT_SCENARIOS)
    stats = load_stats(args.base_dir, scenarios=scenarios)
    names = stats["names"]
    means = stats["means"]
    stds = stats["stds"]
    counts = stats["counts"]

    out_prefix = os.path.join(args.output_dir, "harmonic_amp_5scenarios")
    if args.yscale in ("linear", "both"):
        make_grouped_bar(
            names=names,
            means=means,
            stds=stds,
            out_png=f"{out_prefix}_linear.png",
            yscale="linear",
            show_std=args.show_std,
        )
    if args.yscale in ("log", "both"):
        make_grouped_bar(
            names=names,
            means=means,
            stds=stds,
            out_png=f"{out_prefix}_log.png",
            yscale="log",
            show_std=args.show_std,
        )

    save_stats_csv(
        names=names,
        means=means,
        stds=stds,
        counts=counts,
        out_csv=f"{out_prefix}_stats.csv",
    )
    save_stats_json(
        names=names,
        means=means,
        stds=stds,
        counts=counts,
        out_json=f"{out_prefix}_stats.json",
    )

    print(f"Saved bar figures and stats under: {args.output_dir}")


if __name__ == "__main__":
    main()
