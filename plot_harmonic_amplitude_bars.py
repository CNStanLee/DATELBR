import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_SCENARIOS: Tuple[Tuple[str, str], ...] = (
    ("caseA1", "dataset/caseA1.npz"),
    ("v2g", "dataset/real_v2g.npz"),
    ("g2v", "dataset/real_g2v.npz"),
    ("g2b", "dataset/real_g2b.npz"),
    ("b2g", "dataset/real_b2g.npz"),
)
HARMONIC_ORDERS: Tuple[int, ...] = (1, 3, 5, 7)
HARMONIC_LABELS: Tuple[str, ...] = tuple(str(x) for x in HARMONIC_ORDERS)
OVERLAY_HARMONIC_ORDERS: Tuple[int, ...] = (3, 5, 7)

SCENARIO_DISPLAY_LABELS: Dict[str, str] = {
    "caseA1": "Case A1",
    "v2g": "V2G",
    "g2v": "G2V",
    "g2b": "G2B",
    "b2g": "B2G",
}

METHOD_KEYS: Tuple[str, ...] = (
    "single_cycle_fft",
    "pre_finetune",
    "bp_finetuned",
    "closed_form_finetuned",
)
METHOD_LABELS: Dict[str, str] = {
    "single_cycle_fft": "1-cycle FFT",
    "pre_finetune": "Pre-finetune",
    "bp_finetuned": "BP finetuned",
    "closed_form_finetuned": "Closed-form finetuned",
}
METHOD_COLORS: Tuple[str, ...] = ("tab:cyan", "tab:orange", "tab:blue", "tab:pink")


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
    plt.xticks(x, HARMONIC_LABELS)
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


def split_by_group_ids(
    group_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(split_seed)
    groups_shuffled = unique_groups.copy()
    rng.shuffle(groups_shuffled)

    n_groups = len(groups_shuffled)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    if n_groups >= 1 and n_train == 0:
        n_train = 1
    if n_train + n_val >= n_groups and n_groups >= 2:
        n_val = max(0, n_groups - n_train - 1)

    train_groups = groups_shuffled[:n_train]
    val_groups = groups_shuffled[n_train : n_train + n_val]
    test_groups = groups_shuffled[n_train + n_val :]
    return train_groups, val_groups, test_groups


def resolve_existing_path(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path

    repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    candidates = [
        os.path.join(base_dir, path),
        os.path.join(repo_root, path),
        os.path.join(base_dir, os.path.basename(path)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def safe_float(text: Any, default: float = np.nan) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def read_compare_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing compare summary csv: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_history_json(base_dir: str, output_dir: str, row: Dict[str, str], scenario: str) -> Dict:
    local_hist = os.path.join(output_dir, f"history_compare_{scenario}.json")
    if os.path.exists(local_hist):
        hist_path = local_hist
    else:
        hist_path = resolve_existing_path(base_dir, row.get("history_json", ""))
    if not hist_path or not os.path.exists(hist_path):
        raise FileNotFoundError(
            f"Missing history json for scenario={scenario}, "
            f"tried {local_hist} and {row.get('history_json', '')}"
        )
    with open(hist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_single_cycle_fft_val_mae(
    npz_path: str,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> Tuple[np.ndarray, int]:
    data = np.load(npz_path, allow_pickle=True)
    if "signals" not in data or "labels" not in data:
        raise KeyError(f"npz must contain signals/labels: {npz_path}")

    signals = data["signals"].astype(np.float64)
    labels = data["labels"].astype(np.float64)
    if "group_ids" in data:
        group_ids = data["group_ids"].astype(np.int64)
    else:
        group_ids = np.arange(signals.shape[0], dtype=np.int64)

    if signals.ndim != 2:
        raise ValueError(f"signals must be 2D, got {signals.shape} in {npz_path}")
    if labels.ndim != 2 or labels.shape[1] < len(HARMONIC_ORDERS):
        raise ValueError(f"labels must be [N,{len(HARMONIC_ORDERS)}], got {labels.shape} in {npz_path}")
    if signals.shape[0] != labels.shape[0] or signals.shape[0] != group_ids.shape[0]:
        raise ValueError(
            "mismatched sample dim in "
            f"{npz_path}: signals={signals.shape}, labels={labels.shape}, group_ids={group_ids.shape}"
        )

    _, val_groups, _ = split_by_group_ids(
        group_ids=group_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    val_mask = np.isin(group_ids, val_groups)
    if int(np.sum(val_mask)) == 0:
        raise ValueError(f"empty val split in {npz_path} with split_seed={split_seed}")

    sig = signals[val_mask]
    gt = labels[val_mask, : len(HARMONIC_ORDERS)]
    n = int(sig.shape[1])
    if n <= max(HARMONIC_ORDERS):
        raise ValueError(f"too few samples per cycle={n} for harmonics={HARMONIC_ORDERS} in {npz_path}")

    fft_vals = np.fft.rfft(sig, axis=1)
    bins = np.asarray(HARMONIC_ORDERS, dtype=np.int64)
    if int(np.max(bins)) >= int(fft_vals.shape[1]):
        raise ValueError(f"fft bin out of range in {npz_path}, rfft bins={fft_vals.shape[1]}, need max={int(np.max(bins))}")

    pred = (2.0 / float(n)) * np.abs(fft_vals[:, bins])
    abs_err = np.abs(pred - gt)
    mae = np.mean(abs_err, axis=0)
    return np.asarray(mae, dtype=np.float64), int(abs_err.shape[0])


def collect_method_error_by_scenario(
    base_dir: str,
    output_dir: str,
    scenario_spec: str,
    fallback_scenarios: List[str],
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    compare_csv = os.path.join(output_dir, "compare_summary.csv")
    compare_rows = read_compare_rows(compare_csv)
    compare_rows_ok = [r for r in compare_rows if r.get("status") == "ok"]
    if not compare_rows_ok:
        raise ValueError(f"No status=ok rows found in {compare_csv}")

    if scenario_spec.strip().lower() == "auto":
        scenario_order = [r["scenario"] for r in compare_rows_ok]
        for sc in fallback_scenarios:
            if sc not in scenario_order:
                scenario_order.append(sc)
    else:
        scenario_order = [x.strip() for x in scenario_spec.split(",") if x.strip()]
        if not scenario_order:
            raise ValueError("method_scenarios is empty")

    row_by_scenario = {r["scenario"]: r for r in compare_rows_ok}
    per_scenario: Dict[str, Dict[str, Any]] = {}

    for scenario in scenario_order:
        if scenario in row_by_scenario:
            row = row_by_scenario[scenario]
        elif scenario.lower() == "casea1":
            row = compute_casea1_compare_row_fallback(
                base_dir=base_dir,
                output_dir=output_dir,
                compare_rows_ok=compare_rows_ok,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                split_seed=split_seed,
            )
        else:
            raise KeyError(f"Scenario {scenario} not found in compare_summary.csv status=ok rows")

        npz_from_row = row.get("npz_path", "")
        npz_path = resolve_existing_path(base_dir, npz_from_row)
        if not os.path.exists(npz_path):
            fallback_npz = os.path.join(base_dir, "dataset", f"real_{scenario}.npz")
            if os.path.exists(fallback_npz):
                npz_path = fallback_npz
            else:
                raise FileNotFoundError(
                    f"Missing npz for scenario={scenario}, tried {npz_from_row} and {fallback_npz}"
                )

        fft_mae, fft_val_samples = compute_single_cycle_fft_val_mae(
            npz_path=npz_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
        pre_mae = np.asarray(
            [safe_float(row.get(f"init_val_mae_A{h}")) for h in HARMONIC_ORDERS],
            dtype=np.float64,
        )

        hist = load_history_json(base_dir=base_dir, output_dir=output_dir, row=row, scenario=scenario)
        reg_hist = np.asarray(hist["reg_val_mae_per_epoch_by_channel"], dtype=np.float64)
        bp_hist = np.asarray(hist["bp_val_mae_per_epoch_by_channel"], dtype=np.float64)
        if reg_hist.ndim != 2 or reg_hist.shape[1] < len(HARMONIC_ORDERS):
            raise ValueError(f"Bad reg history shape for {scenario}: {reg_hist.shape}")
        if bp_hist.ndim != 2 or bp_hist.shape[1] < len(HARMONIC_ORDERS):
            raise ValueError(f"Bad bp history shape for {scenario}: {bp_hist.shape}")

        reg_ep = int(round(safe_float(row.get("reg_best_epoch", 0.0), 0.0)))
        bp_ep = int(round(safe_float(row.get("bp_best_epoch", 0.0), 0.0)))
        reg_ep = max(0, min(reg_ep, reg_hist.shape[0] - 1))
        bp_ep = max(0, min(bp_ep, bp_hist.shape[0] - 1))

        reg_best_mae = np.asarray(reg_hist[reg_ep, : len(HARMONIC_ORDERS)], dtype=np.float64)
        bp_best_mae = np.asarray(bp_hist[bp_ep, : len(HARMONIC_ORDERS)], dtype=np.float64)

        per_scenario[scenario] = {
            "single_cycle_fft": fft_mae,
            "pre_finetune": pre_mae,
            "bp_finetuned": bp_best_mae,
            "closed_form_finetuned": reg_best_mae,
            "fft_val_samples": int(fft_val_samples),
            "reg_best_epoch": int(reg_ep),
            "bp_best_epoch": int(bp_ep),
        }

    return scenario_order, per_scenario


def infer_compare_runtime_params(base_dir: str, output_dir: str, compare_rows_ok: List[Dict[str, str]]) -> Dict[str, Any]:
    first = compare_rows_ok[0] if compare_rows_ok else {}
    reg_method = str(first.get("reg_method", "analytic_progressive"))
    bp_lr = safe_float(first.get("bp_lr_used"), 1e-3)
    epochs = 120
    if compare_rows_ok:
        try:
            sc = first["scenario"]
            hist = load_history_json(base_dir=base_dir, output_dir=output_dir, row=first, scenario=sc)
            epochs = max(int(len(hist.get("bp_val_mae_mean_per_epoch", [])) - 1), 1)
        except Exception:
            epochs = 120
    return {
        "reg_method": reg_method,
        "bp_lr": float(bp_lr),
        "epochs": int(epochs),
        "reg_lambda": 1e-2,
        "reg_progressive_min_frac": 0.1,
        "reg_progressive_anchor": "prev",
        "reg_progressive_shuffle_seed": 42,
        "pinv_reg_lambda": 0.0,
        "w_bit": 3,
        "a_bit": 3,
        "seed": 42,
        "label_scale": 100.0,
        "preprocess_mode": "rms",
        "cycle": "half",
        "batch_size": 512,
    }


def ensure_casea1_real_alias(base_dir: str, output_dir: str) -> str:
    src_case = os.path.join(base_dir, "dataset", "caseA1.npz")
    if not os.path.exists(src_case):
        raise FileNotFoundError(f"Missing caseA1 dataset: {src_case}")
    tmp_data_dir = os.path.join(output_dir, "_tmp_caseA1_data")
    os.makedirs(tmp_data_dir, exist_ok=True)
    dst_real_case = os.path.join(tmp_data_dir, "real_caseA1.npz")
    if os.path.lexists(dst_real_case):
        os.remove(dst_real_case)
    d = np.load(src_case, allow_pickle=True)
    if "signals" not in d or "labels" not in d:
        raise KeyError(f"caseA1.npz must contain signals/labels: {src_case}")
    signals = d["signals"].astype(np.float32)
    labels = d["labels"].astype(np.float32)
    if "group_ids" in d:
        group_ids = d["group_ids"].astype(np.int64)
    else:
        group_ids = np.arange(signals.shape[0], dtype=np.int64)
    np.savez_compressed(
        dst_real_case,
        signals=signals,
        labels=labels,
        group_ids=group_ids,
    )
    return tmp_data_dir


def compute_casea1_compare_row_fallback(
    base_dir: str,
    output_dir: str,
    compare_rows_ok: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> Dict[str, Any]:
    cached_row = os.path.join(output_dir, "compare_caseA1_fallback_row.json")
    cached_hist = os.path.join(output_dir, "history_compare_caseA1.json")
    if os.path.exists(cached_row) and os.path.exists(cached_hist):
        with open(cached_row, "r", encoding="utf-8") as f:
            return json.load(f)

    params = infer_compare_runtime_params(base_dir=base_dir, output_dir=output_dir, compare_rows_ok=compare_rows_ok)
    tmp_data_dir = ensure_casea1_real_alias(base_dir=base_dir, output_dir=output_dir)

    from types import SimpleNamespace
    import torch
    from run_compare_reg_vs_bp_same_init import run_one_scenario

    args = SimpleNamespace(
        data_dir=tmp_data_dir,
        cycle=params["cycle"],
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        split_seed=int(split_seed),
        label_scale=params["label_scale"],
        preprocess_mode=params["preprocess_mode"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["bp_lr"],
        bp_lr=params["bp_lr"],
        reg_lambda=params["reg_lambda"],
        reg_method=params["reg_method"],
        reg_progressive_min_frac=params["reg_progressive_min_frac"],
        reg_progressive_anchor=params["reg_progressive_anchor"],
        reg_progressive_shuffle_seed=params["reg_progressive_shuffle_seed"],
        pinv_reg_lambda=params["pinv_reg_lambda"],
        disable_pinv_warm_start=False,
        yscale="log",
        log_switch_ratio=20.0,
        w_bit=params["w_bit"],
        a_bit=params["a_bit"],
        seed=params["seed"],
        use_cpu=True,
        base_ckpt="scripts/DATELBR/model/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb3_ab3_lam0.0.pth",
        output_dir=output_dir,
    )
    row = run_one_scenario(args=args, scenario="caseA1", device=torch.device("cpu"))
    if row.get("status") != "ok":
        raise RuntimeError(f"caseA1 fallback compare failed with status={row.get('status')}")
    with open(cached_row, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    return row


def aggregate_method_error(
    scenario_order: List[str],
    per_scenario: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for method in METHOD_KEYS:
        vals = np.vstack([np.asarray(per_scenario[scenario][method], dtype=np.float64) for scenario in scenario_order])
        means.append(np.mean(vals, axis=0))
        stds.append(np.std(vals, axis=0))
    return np.vstack(means), np.vstack(stds)


def make_method_error_bar(
    method_means: np.ndarray,
    method_stds: np.ndarray,
    out_png: str,
    yscale: str,
    scenario_count: int,
):
    x = np.arange(len(HARMONIC_ORDERS), dtype=np.float64)
    n_methods = len(METHOD_KEYS)
    total_w = 0.84
    bar_w = total_w / max(n_methods, 1)
    offsets = (np.arange(n_methods, dtype=np.float64) - (n_methods - 1) / 2.0) * bar_w
    colors = list(plt.cm.Set2.colors) + list(plt.cm.tab10.colors)

    plt.figure(figsize=(10.5, 5))
    for i, method in enumerate(METHOD_KEYS):
        y = np.asarray(method_means[i], dtype=np.float64)
        yerr = np.asarray(method_stds[i], dtype=np.float64)
        if yscale == "log":
            y = np.maximum(y, 1e-8)
            yerr = np.minimum(yerr, y * 0.95)
        plt.bar(
            x + offsets[i],
            y,
            width=bar_w * 0.9,
            label=METHOD_LABELS[method],
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.25,
            yerr=yerr,
            capsize=2.5,
        )

    if yscale == "log":
        plt.yscale("log")
    plt.xticks(x, HARMONIC_LABELS)
    plt.xlabel("Harmonic Order")
    plt.ylabel("MAE (vs 10-cycle FFT label)")
    plt.title(f"Harmonic MAE by Method ({yscale}, mean±std across {scenario_count} scenarios)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_grouped_bar_with_method_errorbars(
    scenario_order: List[str],
    scenario_stats: Dict[str, Dict[str, np.ndarray]],
    method_per_scenario: Dict[str, Dict[str, Any]],
    out_png: str,
    yscale: str,
    show_std: bool,
):
    n_s = len(scenario_order)
    n_methods = len(METHOD_KEYS)
    h_idx = [HARMONIC_ORDERS.index(h) for h in OVERLAY_HARMONIC_ORDERS]
    x = np.arange(len(OVERLAY_HARMONIC_ORDERS), dtype=np.float64)
    # Overlay all method error-bars on the exact same bar center.
    method_offsets = np.zeros(n_methods, dtype=np.float64)
    method_caps = np.asarray([7.5, 6.0, 4.5, 3.0], dtype=np.float64)

    fig, axes = plt.subplots(1, n_s, figsize=(3.0 * n_s + 1.2, 5.6), squeeze=False)
    axes = axes[0]

    for i, name in enumerate(scenario_order):
        ax = axes[i]
        if name not in scenario_stats:
            raise KeyError(f"Missing scenario stats for {name}")
        if name not in method_per_scenario:
            raise KeyError(f"Missing method errors for {name}")

        bar_top_full = np.asarray(scenario_stats[name]["mean"], dtype=np.float64)
        bar_top = np.asarray(bar_top_full[h_idx], dtype=np.float64)
        bar_std = None
        if show_std:
            bar_std_full = np.asarray(scenario_stats[name]["std"], dtype=np.float64)
            bar_std = np.asarray(bar_std_full[h_idx], dtype=np.float64)

        y_draw = bar_top.copy()
        yerr_draw = None if bar_std is None else bar_std.copy()
        ax.bar(
            x,
            y_draw,
            width=0.42,
            color="#9aa3ad",
            edgecolor="black",
            linewidth=0.45,
            yerr=yerr_draw,
            capsize=2.0 if yerr_draw is not None else 0.0,
            alpha=0.92,
            zorder=1,
        )

        y_low_list = [bar_top]
        y_high_list = [bar_top]
        if show_std and bar_std is not None:
            y_low_list.append(bar_top - bar_std)
            y_high_list.append(bar_top + bar_std)

        for m_idx, method in enumerate(METHOD_KEYS):
            errs_full = np.asarray(method_per_scenario[name][method], dtype=np.float64)
            errs = np.asarray(errs_full[h_idx], dtype=np.float64)
            y_low_list.append(bar_top - errs)
            y_high_list.append(bar_top + errs)
            ax.errorbar(
                x + method_offsets[m_idx],
                bar_top,
                yerr=errs,
                fmt="none",
                color=METHOD_COLORS[m_idx % len(METHOD_COLORS)],
                elinewidth=2.0,
                capsize=float(method_caps[m_idx % len(method_caps)]),
                linewidth=2.0,
                zorder=4,
            )

        # Per-scenario independent linear y-range with tight truncation to improve visibility.
        y_low_ref = float(np.min(np.concatenate(y_low_list)))
        y_high_ref = float(np.max(np.concatenate(y_high_list)))
        span = max(y_high_ref - y_low_ref, max(1e-6, 0.08 * max(abs(y_high_ref), 1.0)))
        pad = 0.06 * span
        ymin = y_low_ref - pad
        ymax = y_high_ref + pad
        if y_low_ref >= 0.0:
            ymin = max(0.0, ymin)
        if ymax <= ymin:
            ymax = ymin + span
        ax.set_yscale("linear")
        ax.set_ylim(ymin, ymax)

        ax.set_xticks(x, [str(h) for h in OVERLAY_HARMONIC_ORDERS])
        ax.set_title(SCENARIO_DISPLAY_LABELS.get(name, name), fontsize=10)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        if i == 0:
            ax.set_ylabel("Amplitude / Error")
        ax.set_xlabel("H")

    handles = [Patch(facecolor="#9aa3ad", edgecolor="black", label="Amplitude bar")]
    handles += [Line2D([0], [0], color=METHOD_COLORS[k], linestyle="-", linewidth=1.5) for k in range(n_methods)]
    labels = ["Amplitude bar"] + [METHOD_LABELS[k] for k in METHOD_KEYS]
    fig.legend(handles, labels, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Scenario-Grouped Harmonic Amplitude (H3/H5/H7) with Raw Method Error-bars", y=1.08, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_method_error_stats(
    out_prefix: str,
    scenario_order: List[str],
    per_scenario: Dict[str, Dict[str, Any]],
    method_means: np.ndarray,
    method_stds: np.ndarray,
):
    agg_csv = f"{out_prefix}_stats.csv"
    per_s_csv = f"{out_prefix}_per_scenario.csv"
    out_json = f"{out_prefix}_stats.json"

    os.makedirs(os.path.dirname(agg_csv), exist_ok=True)
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "scenario_count",
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
        for i, method in enumerate(METHOD_KEYS):
            writer.writerow(
                [
                    method,
                    len(scenario_order),
                    float(method_means[i, 0]),
                    float(method_means[i, 1]),
                    float(method_means[i, 2]),
                    float(method_means[i, 3]),
                    float(method_stds[i, 0]),
                    float(method_stds[i, 1]),
                    float(method_stds[i, 2]),
                    float(method_stds[i, 3]),
                ]
            )

    with open(per_s_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "method",
                "A1",
                "A3",
                "A5",
                "A7",
                "fft_val_samples",
                "reg_best_epoch",
                "bp_best_epoch",
            ]
        )
        for scenario in scenario_order:
            row = per_scenario[scenario]
            for method in METHOD_KEYS:
                vals = np.asarray(row[method], dtype=np.float64)
                writer.writerow(
                    [
                        scenario,
                        method,
                        float(vals[0]),
                        float(vals[1]),
                        float(vals[2]),
                        float(vals[3]),
                        int(row["fft_val_samples"]),
                        int(row["reg_best_epoch"]),
                        int(row["bp_best_epoch"]),
                    ]
                )

    out_obj = {
        "harmonics": list(HARMONIC_LABELS),
        "method_labels": {k: METHOD_LABELS[k] for k in METHOD_KEYS},
        "scenario_order": scenario_order,
        "aggregated_mean": {
            method: {
                "A1": float(method_means[i, 0]),
                "A3": float(method_means[i, 1]),
                "A5": float(method_means[i, 2]),
                "A7": float(method_means[i, 3]),
            }
            for i, method in enumerate(METHOD_KEYS)
        },
        "aggregated_std": {
            method: {
                "A1": float(method_stds[i, 0]),
                "A3": float(method_stds[i, 1]),
                "A5": float(method_stds[i, 2]),
                "A7": float(method_stds[i, 3]),
            }
            for i, method in enumerate(METHOD_KEYS)
        },
        "per_scenario": {
            scenario: {
                method: {
                    "A1": float(np.asarray(per_scenario[scenario][method], dtype=np.float64)[0]),
                    "A3": float(np.asarray(per_scenario[scenario][method], dtype=np.float64)[1]),
                    "A5": float(np.asarray(per_scenario[scenario][method], dtype=np.float64)[2]),
                    "A7": float(np.asarray(per_scenario[scenario][method], dtype=np.float64)[3]),
                }
                for method in METHOD_KEYS
            }
            for scenario in scenario_order
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Draw harmonic bars for scenarios and method-error bars with error bars."
    )
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="output_compare_sameinit_pinv_analytic_vs_bp")
    parser.add_argument("--yscale", type=str, default="both", choices=["linear", "log", "both"])
    parser.add_argument("--show_std", action="store_true", help="Show std error bars.")
    parser.add_argument(
        "--method_error_mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Generate 4-method MAE bars with error bars: auto=if compare_summary exists, on=force, off=disable.",
    )
    parser.add_argument(
        "--method_scenarios",
        type=str,
        default="auto",
        help="Comma-separated scenario list for method-error bars, or 'auto' from compare_summary.csv status=ok rows.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
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

    if args.method_error_mode != "off":
        compare_csv = os.path.join(args.output_dir, "compare_summary.csv")
        should_run_method_error = args.method_error_mode == "on" or os.path.exists(compare_csv)
        if should_run_method_error:
            try:
                scenario_order, per_scenario = collect_method_error_by_scenario(
                    base_dir=args.base_dir,
                    output_dir=args.output_dir,
                    scenario_spec=args.method_scenarios,
                    fallback_scenarios=names,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    split_seed=args.split_seed,
                )
                method_means, method_stds = aggregate_method_error(
                    scenario_order=scenario_order,
                    per_scenario=per_scenario,
                )
                method_prefix = os.path.join(args.output_dir, "harmonic_mae_4methods")
                if args.yscale in ("linear", "both"):
                    make_method_error_bar(
                        method_means=method_means,
                        method_stds=method_stds,
                        out_png=f"{method_prefix}_linear.png",
                        yscale="linear",
                        scenario_count=len(scenario_order),
                    )
                if args.yscale in ("log", "both"):
                    make_method_error_bar(
                        method_means=method_means,
                        method_stds=method_stds,
                        out_png=f"{method_prefix}_log.png",
                        yscale="log",
                        scenario_count=len(scenario_order),
                    )
                save_method_error_stats(
                    out_prefix=method_prefix,
                    scenario_order=scenario_order,
                    per_scenario=per_scenario,
                    method_means=method_means,
                    method_stds=method_stds,
                )
                scenario_stats: Dict[str, Dict[str, np.ndarray]] = {
                    names[i]: {
                        "mean": np.asarray(means[i], dtype=np.float64),
                        "std": np.asarray(stds[i], dtype=np.float64),
                    }
                    for i in range(len(names))
                }
                overlay_order = [sc for sc, _ in DEFAULT_SCENARIOS if sc in scenario_stats and sc in per_scenario]
                for sc in scenario_order:
                    if sc in scenario_stats and sc in per_scenario and sc not in overlay_order:
                        overlay_order.append(sc)
                overlay_prefix = os.path.join(args.output_dir, "harmonic_amp_5scenarios_with_4method_errorbars")
                if args.yscale in ("linear", "both"):
                    make_grouped_bar_with_method_errorbars(
                        scenario_order=overlay_order,
                        scenario_stats=scenario_stats,
                        method_per_scenario=per_scenario,
                        out_png=f"{overlay_prefix}_linear.png",
                        yscale="linear",
                        show_std=args.show_std,
                    )
                if args.yscale in ("log", "both"):
                    make_grouped_bar_with_method_errorbars(
                        scenario_order=overlay_order,
                        scenario_stats=scenario_stats,
                        method_per_scenario=per_scenario,
                        out_png=f"{overlay_prefix}_log.png",
                        yscale="log",
                        show_std=args.show_std,
                    )
                print(f"Saved method-error bar figures and stats under: {args.output_dir}")
            except Exception as ex:
                if args.method_error_mode == "on":
                    raise
                print(f"[WARN] Skip method-error bars in auto mode: {ex}")

    print(f"Saved bar figures and stats under: {args.output_dir}")


if __name__ == "__main__":
    main()
