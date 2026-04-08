import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.models import test_mae_original_scale, test_relative_error_stats
from run_realdata_reg_transfer import (
    HARMONIC_ORDER,
    SCENARIOS,
    build_split_data,
    make_loaders,
    mean_mae_scalar,
    set_global_seed,
)


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_dims(text: str) -> Tuple[int, ...]:
    vals = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("hidden_dims is empty")
    if any(v <= 0 for v in vals):
        raise ValueError(f"hidden dims must be > 0, got {vals}")
    return tuple(vals)


def evaluate_and_fill(row: Dict, prefix: str, mae: np.ndarray, mean_re: np.ndarray, max_re: np.ndarray):
    for i, h in enumerate(HARMONIC_ORDER):
        row[f"{prefix}_mae_A{h}"] = float(mae[i])
        row[f"{prefix}_mean_re_A{h}"] = float(mean_re[i])
        row[f"{prefix}_max_re_A{h}"] = float(max_re[i])


def save_curve_plot(
    train_mae_hist: List[float],
    val_mae_hist: List[float],
    scenario: str,
    out_png: str,
):
    plt.figure(figsize=(8, 4))
    xs = np.arange(1, len(train_mae_hist) + 1)
    plt.plot(xs, train_mae_hist, marker="o", label="Train MAE(mean)")
    plt.plot(xs, val_mae_hist, marker="o", label="Val MAE(mean)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (original scale)")
    plt.title(f"{scenario} MLP BP")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_val_curve_plot(
    val_mae_hist: List[float],
    scenario: str,
    out_png: str,
):
    plt.figure(figsize=(8, 4))
    xs = np.arange(1, len(val_mae_hist) + 1)
    plt.plot(xs, val_mae_hist, marker="o", color="tab:green", label="MLP Val MAE(mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (original scale)")
    plt.title(f"{scenario} MLP val MAE vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_compare_plot(
    transfer_val_hist: List[float],
    mlp_val_hist: List[float],
    scenario: str,
    out_png: str,
):
    plt.figure(figsize=(8, 4))
    xs_t = np.arange(1, len(transfer_val_hist) + 1)
    xs_m = np.arange(1, len(mlp_val_hist) + 1)
    plt.plot(xs_t, transfer_val_hist, label="Transfer Val MAE(mean)", color="tab:orange")
    plt.plot(xs_m, mlp_val_hist, label="MLP BP Val MAE(mean)", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (original scale)")
    plt.title(f"{scenario} Transfer vs MLP (val)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_one_scenario(args, scenario: str, device: torch.device) -> Dict:
    row = {"scenario": scenario, "status": "ok"}
    npz_path = os.path.join(args.data_dir, f"real_{scenario}.npz")
    row["npz_path"] = npz_path

    if not os.path.exists(npz_path):
        row["status"] = "missing_npz"
        return row

    split = build_split_data(
        npz_path=npz_path,
        cycle=args.cycle,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        label_scale=args.label_scale,
        preprocess_mode=args.preprocess_mode,
    )

    row["train_samples"] = int(split.x_train.shape[0])
    row["val_samples"] = int(split.x_val.shape[0])
    row["test_samples"] = int(split.x_test.shape[0])
    row["train_groups"] = int(len(split.train_groups))
    row["val_groups"] = int(len(split.val_groups))
    row["test_groups"] = int(len(split.test_groups))

    if split.x_train.shape[0] == 0 or split.x_test.shape[0] == 0:
        row["status"] = "empty_split"
        return row

    train_loader, val_loader, test_loader = make_loaders(
        split,
        batch_size=args.batch_size,
        train_shuffle=True,
    )

    input_dim = split.x_train.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    set_global_seed(args.seed)
    model = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        out_dim=4,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    train_mae_hist: List[float] = []
    val_mae_hist: List[float] = []

    print(
        f"[{scenario}] MLP cfg: hidden_dims={hidden_dims}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}, wd={args.weight_decay}"
    )

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_mae = test_mae_original_scale(model, train_loader, device, args.label_scale)
        val_mae = test_mae_original_scale(model, val_loader, device, args.label_scale)
        train_mae_mean = mean_mae_scalar(train_mae)
        val_mae_mean = mean_mae_scalar(val_mae)

        train_mae_hist.append(train_mae_mean)
        val_mae_hist.append(val_mae_mean)

        avg_loss = running / max(n_seen, 1)
        print(
            f"[{scenario}][Epoch {ep:03d}/{args.epochs}] "
            f"Loss={avg_loss:.6f} TrainMAE={train_mae} ValMAE={val_mae}"
        )

        if val_mae_mean < best_val:
            best_val = val_mae_mean
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_mae = test_mae_original_scale(model, test_loader, device, args.label_scale)
    test_mean_re, test_max_re = test_relative_error_stats(model, test_loader, device)
    evaluate_and_fill(row, "test", test_mae, test_mean_re, test_max_re)

    row["best_val_mae_mean"] = float(best_val)
    row["best_epoch"] = int(best_epoch)

    os.makedirs(args.output_dir, exist_ok=True)
    out_model = os.path.join(args.output_dir, f"mlp_{scenario}_bp.pth")
    torch.save(model.state_dict(), out_model)
    row["saved_model"] = out_model

    curve_png = os.path.join(args.output_dir, f"mae_vs_epoch_mlp_{scenario}.png")
    save_curve_plot(train_mae_hist, val_mae_hist, scenario=scenario, out_png=curve_png)
    row["mae_curve_png"] = curve_png

    val_png = os.path.join(args.output_dir, f"val_vs_epoch_mlp_{scenario}.png")
    save_val_curve_plot(val_mae_hist, scenario=scenario, out_png=val_png)
    row["val_curve_png"] = val_png

    hist_json = os.path.join(args.output_dir, f"mae_history_mlp_{scenario}.json")
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "train_mae_mean_per_epoch": train_mae_hist,
                "val_mae_mean_per_epoch": val_mae_hist,
                "best_val_mae_mean": float(best_val),
                "best_epoch": int(best_epoch),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    row["mae_history_json"] = hist_json

    compare_curve = None
    if args.compare_transfer_dir:
        transfer_hist_path = os.path.join(args.compare_transfer_dir, f"mae_history_{scenario}.json")
        if os.path.exists(transfer_hist_path):
            with open(transfer_hist_path, "r", encoding="utf-8") as f:
                transfer_hist = json.load(f)
            transfer_val = transfer_hist.get("val_mae_mean_per_epoch", [])
            compare_curve = os.path.join(args.output_dir, f"val_compare_transfer_vs_mlp_{scenario}.png")
            save_compare_plot(
                transfer_val_hist=transfer_val,
                mlp_val_hist=val_mae_hist,
                scenario=scenario,
                out_png=compare_curve,
            )
            row["transfer_mae_history_json"] = transfer_hist_path
            row["compare_curve_png"] = compare_curve
            row["transfer_best_val_mae_mean"] = float(np.min(np.asarray(transfer_val, dtype=np.float64))) if transfer_val else None
        else:
            row["compare_curve_png"] = None
    else:
        row["compare_curve_png"] = None

    return row


def save_summary_csv(rows: List[Dict], out_csv: str):
    keys = set()
    for r in rows:
        keys.update(r.keys())

    ordered = [
        "scenario",
        "status",
        "npz_path",
        "train_samples",
        "val_samples",
        "test_samples",
        "train_groups",
        "val_groups",
        "test_groups",
        "best_val_mae_mean",
        "best_epoch",
        "saved_model",
        "mae_curve_png",
        "val_curve_png",
        "compare_curve_png",
        "mae_history_json",
    ]
    for h in HARMONIC_ORDER:
        ordered.append(f"test_mae_A{h}")
    for h in HARMONIC_ORDER:
        ordered.append(f"test_mean_re_A{h}")
    for h in HARMONIC_ORDER:
        ordered.append(f"test_max_re_A{h}")
    ordered += sorted(k for k in keys if k not in ordered)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="MLP backprop baseline on realdata scenarios")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "v2g", "g2v", "g2b", "b2g"])
    parser.add_argument("--cycle", type=str, default="half", choices=["quarter", "half", "full"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_scale", type=float, default=100.0)
    parser.add_argument("--preprocess_mode", type=str, default="rms", choices=["rms", "maxabs"])

    parser.add_argument("--hidden_dims", type=str, default="128,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--compare_transfer_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_realdata_mlp_bp")
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    rows: List[Dict] = []
    for sc in scenarios:
        print(f"\n===== MLP Scenario: {sc} =====")
        row = run_one_scenario(args, scenario=sc, device=device)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "mlp_bp_summary.json")
    out_csv = os.path.join(args.output_dir, "mlp_bp_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    save_summary_csv(rows, out_csv)

    print(f"\nSaved summary JSON: {out_json}")
    print(f"Saved summary CSV : {out_csv}")


if __name__ == "__main__":
    main()
