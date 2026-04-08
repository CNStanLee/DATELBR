import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model.models import (
    BLSQuantRandomNetNoConcat,
    compute_output_pinv_quant,
    test_mae_original_scale,
    test_relative_error_stats,
)


HARMONIC_ORDER = (1, 3, 5, 7)
SCENARIOS = ("v2g", "g2v", "g2b", "b2g")


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_groups: np.ndarray
    val_groups: np.ndarray
    test_groups: np.ndarray


def resolve_existing_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, path),
        os.path.join(script_dir, os.path.basename(path)),
        os.path.join(script_dir, "model", os.path.basename(path)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def unified_preprocess_signals(signals: np.ndarray, mode: str = "rms", eps: float = 1e-6) -> np.ndarray:
    # Unified preprocessing across amplitudes.
    # mode=rms:
    #   1) remove per-sample DC offset
    #   2) normalize by per-sample RMS
    # mode=maxabs:
    #   align with existing run_all_quant / run_realdata_quant behavior.
    x = signals.astype(np.float32)
    if mode == "rms":
        mean = np.mean(x, axis=1, keepdims=True)
        x = x - mean
        rms = np.sqrt(np.mean(np.square(x), axis=1, keepdims=True))
        x = x / np.maximum(rms, eps)
    elif mode == "maxabs":
        max_abs = float(np.max(np.abs(x)))
        if max_abs > 0:
            x = x / max_abs
    else:
        raise ValueError(f"Unsupported preprocess mode: {mode}")
    return x.astype(np.float32)


def build_split_data(
    npz_path: str,
    cycle: str,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
    label_scale: float,
    preprocess_mode: str,
) -> SplitData:
    data = np.load(npz_path, allow_pickle=True)
    signals = data["signals"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    group_ids = data["group_ids"].astype(np.int64)

    if cycle == "quarter":
        input_len = signals.shape[1] // 4
    elif cycle == "half":
        input_len = signals.shape[1] // 2
    elif cycle == "full":
        input_len = signals.shape[1]
    else:
        raise ValueError(f"Unsupported cycle: {cycle}")

    signals = signals[:, :input_len]
    signals = unified_preprocess_signals(signals, mode=preprocess_mode)
    labels = labels / label_scale

    train_groups, val_groups, test_groups = split_by_group_ids(
        group_ids=group_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    train_mask = np.isin(group_ids, train_groups)
    val_mask = np.isin(group_ids, val_groups)
    test_mask = np.isin(group_ids, test_groups)

    return SplitData(
        x_train=signals[train_mask],
        y_train=labels[train_mask],
        x_val=signals[val_mask],
        y_val=labels[val_mask],
        x_test=signals[test_mask],
        y_test=labels[test_mask],
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
    )


def make_loaders(
    split: SplitData,
    batch_size: int,
    train_shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(split.x_train), torch.from_numpy(split.y_train))
    val_ds = TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val))
    test_ds = TensorDataset(torch.from_numpy(split.x_test), torch.from_numpy(split.y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def freeze_all_but_fc_out(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc_out.parameters():
        p.requires_grad = True


def mean_mae_scalar(mae_vec: np.ndarray) -> float:
    return float(np.mean(mae_vec))


def save_curve_plot(
    train_mae_hist: List[float],
    val_mae_hist: List[float],
    scenario: str,
    out_png: str,
    start_epoch: int = 1,
):
    plt.figure(figsize=(8, 4))
    xs = np.arange(start_epoch, start_epoch + len(train_mae_hist))
    plt.plot(xs, train_mae_hist, marker="o", label="Train MAE(mean)")
    plt.plot(xs, val_mae_hist, marker="o", label="Val MAE(mean)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (original scale)")
    plt.title(f"{scenario} transfer fine-tune (fc_out only)")
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
    start_epoch: int = 1,
):
    plt.figure(figsize=(8, 4))
    xs = np.arange(start_epoch, start_epoch + len(val_mae_hist))
    plt.plot(xs, val_mae_hist, marker="o", color="tab:orange", label="Val MAE(mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (original scale)")
    plt.title(f"{scenario} val MAE vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_preprocessed_split(split: SplitData, out_npz: str):
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        x_train=split.x_train.astype(np.float32),
        y_train=split.y_train.astype(np.float32),
        x_val=split.x_val.astype(np.float32),
        y_val=split.y_val.astype(np.float32),
        x_test=split.x_test.astype(np.float32),
        y_test=split.y_test.astype(np.float32),
        train_groups=split.train_groups.astype(np.int64),
        val_groups=split.val_groups.astype(np.int64),
        test_groups=split.test_groups.astype(np.int64),
    )


def evaluate_and_fill(row: Dict, prefix: str, mae: np.ndarray, mean_re: np.ndarray, max_re: np.ndarray):
    for i, h in enumerate(HARMONIC_ORDER):
        row[f"{prefix}_mae_A{h}"] = float(mae[i])
        row[f"{prefix}_mean_re_A{h}"] = float(mean_re[i])
        row[f"{prefix}_max_re_A{h}"] = float(max_re[i])


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

    # Scenario-aware training config: g2v can use a dedicated stable mode to reduce epoch jitter.
    epochs = args.epochs
    lr = args.lr
    reg_lambda = args.reg_lambda
    batch_size = args.batch_size
    train_shuffle = True

    if args.g2v_stable_mode and scenario == "g2v":
        epochs = args.g2v_epochs
        lr = args.g2v_lr
        reg_lambda = args.g2v_reg_lambda
        if args.g2v_batch_size <= 0:
            batch_size = max(1, int(split.x_train.shape[0]))  # full-batch update by default
        else:
            batch_size = max(1, min(int(args.g2v_batch_size), int(split.x_train.shape[0])))
        train_shuffle = False
        row["g2v_stable_mode"] = True
    else:
        row["g2v_stable_mode"] = False

    row["epochs_used"] = int(epochs)
    row["lr_used"] = float(lr)
    row["reg_lambda_used"] = float(reg_lambda)
    row["batch_size_used"] = int(batch_size)

    prep_npz = os.path.join(args.output_dir, f"preprocessed_{scenario}.npz")
    save_preprocessed_split(split, prep_npz)
    row["preprocessed_npz"] = prep_npz

    train_loader, val_loader, test_loader = make_loaders(
        split,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
    )
    input_dim = split.x_train.shape[1]

    set_global_seed(args.seed)
    model = BLSQuantRandomNetNoConcat(
        input_dim=input_dim,
        a=8,
        b=8,
        m=8,
        out_dim=4,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        freeze_feature=True,
    ).to(device)

    base_ckpt = resolve_existing_path(args.base_ckpt)
    if base_ckpt and os.path.exists(base_ckpt):
        state = torch.load(base_ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        row["status"] = "missing_base_ckpt"
        row["base_ckpt"] = args.base_ckpt
        row["resolved_base_ckpt"] = base_ckpt
        return row

    freeze_all_but_fc_out(model)
    row["base_ckpt"] = args.base_ckpt
    row["resolved_base_ckpt"] = base_ckpt
    row["preprocess_mode"] = args.preprocess_mode
    row["reg_anchor"] = args.reg_anchor
    row["use_pinv_warm_start"] = not args.disable_pinv_warm_start

    base_test_mae = test_mae_original_scale(model, test_loader, device, args.label_scale)
    row["base_test_mae_mean"] = mean_mae_scalar(base_test_mae)

    base_w = model.fc_out.weight.detach().clone()
    base_b = model.fc_out.bias.detach().clone()

    if not args.disable_pinv_warm_start:
        model = compute_output_pinv_quant(
            model,
            train_loader,
            device,
            reg_lambda=args.pinv_reg_lambda,
        )
        pinv_test_mae = test_mae_original_scale(model, test_loader, device, args.label_scale)
        row["pinv_init_test_mae_mean"] = mean_mae_scalar(pinv_test_mae)

    if args.reg_anchor == "base":
        init_w = base_w
        init_b = base_b
    elif args.reg_anchor == "pinv":
        init_w = model.fc_out.weight.detach().clone()
        init_b = model.fc_out.bias.detach().clone()
    else:
        raise ValueError(f"Unsupported reg anchor: {args.reg_anchor}")

    init_w = init_w.to(device)
    init_b = init_b.to(device)

    optimizer = optim.Adam(model.fc_out.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    train_mae_hist = []
    val_mae_hist = []
    history_start_epoch = 1

    # Optional: record the true initialization point before any BP update (epoch 0).
    if args.record_epoch0:
        history_start_epoch = 0
        init_train_mae = test_mae_original_scale(model, train_loader, device, args.label_scale)
        init_val_mae = test_mae_original_scale(model, val_loader, device, args.label_scale)
        init_train_mae_mean = mean_mae_scalar(init_train_mae)
        init_val_mae_mean = mean_mae_scalar(init_val_mae)
        row["init_train_mae_mean"] = float(init_train_mae_mean)
        row["init_val_mae_mean"] = float(init_val_mae_mean)
        train_mae_hist.append(init_train_mae_mean)
        val_mae_hist.append(init_val_mae_mean)
        best_val = init_val_mae_mean
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{scenario}][Epoch 000/init] TrainMAE={init_train_mae} ValMAE={init_val_mae}")

    print(
        f"[{scenario}] train_cfg: epochs={epochs}, batch_size={batch_size}, "
        f"lr={lr}, reg_lambda={reg_lambda}, train_shuffle={train_shuffle}"
    )

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss_main = mse_loss(pred, y)
            reg = torch.mean((model.fc_out.weight - init_w) ** 2) + torch.mean((model.fc_out.bias - init_b) ** 2)
            loss = loss_main + reg_lambda * reg
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
            f"[{scenario}][Epoch {ep:03d}/{epochs}] "
            f"Loss={avg_loss:.6f} TrainMAE={train_mae} ValMAE={val_mae}"
        )

        if val_mae_mean < best_val:
            best_val = val_mae_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_mae = test_mae_original_scale(model, test_loader, device, args.label_scale)
    test_mean_re, test_max_re = test_relative_error_stats(model, test_loader, device)
    evaluate_and_fill(row, "test", test_mae, test_mean_re, test_max_re)
    row["best_val_mae_mean"] = float(best_val)

    out_model = os.path.join(args.output_dir, f"transfer_{scenario}_regft.pth")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), out_model)
    row["saved_model"] = out_model

    curve_png = os.path.join(args.output_dir, f"mae_vs_epoch_{scenario}.png")
    save_curve_plot(
        train_mae_hist,
        val_mae_hist,
        scenario=scenario,
        out_png=curve_png,
        start_epoch=history_start_epoch,
    )
    row["mae_curve_png"] = curve_png

    val_curve_png = os.path.join(args.output_dir, f"val_vs_epoch_{scenario}.png")
    save_val_curve_plot(
        val_mae_hist,
        scenario=scenario,
        out_png=val_curve_png,
        start_epoch=history_start_epoch,
    )
    row["val_curve_png"] = val_curve_png

    hist_json = os.path.join(args.output_dir, f"mae_history_{scenario}.json")
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "history_start_epoch": history_start_epoch,
                "train_mae_mean_per_epoch": train_mae_hist,
                "val_mae_mean_per_epoch": val_mae_hist,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    row["mae_history_json"] = hist_json
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
        "saved_model",
        "mae_curve_png",
        "val_curve_png",
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
    parser = argparse.ArgumentParser(
        description="Unified-preprocess transfer fine-tuning: freeze front layers, reg fine-tune fc_out"
    )
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "v2g", "g2v", "g2b", "b2g"])
    parser.add_argument("--cycle", type=str, default="half", choices=["quarter", "half", "full"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_scale", type=float, default=100.0)
    parser.add_argument("--preprocess_mode", type=str, default="rms", choices=["rms", "maxabs"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg_lambda", type=float, default=1e-2)
    parser.add_argument(
        "--g2v_stable_mode",
        action="store_true",
        help="Use a dedicated low-jitter training config for g2v (full-batch, lower LR, more epochs).",
    )
    parser.add_argument("--g2v_epochs", type=int, default=120)
    parser.add_argument("--g2v_lr", type=float, default=2e-4)
    parser.add_argument("--g2v_reg_lambda", type=float, default=5e-2)
    parser.add_argument(
        "--g2v_batch_size",
        type=int,
        default=0,
        help="Batch size for g2v stable mode. <=0 means full-batch.",
    )
    parser.add_argument("--pinv_reg_lambda", type=float, default=0.0)
    parser.add_argument("--reg_anchor", type=str, default="pinv", choices=["pinv", "base"])
    parser.add_argument("--disable_pinv_warm_start", action="store_true")
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--a_bit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument(
        "--record_epoch0",
        action="store_true",
        help="Record MAE before any BP update as epoch 0 in curves/history.",
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="scripts/DATELBR/model/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb3_ab3_lam0.0.pth",
    )
    parser.add_argument("--output_dir", type=str, default="output_transfer_reg")
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    rows = []
    for sc in scenarios:
        print(f"\n===== Transfer Scenario: {sc} =====")
        row = run_one_scenario(args, scenario=sc, device=device)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "transfer_reg_summary.json")
    out_csv = os.path.join(args.output_dir, "transfer_reg_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    save_summary_csv(rows, out_csv)

    print(f"\nSaved summary JSON: {out_json}")
    print(f"Saved summary CSV : {out_csv}")


if __name__ == "__main__":
    main()
