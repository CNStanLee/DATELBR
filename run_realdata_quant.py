import argparse
import copy
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.data_load_real import create_real_harmonic_datasets
from model.models import (
    BLSQuantRandomNetNoConcat,
    compute_output_pinv_quant,
    test_mae_original_scale,
    test_relative_error_stats,
)
from utils.pruning import analyze_model_sparsity, global_magnitude_prune_with_min


HARMONIC_ORDER = (1, 3, 5, 7)
SCENARIOS = ("v2g", "g2v", "g2b", "b2g")


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_zero_weights(model: nn.Module):
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            mask = (param != 0).float()

            def hook_factory(mask_):
                def hook(grad):
                    return grad * mask_

                return hook

            param.register_hook(hook_factory(mask))


def finetune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    label_scale: float,
    num_epochs: int = 20,
    lr: float = 1e-3,
):
    model.to(device)

    model.train()
    with torch.no_grad():
        for x_warm, _ in train_loader:
            _ = model(x_warm.to(device))
            break

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_train_loss = running_loss / max(len(train_loader.dataset), 1)
        train_mae = test_mae_original_scale(model, train_loader, device, label_scale)
        val_mae = test_mae_original_scale(model, val_loader, device, label_scale)
        print(
            f"  [Finetune][Epoch {epoch}/{num_epochs}] "
            f"TrainLoss={avg_train_loss:.6f} TrainMAE={train_mae} ValMAE={val_mae}"
        )

        val_mae_mean = float(np.mean(val_mae))
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return model


def vector_to_metric_fields(prefix: str, vec: np.ndarray, row: Dict):
    for idx, h in enumerate(HARMONIC_ORDER):
        row[f"{prefix}_A{h}"] = float(vec[idx])


def load_npz_meta(npz_path: str) -> Dict:
    data = np.load(npz_path, allow_pickle=True)
    group_ids = data["group_ids"].astype(np.int64)
    out = {
        "num_samples": int(data["signals"].shape[0]),
        "num_groups": int(np.unique(group_ids).size),
        "channel_c_groups": 0,
        "total_groups": 0,
        "channel_c_group_ratio": float("nan"),
    }

    if "group_channel_source" in data.files:
        src = np.asarray(data["group_channel_source"]).astype(str)
        c_groups = int(np.sum(src == "C_converted"))
        total_groups = int(src.size)
        out["channel_c_groups"] = c_groups
        out["total_groups"] = total_groups
        out["channel_c_group_ratio"] = (
            float(c_groups / total_groups) if total_groups > 0 else float("nan")
        )
    return out


def evaluate_one_scenario(args, scenario: str, device: torch.device) -> Dict:
    row = {
        "scenario": scenario,
        "status": "ok",
    }
    npz_path = os.path.join(args.data_dir, f"real_{scenario}.npz")
    row["npz_path"] = npz_path

    if not os.path.exists(npz_path):
        row["status"] = "missing_npz"
        return row

    row.update(load_npz_meta(npz_path))

    train_ds, val_ds, test_ds = create_real_harmonic_datasets(
        npz_path=npz_path,
        cycle=args.cycle,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    row["train_samples"] = int(len(train_ds))
    row["val_samples"] = int(len(val_ds))
    row["test_samples"] = int(len(test_ds))

    if len(train_ds) == 0 or len(test_ds) == 0:
        row["status"] = "empty_split"
        return row

    # Keep pinv fitting numerically stable/reproducible with fixed sample order.
    train_loader_pinv = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    # Optional finetuning still benefits from shuffled mini-batches.
    train_loader_ft = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds.input_len
    label_scale = train_ds.label_scale

    # Keep each scenario reproducible and independent of call order.
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

    model = compute_output_pinv_quant(model, train_loader_pinv, device, reg_lambda=args.reg_lambda)

    dense_mae = test_mae_original_scale(model, test_loader, device, label_scale)
    dense_mean_re, dense_max_re = test_relative_error_stats(model, test_loader, device)

    vector_to_metric_fields("dense_mae", dense_mae, row)
    vector_to_metric_fields("dense_mean_re", dense_mean_re, row)
    vector_to_metric_fields("dense_max_re", dense_max_re, row)

    print(f"[{scenario}] Dense MAE [A1,A3,A5,A7] = {dense_mae}")
    print(f"[{scenario}] Dense Mean RE(%)      = {dense_mean_re}")
    print(f"[{scenario}] Dense Max  RE(%)      = {dense_max_re}")

    if args.apply_prune_ft:
        pruned_model = global_magnitude_prune_with_min(model, target_sparsity=args.prune_rate)
        if args.print_sparsity:
            print(f"[{scenario}] Sparsity after prune:")
            analyze_model_sparsity(pruned_model)

        freeze_zero_weights(pruned_model)
        pruned_model = finetune_pruned_model(
            pruned_model,
            train_loader=train_loader_ft,
            val_loader=val_loader,
            device=device,
            label_scale=label_scale,
            num_epochs=args.ft_epochs,
            lr=args.ft_lr,
        )

        prune_mae = test_mae_original_scale(pruned_model, test_loader, device, label_scale)
        prune_mean_re, prune_max_re = test_relative_error_stats(pruned_model, test_loader, device)

        vector_to_metric_fields("prune_ft_mae", prune_mae, row)
        vector_to_metric_fields("prune_ft_mean_re", prune_mean_re, row)
        vector_to_metric_fields("prune_ft_max_re", prune_max_re, row)

        print(f"[{scenario}] Prune+FT MAE [A1,A3,A5,A7] = {prune_mae}")
        print(f"[{scenario}] Prune+FT Mean RE(%)      = {prune_mean_re}")
        print(f"[{scenario}] Prune+FT Max  RE(%)      = {prune_max_re}")

    return row


def save_summary_csv(rows: List[Dict], csv_path: str):
    if not rows:
        return

    preferred = [
        "scenario",
        "status",
        "npz_path",
        "num_samples",
        "num_groups",
        "train_samples",
        "val_samples",
        "test_samples",
        "channel_c_groups",
        "total_groups",
        "channel_c_group_ratio",
    ]

    for prefix in (
        "dense_mae",
        "dense_mean_re",
        "dense_max_re",
        "prune_ft_mae",
        "prune_ft_mean_re",
        "prune_ft_max_re",
    ):
        for h in HARMONIC_ORDER:
            preferred.append(f"{prefix}_A{h}")

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    keys = [k for k in preferred if k in all_keys]
    keys += sorted(k for k in all_keys if k not in keys)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run BLS quantized fitting/evaluation on realdata scenarios"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["v2g", "g2v", "g2b", "b2g", "all"],
    )
    parser.add_argument("--cycle", type=str, default="half", choices=["half", "quarter", "full"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--a_bit", type=int, default=3)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--apply_prune_ft", action="store_true")
    parser.add_argument("--prune_rate", type=float, default=0.9)
    parser.add_argument("--ft_lr", type=float, default=1e-3)
    parser.add_argument("--ft_epochs", type=int, default=50)
    parser.add_argument("--print_sparsity", action="store_true")

    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    summary_csv = args.summary_csv or os.path.join(args.data_dir, "realdata_quant_summary.csv")
    summary_json = args.summary_json or os.path.join(args.data_dir, "realdata_quant_summary.json")

    if args.scenario == "all":
        scenarios = list(SCENARIOS)
    else:
        scenarios = [args.scenario]

    rows = []
    for scenario in scenarios:
        print(f"\n===== Scenario: {scenario} =====")
        row = evaluate_one_scenario(args=args, scenario=scenario, device=device)
        rows.append(row)

    os.makedirs(os.path.dirname(summary_json) or ".", exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    save_summary_csv(rows, summary_csv)

    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV : {summary_csv}")


if __name__ == "__main__":
    main()
