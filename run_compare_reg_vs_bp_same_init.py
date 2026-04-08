import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.models import (
    BLSQuantRandomNetNoConcat,
    compute_output_pinv_quant,
    test_mae_original_scale,
)
from run_realdata_reg_transfer import (
    SCENARIOS,
    build_split_data,
    freeze_all_but_fc_out,
    make_loaders,
    mean_mae_scalar,
    resolve_existing_path,
    set_global_seed,
)

HARMONIC_NAMES: Tuple[str, ...] = ("A1", "A3", "A5", "A7")


def make_model(args, input_dim: int, device: torch.device):
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
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Missing base ckpt: {base_ckpt}")
    state = torch.load(base_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    freeze_all_but_fc_out(model)
    return model


def train_one_bp(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    label_scale: float,
    lr: float,
    epochs: int,
    reg_lambda: float,
    anchor_w: torch.Tensor,
    anchor_b: torch.Tensor,
    init_train_mae: float,
    init_val_mae: float,
    init_train_mae_vec: np.ndarray,
    init_val_mae_vec: np.ndarray,
):
    optimizer = optim.Adam(model.fc_out.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_hist: List[float] = [float(init_train_mae)]
    val_hist: List[float] = [float(init_val_mae)]
    train_hist_vec: List[List[float]] = [np.asarray(init_train_mae_vec, dtype=np.float64).tolist()]
    val_hist_vec: List[List[float]] = [np.asarray(init_val_mae_vec, dtype=np.float64).tolist()]

    best_val = float(init_val_mae)
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss_main = criterion(pred, y)
            reg_term = torch.mean((model.fc_out.weight - anchor_w) ** 2) + torch.mean(
                (model.fc_out.bias - anchor_b) ** 2
            )
            loss = loss_main + reg_lambda * reg_term
            loss.backward()
            optimizer.step()

        train_mae_vec = test_mae_original_scale(model, train_loader, device, label_scale)
        val_mae_vec = test_mae_original_scale(model, val_loader, device, label_scale)
        train_mae = mean_mae_scalar(train_mae_vec)
        val_mae = mean_mae_scalar(val_mae_vec)
        train_hist.append(float(train_mae))
        val_hist.append(float(val_mae))
        train_hist_vec.append(np.asarray(train_mae_vec, dtype=np.float64).tolist())
        val_hist_vec.append(np.asarray(val_mae_vec, dtype=np.float64).tolist())

        if val_mae < best_val:
            best_val = float(val_mae)
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=False)
    return {
        "train_hist": train_hist,
        "val_hist": val_hist,
        "train_hist_vec": train_hist_vec,
        "val_hist_vec": val_hist_vec,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
    }


def solve_fc_out_anchored_ridge(
    model: nn.Module,
    train_loader,
    device: torch.device,
    reg_lambda: float,
    anchor_w: torch.Tensor,
    anchor_b: torch.Tensor,
):
    """Closed-form solve for fc_out with optional anchor:
    min ||H_ext W - Y||^2 + lambda ||W - W0||^2
    """
    model.eval()
    h_list = []
    y_list = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            _, h = model.forward_features(x)
            h_list.append(h.cpu())
            y_list.append(y.cpu())

    h_all = torch.cat(h_list, dim=0).double()  # [N, m]
    y_all = torch.cat(y_list, dim=0).double()  # [N, out_dim]
    n = h_all.shape[0]
    ones = torch.ones(n, 1, dtype=h_all.dtype)
    h_ext = torch.cat([h_all, ones], dim=1)  # [N, m+1]

    w0_ext = torch.cat(
        [
            anchor_w.detach().cpu().T.double(),
            anchor_b.detach().cpu().reshape(1, -1).double(),
        ],
        dim=0,
    )  # [m+1, out_dim]

    if reg_lambda > 0.0:
        ht = h_ext.T
        g = ht @ h_ext
        a = g + reg_lambda * torch.eye(g.shape[0], dtype=g.dtype)
        b = ht @ y_all + reg_lambda * w0_ext
        w_ext = torch.linalg.solve(a, b)
    else:
        sol = torch.linalg.lstsq(h_ext, y_all)
        w_ext = sol.solution

    w_ext = w_ext.float()
    w = w_ext[:-1, :]
    b = w_ext[-1, :]
    with torch.no_grad():
        model.fc_out.weight.data.copy_(w.T.to(model.fc_out.weight.device))
        model.fc_out.bias.data.copy_(b.to(model.fc_out.bias.device))


def train_one_reg_analytic(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    label_scale: float,
    epochs: int,
    reg_lambda: float,
    anchor_w: torch.Tensor,
    anchor_b: torch.Tensor,
    init_train_mae: float,
    init_val_mae: float,
    init_train_mae_vec: np.ndarray,
    init_val_mae_vec: np.ndarray,
):
    # epoch 0: shared start
    train_hist: List[float] = [float(init_train_mae)]
    val_hist: List[float] = [float(init_val_mae)]
    train_hist_vec: List[List[float]] = [np.asarray(init_train_mae_vec, dtype=np.float64).tolist()]
    val_hist_vec: List[List[float]] = [np.asarray(init_val_mae_vec, dtype=np.float64).tolist()]

    # epoch 1: one-shot closed-form regression on fc_out
    solve_fc_out_anchored_ridge(
        model=model,
        train_loader=train_loader,
        device=device,
        reg_lambda=reg_lambda,
        anchor_w=anchor_w,
        anchor_b=anchor_b,
    )
    train_mae_vec = np.asarray(test_mae_original_scale(model, train_loader, device, label_scale), dtype=np.float64)
    val_mae_vec = np.asarray(test_mae_original_scale(model, val_loader, device, label_scale), dtype=np.float64)
    train_mae = mean_mae_scalar(train_mae_vec)
    val_mae = mean_mae_scalar(val_mae_vec)
    train_hist.append(float(train_mae))
    val_hist.append(float(val_mae))
    train_hist_vec.append(train_mae_vec.tolist())
    val_hist_vec.append(val_mae_vec.tolist())

    # epoch >=2: closed-form optimum is unchanged with fixed features/data
    for _ in range(2, epochs + 1):
        train_hist.append(float(train_mae))
        val_hist.append(float(val_mae))
        train_hist_vec.append(train_mae_vec.tolist())
        val_hist_vec.append(val_mae_vec.tolist())

    best_epoch = int(np.argmin(np.asarray(val_hist, dtype=np.float64)))
    best_val = float(np.min(np.asarray(val_hist, dtype=np.float64)))
    return {
        "train_hist": train_hist,
        "val_hist": val_hist,
        "train_hist_vec": train_hist_vec,
        "val_hist_vec": val_hist_vec,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }


def _collect_h_ext_y(model: nn.Module, loader, device: torch.device):
    model.eval()
    h_list = []
    y_list = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, h = model.forward_features(x)
            h_list.append(h.cpu())
            y_list.append(y.cpu())
    h_all = torch.cat(h_list, dim=0).double()  # [N, m]
    y_all = torch.cat(y_list, dim=0).double()  # [N, out_dim]
    n = h_all.shape[0]
    ones = torch.ones(n, 1, dtype=h_all.dtype)
    h_ext = torch.cat([h_all, ones], dim=1)  # [N, m+1]
    return h_ext, y_all


def _solve_ridge_anchor(h_ext: torch.Tensor, y_all: torch.Tensor, reg_lambda: float, w_ref_ext: torch.Tensor):
    if reg_lambda > 0.0:
        ht = h_ext.T
        g = ht @ h_ext
        a = g + reg_lambda * torch.eye(g.shape[0], dtype=g.dtype)
        b = ht @ y_all + reg_lambda * w_ref_ext
        w_ext = torch.linalg.solve(a, b)
    else:
        sol = torch.linalg.lstsq(h_ext, y_all)
        w_ext = sol.solution
    return w_ext


def train_one_reg_analytic_progressive(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    label_scale: float,
    epochs: int,
    reg_lambda: float,
    anchor_w: torch.Tensor,
    anchor_b: torch.Tensor,
    init_train_mae: float,
    init_val_mae: float,
    init_train_mae_vec: np.ndarray,
    init_val_mae_vec: np.ndarray,
    min_frac: float = 0.1,
    anchor_mode: str = "prev",
    shuffle_seed: int = 42,
):
    """Progressive analytic ridge:
    each epoch solves closed-form ridge on an increasing subset of target samples.
    This emulates BLS incremental-sample update behavior and gives an epoch-wise process curve.
    """
    train_hist: List[float] = [float(init_train_mae)]
    val_hist: List[float] = [float(init_val_mae)]
    train_hist_vec: List[List[float]] = [np.asarray(init_train_mae_vec, dtype=np.float64).tolist()]
    val_hist_vec: List[List[float]] = [np.asarray(init_val_mae_vec, dtype=np.float64).tolist()]

    h_ext_all, y_all = _collect_h_ext_y(model=model, loader=train_loader, device=device)
    n_total = int(h_ext_all.shape[0])

    w0_ext = torch.cat(
        [
            anchor_w.detach().cpu().T.double(),
            anchor_b.detach().cpu().reshape(1, -1).double(),
        ],
        dim=0,
    )
    w_prev = w0_ext.clone()

    rng = np.random.default_rng(shuffle_seed)
    perm = np.arange(n_total, dtype=np.int64)
    rng.shuffle(perm)

    best_val = float(init_val_mae)
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    safe_min_frac = float(np.clip(min_frac, 1.0 / max(n_total, 1), 1.0))
    denom = max(epochs - 1, 1)
    for ep in range(1, epochs + 1):
        frac = safe_min_frac + (1.0 - safe_min_frac) * ((ep - 1) / denom)
        n_use = max(1, int(round(n_total * frac)))
        idx = perm[:n_use]
        h_ext = h_ext_all[idx]
        y_ep = y_all[idx]

        if anchor_mode == "base":
            w_ref = w0_ext
        elif anchor_mode == "prev":
            w_ref = w_prev
        else:
            raise ValueError(f"Unsupported anchor_mode={anchor_mode}")

        w_ext = _solve_ridge_anchor(h_ext=h_ext, y_all=y_ep, reg_lambda=reg_lambda, w_ref_ext=w_ref)
        w_prev = w_ext

        w_ext = w_ext.float()
        w = w_ext[:-1, :]
        b = w_ext[-1, :]
        with torch.no_grad():
            model.fc_out.weight.data.copy_(w.T.to(model.fc_out.weight.device))
            model.fc_out.bias.data.copy_(b.to(model.fc_out.bias.device))

        train_mae_vec = np.asarray(test_mae_original_scale(model, train_loader, device, label_scale), dtype=np.float64)
        val_mae_vec = np.asarray(test_mae_original_scale(model, val_loader, device, label_scale), dtype=np.float64)
        train_mae = mean_mae_scalar(train_mae_vec)
        val_mae = mean_mae_scalar(val_mae_vec)
        train_hist.append(float(train_mae))
        val_hist.append(float(val_mae))
        train_hist_vec.append(train_mae_vec.tolist())
        val_hist_vec.append(val_mae_vec.tolist())

        if val_mae < best_val:
            best_val = float(val_mae)
            best_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=False)
    return {
        "train_hist": train_hist,
        "val_hist": val_hist,
        "train_hist_vec": train_hist_vec,
        "val_hist_vec": val_hist_vec,
        "best_val": best_val,
        "best_epoch": best_epoch,
    }


def first_epoch_leq(arr: np.ndarray, threshold: float):
    idx = np.where(arr <= threshold)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def choose_yscale(all_vals: np.ndarray, yscale_mode: str, log_switch_ratio: float) -> str:
    if yscale_mode in ("linear", "log"):
        return yscale_mode
    positive = all_vals[all_vals > 0]
    if positive.size == 0:
        return "linear"
    dyn = float(np.max(positive) / np.min(positive))
    return "log" if dyn >= log_switch_ratio else "linear"


def save_curve(
    reg_hist_mean: np.ndarray,
    bp_hist_mean: np.ndarray,
    reg_hist_vec: np.ndarray,
    bp_hist_vec: np.ndarray,
    scenario: str,
    out_png: str,
    yscale_mode: str,
    log_switch_ratio: float,
):
    xs = np.arange(0, reg_hist_mean.size)
    plt.figure(figsize=(8, 4))
    all_vals = np.concatenate(
        [
            reg_hist_vec.reshape(-1),
            bp_hist_vec.reshape(-1),
            reg_hist_mean.reshape(-1),
            bp_hist_mean.reshape(-1),
        ]
    )
    yscale_used = choose_yscale(all_vals=all_vals, yscale_mode=yscale_mode, log_switch_ratio=log_switch_ratio)
    eps = 1e-8
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for idx, name in enumerate(HARMONIC_NAMES):
        reg_y = np.asarray(reg_hist_vec[:, idx], dtype=np.float64)
        bp_y = np.asarray(bp_hist_vec[:, idx], dtype=np.float64)
        if yscale_used == "log":
            reg_y = np.maximum(reg_y, eps)
            bp_y = np.maximum(bp_y, eps)
        plt.plot(xs, reg_y, label=f"{name} Reg", color=colors[idx], linestyle="-", linewidth=2.0)
        plt.plot(xs, bp_y, label=f"{name} BP", color=colors[idx], linestyle="--", linewidth=1.8)
    plt.axvline(0, color="gray", linestyle="--", alpha=0.35)
    if yscale_used == "log":
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE (original scale)")
    plt.title(f"{scenario} same-init comparison ({yscale_used} y-scale)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return yscale_used


def run_one_scenario(args, scenario: str, device: torch.device) -> Dict:
    row: Dict = {"scenario": scenario, "status": "ok"}
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
    if split.x_train.shape[0] == 0:
        row["status"] = "empty_split"
        return row

    batch_size = min(args.batch_size, int(split.x_train.shape[0]))
    train_loader, val_loader, test_loader = make_loaders(
        split,
        batch_size=batch_size,
        train_shuffle=False,
    )
    row["batch_size_used"] = int(batch_size)

    set_global_seed(args.seed)
    model_init = make_model(args, input_dim=split.x_train.shape[1], device=device)
    if not args.disable_pinv_warm_start:
        model_init = compute_output_pinv_quant(
            model_init, train_loader, device, reg_lambda=args.pinv_reg_lambda
        )

    init_train_vec = test_mae_original_scale(model_init, train_loader, device, args.label_scale)
    init_val_vec = test_mae_original_scale(model_init, val_loader, device, args.label_scale)
    init_test_vec = test_mae_original_scale(model_init, test_loader, device, args.label_scale)
    init_train = mean_mae_scalar(init_train_vec)
    init_val = mean_mae_scalar(init_val_vec)
    init_test = mean_mae_scalar(init_test_vec)
    row["init_train_mae_mean"] = float(init_train)
    row["init_val_mae_mean"] = float(init_val)
    row["init_test_mae_mean"] = float(init_test)
    for i, name in enumerate(HARMONIC_NAMES):
        row[f"init_val_mae_{name}"] = float(init_val_vec[i])

    init_state = {k: v.detach().cpu().clone() for k, v in model_init.state_dict().items()}
    anchor_w = model_init.fc_out.weight.detach().clone().to(device)
    anchor_b = model_init.fc_out.bias.detach().clone().to(device)
    bp_lr = float(args.bp_lr if args.bp_lr is not None else args.lr)

    # Reg method
    model_reg = make_model(args, input_dim=split.x_train.shape[1], device=device)
    model_reg.load_state_dict(init_state, strict=False)
    freeze_all_but_fc_out(model_reg)
    if args.reg_method == "analytic":
        reg_res = train_one_reg_analytic(
            model=model_reg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            label_scale=args.label_scale,
            epochs=args.epochs,
            reg_lambda=args.reg_lambda,
            anchor_w=anchor_w,
            anchor_b=anchor_b,
            init_train_mae=init_train,
            init_val_mae=init_val,
            init_train_mae_vec=init_train_vec,
            init_val_mae_vec=init_val_vec,
        )
    elif args.reg_method == "analytic_progressive":
        reg_res = train_one_reg_analytic_progressive(
            model=model_reg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            label_scale=args.label_scale,
            epochs=args.epochs,
            reg_lambda=args.reg_lambda,
            anchor_w=anchor_w,
            anchor_b=anchor_b,
            init_train_mae=init_train,
            init_val_mae=init_val,
            init_train_mae_vec=init_train_vec,
            init_val_mae_vec=init_val_vec,
            min_frac=args.reg_progressive_min_frac,
            anchor_mode=args.reg_progressive_anchor,
            shuffle_seed=args.reg_progressive_shuffle_seed,
        )
    else:
        reg_res = train_one_bp(
            model=model_reg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            label_scale=args.label_scale,
            lr=args.lr,
            epochs=args.epochs,
            reg_lambda=args.reg_lambda,
            anchor_w=anchor_w,
            anchor_b=anchor_b,
            init_train_mae=init_train,
            init_val_mae=init_val,
            init_train_mae_vec=init_train_vec,
            init_val_mae_vec=init_val_vec,
        )

    # BP-only method
    model_bp = make_model(args, input_dim=split.x_train.shape[1], device=device)
    model_bp.load_state_dict(init_state, strict=False)
    freeze_all_but_fc_out(model_bp)
    bp_res = train_one_bp(
        model=model_bp,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        label_scale=args.label_scale,
        lr=bp_lr,
        epochs=args.epochs,
        reg_lambda=0.0,
        anchor_w=anchor_w,
        anchor_b=anchor_b,
        init_train_mae=init_train,
        init_val_mae=init_val,
        init_train_mae_vec=init_train_vec,
        init_val_mae_vec=init_val_vec,
    )

    reg_val = np.asarray(reg_res["val_hist"], dtype=np.float64)
    bp_val = np.asarray(bp_res["val_hist"], dtype=np.float64)
    reg_val_vec = np.asarray(reg_res["val_hist_vec"], dtype=np.float64)
    bp_val_vec = np.asarray(bp_res["val_hist_vec"], dtype=np.float64)

    row["reg_best_val_mae_mean"] = float(reg_res["best_val"])
    row["reg_best_epoch"] = int(reg_res["best_epoch"])
    row["bp_best_val_mae_mean"] = float(bp_res["best_val"])
    row["bp_best_epoch"] = int(bp_res["best_epoch"])
    row["reg_best_first10"] = float(np.min(reg_val[:11]))
    row["bp_best_first10"] = float(np.min(bp_val[:11]))
    row["reg_best_first20"] = float(np.min(reg_val[:21]))
    row["bp_best_first20"] = float(np.min(bp_val[:21]))

    # 10% margin threshold around the better optimum
    thr = float(min(np.min(reg_val), np.min(bp_val)) * 1.10)
    row["threshold_1p1xminbest"] = thr
    row["reg_epoch_reach_thr"] = first_epoch_leq(reg_val, thr)
    row["bp_epoch_reach_thr"] = first_epoch_leq(bp_val, thr)

    curve_png = os.path.join(args.output_dir, f"val_compare_reg_vs_bp_{scenario}.png")
    yscale_used = save_curve(
        reg_hist_mean=reg_val,
        bp_hist_mean=bp_val,
        reg_hist_vec=reg_val_vec,
        bp_hist_vec=bp_val_vec,
        scenario=scenario,
        out_png=curve_png,
        yscale_mode=args.yscale,
        log_switch_ratio=args.log_switch_ratio,
    )
    row["curve_png"] = curve_png
    row["curve_yscale"] = yscale_used
    row["reg_method"] = str(args.reg_method)
    row["used_pinv_warm_start"] = bool(not args.disable_pinv_warm_start)
    row["bp_lr_used"] = float(bp_lr)
    row["reg_epoch1_val_mae"] = float(reg_val[1]) if reg_val.size > 1 else float(reg_val[0])
    row["bp_epoch1_val_mae"] = float(bp_val[1]) if bp_val.size > 1 else float(bp_val[0])
    for i, name in enumerate(HARMONIC_NAMES):
        row[f"reg_epoch1_val_mae_{name}"] = float(reg_val_vec[1, i]) if reg_val_vec.shape[0] > 1 else float(reg_val_vec[0, i])
        row[f"bp_epoch1_val_mae_{name}"] = float(bp_val_vec[1, i]) if bp_val_vec.shape[0] > 1 else float(bp_val_vec[0, i])

    hist_json = os.path.join(args.output_dir, f"history_compare_{scenario}.json")
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario,
                "init_val_mae_mean": float(init_val),
                "reg_val_mae_mean_per_epoch": reg_res["val_hist"],
                "bp_val_mae_mean_per_epoch": bp_res["val_hist"],
                "harmonic_names": list(HARMONIC_NAMES),
                "reg_val_mae_per_epoch_by_channel": reg_res["val_hist_vec"],
                "bp_val_mae_per_epoch_by_channel": bp_res["val_hist_vec"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    row["history_json"] = hist_json

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
        "init_train_mae_mean",
        "init_val_mae_mean",
        "init_test_mae_mean",
        "reg_best_val_mae_mean",
        "reg_best_epoch",
        "bp_best_val_mae_mean",
        "bp_best_epoch",
        "reg_best_first10",
        "bp_best_first10",
        "reg_best_first20",
        "bp_best_first20",
        "threshold_1p1xminbest",
        "reg_epoch_reach_thr",
        "bp_epoch_reach_thr",
        "curve_png",
        "history_json",
    ]
    ordered += sorted(k for k in keys if k not in ordered)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fair speed comparison: analytic-reg vs BP-only from the exact same initialization"
    )
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "v2g", "g2v", "g2b", "b2g"])
    parser.add_argument("--cycle", type=str, default="half", choices=["quarter", "half", "full"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_scale", type=float, default=100.0)
    parser.add_argument("--preprocess_mode", type=str, default="rms", choices=["rms", "maxabs"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--bp_lr",
        type=float,
        default=5e-4,
        help="Learning rate for BP-only baseline. Defaults lower than --lr to reduce overshoot.",
    )
    parser.add_argument("--reg_lambda", type=float, default=1e-2)
    parser.add_argument(
        "--reg_method",
        type=str,
        default="analytic",
        choices=["analytic", "analytic_progressive", "bp_anchor"],
    )
    parser.add_argument("--reg_progressive_min_frac", type=float, default=0.1)
    parser.add_argument("--reg_progressive_anchor", type=str, default="prev", choices=["prev", "base"])
    parser.add_argument("--reg_progressive_shuffle_seed", type=int, default=42)
    parser.add_argument("--pinv_reg_lambda", type=float, default=0.0)
    parser.add_argument("--disable_pinv_warm_start", action="store_true")
    parser.add_argument("--yscale", type=str, default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--log_switch_ratio", type=float, default=20.0)
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--a_bit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="scripts/DATELBR/model/bls_pinv_brevitas_simple_caseA1_cyclehalf_wb3_ab3_lam0.0.pth",
    )
    parser.add_argument("--output_dir", type=str, default="output_compare_sameinit_pinv_analytic_vs_bp")
    args = parser.parse_args()

    if args.disable_pinv_warm_start:
        raise ValueError("This comparison is configured for pinv warm-start only. Remove --disable_pinv_warm_start.")

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    rows: List[Dict] = []
    for sc in scenarios:
        print(f"\n===== Compare Scenario: {sc} =====")
        row = run_one_scenario(args, scenario=sc, device=device)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "compare_summary.json")
    out_csv = os.path.join(args.output_dir, "compare_summary.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    save_summary_csv(rows, out_csv)

    print(f"\nSaved summary JSON: {out_json}")
    print(f"Saved summary CSV : {out_csv}")


if __name__ == "__main__":
    main()
