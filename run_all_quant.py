import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity, QuantConv2d
from brevitas.core.quant import QuantType
from brevitas.export import export_qonnx

from dataset.data_load import create_harmonic_datasets
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from utils.pruning import analyze_model_sparsity, global_magnitude_prune_with_min
from model.models import (
    BLSQuantRandomNetNoConcat,
    compute_output_pinv_quant,
    test_mae_original_scale,
    test_relative_error_stats,
)

# --------------------- 额外用到的 import ---------------------
import torch.optim as optim
import copy

# --------------------- 固定随机种子 ---------------------
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# --------------------- 1. 冻结剪枝后的零权重 ---------------------
def freeze_zero_weights(model: nn.Module):
    """
    对所有 weight 参数：
    - 找出当前为 0 的位置
    - 为这些 weight 注册 grad hook，在反向传播时强制这些位置的梯度为 0
    这样就算继续训练，也不会“长回”非零权重，剪枝结构被保持住。
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            mask = (param != 0).float()  # 非零为1，零为0

            def hook_factory(mask_):
                def hook(grad):
                    return grad * mask_
                return hook

            param.register_hook(hook_factory(mask))

    print("freeze_zero_weights: 已为所有 weight 注册梯度 mask，剪枝结构将保持不变。")


# --------------------- 2. 微调剪枝后模型 ---------------------
def finetune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    label_scale,
    num_epochs: int = 20,
    lr: float = 1e-3,
):
    """
    对剪枝后的模型进行微调：
    - 保持 0 权重为 0（通过 freeze_zero_weights 的 hook）
    - 只更新非零权重，让模型在当前稀疏结构下重新适配数据
    - 使用 MSELoss 做回归训练
    """

    model.to(device)

    # --- 建议：先做一个 warmup forward，确保 Brevitas 所有 lazy 参数都被创建 ---
    model.train()
    with torch.no_grad():
        for x_warm, _ in train_loader:
            x_warm = x_warm.to(device)
            _ = model(x_warm)
            break  # 只跑一小批就够了

    # 只优化 requires_grad=True 的参数
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

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ------- 每个 epoch 后在训练集 / 验证集上评估 MAE（原始尺度） -------
        model.eval()
        with torch.no_grad():
            train_mae = test_mae_original_scale(model, train_loader, device, label_scale)
            val_mae   = test_mae_original_scale(model, val_loader,   device, label_scale)

        print(f"[Finetune][Epoch {epoch}/{num_epochs}] "
              f"TrainLoss={avg_train_loss:.6f} "
              f"TrainMAE={train_mae} "
              f"ValMAE={val_mae}")

        # 按验证集 MAE 平均值选最优
        val_mae_mean = float(torch.tensor(val_mae).mean().item())
        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            best_state = copy.deepcopy(model.state_dict())

    # 恢复验证集表现最好的权重
    if best_state is not None:
        # 关键：strict=False，忽略某些后来新加的量化内部状态 key
        model.load_state_dict(best_state, strict=False)
        print(f"微调结束，恢复验证集最佳模型，最佳验证 MAE(平均) = {best_val_mae:.6f}")
    else:
        print("微调结束，但未记录到更优的验证集模型（这种情况一般不会发生）")

    return model

# --------------------- 3. 主函数 ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="BLS-style random-feature + pinv with Brevitas quantization (no concat, no residual Add)"
    )
    parser.add_argument("--case", type=str, default="A1", choices=["A1", "A2"])
    parser.add_argument("--cycle", type=str, default="half", choices=["half", "quarter", "full"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--w_bit", type=int, default=3)
    parser.add_argument("--a_bit", type=int, default=3)
    parser.add_argument("--reg_lambda", type=float, default=0.0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--prune_rate", type=float, default=0.9)

    # 新增微调相关超参
    parser.add_argument("--ft_lr", type=float, default=1e-3)
    parser.add_argument("--ft_epochs", type=int, default=50)

    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    # ----------------- 数据集 -----------------
    npz_path = os.path.join(args.data_dir, f"case{args.case}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. generate in dataset/ case{args.case}.npz")

    train_ds, val_ds, test_ds = create_harmonic_datasets(
        npz_path=npz_path,
        cycle=args.cycle,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds.input_len
    label_scale = train_ds.label_scale

    print(f"BLS Brevitas-pinv (no concat, no residual) on case{args.case}, cycle={args.cycle}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    print(f"Input dim: {input_dim}, Label scale: {label_scale}")
    print(f"w_bit = {args.w_bit}, a_bit = {args.a_bit}, reg_lambda = {args.reg_lambda}")

    # ----------------- 构建模型，随机特征 + pinv 拟合输出层 -----------------
    model = BLSQuantRandomNetNoConcat(
        input_dim=input_dim,
        a=8,
        b=8,
        m=8,
        out_dim=4,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        freeze_feature=True,  # 如果想连随机特征一起微调，可以改成 False
    ).to(device)

    # dense 版：只用 H 做特征，用最小二乘/岭回归求输出层权重
    model = compute_output_pinv_quant(model, train_loader, device, reg_lambda=args.reg_lambda)
    print("Quantized output layer (on H only) solved by least squares.")

    # dense 模型 Test 性能
    test_mae = test_mae_original_scale(model, test_loader, device, label_scale)
    print("Dense model - Test MAE per harmonic (original scale):")
    print("  [A1, A3, A5, A7] =", test_mae)

    mean_re, max_re = test_relative_error_stats(model, test_loader, device)
    print("Dense model - Test relative error per harmonic (%):")
    print("  Mean RE [A1, A3, A5, A7] =", mean_re)
    print("  Max  RE [A1, A3, A5, A7] =", max_re)

    # ----------------- （可选）导出 dense QONNX -----------------
    os.makedirs("model", exist_ok=True)
    ckpt_name = f"bls_pinv_brevitas_simple_case{args.case}_cycle{args.cycle}_wb{args.w_bit}_ab{args.a_bit}_lam{args.reg_lambda}.pth"
    ckpt_path = os.path.join("model", ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print("Saved dense model to:", ckpt_path)

    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, input_dim)

    onnx_name = f"bls_pinv_brevitas_simple_case{args.case}_cycle{args.cycle}_wb{args.w_bit}_ab{args.a_bit}.onnx"
    onnx_path = os.path.join("model", onnx_name)
    with torch.no_grad():
        export_qonnx(model_cpu, dummy_input, onnx_path)

    qonnx_cleanup(onnx_path, out_file=onnx_path)
    mw = ModelWrapper(onnx_path)
    mw.set_tensor_datatype(mw.graph.input[0].name, DataType["INT8"])
    mw.set_tensor_datatype(mw.graph.output[0].name, DataType["INT8"])
    print("Exported cleaned dense QONNX model to:", onnx_path)

    # 把模型搬回 device，准备剪枝 + 微调
    model = model.to(device)

    # ----------------- 剪枝前 sparsity 分析 -----------------
    print("\n==== Dense model sparsity ====")
    analyze_model_sparsity(model)

    # ----------------- 3. 全局剪枝（一次性 magnitude） -----------------
    print("\n==== Pruning model ====")
    pruned_model = global_magnitude_prune_with_min(model, target_sparsity=args.prune_rate)
    print("\n==== Pruned model sparsity (before finetune) ====")
    analyze_model_sparsity(pruned_model)

    # 剪枝后立即评估一次
    mean_re, max_re = test_relative_error_stats(pruned_model, test_loader, device)
    print("Pruned model (before finetune) - Test relative error per harmonic (%):")
    print("  Mean RE [A1, A3, A5, A7] =", mean_re)
    print("  Max  RE [A1, A3, A5, A7] =", max_re)

    # ----------------- 4. 冻结零权重 + 微调非零权重 -----------------
    freeze_zero_weights(pruned_model)

    pruned_model = finetune_pruned_model(
        pruned_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        label_scale=label_scale,
        num_epochs=args.ft_epochs,
        lr=args.ft_lr,
    )

    print("\n==== Pruned model sparsity (after finetune, 结构不变) ====")
    analyze_model_sparsity(pruned_model)

    # 微调后的性能
    mean_re, max_re = test_relative_error_stats(pruned_model, test_loader, device)
    print("Pruned + Finetuned model - Test relative error per harmonic (%):")
    print("  Mean RE [A1, A3, A5, A7] =", mean_re)
    print("  Max  RE [A1, A3, A5, A7] =", max_re)

    test_mae = test_mae_original_scale(pruned_model, test_loader, device, label_scale)
    print("Pruned + Finetuned model - Test MAE per harmonic (original scale):")
    print("  [A1, A3, A5, A7] =", test_mae)
    analyze_model_sparsity(pruned_model)

    # ----------------- （可选）固化剪枝并导出稀疏 QONNX -----------------
    # 如果你在 utils.pruning 里有 fuse_pruning，可以在这里调用：
    # from utils.pruning import fuse_pruning
    # fuse_pruning(pruned_model)
    #
    # 然后像上面一样 export_qonnx(pruned_model.cpu().eval(), dummy_input, 稀疏 onnx 路径)

    # 保存 pruned + finetuned 模型
    sparse_ckpt_name = (
        f"bls_pinv_brevitas_simple_case{args.case}_cycle{args.cycle}_"
        f"wb{args.w_bit}_ab{args.a_bit}_lam{args.reg_lambda}_"
        f"prune{args.prune_rate}_ft{args.ft_epochs}.pth"
    )
    sparse_ckpt_path = os.path.join("model", sparse_ckpt_name)
    torch.save(pruned_model.state_dict(), sparse_ckpt_path)
    print("Saved pruned + finetuned model to:", sparse_ckpt_path)

    # export sparse QONNX
    sparse_onnx_name = (
        f"bls_pinv_brevitas_simple_case{args.case}_cycle{args.cycle}_"
        f"wb{args.w_bit}_ab{args.a_bit}_prune{args.prune_rate}.onnx"
    )
    sparse_onnx_path = os.path.join("model", sparse_onnx_name)
    model_cpu = pruned_model.to("cpu").eval()
    with torch.no_grad():
        export_qonnx(model_cpu, dummy_input, sparse_onnx_path)  

    qonnx_cleanup(sparse_onnx_path, out_file=sparse_onnx_path)
    mw = ModelWrapper(sparse_onnx_path)
    mw.set_tensor_datatype(mw.graph.input[0].name, DataType["INT8"])
    mw.set_tensor_datatype(mw.graph.output[0].name, DataType["INT8"])
    print("Exported cleaned sparse QONNX model to:", sparse_onnx_path)

if __name__ == "__main__":
    main()
