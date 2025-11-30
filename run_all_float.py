import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.data_load import create_harmonic_datasets


class BLSRandomFeatureNet(nn.Module):
    """
    BLS 风格随机特征网络（浮点）:
    - Feature mapping: Linear + Tanh, 输出 a*b 维
    - Enhancement:    Linear + Tanh, 输出 m 维
    - 拼接 [F|H] 后:  Linear 输出 4 个谐波幅值

    这里默认:
      - 前两层作为随机特征映射层 (初始化后不训练)
      - 输出层权重通过最小二乘 / 伪逆从训练集一次性求解
    """
    def __init__(
        self,
        input_dim: int,
        a: int = 8,
        b: int = 8,
        m: int = 8,
        out_dim: int = 4,
        freeze_feature: bool = True,
    ):
        super().__init__()
        feat_dim = a * b

        self.fc_feature = nn.Linear(input_dim, feat_dim)
        self.act_feature = nn.Tanh()

        self.fc_enh = nn.Linear(feat_dim, m)
        self.act_enh = nn.Tanh()

        self.fc_out = nn.Linear(feat_dim + m, out_dim)

        if freeze_feature:
            for p in self.fc_feature.parameters():
                p.requires_grad = False
            for p in self.fc_enh.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        """
        只做前两层，返回特征 Z = [F | H]，shape = (batch, feat_dim + m)
        """
        F = self.act_feature(self.fc_feature(x))
        H = self.act_enh(self.fc_enh(F))
        Z = torch.cat([F, H], dim=1)
        return Z

    def forward(self, x):
        Z = self.forward_features(x)
        y = self.fc_out(Z)
        return y


def compute_output_pinv(model, loader, device, reg_lambda=0.0):
    """
    使用训练集特征 Z 和标签 Y，通过最小二乘 / Ridge 回归
    一次性求解输出层 fc_out 的权重和偏置。

    Z_ext = [Z, 1]   (扩一列常数 1 用于偏置)
    求解 W_ext 使得:  Z_ext @ W_ext ≈ Y

    如果 reg_lambda > 0，则做 Ridge:
      W_ext = (Z^T Z + λ I)^(-1) Z^T Y
    """
    model.eval()
    Z_list = []
    Y_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            Z = model.forward_features(x)   # [B, D]
            Z_list.append(Z.cpu())
            Y_list.append(y.cpu())

    Z_all = torch.cat(Z_list, dim=0)   # [N, D]
    Y_all = torch.cat(Y_list, dim=0)   # [N, out_dim]

    N, D = Z_all.shape
    _, out_dim = Y_all.shape

    # 扩展一列 1 作为偏置项
    ones = torch.ones(N, 1, dtype=Z_all.dtype)
    Z_ext = torch.cat([Z_all, ones], dim=1)  # [N, D+1]

    # 转到 float64 提高数值稳定性 (可选)
    Z_ext = Z_ext.double()
    Y_all = Y_all.double()

    # 维度
    D_ext = D + 1

    if reg_lambda > 0.0:
        # Ridge: (Z^T Z + λ I) W = Z^T Y
        ZT = Z_ext.T                      # [D_ext, N]
        G = ZT @ Z_ext                    # [D_ext, D_ext]
        A = G + reg_lambda * torch.eye(D_ext, dtype=G.dtype)
        B = ZT @ Y_all                    # [D_ext, out_dim]
        W_ext = torch.linalg.solve(A, B)  # [D_ext, out_dim]
    else:
        # 无正则：最小二乘
        # 使用 torch.linalg.lstsq
        sol = torch.linalg.lstsq(Z_ext, Y_all)
        W_ext = sol.solution              # [D_ext, out_dim]

    # 拆成权重和偏置
    W_ext = W_ext.float()
    W = W_ext[:-1, :].T        # [out_dim, D]
    b = W_ext[-1, :]           # [out_dim]

    # 写回到模型 fc_out
    with torch.no_grad():
        model.fc_out.weight.data.copy_(W)
        model.fc_out.bias.data.copy_(b)

    return model


def test_mae_original_scale(model, loader, device, label_scale: float):
    """
    在测试集上计算原始幅值尺度（未缩放前）的 MAE，
    输出 shape = (4,) 对应 [A1, A3, A5, A7]。
    """
    model.eval()
    total_mae = torch.zeros(4, device=device)
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)           # scaled

            y_pred = model(x)

            # 还原到原始幅值尺度
            y_pred_orig = y_pred * label_scale
            y_orig = y * label_scale

            mae = torch.mean(torch.abs(y_pred_orig - y_orig), dim=0)  # [4]
            bs = x.size(0)

            total_mae += mae * bs
            total_samples += bs

    mean_mae = total_mae / max(total_samples, 1)
    return mean_mae.cpu().numpy()


def test_relative_error_stats(model, loader, device, eps: float = 1e-8):
    """
    按论文定义，在测试集上统计各阶次的 relative error：
        RE_{n,i} = |(Y^0_{n,i} - Y_{n,i}) / Y^0_{n,i}| * 100%
    并计算每个阶次的:
        - Mean relative error (平均相对误差)
        - Max relative error  (最大相对误差)

    注意：
        y, y_pred 都是在同一缩放下(除以 label_scale)，相对误差对全局缩放不敏感，
        所以直接用即可。
    """
    model.eval()
    sum_rel = None
    max_rel = None
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)           # y_true (scaled)
            y_pred = model(x)          # y_pred (scaled)

            rel = torch.abs((y_pred - y) / (y + eps)) * 100.0   # [B, 4]

            if sum_rel is None:
                sum_rel = torch.zeros(rel.shape[1], device=device)
                max_rel = torch.zeros(rel.shape[1], device=device)

            sum_rel += rel.sum(dim=0)
            batch_max, _ = torch.max(rel, dim=0)  # [4]
            max_rel = torch.max(max_rel, batch_max)

            total_samples += x.size(0)

    mean_rel = sum_rel / max(total_samples, 1)      # [4]
    return mean_rel.cpu().numpy(), max_rel.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="BLS-style random-feature + pinv baseline on harmonic dataset")
    parser.add_argument("--case", type=str, default="A1", choices=["A1", "A2"],
                        help="Use which dataset case: A1 or A2")
    parser.add_argument("--cycle", type=str, default="half", choices=["half", "quarter", "full"],
                        help="Input length: half=32, quarter=16, full=64")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory where caseA1.npz / caseA2.npz are stored")
    parser.add_argument("--reg_lambda", type=float, default=0.0,
                        help="Ridge regularization lambda for output layer (0 = pure least squares)")
    parser.add_argument("--use_cpu", action="store_true",
                        help="Force to use CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)

    npz_path = os.path.join(args.data_dir, f"case{args.case}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. 请先在 dataset/ 中生成 case{args.case}.npz")

    # 创建 train / val / test 三个 Dataset （这里主要用 train/test，val 可忽略）
    train_ds, val_ds, test_ds = create_harmonic_datasets(
        npz_path=npz_path,
        cycle=args.cycle,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = train_ds.input_len
    label_scale = train_ds.label_scale

    print(f"BLS pinv baseline on case{args.case}, cycle={args.cycle}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    print(f"Input dim: {input_dim}, Label scale: {label_scale}")

    # 构造随机特征 BLS 网络（前两层冻结）
    model = BLSRandomFeatureNet(
        input_dim=input_dim,
        a=8,
        b=8,
        m=8,
        out_dim=4,
        freeze_feature=True,
    ).to(device)

    # 1) 用训练集特征 + 标签，伪逆/最小二乘求输出层权重
    model = compute_output_pinv(model, train_loader, device, reg_lambda=args.reg_lambda)
    print("Output layer solved by least squares (pinv).")

    # 2) 在测试集上评估 MAE（原始幅值）
    test_mae = test_mae_original_scale(model, test_loader, device, label_scale)
    print("BLS pinv baseline - Test MAE per harmonic (original scale):")
    print("  [A1, A3, A5, A7] =", test_mae)

    # 3) 在测试集上按论文方式统计 relative error mean / max
    mean_re, max_re = test_relative_error_stats(model, test_loader, device)
    print("BLS pinv baseline - Test relative error per harmonic (%):")
    print("  Mean RE [A1, A3, A5, A7] =", mean_re)
    print("  Max  RE [A1, A3, A5, A7] =", max_re)

    # 4) 保存模型（包括随机特征层和求解好的输出层）
    os.makedirs("model", exist_ok=True)
    save_name = f"bls_pinv_case{args.case}_cycle{args.cycle}_lam{args.reg_lambda}.pth"
    save_path = os.path.join("model", save_name)
    torch.save(model.state_dict(), save_path)
    print("Saved BLS pinv baseline model to:", save_path)


if __name__ == "__main__":
    main()
#python run_all_float.py --case A1 --cycle half --reg_lambda 1e-3