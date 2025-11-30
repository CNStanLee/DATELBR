import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity
from brevitas.core.quant import QuantType
from brevitas.export import export_qonnx

from dataset.data_load import create_harmonic_datasets
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import cleanup as qonnx_cleanup

class BLSQuantRandomNetNoConcat(nn.Module):
    """
    精简版：BLS 风格随机特征网络（无 concat，无残差 Add）
    结构：
        x -> input_quant -> fc_feature -> act_feature -> F
        F -> fc_enh -> act_enh -> H
        H -> fc_out -> y
    """

    def __init__(
        self,
        input_dim: int,
        a: int = 8,
        b: int = 8,
        m: int = 8,
        out_dim: int = 4,
        w_bit: int = 8,
        a_bit: int = 8,
        freeze_feature: bool = True,
    ):
        super().__init__()
        feat_dim = a * b
        self.feat_dim = feat_dim
        self.m = m
        self.out_dim = out_dim

        # 输入量化，确保整个图从一开始就是量化路径
        self.input_quant = QuantIdentity(
            bit_width=3,
            return_quant_tensor=False,
        )
        self.output_quant = QuantIdentity(
            bit_width=8,
            return_quant_tensor=False,
        )

        # x -> F
        self.fc_feature = QuantLinear(
            in_features=input_dim,
            out_features=feat_dim,
            bias=True,
            weight_bit_width=w_bit,
            bias_bit_width=3,
            weight_quant_type=QuantType.INT,
        )
        self.act_feature = QuantReLU(bit_width=a_bit)

        # F -> H
        self.fc_enh = QuantLinear(
            in_features=feat_dim,
            out_features=m,
            bias=True,
            weight_bit_width=w_bit,
            bias_bit_width=3,
            weight_quant_type=QuantType.INT,
        )
        self.act_enh = QuantReLU(bit_width=a_bit)

        # 最终只保留一个输出头：H -> y
        self.fc_out = QuantLinear(
            in_features=m,
            out_features=out_dim,
            bias=True,
            weight_bit_width=w_bit,
            bias_bit_width=16,
            weight_quant_type=QuantType.INT,
        )

        if freeze_feature:
            for p in self.fc_feature.parameters():
                p.requires_grad = False
            for p in self.fc_enh.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        """
        返回 F, H（给 pinv 用），注意也要走 input_quant
        """
        x = self.input_quant(x)
        F = self.act_feature(self.fc_feature(x))
        H = self.act_enh(self.fc_enh(F))
        return F, H

    def forward(self, x):
        x = self.input_quant(x)
        F = self.act_feature(self.fc_feature(x))
        H = self.act_enh(self.fc_enh(F))
        y = self.fc_out(H)  
        # y = self.output_quant(y)
        return y


def compute_output_pinv_quant(model, loader, device, reg_lambda=0.0):
    """
    使用 H 特征和标签 Y，最小二乘 / Ridge 求解 fc_out 的权重：
        H_ext = [H, 1]  (扩一列常数项)
        求解 W_ext 使 H_ext W_ext ≈ Y
        然后：
            W = W_ext[:-1, :]
            b = W_ext[-1, :]
        写回：
            fc_out.weight = W^T
            fc_out.bias   = b
    """
    model.eval()
    H_list = []
    Y_list = []

    m = model.m

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            _, H = model.forward_features(x)   # 只用 H 做特征
            H_list.append(H.cpu())
            Y_list.append(y.cpu())

    H_all = torch.cat(H_list, dim=0)   # [N, m]
    Y_all = torch.cat(Y_list, dim=0)   # [N, out_dim]

    N, D = H_all.shape
    _, out_dim = Y_all.shape
    assert D == m

    # 扩展偏置列
    ones = torch.ones(N, 1, dtype=H_all.dtype)
    H_ext = torch.cat([H_all, ones], dim=1)   # [N, D+1]

    H_ext = H_ext.double()
    Y_all = Y_all.double()

    D_ext = D + 1

    if reg_lambda > 0.0:
        HT = H_ext.T
        G = HT @ H_ext
        A = G + reg_lambda * torch.eye(D_ext, dtype=G.dtype)
        B = HT @ Y_all
        W_ext = torch.linalg.solve(A, B)
    else:
        sol = torch.linalg.lstsq(H_ext, Y_all)
        W_ext = sol.solution

    W_ext = W_ext.float()
    W = W_ext[:-1, :]   # [D, out_dim]
    b = W_ext[-1, :]    # [out_dim]

    with torch.no_grad():
        # fc_out: weight [out_dim, m], bias [out_dim]
        model.fc_out.weight.data.copy_(W.T)
        model.fc_out.bias.data.copy_(b)

    return model


def test_mae_original_scale(model, loader, device, label_scale: float):
    model.eval()
    total_mae = torch.zeros(4, device=device)
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            y_pred_orig = y_pred * label_scale
            y_orig = y * label_scale

            mae = torch.mean(torch.abs(y_pred_orig - y_orig), dim=0)
            bs = x.size(0)

            total_mae += mae * bs
            total_samples += bs

    mean_mae = total_mae / max(total_samples, 1)
    return mean_mae.cpu().numpy()


def test_relative_error_stats(model, loader, device, eps: float = 1e-8):
    model.eval()
    sum_rel = None
    max_rel = None
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            rel = torch.abs((y_pred - y) / (y + eps)) * 100.0

            if sum_rel is None:
                sum_rel = torch.zeros(rel.shape[1], device=device)
                max_rel = torch.zeros(rel.shape[1], device=device)

            sum_rel += rel.sum(dim=0)
            batch_max, _ = torch.max(rel, dim=0)
            max_rel = torch.max(max_rel, batch_max)

            total_samples += x.size(0)

    mean_rel = sum_rel / max(total_samples, 1)
    return mean_rel.cpu().numpy(), max_rel.cpu().numpy()