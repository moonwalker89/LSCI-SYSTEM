import torch
import numpy as np
from torch.nn import functional as F

def cal_grad_penalty(critic, real_samples, fake_samples):
    """计算critic的惩罚项"""
    device = real_samples.device
    # 定义alpha
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)

    # 从真实数据和生成数据中的连线采样
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)  # 输出形状：[B, 1]

    fake = torch.ones(real_samples.shape[0], 1, device=device)

    # 对采样数据进行求导
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # 返回一个元组(value, )

    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def quantizer(w, L, device):
    top = (L-1)//2
    down = top - L + 1

    [B, W, H, C] = w.shape
    centers = torch.arange(down, top + 1, dtype=torch.float32).to(device)

    centers_stack = centers.reshape(1, 1, 1, 1, L)
    centers_stack = centers_stack.repeat(B, W, H, C, 1)
    # Partition W into the Voronoi tesellation over the centers
    w = w.reshape(B, W, H, C, 1)
    w_stack = w.repeat(1, 1, 1, 1, L)

    w_hard = torch.argmin(torch.abs(w_stack - centers_stack), dim=-1) + down #hard quantization

    smx = F.softmax(1.0 / torch.abs(w_stack - centers_stack + 10e-7), dim=-1)

    # Contract last dimension
    w_soft = torch.einsum('ijklm,m->ijkl', smx, centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

    # Treat quantization as differentiable for optimization
    # STE 修复（2026-09-21 STE_grad_check 实锤原实现缺陷）：
    #   原实现 w_bar = torch.round(w_detch + w_soft)：torch.round 梯度恒为 0，
    #   且 w_detch 已 detach → w_bar 对 w 的梯度为 0 → Encoder 在量化训练中完全无梯度（只训 Decoder）。
    # 修复：前向仍 = 硬量化 argmin（值 = w_hard，不变，下游推理/ONNX/RKNN 零影响），
    #       反向梯度经 w_soft（soft 近似）直通回 Encoder。
    w_detch = (w_hard - w_soft).detach()
    w_bar = w_detch + w_soft

    return w_bar

