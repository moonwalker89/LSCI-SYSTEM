# inference.py — LSCI 端到端推理脚本（PC / Orin 通用）
# 链路: 编码 → 硬量化(argmin 最近邻, 不含 einsum) → 解码；输出重建图 + 分段时间 + PSNR/SSIM
# 指标口径: [-1,1]→[0,1] 归一化下算 PSNR/SSIM（与训练一致）
# 用法:
#   python inference.py --image_dir <测试集> --save_dir <out>                    # 默认 light C16（STE+E2 主力）
#   python inference.py --image_dir <测试集> --save_dir <out> --weights_dir ./model_v8l_ste_multil
#   python inference.py --C 8 --weights_dir ./model_v8l_ste_c8_multil ...        # C8/C4 低码率档
#   python inference.py --image <单张路径> --weights_dir ./model_v8l_ste_multil --C 16 --device cuda
#   python inference.py ... --keep_res      # 保持原图分辨率（裁剪到 16 倍数），不强制 resize 到 1024x512
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from model import Decoder_2, Encoder_2, Light_Encoder, Light_Decoder
from SSIM import SSIM


def hard_quantize(w, L=7):
    """推理专用硬量化（argmin 最近邻星座映射，不含 einsum）。
    与训练软量化在推理路径上数值等价——这是 ONNX/RKNN 阶段要用的路径。"""
    top = (L - 1) // 2
    down = top - L + 1
    centers = torch.arange(down, top + 1, dtype=torch.float32, device=w.device)  # L=7 -> [-3..3]
    dist = torch.abs(w.unsqueeze(-1) - centers)   # [B,C,H,W,L]
    idx = torch.argmin(dist, dim=-1) + down       # [B,C,H,W]
    return idx.float()


def preprocess(image_path, preprocess='cityscapes', keep_res=False):
    """读取图片并返回 [1,3,H,W] float32，像素归一化到 [-1,1]（与训练预处理一致）。
    keep_res=True 时保持原图分辨率（仅居中裁剪到 16 的倍数），不强制 resize。"""
    im = Image.open(image_path).convert('RGB')
    if keep_res:
        w, h = im.size
        w16, h16 = w - w % 16, h - h % 16
        if w16 == 0 or h16 == 0:
            raise SystemExit(f'原图过小: {w}x{h}（需至少 16x16）')
        if (w16, h16) != (w, h):
            left, top = (w - w16) // 2, (h - h16) // 2
            im = im.crop((left, top, left + w16, top + h16))
            print(f'[keep_res] 原图 {w}x{h} → 居中裁剪到 {w16}x{h16}（16 倍数）')
    elif preprocess == 'cityscapes':
        im = im.resize((1024, 512))   # (宽, 高)
    else:  # openimage 预留
        im = im.resize((512, 512))
    arr = np.asarray(im, dtype=np.float32)
    arr = (arr / 255.0) * 2 - 1
    x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x


def psnr(a, b, max_val=1.0):
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', default='./model_v8l_ste_multil', help='权重目录（默认 STE+E2 C16 主力）')
    parser.add_argument('--C', type=int, default=16)
    parser.add_argument('--image', default=None, help='单张图片路径')
    parser.add_argument('--image_dir', default=None, help='图片目录（批量推理）')
    parser.add_argument('--save_dir', default='./outputs', help='重建图输出目录')
    parser.add_argument('--preprocess', default='cityscapes', choices=['cityscapes', 'openimage'])
    parser.add_argument('--keep_res', action='store_true',
                        help='保持原图分辨率（仅居中裁剪到 16 倍数），不强制 resize 到 1024x512')
    parser.add_argument('--arch', default='light', choices=['e2', 'light'], help='light=轻量版（主力）；e2=原版（历史）')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise SystemExit('请用 --image 或 --image_dir 指定输入图片')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    if args.arch == 'light':
        E = Light_Encoder(args.C).to(device).eval()
        G = Light_Decoder(args.C).to(device).eval()
        arch_tag = 'Light'
    else:
        E = Encoder_2(args.C).to(device).eval()
        G = Decoder_2(args.C).to(device).eval()
        arch_tag = 'Cityscapes'
    enc_path = os.path.join(args.weights_dir, f'{arch_tag}_Encoder_q_{args.C}.pkl')
    dec_path = os.path.join(args.weights_dir, f'{arch_tag}_Decoder_q_{args.C}.pkl')
    if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
        raise SystemExit(f'找不到权重: {enc_path} 或 {dec_path}')
    E.load_state_dict(torch.load(enc_path, map_location=device))
    G.load_state_dict(torch.load(dec_path, map_location=device))
    print(f'权重加载: {enc_path} / {dec_path}')

    os.makedirs(args.save_dir, exist_ok=True)
    ssim_fn = SSIM().to(device)

    if args.image:
        paths = [args.image]
    else:
        paths = [os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not paths:
            raise SystemExit(f'目录中没有图片: {args.image_dir}')

    for p in paths:
        x = preprocess(p, args.preprocess, args.keep_res).to(device)   # [1,3,H,W]
        with torch.no_grad():
            t0 = time.time()
            w = E(x) * 3
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            wq = hard_quantize(w)                       # 硬量化，不含 einsum
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.time()
            recon = G(wq)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t3 = time.time()

        r01 = ((recon[0] + 1) / 2).clamp(0, 1)          # [C,H,W] 反归一化
        x01 = ((x[0] + 1) / 2).clamp(0, 1)
        ps = psnr(r01.cpu().numpy(), x01.cpu().numpy())
        ss = ssim_fn(x01.unsqueeze(0), r01.unsqueeze(0)).item()

        name = os.path.splitext(os.path.basename(p))[0]
        out = np.uint8(r01.permute(1, 2, 0).cpu().numpy() * 255)
        out_path = os.path.join(args.save_dir, f'{name}_recon.png')
        Image.fromarray(out).save(out_path)

        print(f'[{name}]')
        print(f'  重建图: {out_path}')
        print(f'  PSNR = {ps:.2f} dB, SSIM = {ss:.4f}')
        print(f'  耗时: 编码 {t1-t0:.3f}s | 量化 {(t2-t1)*1000:.2f}ms | 解码 {t3-t2:.3f}s | 端到端 {t3-t0:.3f}s')


if __name__ == '__main__':
    main()
