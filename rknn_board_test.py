# rknn_board_test.py — RK3588 板端实测脚本（rknn-toolkit-lite2 2.3.2，纯 numpy 无 torch）
# 链路: encoder_int8.rknn → 硬量化(argmin) → decoder_int8.rknn；输入 NHWC uint8（mean/std 已烘焙进模型）
# 输出: 三平台测试集逐图 PSNR/SSIM + 四段时延(编码/量化/解码/端到端) + 重建图 + CSV
# 与 PC 侧 verify_rknn.py 同链路/布局约定（encoder 输出 NCHW；decoder 输入转 NHWC；已自适应）
#
# 用法（板上，python3.8~3.12）:
#   python3 rknn_board_test.py --image_dir ./testset --save_dir ./out \
#       --encoder encoder_int8.rknn --decoder decoder_int8.rknn
# 依赖: rknn-toolkit-lite2==2.3.2 + pillow + numpy（lite2 版本与 PC toolkit2/板上 NPU 驱动对齐 2.3.2）
import argparse
import csv
import os
import time

import numpy as np
from PIL import Image


# ---------- numpy SSIM（与 PC SSIM.py 移植一致；12 张验证最大差 7.4e-6） ----------
def numpy_ssim(img1, img2, window_size=11, sigma=1.5):
    """高斯窗 SSIM，等价 torch SSIM.py（窗口 11 / σ1.5 / valid 卷积 / 按通道独立）。输入 [0,1]，(C,H,W) 或 (1,C,H,W)。"""
    if img1.ndim == 3:
        img1, img2 = img1[None], img2[None]
    _, C, H, W = img1.shape
    L = 1.0
    K = min(window_size, H, W)
    x = np.arange(K) - K // 2
    w1 = np.exp(-(x ** 2) / (2 * sigma ** 2))
    w1 = w1 / w1.sum()

    def conv2d_sep(im):
        im = im.astype(np.float64)
        v0 = np.stack([np.convolve(im[:, j], w1, 'valid') for j in range(W)], axis=1)
        return np.stack([np.convolve(v0[i, :], w1, 'valid') for i in range(v0.shape[0])], axis=0)

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    mu1 = np.stack([conv2d_sep(img1[0, c]) for c in range(C)])
    mu2 = np.stack([conv2d_sep(img2[0, c]) for c in range(C)])
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = np.stack([conv2d_sep(img1[0, c] ** 2) for c in range(C)]) - mu1_sq
    sigma2_sq = np.stack([conv2d_sep(img2[0, c] ** 2) for c in range(C)]) - mu2_sq
    sigma12 = np.stack([conv2d_sep(img1[0, c] * img2[0, c]) for c in range(C)]) - mu1_mu2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    return float(((2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)).mean())


# ---------- 硬量化（与推理/ONNX/RKNN 完全一致，不含 einsum） ----------
def hard_quantize(w, L=7):
    top = (L - 1) // 2
    down = top - L + 1
    centers = np.arange(down, top + 1, dtype=np.float32)
    return (np.argmin(np.abs(w[..., None] - centers), axis=-1) + down).astype(np.float32)


def psnr(a, b, max_val=1.0):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')


def load_rknn(path):
    from rknnlite.api import RKNNLite
    rk = RKNNLite()
    if rk.load_rknn(path) != 0:
        raise SystemExit(f'load_rknn 失败: {path}')
    if rk.init_runtime() != 0:
        raise SystemExit(f'init_runtime 失败: {path}')
    return rk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', required=True, help='三平台测试集目录（0-255 PNG）')
    p.add_argument('--save_dir', required=True, help='输出目录（重建图 + CSV）')
    p.add_argument('--encoder', required=True, help='encoder_int8.rknn')
    p.add_argument('--decoder', required=True, help='decoder_int8.rknn')
    p.add_argument('--target', default='rk3568', help='目标平台名（用于 CSV 命名与打印，如 rk3568）')
    return p.parse_args()


def main():
    args = parse_args()
    enc = load_rknn(args.encoder)
    dec = load_rknn(args.decoder)
    os.makedirs(args.save_dir, exist_ok=True)

    # 预热（首张推理含初始化开销，单独跑掉，不计时）
    _ = enc.inference(inputs=[np.zeros((1, 512, 1024, 3), np.uint8)], data_format='nhwc')

    imgs = sorted(f for f in os.listdir(args.image_dir)
                  if f.lower().endswith(('.png', '.jpg')) and 'recon' not in f)
    rows = []
    for fn in imgs:
        im = Image.open(os.path.join(args.image_dir, fn)).convert('RGB').resize((1024, 512))
        x_u8 = np.asarray(im, dtype=np.uint8)[None]            # (1,512,1024,3) NHWC 0-255
        x01 = np.asarray(im, dtype=np.float32) / 255.0         # (512,1024,3) [0,1]

        t0 = time.perf_counter()
        w = np.asarray(enc.inference(inputs=[x_u8], data_format='nhwc')[0])
        t1 = time.perf_counter()
        if w.shape == (1, 32, 64, 16):                         # 保险：若 NPU 输出 NHWC 则转 NCHW
            w = w.transpose(0, 3, 1, 2)
        wq = hard_quantize(w * 3.0)
        t2 = time.perf_counter()
        recon = np.asarray(dec.inference(inputs=[wq.transpose(0, 2, 3, 1)], data_format='nhwc')[0])
        t3 = time.perf_counter()
        if recon.shape == (1, 512, 1024, 3):                   # 保险：若输出 NHWC 则转 NCHW
            recon = recon.transpose(0, 3, 1, 2)

        r01 = np.clip((recon[0] + 1) / 2, 0, 1)                # (3,H,W) [0,1]
        x01_nchw = x01.transpose(2, 0, 1)
        p = psnr(r01, x01_nchw)
        s = numpy_ssim(r01, x01_nchw)
        e_ms = (t1 - t0) * 1000; q_ms = (t2 - t1) * 1000; d_ms = (t3 - t2) * 1000; e2e = (t3 - t0) * 1000

        name = os.path.splitext(fn)[0]
        Image.fromarray(np.uint8(r01.transpose(1, 2, 0) * 255)).save(
            os.path.join(args.save_dir, f'{name}_recon.png'))
        rows.append([name, p, s, e_ms, q_ms, d_ms, e2e])
        print(f'{name}: PSNR={p:.2f} SSIM={s:.4f} | 编码 {e_ms:.1f}ms 量化 {q_ms:.2f}ms 解码 {d_ms:.1f}ms 端到端 {e2e:.1f}ms')

    csv_path = os.path.join(args.save_dir, f'{args.target}_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['文件名', 'PSNR', 'SSIM', '编码ms', '量化ms', '解码ms', '端到端ms'])
        wr.writerows(rows)

    print(f'\n===== {args.target.upper()} 板上实测汇总（{len(imgs)} 张，已预热）=====')
    print(f'PSNR 平均: {np.mean([r[1] for r in rows]):.3f} | SSIM 平均: {np.mean([r[2] for r in rows]):.4f}')
    print(f'编码 {np.mean([r[3] for r in rows]):.1f}ms | 解码 {np.mean([r[5] for r in rows]):.1f}ms | 端到端 {np.mean([r[6] for r in rows]):.1f}ms')
    print('对比参考: PC/Orin(FP32) 24.06/0.7593 ~27ms/~120ms | PC 模拟器 INT8 23.917/0.7514')
    print(f'明细已存: {csv_path}')


if __name__ == '__main__':
    main()
