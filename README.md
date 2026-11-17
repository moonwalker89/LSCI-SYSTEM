# LSCI-SYSTEM · 边缘部署发布包

> 语义图像压缩-重建系统的**权重与推理复现包**（LSCI: Layer-based Semantic Communication system for Images，IEEE JSAC 2023）。
> 本包只包含：**C16 定稿权重、PyTorch 推理/评测脚本、RK3576 INT8 部署件**；不含训练代码与实验记录。

## 结果（12 张固定测试集）

| 平台 | 推理精度 | PSNR / SSIM | 端到端时延 |
|---|---|---|---|
| PC（RTX 4060，PyTorch） | FP32 | **27.36 / 0.8457** | ~24 ms |
| Jetson AGX Orin（PyTorch） | FP16 | 27.36 / 0.8457（与 PC 逐位一致） | ~90 ms |
| **RK3576（RKNN INT8 混合）** | INT8 混合（阈值 0.999） | **26.979 / 0.8403** | ~970 ms（≈1 fps） |

- 模型：Light 编解码器（**4.84 M 参数**），语义特征 16×32×64，码本 L=7（**0.176 bpp**）；
- 方法要点：修复量化器 STE 缺陷使编码器参与量化训练（**+3.3 dB**）；INT8 混合精度（InstanceNorm/ConvTranspose 敏感层 FP16 回退），RKNN 相对 ONNX 掉幅仅 0.38 dB；
- 可伸缩：同一模型支持 L3/L5/L7 多级量化（0.099 / 0.145 / 0.176 bpp）。

## 目录

```
model.py / MyFun.py / SSIM.py    # 网络定义 / 量化器（STE 修复版）/ 指标
inference.py                     # 12 图评测（PSNR/SSIM/耗时/重建图）
rknn_board_test.py               # RK3576 板端实测（RKNN-Lite2 + 纯 numpy）
weights/                         # Light C16 定稿权重（Light_Encoder/Decoder_q_16.pkl）
rknn/                            # RK3576 部署件（ste_encoder/decoder_int8.rknn + MD5）
```

## 快速开始

```bash
pip install -r requirements.txt
# 批量评测（图像目录内任意 512×1024 PNG/JPG）
python inference.py --C 16 --arch light --weights_dir weights \
    --image_dir <你的图像目录> --save_dir outputs
```

板端（RK3576，需 Rockchip 官方 `rknn_toolkit_lite2` aarch64 wheel）：

```bash
python3 rknn_board_test.py --image_dir <测试图目录> --save_dir out \
    --encoder rknn/ste_encoder_int8.rknn --decoder rknn/ste_decoder_int8.rknn --target rk3576
```

## 数据说明

测试图像（Cityscapes）因数据集许可**不随包分发**，请自行获取；脚本接受 512×1024 尺寸的任意图像。

## 来源与致谢

- 方法：Dong C, et al. *Semantic Communication System Based on Semantic Slice Models Propagation*. **IEEE JSAC**, 2023.
- 模型结构与量化实现参考论文参考实现 [TaiGabriel/LSCI-SYSTEM](https://github.com/TaiGabriel/LSCI-SYSTEM)；本包为其**轻量化 / 量化鲁棒性改进 / 边缘部署**方向的复现改进版本。
- 仅用于学术研究。
