"""追踪 tmix_out 166% 误差的级联放大链路。
关键路径: gated @ O_ 中, gated = gn_plus_bonus * g_vec
g_vec 本身经过了 sigmoid(xg@g1) @ g2 两级 matmul。
"""
import numpy as np
import torch
import torch.nn.functional as F
from golden.generate_rwkv_layer import (
    quantize_and_save, dequantize_matrix, quantize_matrix,
    compute_metrics, print_metrics,
    TILE_DIM, C, D_G, H, N, DTYPE, OUT_DIR
)

torch.manual_seed(42)
device = "cpu"

def rp(*shape):
    return (torch.randn(*shape, device=device) * 0.02).to(dtype=DTYPE)

# 重建和 main() 完全相同的权重
torch.manual_seed(42)
x_in = (torch.randn(C, device=device) * 0.5).to(DTYPE)
tmix_x_prev = (torch.randn(C, device=device) * 0.5).to(DTYPE)
_ = (torch.randn(C, device=device) * 0.5).to(DTYPE)  # cmix_x_prev
_ = torch.zeros(C, device=device, dtype=DTYPE)  # v_first_in
_ = torch.zeros(H, N, N, device=device, dtype=torch.float32)  # att_state

# 跳过 scalar weights 的生成 (需要和 main 一致)
# 太复杂了, 直接跑 main 然后分析中间值更好

# 换个思路: 直接测试 "单次 64 维 matmul 的误差放大倍数"
print("=== 单次 matmul 误差放大测试 ===\n")

for scale_factor, label in [(0.1, "randn*0.1 (R/K/V/O)"), (0.3, "randn*0.3 (lowrank)")]:
    W = (torch.randn(C, C, device=device) * scale_factor).to(DTYPE)
    W_np = W.numpy()
    w_int, sw, mw = quantize_matrix(W_np, label, C, C)
    W_dq = dequantize_matrix(w_int, sw, mw, C, C)

    # 权重误差
    w_cos, w_max, w_mrel, w_mrel_sig, _ = compute_metrics(W_dq, W)

    # 用不同幅度的输入测试 matmul 误差
    for x_scale, x_label in [(0.5, "|x|~0.4"), (1.0, "|x|~0.8"), (2.0, "|x|~1.6")]:
        x = (torch.randn(C, device=device) * x_scale).to(DTYPE)
        y_fp16 = x @ W
        y_q5k = x @ W_dq
        _, y_max, y_mrel, y_mrel_sig, _ = compute_metrics(y_q5k, y_fp16)
        print(f"  {label:30s} x={x_label:10s} w_mrel_sig={w_mrel_sig:.1f}%  "
              f"y_mrel_sig={y_mrel_sig:.1f}%  amplification={y_mrel_sig/w_mrel_sig:.1f}x")

print("\n=== 两级级联 matmul 误差 ===\n")
# 模拟 g 路径: tg = xg @ g1, tg_sig = sigmoid(tg), g_vec = tg_sig @ g2
g1 = (torch.randn(C, D_G, device=device) * 0.3).to(DTYPE)
g2 = (torch.randn(D_G, C, device=device) * 0.3).to(DTYPE)

g1_np = g1.numpy()
g2_np = g2.numpy()
g1_int, g1_sw, g1_mw = quantize_matrix(g1_np, 'g1', C, D_G)
g2_int, g2_sw, g2_mw = quantize_matrix(g2_np, 'g2', D_G, C)
g1_dq = dequantize_matrix(g1_int, g1_sw, g1_mw, C, D_G)
g2_dq = dequantize_matrix(g2_int, g2_sw, g2_mw, D_G, C)

xg = (torch.randn(C, device=device) * 0.5).to(DTYPE)

# FP16 路径
tg_fp16 = xg @ g1
tg_sig_fp16 = torch.sigmoid(tg_fp16)
g_vec_fp16 = tg_sig_fp16 @ g2

# Q5K 路径
tg_q5k = xg @ g1_dq
tg_sig_q5k = torch.sigmoid(tg_q5k)
g_vec_q5k = tg_sig_q5k @ g2_dq

print_metrics("tg (xg@g1)", tg_q5k, tg_fp16)
print_metrics("tg_sig (sigmoid)", tg_sig_q5k, tg_sig_fp16)
print_metrics("g_vec (tg_sig@g2)", g_vec_q5k, g_vec_fp16)

# 分离: 只有 g2 误差
g_vec_only_g2 = tg_sig_fp16 @ g2_dq
print_metrics("g_vec (g2 only)", g_vec_only_g2, g_vec_fp16)

# 分离: 只有 g1 误差 (通过 tg_sig)
g_vec_only_g1 = tg_sig_q5k @ g2
print_metrics("g_vec (g1 only via sig)", g_vec_only_g1, g_vec_fp16)

print("\n=== 三级级联: gated @ O_ ===\n")
# gated = gn_plus_bonus * g_vec, tmix_out = gated @ O_
# 假设 gn_plus_bonus 有 ~13% 误差
O_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
O_np = O_.numpy()
O_int, O_sw, O_mw = quantize_matrix(O_np, 'O', C, C)
O_dq = dequantize_matrix(O_int, O_sw, O_mw, C, C)

gn_plus_bonus = (torch.randn(C, device=device) * 1.0).to(DTYPE)
# 模拟 13% 误差
gn_err = gn_plus_bonus * (1 + 0.13 * torch.randn(C, device=device).to(DTYPE))

# g_vec 有 ~17% 误差
g_vec_true = (torch.randn(C, device=device) * 1.0).to(DTYPE)
g_vec_err = g_vec_true * (1 + 0.17 * torch.randn(C, device=device).to(DTYPE))

gated_true = gn_plus_bonus * g_vec_true
gated_err = gn_err * g_vec_err

tmix_true = gated_true @ O_
tmix_q5k = gated_err @ O_dq

print_metrics("gated (simulated)", gated_err, gated_true)
print_metrics("tmix (simulated)", tmix_q5k, tmix_true)
