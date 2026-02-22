"""快速验证 Q5K 量化误差 vs 权重分布的关系。"""
import numpy as np
import torch
from golden.generate_rwkv_layer import quantize_tile_weights, TILE_DIM

def test_tile(label, w_fp16):
    """量化一个 tile 并报告误差。"""
    w_int, scale_w, min_w = quantize_tile_weights(w_fp16)
    # 反量化
    w_recon = np.zeros_like(w_fp16)
    for i in range(TILE_DIM):
        for j in range(TILE_DIM):
            w_recon[i, j] = np.float16(float(scale_w[j]) * w_int[i, j] + float(min_w[j]))

    diff = np.abs(w_fp16.astype(np.float64) - w_recon.astype(np.float64))
    abs_orig = np.abs(w_fp16.astype(np.float64))
    nz = abs_orig > 1e-6

    max_abs = diff.max()
    mean_rel = np.mean(diff[nz] / abs_orig[nz]) * 100 if nz.any() else 0
    # 排除近零
    sig = abs_orig > 1e-3
    mean_rel_sig = np.mean(diff[sig] / abs_orig[sig]) * 100 if sig.any() else 0

    # matmul 误差: x @ W vs x @ W_recon
    x = np.random.randn(TILE_DIM).astype(np.float16)
    y_orig = x.astype(np.float64) @ w_fp16.astype(np.float64)
    y_recon = x.astype(np.float64) @ w_recon.astype(np.float64)
    y_diff = np.abs(y_orig - y_recon)
    y_abs = np.abs(y_orig)
    y_nz = y_abs > 1e-6
    matmul_mrel = np.mean(y_diff[y_nz] / y_abs[y_nz]) * 100 if y_nz.any() else 0

    print(f"  {label:40s} max_abs={max_abs:.6f}  mrel={mean_rel:.1f}%  "
          f"mrel_sig={mean_rel_sig:.1f}%  matmul_mrel={matmul_mrel:.1f}%")

np.random.seed(42)

print("=== Q5K tile quantization error vs weight distribution ===\n")

# 1. randn * 0.1 (当前用法)
w = np.random.randn(TILE_DIM, TILE_DIM).astype(np.float16) * np.float16(0.1)
test_tile("randn*0.1 (current)", w)

# 2. randn * 0.3 (低秩投影当前用法)
w = np.random.randn(TILE_DIM, TILE_DIM).astype(np.float16) * np.float16(0.3)
test_tile("randn*0.3 (current lowrank)", w)

# 3. randn * 1.0
w = np.random.randn(TILE_DIM, TILE_DIM).astype(np.float16)
test_tile("randn*1.0", w)

# 4. uniform [0, 1]
w = np.random.uniform(0, 1, (TILE_DIM, TILE_DIM)).astype(np.float16)
test_tile("uniform [0,1]", w)

# 5. uniform [-1, 1]
w = np.random.uniform(-1, 1, (TILE_DIM, TILE_DIM)).astype(np.float16)
test_tile("uniform [-1,1]", w)

# 6. randn * 0.1 + 0.5 (偏移, 避免零附近)
w = (np.random.randn(TILE_DIM, TILE_DIM) * 0.1 + 0.5).astype(np.float16)
test_tile("randn*0.1+0.5 (shifted)", w)

# 7. 真实模型权重模拟: 大部分值集中在小范围
w = np.random.randn(TILE_DIM, TILE_DIM).astype(np.float16) * np.float16(0.01)
test_tile("randn*0.01 (very small)", w)

print("\n=== 关键观察 ===")
print("Q5K 只有 31 级, 量化步长 = range/31")
print("对于 randn*0.1, range≈0.6, step≈0.019")
print("权重值 ~0.05 时, 相对误差 = 0.019/2/0.05 = 19%")
print("经过 64 维 matmul, 误差不会抵消(因为是系统性偏差), 会累加")
