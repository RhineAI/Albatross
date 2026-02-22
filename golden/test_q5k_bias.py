"""分析 Q5K 量化误差的系统性偏差 (min_w bias) 效应。"""
import numpy as np
import torch
from golden.generate_rwkv_layer import (
    quantize_tile_weights, dequantize_matrix, quantize_matrix,
    TILE_DIM, C, DTYPE
)

np.random.seed(42)
torch.manual_seed(42)

# 模拟 O_ 矩阵 [64, 64]
W = (torch.randn(C, C) * 0.1).to(DTYPE).numpy()

# 量化
w_int, scale_w, min_w = quantize_matrix(W, 'test', C, C)

# 反量化
W_dq = dequantize_matrix(w_int, scale_w, min_w, C, C).numpy()

# 误差分解
W_err = W_dq.astype(np.float64) - W.astype(np.float64)

# 模拟 gated 向量
x = (torch.randn(C) * 1.0).to(DTYPE).numpy().astype(np.float64)

# 原始 matmul
y_orig = x @ W.astype(np.float64)
y_dq = x @ W_dq.astype(np.float64)
y_err = y_dq - y_orig

print("=== 误差分解: y_err = x @ W_err ===\n")
print(f"x sum = {x.sum():.4f}")
print(f"|x| mean = {np.abs(x).mean():.4f}")
print(f"|y_orig| mean = {np.abs(y_orig).mean():.6f}")
print(f"|y_err| mean = {np.abs(y_err).mean():.6f}")
print(f"mean_rel = {np.mean(np.abs(y_err[y_orig!=0]) / np.abs(y_orig[y_orig!=0])) * 100:.1f}%")

# 分解 W_err 为 bias 部分和 noise 部分
# W_dq[i,j] = scale[j] * q[i,j] + min_dq[j]
# W[i,j] = scale_true[j] * q_true[i,j] + min_true[j]  (不完全成立, 但近似)
# 误差 = (scale_dq - scale_true) * q + (min_dq - min_true) + ...

# 更直接: 看 W_err 每列的均值 (系统性偏差) vs 标准差 (随机噪声)
col_bias = W_err.mean(axis=0)  # 每列的平均误差
col_std = W_err.std(axis=0)    # 每列的误差标准差

print(f"\n--- 每列误差分析 ---")
print(f"|col_bias| mean = {np.abs(col_bias).mean():.6f}")
print(f"col_std mean = {col_std.mean():.6f}")
print(f"bias/std ratio = {np.abs(col_bias).mean() / col_std.mean():.2f}")

# 分离 bias 贡献和 noise 贡献
# y_err[j] = Σ_i x[i] * W_err[i,j]
#           = (Σ_i x[i]) * col_bias[j]  +  Σ_i x[i] * (W_err[i,j] - col_bias[j])
bias_contribution = x.sum() * col_bias
noise_err = W_err - col_bias[np.newaxis, :]
noise_contribution = x @ noise_err

print(f"\n--- y_err 分解 ---")
print(f"|bias_contribution| mean = {np.abs(bias_contribution).mean():.6f}")
print(f"|noise_contribution| mean = {np.abs(noise_contribution).mean():.6f}")
print(f"|total y_err| mean = {np.abs(y_err).mean():.6f}")
print(f"bias 占比 = {np.abs(bias_contribution).mean() / np.abs(y_err).mean() * 100:.1f}%")

# 如果去掉 bias, mean_rel 会是多少?
y_no_bias = y_orig + noise_contribution
y_err_no_bias = y_no_bias - y_orig
nz = y_orig != 0
print(f"\n去掉 bias 后 mean_rel = {np.mean(np.abs(noise_contribution[nz]) / np.abs(y_orig[nz])) * 100:.1f}%")
print(f"原始 mean_rel = {np.mean(np.abs(y_err[nz]) / np.abs(y_orig[nz])) * 100:.1f}%")

# 看看 min_w 的 FP16 截断误差
print(f"\n--- min_w FP16 截断分析 ---")
tr, tc = C // TILE_DIM, C // TILE_DIM
for r in range(tr):
    for c in range(tc):
        tile = W[r*TILE_DIM:(r+1)*TILE_DIM, c*TILE_DIM:(c+1)*TILE_DIM]
        for j in range(min(4, TILE_DIM)):  # 只看前4列
            col = tile[:, j].astype(np.float64)
            col_min = col.min()
            col_max = col.max()
            s = (col_max - col_min) / 31.0
            s_fp16 = float(np.float16(s))
            m_fp16 = float(np.float16(col_min))
            print(f"  tile[{r},{c}] col{j}: range=[{col_min:.6f},{col_max:.6f}] "
                  f"scale={s:.6f}->fp16={s_fp16:.6f} "
                  f"min={col_min:.6f}->fp16={m_fp16:.6f} "
                  f"Δmin={m_fp16-col_min:.6f}")
        break  # 只看第一个 tile
    break
