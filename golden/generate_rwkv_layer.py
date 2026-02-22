"""
RWKV7 Layer golden data generator — 为 rwkv_layer.sv RTL 测试生成数据。

维度: C=64, H=1, N=64, FFN_DIM=128, D_W/A/V/G=32, LAYER_ID=0
矩阵权重做 Q5K 分块量化后导出。

数据流: x_in -> LN1 -> TMix -> +residual -> LN2 -> CMix -> +residual -> x_out

输出: D:/VMwareShare/rhine-chip-rwkv-design/golden/rwkv_layer/ 下的 hex 文件
用法: cd C:/Projects/Albatross && uv run -m golden.generate_rwkv_layer
"""

import math
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from golden.utils import save
from reference.rwkv7 import DTYPE

# ── 维度常量 (与 RTL rwkv_layer.sv 默认参数一致) ──
C = 64
H = 1
N = 64
FFN_DIM = 128
D_W = 32
D_A = 32
D_V = 32
D_G = 32
LAYER_ID = 0

TILE_DIM = 32
Q_BITS = 5
Q_MAX = 31
Q_MIN = 0

EXP_NEG_HALF = math.exp(-0.5)

OUT_DIR = r"D:\VMwareShare\rhine-chip-rwkv-design\golden\rwkv_layer"


# ══════════════════════════════════════════════════════════════
# Q5K 量化工具 (从 gen_rhine_vmm_v1_golden.py 移植)
# ══════════════════════════════════════════════════════════════

def fp16_to_uint16(val):
    return int(np.float16(val).view(np.uint16))


def quantize_tile_weights(w_tile_fp16):
    """Q5_K 非对称量化 (按列): w_real[i][j] = scale_w[j] * w_int[i][j] + min_w[j]"""
    w_int = np.zeros((TILE_DIM, TILE_DIM), dtype=np.int32)
    scale_w = np.zeros(TILE_DIM, dtype=np.float16)
    min_w = np.zeros(TILE_DIM, dtype=np.float16)

    for j in range(TILE_DIM):
        col = w_tile_fp16[:, j].astype(np.float64)
        col_min = np.min(col)
        col_max = np.max(col)

        if col_max == col_min:
            scale_w[j] = np.float16(0.0)
            min_w[j] = np.float16(col_min)
            w_int[:, j] = 0
            continue

        s = (col_max - col_min) / float(Q_MAX - Q_MIN)
        scale_w[j] = np.float16(s)
        min_w[j] = np.float16(col_min - s * Q_MIN)
        s_f = float(scale_w[j])
        m_f = float(min_w[j])

        if s_f == 0.0:
            w_int[:, j] = 0
            continue

        for i in range(TILE_DIM):
            q = round((float(col[i]) - m_f) / s_f)
            w_int[i, j] = max(Q_MIN, min(Q_MAX, int(q)))

    return w_int, scale_w, min_w


def quantize_matrix(W_fp16_np, name, rows, cols):
    """对整个矩阵做 Q5K 分块量化, 返回 tile 化的量化数据。
    W_fp16_np: [rows, cols] numpy float16
    返回: w_int[tr][tc], scale_w[tr][tc], min_w[tr][tc]
    """
    tr = rows // TILE_DIM
    tc = cols // TILE_DIM
    w_int = np.zeros((tr, tc, TILE_DIM, TILE_DIM), dtype=np.int32)
    scale_w_arr = np.zeros((tr, tc, TILE_DIM), dtype=np.float16)
    min_w_arr = np.zeros((tr, tc, TILE_DIM), dtype=np.float16)

    for r in range(tr):
        for c in range(tc):
            tile = W_fp16_np[r*TILE_DIM:(r+1)*TILE_DIM, c*TILE_DIM:(c+1)*TILE_DIM]
            w_int[r][c], scale_w_arr[r][c], min_w_arr[r][c] = quantize_tile_weights(tile)

    print(f"  Quantized {name}: [{rows},{cols}] -> {tr}x{tc} tiles")
    return w_int, scale_w_arr, min_w_arr


def save_q5k(out_dir, name, w_int, scale_w, min_w, tr, tc):
    """导出 Q5K 量化数据为 hex 文件。"""
    # weight_int5.hex
    path = os.path.join(out_dir, f"{name}.weight_int5.hex")
    with open(path, 'w') as f:
        for r in range(tr):
            for c in range(tc):
                for i in range(TILE_DIM):
                    for j in range(TILE_DIM):
                        f.write(f"{int(w_int[r][c][i][j]) & 0x1F:02x}\n")
    cnt = tr * tc * TILE_DIM * TILE_DIM
    print(f"  {name}.weight_int5.hex: {cnt} values")

    # scale_w.hex
    path = os.path.join(out_dir, f"{name}.scale_w.hex")
    with open(path, 'w') as f:
        for r in range(tr):
            for c in range(tc):
                for j in range(TILE_DIM):
                    f.write(f"{fp16_to_uint16(scale_w[r][c][j]):04x}\n")
    cnt = tr * tc * TILE_DIM
    print(f"  {name}.scale_w.hex: {cnt} values")

    # min_w.hex
    path = os.path.join(out_dir, f"{name}.min_w.hex")
    with open(path, 'w') as f:
        for r in range(tr):
            for c in range(tc):
                for j in range(TILE_DIM):
                    f.write(f"{fp16_to_uint16(min_w[r][c][j]):04x}\n")
    cnt = tr * tc * TILE_DIM
    print(f"  {name}.min_w.hex: {cnt} values")


def dequantize_matrix(w_int, scale_w, min_w, rows, cols):
    """Q5K 反量化: 从量化数据重建 FP16 矩阵, 返回 torch DTYPE tensor。"""
    tr = rows // TILE_DIM
    tc = cols // TILE_DIM
    W_recon = np.zeros((rows, cols), dtype=np.float16)
    for r in range(tr):
        for c in range(tc):
            for i in range(TILE_DIM):
                for j in range(TILE_DIM):
                    s = float(scale_w[r][c][j])
                    m = float(min_w[r][c][j])
                    W_recon[r*TILE_DIM+i, c*TILE_DIM+j] = np.float16(
                        s * w_int[r][c][i][j] + m
                    )
    return torch.from_numpy(W_recon).to(DTYPE)


def quantize_and_save(out_dir, name, W_tensor, rows, cols):
    """量化矩阵并导出, 返回 (量化数据, 反量化 tensor)。"""
    W_np = W_tensor.detach().cpu().to(torch.float16).numpy()
    w_int, scale_w, min_w = quantize_matrix(W_np, name, rows, cols)
    tr, tc = rows // TILE_DIM, cols // TILE_DIM
    save_q5k(out_dir, name, w_int, scale_w, min_w, tr, tc)
    W_dq = dequantize_matrix(w_int, scale_w, min_w, rows, cols)
    return w_int, scale_w, min_w, W_dq


# ══════════════════════════════════════════════════════════════
# 误差指标 (与 RTL error_metrics.svh 一致)
# ══════════════════════════════════════════════════════════════

def compute_metrics(a: torch.Tensor, b: torch.Tensor):
    """计算 cos_sim, max_abs, mean_rel (与 error_metrics.svh 一致)。
    a = RTL/Q5K 侧, b = golden/FP16 侧。"""
    a_np = a.detach().cpu().float().numpy().flatten()
    b_np = b.detach().cpu().float().numpy().flatten()

    dot = np.sum(a_np * b_np)
    norm_a = np.sqrt(np.sum(a_np * a_np))
    norm_b = np.sqrt(np.sum(b_np * b_np))
    cos_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else (1.0 if norm_a == 0 and norm_b == 0 else 0.0)

    diff = np.abs(a_np - b_np)
    max_abs = float(np.max(diff))

    abs_b = np.abs(b_np)
    nz = abs_b > 0
    mean_rel = float(np.mean(diff[nz] / abs_b[nz]) * 100.0) if np.any(nz) else 0.0

    return cos_sim, max_abs, mean_rel


def print_metrics(label: str, a: torch.Tensor, b: torch.Tensor):
    """打印误差指标, 格式与 RTL testbench 一致。"""
    cos_sim, max_abs, mean_rel = compute_metrics(a, b)
    print(f"  {label:35s} cos_sim={cos_sim:.10f}  max_abs={max_abs:.6e}  mean_rel={mean_rel:.4f}%")
    return cos_sim, max_abs, mean_rel


# ══════════════════════════════════════════════════════════════
# WKV7 核心 (从 generate_tmix_mini.py 复制)
# ══════════════════════════════════════════════════════════════

def _wkv7_core(state, r, w, k, v, a, b):
    _H, _N = r.shape
    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    w = torch.exp(-EXP_NEG_HALF * w.float())

    sa = torch.einsum('hn,hin->hi', a, state)
    state.mul_(w.unsqueeze(1))
    state.add_(k.unsqueeze(1) * v.unsqueeze(2))
    state.add_(sa.unsqueeze(2) * b.unsqueeze(1))

    y = torch.einsum('hin,hn->hi', state, r)
    return y.to(DTYPE)


def RWKV7_ONE_OP(state, r, w, k, v, a, b):
    _C = r.shape[0]
    _H = _C // N
    y = _wkv7_core(state, r.view(_H, N), w.view(_H, N), k.view(_H, N),
                    v.view(_H, N), a.view(_H, N), b.view(_H, N))
    return y.view(_C)


# ══════════════════════════════════════════════════════════════
# TMix forward (从 generate_tmix_mini.py 改写, 维度 C=64)
# ══════════════════════════════════════════════════════════════

def tmix_forward(D, layer_id, x, x_prev, v_first, state,
                 x_r, x_w, x_k, x_v, x_a, x_g,
                 w0, w1, w2, a0, a1, a2, v0, v1, v2,
                 g1, g2, k_k, k_a, r_k,
                 R_, K_, V_, O_,
                 ln_w, ln_b):
    """TMix forward, 导出中间结果。"""
    # Stage 1: Delta Mix
    xx = x_prev[0] - x
    x_prev[0] = x

    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    # Stage 2: Projections
    r = xr @ R_
    k = xk @ K_
    v = xv @ V_

    tw = xw @ w1
    ta = xa @ a1
    tv = xv @ v1
    tg = xg @ g1

    # Stage 3a: Decay
    tw_tanh = torch.tanh(tw)
    w_proj2 = tw_tanh @ w2
    w_sum = w0 + w_proj2
    w_sigmoid = torch.sigmoid(w_sum)

    # Stage 3b: Alpha
    a_proj2 = ta @ a2
    a_sum = a0 + a_proj2
    a_vec = torch.sigmoid(a_sum)

    # Stage 3c: Gate
    tg_sig = torch.sigmoid(tg)
    g_vec = tg_sig @ g2

    # Stage 3d: V-first coeff
    v_proj2 = tv @ v2
    vf_sum = v0 + v_proj2
    vf_coeff = torch.sigmoid(vf_sum)

    # Stage 4: Key norm + modulation
    kk_raw = k * k_k
    kk_norm = F.normalize(kk_raw.view(H, N), dim=-1, p=2.0).view(H * N)

    a_minus_1 = a_vec - 1.0
    scale = 1.0 + a_minus_1 * k_a
    k_mod = k * scale

    # Stage 5: V-first mix
    if layer_id == 0:
        v_first = v.clone()
    else:
        v = v + (v_first - v) * vf_coeff

    # Stage 6: WKV7 core
    neg_kk = -kk_norm
    kk_alpha = kk_norm * a_vec

    wkv_out = RWKV7_ONE_OP(state, r, w_sigmoid, k_mod, v, neg_kk, kk_alpha)
    if D is not None:
        save(D, 'att.wkv.out', wkv_out)
        save(D, 'att_state.out', state)

    # Stage 7: GroupNorm
    gn_out = F.group_norm(
        wkv_out.view(1, H * N), num_groups=H,
        weight=ln_w, bias=ln_b, eps=64e-5
    ).view(H * N)

    # Stage 8: Bonus
    bonus = ((r * k_mod * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    gn_plus_bonus = gn_out + bonus

    # Stage 9: Gate + O_ projection
    gated = gn_plus_bonus * g_vec
    tmix_out = gated @ O_
    if D is not None:
        save(D, 'att.out', tmix_out)

    return tmix_out, v_first


# ══════════════════════════════════════════════════════════════
# CMix forward (从 generate_cmix_mini.py 改写, 维度 C=64, FFN_DIM=128)
# ══════════════════════════════════════════════════════════════

def cmix_forward(D, x, x_prev, x_k, K_, V_):
    """CMix forward, 导出中间结果。"""
    xx = x_prev[1] - x
    x_prev[1] = x

    k = x + xx * x_k

    k_expanded = k @ K_
    k_act = torch.relu(k_expanded) ** 2

    out = k_act @ V_
    if D is not None:
        save(D, 'ffn.out', out)

    return out


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    if os.path.exists(OUT_DIR):
        for f in glob.glob(os.path.join(OUT_DIR, '*.hex')):
            os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    device = "cpu"

    print("=" * 60)
    print("Generating rwkv_layer golden data")
    print(f"  C={C}, H={H}, N={N}, FFN_DIM={FFN_DIM}")
    print(f"  D_W={D_W}, D_A={D_A}, D_V={D_V}, D_G={D_G}")
    print(f"  LAYER_ID={LAYER_ID}")
    print(f"  Output: {OUT_DIR}/")
    print("=" * 60)

    # ── helper ──
    def rp(*shape):
        return (torch.randn(*shape, device=device) * 0.02).to(dtype=DTYPE)

    # ================================================================
    # 生成随机输入 & 状态
    # ================================================================
    print("\n--- Inputs & States ---")
    x_in = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    tmix_x_prev = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    cmix_x_prev = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    v_first_in = torch.zeros(C, device=device, dtype=DTYPE)
    att_state = torch.zeros(H, N, N, device=device, dtype=torch.float32)

    save(OUT_DIR, 'x_in', x_in)
    save(OUT_DIR, 'tmix_x_prev.in', tmix_x_prev)
    save(OUT_DIR, 'cmix_x_prev.in', cmix_x_prev)
    save(OUT_DIR, 'v_first.in', v_first_in)
    save(OUT_DIR, 'att_state.in', att_state)

    # ================================================================
    # 生成标量向量权重 (FP16, 直接输入端口)
    # ================================================================
    print("\n--- Scalar Vector Weights ---")
    ln1_w = (torch.rand(C, device=device) * 0.5 + 0.75).to(DTYPE)
    ln1_b = (torch.randn(C, device=device) * 0.1).to(DTYPE)
    ln2_w = (torch.rand(C, device=device) * 0.5 + 0.75).to(DTYPE)
    ln2_b = (torch.randn(C, device=device) * 0.1).to(DTYPE)

    save(OUT_DIR, 'ln1.weight', ln1_w)
    save(OUT_DIR, 'ln1.bias', ln1_b)
    save(OUT_DIR, 'ln2.weight', ln2_w)
    save(OUT_DIR, 'ln2.bias', ln2_b)

    # TMix delta mix 系数
    att_x_r = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    att_x_w = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    att_x_k = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    att_x_v = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    att_x_a = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    att_x_g = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)

    for name, val in [('att.x_r', att_x_r), ('att.x_w', att_x_w), ('att.x_k', att_x_k),
                       ('att.x_v', att_x_v), ('att.x_a', att_x_a), ('att.x_g', att_x_g)]:
        save(OUT_DIR, name, val)

    # TMix decay/alpha/v-first 偏置
    att_w0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    att_a0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)
    att_v0 = (torch.randn(C, device=device) * 0.5).to(DTYPE)

    save(OUT_DIR, 'att.w0', att_w0)
    save(OUT_DIR, 'att.a0', att_a0)
    save(OUT_DIR, 'att.v0', att_v0)

    # TMix key/bonus
    att_k_k = (torch.randn(C, device=device) * 0.3).to(DTYPE)
    att_k_a = (torch.randn(C, device=device) * 0.3).to(DTYPE)
    att_r_k = (torch.randn(C, device=device) * 0.3).to(DTYPE)

    save(OUT_DIR, 'att.k_k', att_k_k)
    save(OUT_DIR, 'att.k_a', att_k_a)
    save(OUT_DIR, 'att.r_k', att_r_k)

    # TMix GroupNorm
    att_ln_w = torch.ones(C, device=device, dtype=DTYPE)
    att_ln_b = torch.zeros(C, device=device, dtype=DTYPE)

    save(OUT_DIR, 'att.ln_x.weight', att_ln_w)
    save(OUT_DIR, 'att.ln_x.bias', att_ln_b)

    # CMix delta mix 系数
    ffn_x_k = (torch.rand(C, device=device) * 0.8 + 0.1).to(DTYPE)
    save(OUT_DIR, 'ffn.x_k', ffn_x_k)

    # ================================================================
    # 生成矩阵权重 (FP16) 并做 Q5K 量化导出
    # ================================================================
    print("\n--- Matrix Weights (Q5K quantized) ---")

    # TMix 投影矩阵 [C, C]
    R_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    K_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    V_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)
    O_ = (torch.randn(C, C, device=device) * 0.1).to(DTYPE)

    # TMix 低秩投影
    w1 = (torch.randn(C, D_W, device=device) * 0.3).to(DTYPE)
    w2 = (torch.randn(D_W, C, device=device) * 0.3).to(DTYPE)
    a1 = (torch.randn(C, D_A, device=device) * 0.3).to(DTYPE)
    a2 = (torch.randn(D_A, C, device=device) * 0.3).to(DTYPE)
    v1 = (torch.randn(C, D_V, device=device) * 0.3).to(DTYPE)
    v2 = (torch.randn(D_V, C, device=device) * 0.3).to(DTYPE)
    g1 = (torch.randn(C, D_G, device=device) * 0.3).to(DTYPE)
    g2 = (torch.randn(D_G, C, device=device) * 0.3).to(DTYPE)

    # CMix 投影矩阵
    ffn_K = (torch.randn(C, FFN_DIM, device=device) * 0.3).to(DTYPE)
    ffn_V = (torch.randn(FFN_DIM, C, device=device) * 0.3).to(DTYPE)

    # 量化并导出所有矩阵, 同时获取反量化权重
    # TMix grid 0-3: R_, K_, V_, O_ [C, C]
    _, _, _, R_dq = quantize_and_save(OUT_DIR, 'att.R', R_, C, C)
    _, _, _, K_dq = quantize_and_save(OUT_DIR, 'att.K', K_, C, C)
    _, _, _, V_dq = quantize_and_save(OUT_DIR, 'att.V', V_, C, C)
    _, _, _, O_dq = quantize_and_save(OUT_DIR, 'att.O', O_, C, C)

    # TMix grid 4-7: w1 [C, D_W], a1 [C, D_A], v1 [C, D_V], g1 [C, D_G]
    _, _, _, w1_dq = quantize_and_save(OUT_DIR, 'att.w1', w1, C, D_W)
    _, _, _, a1_dq = quantize_and_save(OUT_DIR, 'att.a1', a1, C, D_A)
    _, _, _, v1_dq = quantize_and_save(OUT_DIR, 'att.v1', v1, C, D_V)
    _, _, _, g1_dq = quantize_and_save(OUT_DIR, 'att.g1', g1, C, D_G)

    # TMix grid 8-11: w2 [D_W, C], a2 [D_A, C], v2 [D_V, C], g2 [D_G, C]
    _, _, _, w2_dq = quantize_and_save(OUT_DIR, 'att.w2', w2, D_W, C)
    _, _, _, a2_dq = quantize_and_save(OUT_DIR, 'att.a2', a2, D_A, C)
    _, _, _, v2_dq = quantize_and_save(OUT_DIR, 'att.v2', v2, D_V, C)
    _, _, _, g2_dq = quantize_and_save(OUT_DIR, 'att.g2', g2, D_G, C)

    # CMix: ffn_K [C, FFN_DIM], ffn_V [FFN_DIM, C]
    _, _, _, ffn_K_dq = quantize_and_save(OUT_DIR, 'ffn.K', ffn_K, C, FFN_DIM)
    _, _, _, ffn_V_dq = quantize_and_save(OUT_DIR, 'ffn.V', ffn_V, FFN_DIM, C)

    # ================================================================
    # Forward: LN1
    # ================================================================
    print("\n--- Forward Pass ---")
    ln1_out = F.layer_norm(x_in.float(), (C,),
                           weight=ln1_w.float(), bias=ln1_b.float()).to(DTYPE)
    save(OUT_DIR, 'ln1.out', ln1_out)

    # ================================================================
    # Forward: TMix
    # ================================================================
    x_prev = [tmix_x_prev.clone(), cmix_x_prev.clone()]
    v_first = v_first_in.clone()

    tmix_out, v_first = tmix_forward(
        OUT_DIR, layer_id=LAYER_ID,
        x=ln1_out, x_prev=x_prev, v_first=v_first, state=att_state,
        x_r=att_x_r, x_w=att_x_w, x_k=att_x_k, x_v=att_x_v, x_a=att_x_a, x_g=att_x_g,
        w0=att_w0, w1=w1, w2=w2,
        a0=att_a0, a1=a1, a2=a2,
        v0=att_v0, v1=v1, v2=v2,
        g1=g1, g2=g2,
        k_k=att_k_k, k_a=att_k_a, r_k=att_r_k,
        R_=R_, K_=K_, V_=V_, O_=O_,
        ln_w=att_ln_w, ln_b=att_ln_b
    )

    # ================================================================
    # Forward: Residual 1
    # ================================================================
    x_after_att = (x_in.float() + tmix_out.float()).to(DTYPE)
    save(OUT_DIR, 'x_after_att', x_after_att)

    # ================================================================
    # Forward: LN2
    # ================================================================
    ln2_out = F.layer_norm(x_after_att.float(), (C,),
                           weight=ln2_w.float(), bias=ln2_b.float()).to(DTYPE)
    save(OUT_DIR, 'ln2.out', ln2_out)

    # ================================================================
    # Forward: CMix
    # ================================================================
    cmix_out = cmix_forward(OUT_DIR, ln2_out, x_prev, ffn_x_k, ffn_K, ffn_V)

    # ================================================================
    # Forward: Residual 2
    # ================================================================
    x_out = (x_after_att.float() + cmix_out.float()).to(DTYPE)
    save(OUT_DIR, 'x_out', x_out)

    # ================================================================
    # 导出状态输出
    # ================================================================
    print("\n--- State Outputs ---")
    save(OUT_DIR, 'tmix_x_prev.out', x_prev[0])
    save(OUT_DIR, 'cmix_x_prev.out', x_prev[1])
    save(OUT_DIR, 'v_first.out', v_first)
    save(OUT_DIR, 'att_state.out', att_state)

    # ================================================================
    # Q5K Forward: 用反量化权重重新跑一遍, 生成 q5k.* golden data
    # ================================================================
    print("\n--- Q5K Forward Pass (dequantized weights) ---")

    # 重置状态 (与原始 forward 相同初始条件)
    att_state_q5k = torch.zeros(H, N, N, device=device, dtype=torch.float32)
    x_prev_q5k = [tmix_x_prev.clone(), cmix_x_prev.clone()]
    v_first_q5k = v_first_in.clone()

    # LN1 (与原始相同, 不涉及矩阵权重)
    q5k_ln1_out = ln1_out  # 完全相同

    # TMix with Q5K weights
    q5k_tmix_out, q5k_v_first = tmix_forward(
        None, layer_id=LAYER_ID,
        x=q5k_ln1_out, x_prev=x_prev_q5k, v_first=v_first_q5k, state=att_state_q5k,
        x_r=att_x_r, x_w=att_x_w, x_k=att_x_k, x_v=att_x_v, x_a=att_x_a, x_g=att_x_g,
        w0=att_w0, w1=w1_dq, w2=w2_dq,
        a0=att_a0, a1=a1_dq, a2=a2_dq,
        v0=att_v0, v1=v1_dq, v2=v2_dq,
        g1=g1_dq, g2=g2_dq,
        k_k=att_k_k, k_a=att_k_a, r_k=att_r_k,
        R_=R_dq, K_=K_dq, V_=V_dq, O_=O_dq,
        ln_w=att_ln_w, ln_b=att_ln_b
    )

    # Residual 1
    q5k_x_after_att = (x_in.float() + q5k_tmix_out.float()).to(DTYPE)

    # LN2
    q5k_ln2_out = F.layer_norm(q5k_x_after_att.float(), (C,),
                                weight=ln2_w.float(), bias=ln2_b.float()).to(DTYPE)

    # CMix with Q5K weights
    q5k_cmix_out = cmix_forward(None, q5k_ln2_out, x_prev_q5k, ffn_x_k, ffn_K_dq, ffn_V_dq)

    # Residual 2
    q5k_x_out = (q5k_x_after_att.float() + q5k_cmix_out.float()).to(DTYPE)

    # 保存 Q5K golden data (覆盖 tmix_forward/cmix_forward 内部 save 的文件)
    save(OUT_DIR, 'q5k.att.wkv.out', q5k_tmix_out)  # tmix_forward 已保存了 att.wkv.out, 这里用 q5k 前缀
    save(OUT_DIR, 'q5k.att.out', q5k_tmix_out)
    save(OUT_DIR, 'q5k.x_after_att', q5k_x_after_att)
    save(OUT_DIR, 'q5k.ln2.out', q5k_ln2_out)
    save(OUT_DIR, 'q5k.ffn.out', q5k_cmix_out)
    save(OUT_DIR, 'q5k.x_out', q5k_x_out)
    save(OUT_DIR, 'q5k.att_state.out', att_state_q5k)

    # ================================================================
    # 误差对比: FP16 原始 vs Q5K 反量化
    # ================================================================
    print("\n" + "=" * 60)
    print("Error Metrics: FP16 original vs Q5K dequantized")
    print("=" * 60)
    print_metrics("TMix output", q5k_tmix_out, tmix_out)
    print_metrics("x_after_att", q5k_x_after_att, x_after_att)
    print_metrics("LN2 output", q5k_ln2_out, ln2_out)
    print_metrics("CMix output", q5k_cmix_out, cmix_out)
    print_metrics("x_out (end-to-end)", q5k_x_out, x_out)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
