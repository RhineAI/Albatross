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
Q_MAX = 15
Q_MIN = -16

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


# ══════════════════════════════════════════════════════════════
# 硬件级 tile VMM 模拟 (从 gen_rhine_vmm_v1_golden.py 移植)
# 精确模拟 fp16_q5k_vmm32_tile 的完整流水线:
#   1. fp16_q5k_vq: 输入向量量化 FP16 → INT5 + scale_x
#   2. int_vmm: INT5 × INT5 整数矩阵乘
#   3. fp16_q5k_vdq: 反量化 → FP16 输出
# ══════════════════════════════════════════════════════════════

def _uint16_to_fp16(val):
    return np.uint16(val).view(np.float16)


def _fp16_mul(a, b):
    return np.float16(np.float64(a) * np.float64(b))


def _fp16_add(a, b):
    return np.float16(np.float64(a) + np.float64(b))


def _fp16_recip(a):
    if float(a) == 0.0:
        return np.float16(0.0)
    return np.float16(1.0 / np.float64(a))


def _quantize_block(x_fp16_vec):
    """模拟 fp16_q5k_vq: 对称量化输入向量 → INT5 + scale_x"""
    abs_vals = np.abs(x_fp16_vec).astype(np.float16)
    amax = np.float16(np.max(abs_vals))
    if float(amax) == 0.0:
        return np.zeros(TILE_DIM, dtype=np.int32), np.float16(0.0)

    recip_amax = _fp16_recip(amax)
    iscale = _fp16_mul(np.float16(15.0), recip_amax)
    scale = _fp16_mul(amax, _uint16_to_fp16(0x2C44))  # amax * (1/15)

    q_int = np.zeros(TILE_DIM, dtype=np.int32)
    for i in range(TILE_DIM):
        prod = _fp16_mul(x_fp16_vec[i], iscale)
        prod_f = float(prod)
        q_val = int(np.trunc(prod_f))
        frac = prod_f - q_val
        if frac > 0.5:
            q_val += 1
        elif frac < -0.5:
            q_val -= 1
        q_int[i] = max(-16, min(15, q_val))
    return q_int, scale


def _int_vmm(q_x, w_int):
    """模拟 int_vmm: y[j] = Σ_i q_x[i] * w_int[i][j]"""
    sumi = np.zeros(TILE_DIM, dtype=np.int64)
    for j in range(TILE_DIM):
        for i in range(TILE_DIM):
            sumi[j] += int(q_x[i]) * int(w_int[i][j])
    return sumi


def _dequantize_output(sumi, q_x, scale_x, scale_w, min_w):
    """模拟 fp16_q5k_vdq: 反量化整数累加结果 → FP16"""
    sum_qx = int(np.sum(q_x.astype(np.int64)))
    sum_qx_fp16 = np.float16(float(sum_qx))
    sx_sum_qx = _fp16_mul(sum_qx_fp16, scale_x)

    y = np.zeros(TILE_DIM, dtype=np.float16)
    for j in range(TILE_DIM):
        coeff_j = _fp16_mul(scale_x, scale_w[j])
        bias_j = _fp16_mul(sx_sum_qx, min_w[j])
        sumi_j_fp16 = np.float16(float(int(sumi[j])))
        term_a = _fp16_mul(sumi_j_fp16, coeff_j)
        y[j] = _fp16_add(term_a, bias_j)
    return y


def _tile_vmm(x_group_fp16, w_int, scale_w, min_w):
    """模拟单个 fp16_q5k_vmm32_tile 的完整流水线"""
    q_x, scale_x = _quantize_block(x_group_fp16)
    sumi = _int_vmm(q_x, w_int)
    return _dequantize_output(sumi, q_x, scale_x, scale_w, min_w)


def _fp16_sum_tree(values):
    """模拟 fp16_sum_tree: 二叉树逐级 FP16 加法"""
    current = [np.float16(v) for v in values]
    while len(current) > 1:
        nxt = []
        for k in range(0, len(current), 2):
            if k + 1 < len(current):
                nxt.append(_fp16_add(current[k], current[k + 1]))
            else:
                nxt.append(current[k])
        current = nxt
    return current[0]


def hw_matmul(x_tensor, w_int, scale_w, min_w, rows, cols):
    """硬件级矩阵向量乘: 精确模拟 tile 网格阵列。
    x_tensor: [rows] FP16 torch tensor (输入向量)
    w_int/scale_w/min_w: [tr][tc] 量化数据 (来自 quantize_matrix)
    返回: [cols] FP16 torch tensor (输出向量)
    """
    tr = rows // TILE_DIM
    tc = cols // TILE_DIM
    x_np = x_tensor.detach().cpu().to(torch.float16).numpy()

    y = np.zeros(cols, dtype=np.float16)
    for gc in range(tc):
        for gk in range(TILE_DIM):
            col_vals = []
            for gr in range(tr):
                x_group = x_np[gr*TILE_DIM:(gr+1)*TILE_DIM]
                tile_y = _tile_vmm(x_group, w_int[gr][gc], scale_w[gr][gc], min_w[gr][gc])
                col_vals.append(tile_y[gk])
            y[gc*TILE_DIM + gk] = _fp16_sum_tree(col_vals)

    return torch.from_numpy(y).to(DTYPE)


def quantize_and_save(out_dir, name, W_tensor, rows, cols):
    """量化矩阵并导出, 返回 (量化数据, 反量化 tensor, hw_matmul 可用的量化数据)。"""
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

    # 排除 |b| < threshold 的近零值后的 mean_rel
    threshold = 0.1
    sig = abs_b > threshold
    mean_rel_sig = float(np.mean(diff[sig] / abs_b[sig]) * 100.0) if np.any(sig) else 0.0
    n_small = int(np.sum(nz & ~sig))  # 非零但很小的值数量

    return cos_sim, max_abs, mean_rel, mean_rel_sig, n_small


def print_metrics(label: str, a: torch.Tensor, b: torch.Tensor):
    """打印误差指标, 格式与 RTL testbench 一致。"""
    cos_sim, max_abs, mean_rel, mean_rel_sig, n_small = compute_metrics(a, b)
    print(f"  {label:35s} cos={cos_sim:.10f}  max_abs={max_abs:.6e}  "
          f"mrel={mean_rel:.1f}%  mrel_sig={mean_rel_sig:.1f}%  n_small={n_small}")
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
    """TMix forward, 导出中间结果, 返回 (tmix_out, v_first, intermediates_dict)。"""
    mid = {}
    # Stage 1: Delta Mix
    xx = x_prev[0] - x
    x_prev[0] = x

    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g
    mid['xr'] = xr.clone(); mid['xw'] = xw.clone(); mid['xk'] = xk.clone()
    mid['xv'] = xv.clone(); mid['xa'] = xa.clone(); mid['xg'] = xg.clone()

    # Stage 2: Projections
    r = xr @ R_
    k = xk @ K_
    v = xv @ V_
    mid['r'] = r.clone(); mid['k'] = k.clone(); mid['v'] = v.clone()

    tw = xw @ w1
    ta = xa @ a1
    tv = xv @ v1
    tg = xg @ g1
    mid['tw'] = tw.clone(); mid['ta'] = ta.clone()
    mid['tv'] = tv.clone(); mid['tg'] = tg.clone()

    # Stage 3a: Decay
    tw_tanh = torch.tanh(tw)
    w_proj2 = tw_tanh @ w2
    w_sum = w0 + w_proj2
    w_sigmoid = torch.sigmoid(w_sum)
    mid['tw_tanh'] = tw_tanh.clone(); mid['w_proj2'] = w_proj2.clone()
    mid['w_sum'] = w_sum.clone(); mid['w_sigmoid'] = w_sigmoid.clone()

    # Stage 3b: Alpha
    a_proj2 = ta @ a2
    a_sum = a0 + a_proj2
    a_vec = torch.sigmoid(a_sum)
    mid['a_proj2'] = a_proj2.clone(); mid['a_sum'] = a_sum.clone()
    mid['a_vec'] = a_vec.clone()

    # Stage 3c: Gate
    tg_sig = torch.sigmoid(tg)
    g_vec = tg_sig @ g2
    mid['tg_sig'] = tg_sig.clone(); mid['g_vec'] = g_vec.clone()

    # Stage 3d: V-first coeff
    v_proj2 = tv @ v2
    vf_sum = v0 + v_proj2
    vf_coeff = torch.sigmoid(vf_sum)
    mid['v_proj2'] = v_proj2.clone(); mid['vf_sum'] = vf_sum.clone()
    mid['vf_coeff'] = vf_coeff.clone()

    # Stage 4: Key norm + modulation
    kk_raw = k * k_k
    kk_norm = F.normalize(kk_raw.view(H, N), dim=-1, p=2.0).view(H * N)
    mid['kk_raw'] = kk_raw.clone(); mid['kk_norm'] = kk_norm.clone()

    a_minus_1 = a_vec - 1.0
    scale = 1.0 + a_minus_1 * k_a
    k_mod = k * scale
    mid['k_mod'] = k_mod.clone()

    # Stage 5: V-first mix
    if layer_id == 0:
        v_first = v.clone()
    else:
        v = v + (v_first - v) * vf_coeff
    mid['v_after_mix'] = v.clone()

    # Stage 6: WKV7 core
    neg_kk = -kk_norm
    kk_alpha = kk_norm * a_vec

    wkv_out = RWKV7_ONE_OP(state, r, w_sigmoid, k_mod, v, neg_kk, kk_alpha)
    mid['wkv_out'] = wkv_out.clone()
    mid['state_after'] = state.clone()
    if D is not None:
        save(D, 'att.wkv.out', wkv_out)
        save(D, 'att_state.out', state)

    # Stage 7: GroupNorm
    gn_out = F.group_norm(
        wkv_out.view(1, H * N), num_groups=H,
        weight=ln_w, bias=ln_b, eps=64e-5
    ).view(H * N)
    mid['gn_out'] = gn_out.clone()

    # Stage 8: Bonus
    bonus = ((r * k_mod * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    gn_plus_bonus = gn_out + bonus
    mid['bonus'] = bonus.clone(); mid['gn_plus_bonus'] = gn_plus_bonus.clone()

    # Stage 9: Gate + O_ projection
    gated = gn_plus_bonus * g_vec
    tmix_out = gated @ O_
    mid['gated'] = gated.clone(); mid['tmix_out'] = tmix_out.clone()
    if D is not None:
        save(D, 'att.out', tmix_out)

    return tmix_out, v_first, mid


# ══════════════════════════════════════════════════════════════
# CMix forward (从 generate_cmix_mini.py 改写, 维度 C=64, FFN_DIM=128)
# ══════════════════════════════════════════════════════════════

def cmix_forward(D, x, x_prev, x_k, K_, V_):
    """CMix forward, 导出中间结果, 返回 (out, intermediates_dict)。"""
    mid = {}
    xx = x_prev[1] - x
    x_prev[1] = x

    k = x + xx * x_k
    mid['k'] = k.clone()

    k_expanded = k @ K_
    mid['k_expanded'] = k_expanded.clone()

    k_act = torch.relu(k_expanded) ** 2
    mid['k_act'] = k_act.clone()

    out = k_act @ V_
    mid['out'] = out.clone()
    if D is not None:
        save(D, 'ffn.out', out)

    return out, mid


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

    tmix_out, v_first, fp16_tmix_mid = tmix_forward(
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
    cmix_out, fp16_cmix_mid = cmix_forward(OUT_DIR, ln2_out, x_prev, ffn_x_k, ffn_K, ffn_V)

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
    q5k_tmix_out, q5k_v_first, q5k_tmix_mid = tmix_forward(
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
    q5k_cmix_out, q5k_cmix_mid = cmix_forward(None, q5k_ln2_out, x_prev_q5k, ffn_x_k, ffn_K_dq, ffn_V_dq)

    # Residual 2
    q5k_x_out = (q5k_x_after_att.float() + q5k_cmix_out.float()).to(DTYPE)

    # 保存 Q5K golden data — TMix 中间信号 (逐级调试用)
    save(OUT_DIR, 'q5k.tmix.r', q5k_tmix_mid['r'])
    save(OUT_DIR, 'q5k.tmix.k', q5k_tmix_mid['k'])
    save(OUT_DIR, 'q5k.tmix.v', q5k_tmix_mid['v'])
    save(OUT_DIR, 'q5k.tmix.w_sigmoid', q5k_tmix_mid['w_sigmoid'])
    save(OUT_DIR, 'q5k.tmix.a_vec', q5k_tmix_mid['a_vec'])
    save(OUT_DIR, 'q5k.tmix.g_vec', q5k_tmix_mid['g_vec'])
    save(OUT_DIR, 'q5k.tmix.kk_norm', q5k_tmix_mid['kk_norm'])
    save(OUT_DIR, 'q5k.tmix.k_mod', q5k_tmix_mid['k_mod'])
    save(OUT_DIR, 'q5k.tmix.v_after_mix', q5k_tmix_mid['v_after_mix'])
    save(OUT_DIR, 'q5k.tmix.gn_out', q5k_tmix_mid['gn_out'])
    save(OUT_DIR, 'q5k.tmix.gated', q5k_tmix_mid['gated'])

    # 保存 Q5K golden data — 最终输出
    save(OUT_DIR, 'q5k.att.wkv.out', q5k_tmix_mid['wkv_out'])
    save(OUT_DIR, 'q5k.att.out', q5k_tmix_out)
    save(OUT_DIR, 'q5k.x_after_att', q5k_x_after_att)
    save(OUT_DIR, 'q5k.ln2.out', q5k_ln2_out)
    save(OUT_DIR, 'q5k.ffn.out', q5k_cmix_out)
    save(OUT_DIR, 'q5k.x_out', q5k_x_out)
    save(OUT_DIR, 'q5k.att_state.out', att_state_q5k)
    save(OUT_DIR, 'q5k.tmix_x_prev.out', x_prev_q5k[0])
    save(OUT_DIR, 'q5k.cmix_x_prev.out', x_prev_q5k[1])
    save(OUT_DIR, 'q5k.v_first.out', q5k_v_first)

    # ================================================================
    # 误差对比: FP16 原始 vs Q5K 反量化 — 逐阶段
    # ================================================================
    print("\n" + "=" * 60)
    print("Error Metrics: FP16 original vs Q5K dequantized")
    print("=" * 60)

    # --- 权重反量化误差 ---
    print("\n--- Weight dequantization error ---")
    for wname, w_orig, w_dq in [
        ('R_', R_, R_dq), ('K_', K_, K_dq), ('V_', V_, V_dq), ('O_', O_, O_dq),
        ('w1', w1, w1_dq), ('w2', w2, w2_dq),
        ('a1', a1, a1_dq), ('a2', a2, a2_dq),
        ('v1', v1, v1_dq), ('v2', v2, v2_dq),
        ('g1', g1, g1_dq), ('g2', g2, g2_dq),
        ('ffn_K', ffn_K, ffn_K_dq), ('ffn_V', ffn_V, ffn_V_dq),
    ]:
        print_metrics(f"weight {wname}", w_dq, w_orig)

    # --- TMix 中间值逐步对比 ---
    print("\n--- TMix stage-by-stage ---")
    tmix_stages = [
        'xr', 'xw', 'xk', 'xv', 'xa', 'xg',
        'r', 'k', 'v',
        'tw', 'ta', 'tv', 'tg',
        'tw_tanh', 'w_proj2', 'w_sum', 'w_sigmoid',
        'a_proj2', 'a_sum', 'a_vec',
        'tg_sig', 'g_vec',
        'v_proj2', 'vf_sum', 'vf_coeff',
        'kk_raw', 'kk_norm', 'k_mod',
        'v_after_mix',
        'wkv_out', 'state_after',
        'gn_out', 'bonus', 'gn_plus_bonus',
        'gated', 'tmix_out',
    ]
    for stage in tmix_stages:
        if stage in fp16_tmix_mid and stage in q5k_tmix_mid:
            print_metrics(f"tmix.{stage}", q5k_tmix_mid[stage], fp16_tmix_mid[stage])

    # --- 残差 + LN2 ---
    print("\n--- Residual & LN2 ---")
    print_metrics("x_after_att", q5k_x_after_att, x_after_att)
    print_metrics("LN2 output", q5k_ln2_out, ln2_out)

    # --- CMix 中间值逐步对比 ---
    print("\n--- CMix stage-by-stage ---")
    cmix_stages = ['k', 'k_expanded', 'k_act', 'out']
    for stage in cmix_stages:
        if stage in fp16_cmix_mid and stage in q5k_cmix_mid:
            print_metrics(f"cmix.{stage}", q5k_cmix_mid[stage], fp16_cmix_mid[stage])

    # --- 误差分离分析: gated @ O_ ---
    print("\n--- Error decomposition: tmix_out = gated @ O_ ---")
    tmix_only_O_err = fp16_tmix_mid['gated'] @ O_dq
    print_metrics("fp16_gated @ O_dq (O only)", tmix_only_O_err, fp16_tmix_mid['tmix_out'])
    tmix_only_gated_err = q5k_tmix_mid['gated'] @ O_
    print_metrics("q5k_gated @ fp16_O (gated only)", tmix_only_gated_err, fp16_tmix_mid['tmix_out'])
    print_metrics("q5k_gated @ O_dq (both)", q5k_tmix_mid['tmix_out'], fp16_tmix_mid['tmix_out'])

    # 值域统计
    def val_stats(name, t):
        a = t.detach().cpu().float().numpy().flatten()
        print(f"  {name:35s} range=[{a.min():+.6f}, {a.max():+.6f}]  "
              f"|mean|={np.abs(a).mean():.6f}  std={a.std():.6f}")
    print("\n--- Value range stats ---")
    val_stats("fp16 gated", fp16_tmix_mid['gated'])
    val_stats("q5k  gated", q5k_tmix_mid['gated'])
    val_stats("fp16 tmix_out", fp16_tmix_mid['tmix_out'])
    val_stats("q5k  tmix_out", q5k_tmix_mid['tmix_out'])
    val_stats("fp16 gn_plus_bonus", fp16_tmix_mid['gn_plus_bonus'])
    val_stats("fp16 g_vec", fp16_tmix_mid['g_vec'])
    val_stats("fp16 r", fp16_tmix_mid['r'])
    val_stats("fp16 k", fp16_tmix_mid['k'])
    val_stats("fp16 v", fp16_tmix_mid['v'])

    # 逐元素误差分布
    print("\n--- Per-element error distribution: gated ---")
    gated_fp = fp16_tmix_mid['gated'].float().numpy()
    gated_q5 = q5k_tmix_mid['gated'].float().numpy()
    gated_diff = np.abs(gated_q5 - gated_fp)
    gated_rel = np.zeros_like(gated_diff)
    nz = np.abs(gated_fp) > 0
    gated_rel[nz] = gated_diff[nz] / np.abs(gated_fp[nz])
    # 找最大误差的元素
    worst_idx = np.argsort(gated_rel)[::-1][:5]
    for idx in worst_idx:
        print(f"  idx={idx}: fp16={gated_fp[idx]:+.6f} q5k={gated_q5[idx]:+.6f} "
              f"abs_err={gated_diff[idx]:.6f} rel_err={gated_rel[idx]*100:.1f}%")

    print("\n--- Per-element error distribution: tmix_out ---")
    tout_fp = fp16_tmix_mid['tmix_out'].float().numpy()
    tout_q5 = q5k_tmix_mid['tmix_out'].float().numpy()
    tout_diff = np.abs(tout_q5 - tout_fp)
    tout_rel = np.zeros_like(tout_diff)
    nz = np.abs(tout_fp) > 0
    tout_rel[nz] = tout_diff[nz] / np.abs(tout_fp[nz])
    worst_idx = np.argsort(tout_rel)[::-1][:5]
    for idx in worst_idx:
        print(f"  idx={idx}: fp16={tout_fp[idx]:+.6f} q5k={tout_q5[idx]:+.6f} "
              f"abs_err={tout_diff[idx]:.6f} rel_err={tout_rel[idx]*100:.1f}%")

    # --- 误差分离分析: 各级矩阵乘 ---
    print("\n--- Error decomposition: r = xr @ R_ ---")
    r_only_R_err = fp16_tmix_mid['xr'] @ R_dq
    print_metrics("fp16_xr @ R_dq (R only)", r_only_R_err, fp16_tmix_mid['r'])

    print("\n--- Error decomposition: w path (tw->w2->sigmoid) ---")
    # tw = xw @ w1: 只看 w1 误差
    tw_only_w1 = fp16_tmix_mid['xw'] @ w1_dq
    print_metrics("fp16_xw @ w1_dq (w1 only)", tw_only_w1, fp16_tmix_mid['tw'])
    # tw_tanh -> w_proj2 = tanh(tw) @ w2: 用 fp16 tw_tanh + q5k w2
    wp2_only_w2 = fp16_tmix_mid['tw_tanh'] @ w2_dq
    print_metrics("fp16_tw_tanh @ w2_dq (w2 only)", wp2_only_w2, fp16_tmix_mid['w_proj2'])

    # --- 最终输出 ---
    print("\n--- Final ---")
    print_metrics("x_out (end-to-end)", q5k_x_out, x_out)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
