"""
TMix mini 核心计算函数 — 基于 RWKV_x070_TMix_one 改写，加入中间结果导出。

提供 RWKV_x070_TMix_one_export 供 generate_layer_mini.py 调用。
维度常量: C=128, H=2, N=64, D_W/A/V=16, D_G=32
"""

import math
import torch
import torch.nn.functional as F
from golden.utils import save
from reference.rwkv7 import DTYPE

C, H, N = 128, 2, 64
D_W, D_A, D_V, D_G = 16, 16, 16, 32
EXP_NEG_HALF = math.exp(-0.5)


def _wkv7_core_export(out_dir, state, r, w, k, v, a, b):
    """
    从 reference/rwkv7.py 的 _wkv7_core 复制，加入 save()。
    state: [H, N, N] fp32, in-place 修改
    r, w, k, v, a, b: [H, N] fp16
    返回 y: [H, N] fp16
    """
    H, N = r.shape
    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    # w 变换: exp(-exp(-0.5) * w), w 已是 sigmoid 输出
    w = torch.exp(-EXP_NEG_HALF * w.float())

    # sa[h,i] = sum_j a[h,j] * state[h,i,j]
    sa = torch.einsum('hn,hin->hi', a, state)

    # state[h,i,j] = state[h,i,j] * w[h,j] + k[h,j] * v[h,i] + sa[h,i] * b[h,j]
    state.mul_(w.unsqueeze(1))
    state.add_(k.unsqueeze(1) * v.unsqueeze(2))
    state.add_(sa.unsqueeze(2) * b.unsqueeze(1))

    # y[h,i] = sum_j(state[h,i,j] * r[h,j])
    y = torch.einsum('hin,hn->hi', state, r)
    return y.to(DTYPE)


def RWKV7_ONE_OP_export(out_dir, state, r, w, k, v, a, b):
    """包装: [C] -> [H,N] -> _wkv7_core_export -> [C]"""
    _C = r.shape[0]
    _H = _C // N
    r_ = r.view(_H, N)
    w_ = w.view(_H, N)
    k_ = k.view(_H, N)
    v_ = v.view(_H, N)
    a_ = a.view(_H, N)
    b_ = b.view(_H, N)
    y = _wkv7_core_export(out_dir, state, r_, w_, k_, v_, a_, b_)
    return y.view(_C)


def RWKV_x070_TMix_one_export(
    out_dir, layer_id, x, x_prev, v_first, state,
    x_r, x_w, x_k, x_v, x_a, x_g,
    w0, w1, w2, a0, a1, a2, v0, v1, v2,
    g1, g2, k_k, k_a, r_k,
    R_, K_, V_, O_,
    ln_w, ln_b
):
    """从 RWKV_x070_TMix_one 复制，每步插入 save()"""
    D = out_dir

    # --- Stage 1: Delta Mix ---
    xx = x_prev[0] - x
    x_prev[0] = x

    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    for suffix, val in [('x_r', xr), ('x_w', xw), ('x_k', xk),
                         ('x_v', xv), ('x_a', xa), ('x_g', xg)]:
        save(D, 'att.' + suffix + '.out', val)

    # --- Stage 2: 7-way projections ---
    r = xr @ R_
    k = xk @ K_
    v = xv @ V_

    save(D, 'att.receptance.weight.out', r)
    save(D, 'att.key.weight.out', k)
    save(D, 'att.value.weight.out', v)

    tw = xw @ w1
    save(D, 'att.w1.out', tw)

    ta = xa @ a1
    save(D, 'att.a1.out', ta)

    tv = xv @ v1
    save(D, 'att.v1.out', tv)

    tg = xg @ g1
    save(D, 'att.g1.out', tg)

    # --- Stage 3a: Decay ---
    tw_tanh = torch.tanh(tw)
    save(D, 'att.w1.tanh', tw_tanh)

    w_proj2 = tw_tanh @ w2
    save(D, 'att.w2.out', w_proj2)

    w_sum = w0 + w_proj2
    save(D, 'att.w0.sum', w_sum)

    w_sigmoid = torch.sigmoid(w_sum)
    save(D, 'att.w0.sigmoid', w_sigmoid)

    # RTL decay_lut 输出: exp(-exp(-0.5) * sigmoid(x))
    w_decay_rtl = torch.exp(-EXP_NEG_HALF * w_sigmoid.float()).to(DTYPE)
    save(D, 'att.wkv.in.w_decay', w_decay_rtl)

    # --- Stage 3b: Alpha ---
    a_proj2 = ta @ a2
    save(D, 'att.a2.out', a_proj2)

    a_sum = a0 + a_proj2
    save(D, 'att.a0.sum', a_sum)

    a_vec = torch.sigmoid(a_sum)
    save(D, 'att.a0.sigmoid', a_vec)

    # --- Stage 3c: Gate ---
    tg_sig = torch.sigmoid(tg)
    save(D, 'att.g1.sigmoid', tg_sig)

    g_vec = tg_sig @ g2
    save(D, 'att.g2.out', g_vec)

    # --- Stage 3d: V-first coeff (layer > 0 only) ---
    # 仍然计算 vf_coeff，但 layer 0 不使用
    v_proj2 = tv @ v2
    save(D, 'att.v2.out', v_proj2)

    vf_sum = v0 + v_proj2
    save(D, 'att.v0.sum', vf_sum)

    vf_coeff = torch.sigmoid(vf_sum)
    save(D, 'att.v0.sigmoid', vf_coeff)

    # --- Stage 4: Key norm + modulation ---
    kk_raw = k * k_k
    save(D, 'att.k_k.out', kk_raw)

    kk_norm = F.normalize(kk_raw.view(H, N), dim=-1, p=2.0).view(H * N)
    save(D, 'att.k_k.norm', kk_norm)

    a_minus_1 = a_vec - 1.0
    scale = 1.0 + a_minus_1 * k_a
    save(D, 'att.k_a.scale', scale)

    k_mod = k * scale
    save(D, 'att.key.weight.mod', k_mod)

    # --- Stage 5: V-first mix ---
    if layer_id == 0:
        v_first = v.clone()
    else:
        v = v + (v_first - v) * vf_coeff

    save(D, 'att.v_first.out', v_first)
    save(D, 'att.value.weight.mixed', v)

    # --- Stage 6: WKV7 core ---
    neg_kk = -kk_norm
    kk_alpha = kk_norm * a_vec

    save(D, 'att.wkv.in.r', r)
    save(D, 'att.wkv.in.k', k_mod)
    save(D, 'att.wkv.in.v', v)
    save(D, 'att.wkv.in.neg_kk', neg_kk)
    save(D, 'att.wkv.in.kk_alpha', kk_alpha)

    wkv_out = RWKV7_ONE_OP_export(D, state, r, w_sigmoid, k_mod, v, neg_kk, kk_alpha)
    save(D, 'att.wkv.out', wkv_out)
    save(D, 'att.state.out', state)  # FP32

    # --- Stage 7: GroupNorm ---
    gn_out = F.group_norm(
        wkv_out.view(1, H * N), num_groups=H,
        weight=ln_w, bias=ln_b, eps=64e-5
    ).view(H * N)
    save(D, 'att.ln_x.out', gn_out)

    # --- Stage 8: Bonus ---
    bonus = ((r * k_mod * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    save(D, 'att.r_k.bonus', bonus)

    gn_plus_bonus = gn_out + bonus
    save(D, 'att.ln_x.gated', gn_plus_bonus)

    # --- Stage 9: Gate + O_ projection ---
    gated = gn_plus_bonus * g_vec
    save(D, 'att.output.weight.in', gated)

    tmix_out = gated @ O_
    save(D, 'att.output.weight.out', tmix_out)

    return tmix_out, v_first
