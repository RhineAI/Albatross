"""
RWKV7 全层逐级验证脚本 — 导出每步中间激活值。

逐步计算每个中间结果，导出 hex 到 golden/task0/{emb,layer0,layer1,...,output}/。
文件名不带 blocks.X 前缀（每层已有独立目录），用 .in/.out 表示部件的输入输出。
最后输出 logits 并打印对应的 token。
"""

import os
import torch
from torch.nn import functional as F
from reference.rwkv7 import RWKV7_ONE_OP, DTYPE
from reference.utils import TRIE_TOKENIZER
from golden.utils import GOLDEN_ROOT, save, compare, load_model

# ================= 加载模型 =================
model, z = load_model()
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

C = model.n_embd
H = model.n_head
N = model.head_size
n_layer = model.n_layer

print(f"C={C}, H={H}, N={N}, n_layer={n_layer}")

# ================= 编码 =================
context = "1"
input_tokens = tokenizer.encode(context)
token_id = input_tokens[0]
print(f"Token: '{context}' -> id={token_id}")

# ================= 初始化 =================
state = model.generate_zero_state(0)
x = z['emb.weight'][token_id]  # [C] fp16
v_first = torch.empty_like(x)

# ================= 导出 embedding =================
TASK_EMB_DIR = os.path.join(GOLDEN_ROOT, "task0", "emb")
os.makedirs(TASK_EMB_DIR, exist_ok=True)

# token_id 是 int，单独写一行 hex
with open(os.path.join(TASK_EMB_DIR, 'token_id.hex'), 'w') as f:
    f.write(f"{token_id:08x}\n")
print(f"  token_id.hex: 1 value (id={token_id})")

save(TASK_EMB_DIR, 'emb.weight.out', x)

# ================= 逐层计算 =================
for layer_id in range(n_layer):
    # 权重 key 前缀（用于从 z 中取权重）
    bbb = f'blocks.{layer_id}.'
    att = f'blocks.{layer_id}.att.'
    ffn = f'blocks.{layer_id}.ffn.'
    FFN_DIM = z[ffn+'key.weight'].shape[1]

    D = os.path.join(GOLDEN_ROOT, "task0", f"layer{layer_id}")
    os.makedirs(D, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Layer {layer_id}  (FFN_DIM={FFN_DIM})")
    print(f"{'='*60}")

    # --- 导出初始状态 ---
    save(D, 'att.x_prev.in', state[0][layer_id][0])
    save(D, 'ffn.x_prev.in', state[0][layer_id][1])
    if layer_id == 0:
        save(D, 'att.v_first.in', torch.zeros(C, dtype=DTYPE, device='cuda'))
    else:
        save(D, 'att.v_first.in', v_first)
    save(D, 'att.state.in', state[1][layer_id])

    # --- Step 1: LayerNorm 1 ---
    ln1_out = F.layer_norm(x, (C,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
    compare("ln1.out", ln1_out)
    save(D, 'ln1.in', x)
    save(D, 'ln1.out', ln1_out)

    # --- Step 2: TMix -- Delta Mix ---
    x_tmix = ln1_out
    x_prev_tmix = state[0][layer_id][0].clone()
    xx = x_prev_tmix - x_tmix
    state[0][layer_id][0] = x_tmix

    xr = x_tmix + xx * z[att+'x_r']
    xw = x_tmix + xx * z[att+'x_w']
    xk = x_tmix + xx * z[att+'x_k']
    xv = x_tmix + xx * z[att+'x_v']
    xa = x_tmix + xx * z[att+'x_a']
    xg = x_tmix + xx * z[att+'x_g']

    for suffix, val in [('x_r', xr), ('x_w', xw), ('x_k', xk), ('x_v', xv), ('x_a', xa), ('x_g', xg)]:
        save(D, 'att.'+suffix+'.out', val)

    # --- Step 3: TMix -- Projections ---
    r = xr @ z[att+'receptance.weight']
    k = xk @ z[att+'key.weight']
    v = xv @ z[att+'value.weight']

    for suffix, inv, outv in [('receptance.weight', xr, r), ('key.weight', xk, k), ('value.weight', xv, v)]:
        save(D, 'att.'+suffix+'.in', inv)
        save(D, 'att.'+suffix+'.out', outv)

    # --- Step 4: TMix -- Low-rank paths ---

    # Decay: w = sigmoid(w0 + tanh(xw @ w1) @ w2)
    tw = xw @ z[att+'w1']
    tw_tanh = torch.tanh(tw)
    w_proj2 = tw_tanh @ z[att+'w2']
    w_sum = z[att+'w0'] + w_proj2
    w_decay = torch.sigmoid(w_sum)

    save(D, 'att.w1.in', xw)
    save(D, 'att.w1.out', tw)
    save(D, 'att.w1.tanh', tw_tanh)
    save(D, 'att.w2.in', tw_tanh)
    save(D, 'att.w2.out', w_proj2)
    save(D, 'att.w0.sum', w_sum)
    save(D, 'att.w0.sigmoid', w_decay)

    # Alpha: a = sigmoid(a0 + (xa @ a1) @ a2)
    ta = xa @ z[att+'a1']
    a_proj2 = ta @ z[att+'a2']
    a_sum = z[att+'a0'] + a_proj2
    a_vec = torch.sigmoid(a_sum)

    save(D, 'att.a1.in', xa)
    save(D, 'att.a1.out', ta)
    save(D, 'att.a2.in', ta)
    save(D, 'att.a2.out', a_proj2)
    save(D, 'att.a0.sum', a_sum)
    save(D, 'att.a0.sigmoid', a_vec)

    # Gate: g = sigmoid(xg @ g1) @ g2
    tg = xg @ z[att+'g1']
    tg_sig = torch.sigmoid(tg)
    g_vec = tg_sig @ z[att+'g2']

    save(D, 'att.g1.in', xg)
    save(D, 'att.g1.out', tg)
    save(D, 'att.g1.sigmoid', tg_sig)
    save(D, 'att.g2.in', tg_sig)
    save(D, 'att.g2.out', g_vec)

    # V-first
    if layer_id == 0:
        v_first = v.clone()
    else:
        tv = xv @ z[att+'v1']
        v_proj2 = tv @ z[att+'v2']
        v_mix = torch.sigmoid(z[att+'v0'] + v_proj2)
        v = v + (v_first - v) * v_mix

        save(D, 'att.v1.in', xv)
        save(D, 'att.v1.out', tv)
        save(D, 'att.v2.in', tv)
        save(D, 'att.v2.out', v_proj2)
        save(D, 'att.v0.sum', z[att+'v0'] + v_proj2)
        save(D, 'att.v0.sigmoid', v_mix)
        save(D, 'att.value.weight.mixed', v)

    save(D, 'att.v_first.out', v_first)

    # --- Step 5: Key normalization & modulation ---
    kk_raw = k * z[att+'k_k']
    kk_norm = F.normalize(kk_raw.view(H, N), dim=-1, p=2.0).view(H*N)
    a_minus_1 = a_vec - 1.0
    am1_ka = a_minus_1 * z[att+'k_a']
    scale = 1.0 + am1_ka
    k_mod = k * scale

    save(D, 'att.k_k.out', kk_raw)
    save(D, 'att.k_k.norm', kk_norm)
    save(D, 'att.k_a.scale', scale)
    save(D, 'att.key.weight.mod', k_mod)

    # --- Step 6: WKV7 core ---
    neg_kk = -kk_norm
    kk_alpha = kk_norm * a_vec

    save(D, 'att.wkv.in.r', r)
    save(D, 'att.wkv.in.w', w_decay)
    save(D, 'att.wkv.in.k', k_mod)
    save(D, 'att.wkv.in.v', v)
    save(D, 'att.wkv.in.neg_kk', neg_kk)
    save(D, 'att.wkv.in.kk_alpha', kk_alpha)
    save(D, 'att.wkv.state.in', state[1][layer_id])

    wkv_out = RWKV7_ONE_OP(state[1][layer_id], r, w_decay, k_mod, v, neg_kk, kk_alpha)

    compare("att.wkv.out", wkv_out)
    save(D, 'att.wkv.out', wkv_out)
    save(D, 'att.wkv.state.out', state[1][layer_id])

    # --- Step 7: GroupNorm ---
    gn_out = F.group_norm(wkv_out.view(1, H*N), num_groups=H,
                           weight=z[att+'ln_x.weight'], bias=z[att+'ln_x.bias'],
                           eps=64e-5).view(H*N)

    save(D, 'att.ln_x.in', wkv_out)
    save(D, 'att.ln_x.out', gn_out)

    # --- Step 8: Bonus ---
    bonus = ((r * k_mod * z[att+'r_k']).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H*N)
    gn_plus_bonus = gn_out + bonus

    save(D, 'att.r_k.bonus', bonus)
    save(D, 'att.ln_x.gated', gn_plus_bonus)

    # --- Step 9: Gate + Output projection ---
    gated = gn_plus_bonus * g_vec
    tmix_out = gated @ z[att+'output.weight']

    save(D, 'att.output.weight.in', gated)
    save(D, 'att.output.weight.out', tmix_out)

    # --- Step 10: Residual 1 ---
    x_after_att = x + tmix_out
    compare("att.residual.out", x_after_att)
    save(D, 'att.residual.out', x_after_att)

    # --- Step 11: LayerNorm 2 ---
    ln2_out = F.layer_norm(x_after_att, (C,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
    save(D, 'ln2.in', x_after_att)
    save(D, 'ln2.out', ln2_out)

    # --- Step 12: CMix -- Delta Mix ---
    x_cmix = ln2_out
    x_prev_cmix = state[0][layer_id][1].clone()
    xx_cmix = x_prev_cmix - x_cmix
    state[0][layer_id][1] = x_cmix

    k_mixed = x_cmix + xx_cmix * z[ffn+'x_k']
    save(D, 'ffn.x_k.out', k_mixed)

    # --- Step 13: CMix -- Expand + ReLU^2 ---
    k_expanded = k_mixed @ z[ffn+'key.weight']
    k_act = torch.relu(k_expanded) ** 2

    save(D, 'ffn.key.weight.in', k_mixed)
    save(D, 'ffn.key.weight.out', k_expanded)
    save(D, 'ffn.key.weight.act', k_act)

    # --- Step 14: CMix -- Contract ---
    cmix_out = k_act @ z[ffn+'value.weight']
    save(D, 'ffn.value.weight.in', k_act)
    save(D, 'ffn.value.weight.out', cmix_out)

    # --- Step 15: Residual 2 ---
    x = x_after_att + cmix_out
    compare("ffn.residual.out", x)
    save(D, 'ffn.residual.out', x)

    # --- 导出输出状态 ---
    save(D, 'att.x_prev.out', state[0][layer_id][0])
    save(D, 'ffn.x_prev.out', state[0][layer_id][1])
    save(D, 'att.state.out', state[1][layer_id])

# ================= Final: ln_out + head =================
print(f"\n{'='*60}")
print("  Final Output")
print(f"{'='*60}")

TASK_OUT_DIR = os.path.join(GOLDEN_ROOT, "task0", "output")
os.makedirs(TASK_OUT_DIR, exist_ok=True)

save(TASK_OUT_DIR, 'ln_out.in', x)
x = F.layer_norm(x, (C,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
save(TASK_OUT_DIR, 'ln_out.out', x)

logits = x @ z['head.weight']
save(TASK_OUT_DIR, 'head.weight.out', logits)

# 打印 top token
top_id = logits.argmax().item()
top_token = tokenizer.decode([top_id])
print(f"\nLogits shape: {list(logits.shape)}")
print(f"Top token: id={top_id}, text='{top_token}'")
compare("logits", logits)

print(f"\nDone! Activation data exported to {GOLDEN_ROOT}/task0/")
