"""
RWKV7 Layer 0 逐级验证脚本
逐步计算每个中间结果，导出 hex，并与 RTL 中间信号一一对应。
用于定位 RTL 与参考实现之间的误差来源。
"""

import os, struct, types
import numpy as np
import torch
from torch.nn import functional as F

# ================= 配置 =================
args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = r"D:\Development\models\rwkv\rwkv7-g1d-0.1b-20260129-ctx8192"

EXPORT_DIR = r"C:\Projects\verilog-test\sim\layer0_golden"
os.makedirs(EXPORT_DIR, exist_ok=True)

LAYER_ID = 0

# ================= 加载模型 =================
from reference.rwkv7 import RWKV_x070, RWKV7_ONE_OP
from reference.rwkv7 import DTYPE, HEAD_SIZE
import reference.rwkv7 as rwkv7_mod
rwkv7_mod.enable_print = False

model = RWKV_x070(args)
z = model.z

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

# ================= 工具函数 =================
def fp16_to_hex(t: torch.Tensor) -> list:
    t = t.detach().cpu().to(torch.float16).contiguous()
    raw = t.numpy().view(np.uint16).flatten()
    return [f"{v:04x}" for v in raw]

def fp32_to_hex(t: torch.Tensor) -> list:
    t = t.detach().cpu().to(torch.float32).contiguous()
    raw = t.numpy().view(np.uint32).flatten()
    return [f"{v:08x}" for v in raw]

def save_hex(name, lines):
    path = os.path.join(EXPORT_DIR, f"{name}.hex")
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def save_fp16(name, t):
    save_hex(name, fp16_to_hex(t))

def save_fp32(name, t):
    save_hex(name, fp32_to_hex(t))

def load_fp16(name):
    path = os.path.join(EXPORT_DIR, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint16).view(np.float16).astype(np.float64)

def load_fp32(name):
    path = os.path.join(EXPORT_DIR, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint32).view(np.float32).astype(np.float64)

def compare(name, tensor, size=None, tol=0.02):
    """Compare a tensor against its saved hex file."""
    arr = tensor.detach().cpu().to(torch.float16).numpy().astype(np.float64).flatten()
    if size:
        arr = arr[:size]
    n = len(arr)

    # Also compute abs stats
    abs_arr = np.abs(arr)
    nz = arr[arr != 0]

    print(f"  {name:30s}: n={n:6d}  range=[{arr.min():+.6f}, {arr.max():+.6f}]  "
          f"|mean|={abs_arr.mean():.6f}  nonzero={len(nz)}/{n}")
    return arr

def compare_two(name, a_np, b_np, tol=0.05):
    """Compare two numpy arrays element-wise, report errors."""
    diff = np.abs(a_np - b_np)
    # relative error where b != 0
    mask = b_np != 0
    rel_err = np.zeros_like(diff)
    rel_err[mask] = diff[mask] / np.abs(b_np[mask])
    rel_err[~mask & (a_np != 0)] = 1.0

    max_rel = rel_err.max()
    max_abs = diff.max()
    n_exceed = np.sum(rel_err > tol)

    status = "PASS" if n_exceed == 0 else "FAIL"
    print(f"  {status} {name:30s}: max_rel={max_rel*100:.4f}%  max_abs={max_abs:.6f}  "
          f"exceed_{tol*100:.0f}%={n_exceed}/{len(a_np)}")

    if n_exceed > 0 and n_exceed <= 10:
        idxs = np.where(rel_err > tol)[0]
        for idx in idxs[:5]:
            print(f"       idx={idx}: a={a_np[idx]:.6f} b={b_np[idx]:.6f} rel={rel_err[idx]*100:.2f}%")

    return max_rel, n_exceed

# ================= 编码 =================
context = "1"
input_tokens = tokenizer.encode(context)
token_id = input_tokens[0]
print(f"Token: '{context}' -> id={token_id}")

# ================= 初始化 =================
state = model.generate_zero_state(0)
C = model.n_embd
H = model.n_head
N = model.head_size
FFN_DIM = z[f'blocks.{LAYER_ID}.ffn.key.weight'].shape[1]

print(f"C={C}, H={H}, N={N}, FFN_DIM={FFN_DIM}")

x = z['emb.weight'][token_id]  # [C] fp16

bbb = f'blocks.{LAYER_ID}.'
att = f'blocks.{LAYER_ID}.att.'
ffn = f'blocks.{LAYER_ID}.ffn.'

# ================= Step 1: LayerNorm 1 =================
print("\n" + "="*60)
print("Step 1: LayerNorm 1")
print("="*60)

ln1_out = F.layer_norm(x, (C,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
compare("ln1_out", ln1_out)
save_fp16('dbg_ln1_out', ln1_out)

# ================= Step 2: TMix -- Delta Mix =================
print("\n" + "="*60)
print("Step 2: TMix -- Delta Mix (6-way)")
print("="*60)

x_tmix = ln1_out
x_prev_tmix = state[0][LAYER_ID][0].clone()  # zero

xx = x_prev_tmix - x_tmix
state[0][LAYER_ID][0] = x_tmix  # update x_prev

xr = x_tmix + xx * z[att+'x_r']
xw = x_tmix + xx * z[att+'x_w']
xk = x_tmix + xx * z[att+'x_k']
xv = x_tmix + xx * z[att+'x_v']
xa = x_tmix + xx * z[att+'x_a']
xg = x_tmix + xx * z[att+'x_g']

for name, val in [('xr', xr), ('xw', xw), ('xk', xk), ('xv', xv), ('xa', xa), ('xg', xg)]:
    compare(f"delta_mix_{name}", val)
    save_fp16(f'dbg_{name}', val)

# ================= Step 3: TMix -- Projections =================
print("\n" + "="*60)
print("Step 3: TMix -- Main Projections (R, K, V)")
print("="*60)

r = xr @ z[att+'receptance.weight']
k = xk @ z[att+'key.weight']
v = xv @ z[att+'value.weight']

for name, val in [('r_vec', r), ('k_vec', k), ('v_vec', v)]:
    compare(f"proj_{name}", val)
    save_fp16(f'dbg_{name}', val)

# ================= Step 4: TMix -- Low-rank paths =================
print("\n" + "="*60)
print("Step 4: TMix -- Low-rank paths (w, a, g, v_first)")
print("="*60)

# Decay: w = sigmoid(w0 + tanh(xw @ w1) @ w2)
tw = xw @ z[att+'w1']
tw_tanh = torch.tanh(tw)
w_proj2 = tw_tanh @ z[att+'w2']
w_sum = z[att+'w0'] + w_proj2
w_decay = torch.sigmoid(w_sum)

for name, val in [('tw', tw), ('tw_tanh', tw_tanh), ('w_proj2', w_proj2), ('w_sum', w_sum), ('w_decay', w_decay)]:
    compare(f"decay_{name}", val)
    save_fp16(f'dbg_{name}', val)

# Alpha: a = sigmoid(a0 + (xa @ a1) @ a2)
ta = xa @ z[att+'a1']
a_proj2 = ta @ z[att+'a2']
a_sum = z[att+'a0'] + a_proj2
a_vec = torch.sigmoid(a_sum)

for name, val in [('ta', ta), ('a_proj2', a_proj2), ('a_sum', a_sum), ('a_vec', a_vec)]:
    compare(f"alpha_{name}", val)
    save_fp16(f'dbg_{name}', val)

# Gate: g = sigmoid(xg @ g1) @ g2
tg = xg @ z[att+'g1']
tg_sig = torch.sigmoid(tg)
g_vec = tg_sig @ z[att+'g2']

for name, val in [('tg', tg), ('tg_sig', tg_sig), ('g_vec', g_vec)]:
    compare(f"gate_{name}", val)
    save_fp16(f'dbg_{name}', val)

# V-first (layer 0: v_first = v, no blending)
v_first = v.clone()
save_fp16('dbg_v_first', v_first)

# ================= Step 5: TMix -- Key normalization & modulation =================
print("\n" + "="*60)
print("Step 5: TMix -- Key normalization & modulation")
print("="*60)

kk_raw = k * z[att+'k_k']
kk_norm = F.normalize(kk_raw.view(H, N), dim=-1, p=2.0).view(H*N)
a_minus_1 = a_vec - 1.0
am1_ka = a_minus_1 * z[att+'k_a']
scale = 1.0 + am1_ka
k_mod = k * scale

for name, val in [('kk_raw', kk_raw), ('kk_norm', kk_norm), ('a_minus_1', a_minus_1),
                   ('am1_ka', am1_ka), ('scale', scale), ('k_mod', k_mod)]:
    compare(f"key_{name}", val)
    save_fp16(f'dbg_{name}', val)

# ================= Step 6: TMix -- WKV7 core =================
print("\n" + "="*60)
print("Step 6: TMix -- WKV7 core (12 heads)")
print("="*60)

neg_kk = -kk_norm
kk_alpha = kk_norm * a_vec

save_fp16('dbg_neg_kk', neg_kk)
save_fp16('dbg_kk_alpha', kk_alpha)

# Save pre-WKV state
save_fp32('dbg_att_state_pre', state[1][LAYER_ID])

wkv_out = RWKV7_ONE_OP(state[1][LAYER_ID], r, w_decay, k_mod, v, neg_kk, kk_alpha)

compare("wkv_out", wkv_out)
save_fp16('dbg_wkv_out', wkv_out)
save_fp32('dbg_att_state_post', state[1][LAYER_ID])

# ================= Step 7: TMix -- GroupNorm =================
print("\n" + "="*60)
print("Step 7: TMix -- GroupNorm")
print("="*60)

gn_out = F.group_norm(wkv_out.view(1, H*N), num_groups=H,
                       weight=z[att+'ln_x.weight'], bias=z[att+'ln_x.bias'],
                       eps=64e-5).view(H*N)

compare("gn_out", gn_out)
save_fp16('dbg_gn_out', gn_out)

# ================= Step 8: TMix -- Bonus =================
print("\n" + "="*60)
print("Step 8: TMix -- Bonus")
print("="*60)

bonus = ((r * k_mod * z[att+'r_k']).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H*N)
gn_plus_bonus = gn_out + bonus

compare("bonus", bonus)
compare("gn_plus_bonus", gn_plus_bonus)
save_fp16('dbg_bonus', bonus)
save_fp16('dbg_gn_plus_bonus', gn_plus_bonus)

# ================= Step 9: TMix -- Gate + Output projection =================
print("\n" + "="*60)
print("Step 9: TMix -- Gate + Output projection")
print("="*60)

gated = gn_plus_bonus * g_vec
tmix_out = gated @ z[att+'output.weight']

compare("gated", gated)
compare("tmix_out", tmix_out)
save_fp16('dbg_gated', gated)
save_fp16('dbg_tmix_out', tmix_out)

# ================= Step 10: Residual 1 =================
print("\n" + "="*60)
print("Step 10: Residual 1 (x + tmix_out)")
print("="*60)

x_after_att = x + tmix_out
compare("x_after_att", x_after_att)
save_fp16('dbg_x_after_att', x_after_att)

# ================= Step 11: LayerNorm 2 =================
print("\n" + "="*60)
print("Step 11: LayerNorm 2")
print("="*60)

ln2_out = F.layer_norm(x_after_att, (C,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
compare("ln2_out", ln2_out)
save_fp16('dbg_ln2_out', ln2_out)

# ================= Step 12: CMix -- Delta Mix =================
print("\n" + "="*60)
print("Step 12: CMix -- Delta Mix")
print("="*60)

x_cmix = ln2_out
x_prev_cmix = state[0][LAYER_ID][1].clone()  # zero
xx_cmix = x_prev_cmix - x_cmix
state[0][LAYER_ID][1] = x_cmix

k_mixed = x_cmix + xx_cmix * z[ffn+'x_k']
compare("cmix_k_mixed", k_mixed)
save_fp16('dbg_cmix_k_mixed', k_mixed)

# ================= Step 13: CMix -- Expand + ReLU² =================
print("\n" + "="*60)
print("Step 13: CMix -- Expand + ReLU^2")
print("="*60)

k_expanded = k_mixed @ z[ffn+'key.weight']
k_act = torch.relu(k_expanded) ** 2

compare("cmix_k_expanded", k_expanded)
compare("cmix_k_act", k_act)
save_fp16('dbg_cmix_k_expanded', k_expanded)
save_fp16('dbg_cmix_k_act', k_act)

# ================= Step 14: CMix -- Contract =================
print("\n" + "="*60)
print("Step 14: CMix -- Contract")
print("="*60)

cmix_out = k_act @ z[ffn+'value.weight']
compare("cmix_out", cmix_out)
save_fp16('dbg_cmix_out', cmix_out)

# ================= Step 15: Residual 2 =================
print("\n" + "="*60)
print("Step 15: Residual 2 (x_after_att + cmix_out)")
print("="*60)

x_out = x_after_att + cmix_out
compare("x_out", x_out)
save_fp16('dbg_x_out', x_out)

# ================= 交叉验证: 与之前导出的 golden 对比 =================
print("\n" + "="*60)
print("Cross-validation: dbg vs golden")
print("="*60)

pairs = [
    ('ln1_out', 'dbg_ln1_out'),
    ('tmix_out', 'dbg_tmix_out'),
    ('x_after_att', 'dbg_x_after_att'),
    ('ln2_out', 'dbg_ln2_out'),
    ('cmix_out', 'dbg_cmix_out'),
    ('x_out', 'dbg_x_out'),
    ('v_first_out', 'dbg_v_first'),
    ('tmix_x_prev_out', 'dbg_ln1_out'),  # x_prev = ln1_out for tmix
]

for golden_name, dbg_name in pairs:
    g = load_fp16(golden_name)
    d = load_fp16(dbg_name)
    compare_two(f"{golden_name} vs {dbg_name}", g, d, tol=0.001)

# FP32 att_state
g_st = load_fp32('att_state_out')
d_st = load_fp32('dbg_att_state_post')
compare_two("att_state_out vs dbg", g_st, d_st, tol=0.001)

print("\nDone! All debug hex files saved to", EXPORT_DIR)
