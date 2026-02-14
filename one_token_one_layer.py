"""
Export RWKV7 Layer 0 single-token golden data for RTL verification.

Outputs hex files to EXPORT_DIR that tb_rwkv7_layer.sv can load via $readmemh.

Data format: FP16 → 4-digit hex, FP32 → 8-digit hex, one element per line.
2D matrices stored row-major: mat[R][C] → R*C lines.
3D att_state[H][N][N] → H*N*N lines.
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
from reference.rwkv7 import RWKV_x070, RWKV_x070_TMix_one, RWKV_x070_CMix_one, RWKV7_ONE_OP
from reference.rwkv7 import enable_print, DTYPE, HEAD_SIZE
import reference.rwkv7 as rwkv7_mod
rwkv7_mod.enable_print = False

model = RWKV_x070(args)
z = model.z

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

# ================= 工具函数 =================
def fp16_to_hex(t: torch.Tensor) -> list:
    """FP16 tensor → list of 4-char hex strings."""
    t = t.detach().cpu().to(torch.float16).contiguous()
    raw = t.numpy().view(np.uint16).flatten()
    return [f"{v:04x}" for v in raw]

def fp32_to_hex(t: torch.Tensor) -> list:
    """FP32 tensor → list of 8-char hex strings."""
    t = t.detach().cpu().to(torch.float32).contiguous()
    raw = t.numpy().view(np.uint32).flatten()
    return [f"{v:08x}" for v in raw]

def save_hex(name: str, lines: list):
    path = os.path.join(EXPORT_DIR, f"{name}.hex")
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  {name}.hex: {len(lines)} values")

def save_fp16(name: str, t: torch.Tensor):
    save_hex(name, fp16_to_hex(t))

def save_fp32(name: str, t: torch.Tensor):
    save_hex(name, fp32_to_hex(t))

# ================= 编码 =================
context = "1"
input_tokens = tokenizer.encode(context)
token_id = input_tokens[0]
print(f"Token: '{context}' → id={token_id}")

# ================= 初始化状态 =================
state = model.generate_zero_state(0)
# state[0]: [n_layer, 2, C] — x_prev for tmix(slot0) and cmix(slot1)
# state[1]: [n_layer, H, N, N] — att_state (fp32)

C = model.n_embd
H = model.n_head
N = model.head_size
FFN_DIM = z[f'blocks.{LAYER_ID}.ffn.key.weight'].shape[1]  # transposed: [C, FFN_DIM]

print(f"C={C}, H={H}, N={N}, FFN_DIM={FFN_DIM}")

# ================= 获取 embedding =================
x = z['emb.weight'][token_id]  # [C] fp16

# ================= 导出权重 =================
print("\n--- Exporting weights ---")
bbb = f'blocks.{LAYER_ID}.'
att = f'blocks.{LAYER_ID}.att.'
ffn = f'blocks.{LAYER_ID}.ffn.'

# LayerNorm 1 & 2
save_fp16('ln1_w', z[bbb+'ln1.weight'])
save_fp16('ln1_b', z[bbb+'ln1.bias'])
save_fp16('ln2_w', z[bbb+'ln2.weight'])
save_fp16('ln2_b', z[bbb+'ln2.bias'])

# TMix delta mix coefficients
save_fp16('att_x_r', z[att+'x_r'])
save_fp16('att_x_w', z[att+'x_w'])
save_fp16('att_x_k', z[att+'x_k'])
save_fp16('att_x_v', z[att+'x_v'])
save_fp16('att_x_a', z[att+'x_a'])
save_fp16('att_x_g', z[att+'x_g'])

# TMix large projection matrices
# Python: x @ W  where W is [C, C] (model loader transposed)
# RTL matvec: y[m] = Σ weight[m][n] * x[n], weight is [out, in]
# So RTL weight = Python W.T
save_fp16('att_R', z[att+'receptance.weight'].T.contiguous())  # RTL: [C, C]
save_fp16('att_K', z[att+'key.weight'].T.contiguous())          # RTL: [C, C]
save_fp16('att_V', z[att+'value.weight'].T.contiguous())        # RTL: [C, C]
save_fp16('att_O', z[att+'output.weight'].T.contiguous())       # RTL: [C, C]

# TMix decay low-rank
# Python: xw @ w1 where w1 is [C, D_W] → RTL: w_w1[D_W][C] = w1.T
save_fp16('att_w0', z[att+'w0'])
save_fp16('att_w1', z[att+'w1'].T.contiguous())  # RTL: [D_W, C]
save_fp16('att_w2', z[att+'w2'].T.contiguous())  # RTL: [C, D_W]

# TMix alpha low-rank
save_fp16('att_a0', z[att+'a0'])
save_fp16('att_a1', z[att+'a1'].T.contiguous())  # RTL: [D_A, C]
save_fp16('att_a2', z[att+'a2'].T.contiguous())  # RTL: [C, D_A]

# TMix v_first low-rank (layer 0: v0/v1/v2 = a0/a1/a2, actually ignored)
save_fp16('att_v0', z[att+'v0'])
save_fp16('att_v1', z[att+'v1'].T.contiguous())  # RTL: [D_V, C]
save_fp16('att_v2', z[att+'v2'].T.contiguous())  # RTL: [C, D_V]

# TMix gate low-rank
save_fp16('att_g1', z[att+'g1'].T.contiguous())  # RTL: [D_G, C]
save_fp16('att_g2', z[att+'g2'].T.contiguous())  # RTL: [C, D_G]

# TMix key/bonus
save_fp16('att_k_k', z[att+'k_k'])
save_fp16('att_k_a', z[att+'k_a'])
save_fp16('att_r_k', z[att+'r_k'])

# TMix GroupNorm
save_fp16('att_ln_w', z[att+'ln_x.weight'])
save_fp16('att_ln_b', z[att+'ln_x.bias'])

# CMix weights
# Python: k @ K_ where K_ is [C, FFN_DIM] (transposed by loader)
# RTL: w_K[FFN_DIM][C] = K_.T
save_fp16('ffn_x_k', z[ffn+'x_k'])
save_fp16('ffn_K', z[ffn+'key.weight'].T.contiguous())    # RTL: [FFN_DIM, C]
save_fp16('ffn_V', z[ffn+'value.weight'].T.contiguous())  # RTL: [C, FFN_DIM]

# ================= 导出输入状态 =================
print("\n--- Exporting input data & state ---")
save_fp16('x_in', x)

# TMix x_prev (slot 0) — zero state
save_fp16('tmix_x_prev_in', state[0][LAYER_ID][0])
# CMix x_prev (slot 1) — zero state
save_fp16('cmix_x_prev_in', state[0][LAYER_ID][1])
# v_first — uninitialized, layer 0 will set it = v
save_fp16('v_first_in', torch.zeros(C, dtype=DTYPE, device='cuda'))
# att_state — zero [H, N, N] fp32
save_fp32('att_state_in', state[1][LAYER_ID])

# ================= 运行 Layer 0 推理 =================
print("\n--- Running Layer 0 inference ---")

# LN1
xx = F.layer_norm(x, (C,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
save_fp16('ln1_out', xx)

# TMix
v_first = torch.empty_like(x)
tmix_out, v_first = RWKV_x070_TMix_one(
    LAYER_ID, H, N, xx,
    state[0][LAYER_ID], v_first, state[1][LAYER_ID],
    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
    z[att+'w0'], z[att+'w1'], z[att+'w2'],
    z[att+'a0'], z[att+'a1'], z[att+'a2'],
    z[att+'v0'], z[att+'v1'], z[att+'v2'],
    z[att+'g1'], z[att+'g2'],
    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
    z[att+'ln_x.weight'], z[att+'ln_x.bias']
)
save_fp16('tmix_out', tmix_out)

# Residual 1
x_after_att = x + tmix_out
save_fp16('x_after_att', x_after_att)

# LN2
xx2 = F.layer_norm(x_after_att, (C,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
save_fp16('ln2_out', xx2)

# CMix
cmix_out = RWKV_x070_CMix_one(xx2, state[0][LAYER_ID], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
save_fp16('cmix_out', cmix_out)

# Residual 2
x_out = x_after_att + cmix_out
save_fp16('x_out', x_out)

# ================= 导出输出状态 =================
print("\n--- Exporting output state ---")
save_fp16('tmix_x_prev_out', state[0][LAYER_ID][0])
save_fp16('cmix_x_prev_out', state[0][LAYER_ID][1])
save_fp16('v_first_out', v_first)
save_fp32('att_state_out', state[1][LAYER_ID])

# ================= 打印维度信息 =================
print("\n--- Weight dimensions ---")
for name, key in [
    ('att_w1', att+'w1'), ('att_w2', att+'w2'),
    ('att_a1', att+'a1'), ('att_a2', att+'a2'),
    ('att_v1', att+'v1'), ('att_v2', att+'v2'),
    ('att_g1', att+'g1'), ('att_g2', att+'g2'),
    ('att_R', att+'receptance.weight'),
    ('att_K', att+'key.weight'),
    ('att_V', att+'value.weight'),
    ('att_O', att+'output.weight'),
    ('ffn_K', ffn+'key.weight'),
    ('ffn_V', ffn+'value.weight'),
]:
    print(f"  {name}: {list(z[key].shape)}")

print(f"\nDone! Golden data exported to {EXPORT_DIR}")
print(f"Total files: {len(os.listdir(EXPORT_DIR))}")
