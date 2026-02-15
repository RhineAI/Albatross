"""
Export RWKV7 model weights for RTL verification.

Outputs hex files to golden/weights/{emb,layer0,layer1,...,output}/ for $readmemh loading.
文件名不带 blocks.X 前缀（每层已有独立目录），数据按 Python 加载后的原始布局存储。
"""

import os
from golden.utils import GOLDEN_ROOT, save, load_model

# ================= 加载模型 =================
model, z = load_model()

C = model.n_embd
H = model.n_head
N = model.head_size
n_layer = model.n_layer

print(f"C={C}, H={H}, N={N}, n_layer={n_layer}")

# ================= Embedding =================
WEIGHTS_EMB_DIR = os.path.join(GOLDEN_ROOT, "weights", "emb")
os.makedirs(WEIGHTS_EMB_DIR, exist_ok=True)

print("\n--- Exporting embedding weights ---")
save(WEIGHTS_EMB_DIR, 'emb.weight', z['emb.weight'])

# ================= 逐层导出 =================
for layer_id in range(n_layer):
    bbb = f'blocks.{layer_id}.'
    att = f'blocks.{layer_id}.att.'
    ffn = f'blocks.{layer_id}.ffn.'
    FFN_DIM = z[ffn+'key.weight'].shape[1]

    W = os.path.join(GOLDEN_ROOT, "weights", f"layer{layer_id}")
    os.makedirs(W, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Layer {layer_id}  (FFN_DIM={FFN_DIM})")
    print(f"{'='*60}")

    # 1D 权重
    KEYS_1D = [
        ('ln1.weight', bbb+'ln1.weight'), ('ln1.bias', bbb+'ln1.bias'),
        ('ln2.weight', bbb+'ln2.weight'), ('ln2.bias', bbb+'ln2.bias'),
        ('att.x_r', att+'x_r'), ('att.x_w', att+'x_w'), ('att.x_k', att+'x_k'),
        ('att.x_v', att+'x_v'), ('att.x_a', att+'x_a'), ('att.x_g', att+'x_g'),
        ('att.w0', att+'w0'), ('att.a0', att+'a0'), ('att.v0', att+'v0'),
        ('att.k_k', att+'k_k'), ('att.k_a', att+'k_a'), ('att.r_k', att+'r_k'),
        ('att.ln_x.weight', att+'ln_x.weight'), ('att.ln_x.bias', att+'ln_x.bias'),
        ('ffn.x_k', ffn+'x_k'),
    ]
    for name, key in KEYS_1D:
        save(W, name, z[key])

    # 2D 矩阵
    KEYS_2D = [
        ('att.receptance.weight', att+'receptance.weight'),
        ('att.key.weight', att+'key.weight'),
        ('att.value.weight', att+'value.weight'),
        ('att.output.weight', att+'output.weight'),
        ('att.w1', att+'w1'), ('att.w2', att+'w2'),
        ('att.a1', att+'a1'), ('att.a2', att+'a2'),
        ('att.v1', att+'v1'), ('att.v2', att+'v2'),
        ('att.g1', att+'g1'), ('att.g2', att+'g2'),
        ('ffn.key.weight', ffn+'key.weight'),
        ('ffn.value.weight', ffn+'value.weight'),
    ]
    for name, key in KEYS_2D:
        save(W, name, z[key])

    # 打印维度
    print("  --- dimensions ---")
    for name, key in KEYS_2D:
        print(f"    {name}: {list(z[key].shape)}")

# ================= Output weights =================
print(f"\n{'='*60}")
print("  Output weights")
print(f"{'='*60}")

WEIGHTS_OUT_DIR = os.path.join(GOLDEN_ROOT, "weights", "output")
os.makedirs(WEIGHTS_OUT_DIR, exist_ok=True)

save(WEIGHTS_OUT_DIR, 'ln_out.weight', z['ln_out.weight'])
save(WEIGHTS_OUT_DIR, 'ln_out.bias', z['ln_out.bias'])
save(WEIGHTS_OUT_DIR, 'head.weight', z['head.weight'])

print(f"\nDone! Weights exported to {GOLDEN_ROOT}/weights/")
