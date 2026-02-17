"""
CMix mini 核心计算函数 — 基于 RWKV_x070_CMix_one 改写，加入中间结果导出。

提供 RWKV_x070_CMix_one_export 供 generate_layer_mini.py 调用。
维度常量: C=128, FFN_DIM=512
"""

import torch
from golden.utils import save
from reference.rwkv7 import DTYPE

C = 128
FFN_DIM = 512


# 从 RWKV_x070_CMix_one 复制，加入中间结果导出
def RWKV_x070_CMix_one_export(out_dir, x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x

    k = x + xx * x_k
    save(out_dir, 'ffn.x_k.out', k)

    k_expanded = k @ K_
    save(out_dir, 'ffn.key.weight.in', k)
    save(out_dir, 'ffn.key.weight.out', k_expanded)

    k = torch.relu(k_expanded) ** 2
    save(out_dir, 'ffn.key.weight.act', k)

    save(out_dir, 'ffn.value.weight.in', k)
    out = k @ V_
    save(out_dir, 'ffn.value.weight.out', out)

    return out
