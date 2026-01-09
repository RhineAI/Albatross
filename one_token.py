import numpy as np
import types, torch
from torch.nn import functional as F

# ================= 配置部分 =================
args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
# 请确保路径与你实际存放模型的位置一致
args.MODEL_NAME = "/root/models/rwkv7-g1a-0.1b-20250728-ctx4096"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

# ================= 加载模型 =================
from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

# ================= 核心任务 =================

# 1. 定义前文 (Context)
context = "1+3="
print(f"Input: '{context}'")

# 2. 编码 (Encoding)
# 将文本转换为 token id 列表
input_tokens = tokenizer.encode(context)

# 3. 初始化状态 (State)
# Batch Size = 1
state = model.generate_zero_state(1)

# 4. 推理 (Inference)
# forward_batch 接受 token 列表的列表，这里放入一个样本 [input_tokens]
# out 包含了 logits (预测概率的未归一化值)
out_batch = model.forward_batch([input_tokens], state)
logits = out_batch[0] # 取出这一条数据的输出

# 5. 获取下一个 Token
probs = F.softmax(logits.float(), dim=-1) # 转为概率分布
_, indices = torch.topk(probs, 1)         # 取概率最高的一个
next_token_id = indices[0].item()
next_token_str = tokenizer.decode([next_token_id])

# 6. 输出结果
print(f"Next Token: '{next_token_str}'")
