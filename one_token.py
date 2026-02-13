import numpy as np
import types, torch
from torch.nn import functional as F

# ================= 配置部分 =================
args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "D:\\Development\\models\\rwkv\\rwkv7-g1d-0.1b-20260129-ctx8192"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

# ================= 加载模型 =================
from reference.rwkv7 import RWKV_x070, print_information
model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

# ================= 核心任务 =================

# 1. 定义前文 (Context)
context = "1"
print(f"Input: '{context}'")

# 2. 编码 (Encoding)
print_information('START Encoding')
input_tokens = tokenizer.encode(context)
print_information('input_tokens', len(input_tokens))
print_information('END Encoding')

# 3. 初始化状态 (State)
print_information('START Init State')
state = model.generate_zero_state(0)
print_information('state[0] (x_prev)', state[0])
print_information('state[1] (att_state)', state[1])
print_information('END Init State')

# 4. 推理 (Inference)
print_information('START Inference')
logits = model.forward(input_tokens, state)
print_information('logits', logits)
print_information('END Inference')

# 5. 获取下一个 Token
print_information('START Sampling')
probs = F.softmax(logits.float(), dim=-1)
print_information('probs', probs)
_, indices = torch.topk(probs, 1)
next_token_id = indices[0].item()
print_information('next_token_id', next_token_id)
next_token_str = tokenizer.decode([next_token_id])
print_information('END Sampling')

# 6. 输出结果
print(f"Next Token: '{next_token_str}'")