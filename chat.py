import sys, types, torch
from torch.nn import functional as F

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "D:\\Development\\models\\rwkv\\rwkv7-g1d-0.1b-20260129-ctx8192"

print(f'\nLoading {args.MODEL_NAME} ...')

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple

model = RWKV_x070(args)
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

TEMPERATURE = 1.0
TOP_P = 0.7
MAX_TOKENS = 2048

def sample(logits, temp=TEMPERATURE, top_p=TOP_P):
    probs = F.softmax(logits.float() / temp, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[idx].item()

def generate(prompt, state, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(prompt)
    logits = model.forward(tokens, state)

    out_tokens = []
    for _ in range(max_tokens):
        token = sample(logits)
        out_tokens.append(token)

        # 流式输出
        try:
            text = tokenizer.decode(out_tokens, utf8_errors="strict")
            sys.stdout.write(text)
            sys.stdout.flush()
            out_tokens = []
        except:
            pass

        # 遇到 \n\n 或特殊结束标记就停
        if token == 0:
            break

        logits = model.forward([token], state)

    # flush 剩余
    if out_tokens:
        text = tokenizer.decode(out_tokens, utf8_errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
    print()

print("Model loaded. Type your message, or 'quit' to exit.\n")

state = model.generate_zero_state(0)

PROMPT_TEMPLATE = "User: {user}\n\nAssistant:"

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Bye!")
        break

    prompt = PROMPT_TEMPLATE.format(user=user_input)
    print("Assistant: ", end="", flush=True)
    generate(prompt, state)
