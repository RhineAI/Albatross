########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import random, math, time, json
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SHOW_SPEED_PERCENTILE = 50

from mlx_lm import load, generate
import mlx.core as mx
import mlx.nn as nn
import numpy as np
mx.random.seed(SEED)

########################################################################################################

model, _ = load("/Users/molly/rwkv7-2.9B-g1a", tokenizer_config={"trust_remote_code": True})

from reference.utils import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################

def xprint(s):
    c0, c1 = 3, 80-len(s)-3
    print(f"\n{'#'*c0} {s} {'#'*c1}\n")

#######################################################################################################

# xprint("Prefill")

# raw = open("eval/calibration_data_v5_rc.txt").read()
# tokens = tokenizer.encode(raw)
# # print(len(tokens))

# for stage in range(12, 12+1):
#     CTX_LEN = 2**stage
#     loss = 0
#     a = 0
#     cnt = 0

#     times = []
#     step_size = 2048
#     while a+CTX_LEN < len(tokens):
#         src = tokens[a:a+CTX_LEN]

#         t0 = time.perf_counter()
#         cache = model.make_cache()
#         logits = model(mx.array(src[:-1]).astype(mx.int32).reshape(1, -1), cache=cache)
#         mx.eval([c.state for c in cache])
#         mx.clear_cache()
#         t1 = time.perf_counter()
#         times.append(t1 - t0)

#         probs = nn.softmax(logits.astype(mx.float32))
#         for j in range(CTX_LEN-1):
#             loss -= math.log(probs[0][j][src[j+1]])
#             cnt += 1
#         a += CTX_LEN

#     times = np.percentile(times, SHOW_SPEED_PERCENTILE)
#     print(f'CTX_LEN {CTX_LEN} : avg loss {round(loss/cnt,4)} || prefill {round((CTX_LEN-1)/times)} token/s')# = {round((CTX_LEN-1)/times * active_params * 2/1e12, 2)} TFLOPS')

#######################################################################################################

# xprint("Arithmetic")

def eval_qa(todo, print_interval, pad_eod = True, loss_mode = False):
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in todo:
        if pad_eod:
            src = [0] + tokenizer.encode(d[0])
        else:
            src = tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        logits = 0
        correct = True

        out = model(mx.array(src+dst).astype(mx.int32).reshape(1, -1))
        mx.eval(out)

        for i in range(len(dst)):
            probs = mx.softmax(out[0][len(src)-1+i].astype(mx.float32), axis=-1)
            logits += math.log(probs[dst[i]])
            if mx.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % print_interval == 0 or xcnt == len(todo):
            if loss_mode:
                print('loss', round(-xsum / xcnt, 2), 'acc', round(xacc/xcnt*100, 1))
            else:
                print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 1))

# x1, x2 = 1, 2
# magic = (5**(0.5)-1)/2
# for stage in range(2,4+1):
#     todo = []
#     NUMBER_LIMIT = 10**stage
#     for i in range(200):
#         x1 += i
#         x2 += i*i
#         s1 = int(magic * x1 * NUMBER_LIMIT) % NUMBER_LIMIT
#         s2 = int(magic * x2 * NUMBER_LIMIT) % NUMBER_LIMIT
#         # todo.append([f'\nAssistant: {s1}+{s2}=',str(s1+s2)])
#         # todo.append([f'\nAssistant: {s1}-{s2}=',str(s1-s2)])
#         todo.append([f'\nA: 123+321=444\n{s1}+{s2}=',str(s1+s2)]) # better prompt
#         todo.append([f'\nA: 123-321=-198\n{s1}-{s2}=',str(s1-s2)]) # better prompt
#     # print(todo)
#     print(f"Len {stage} : ", end="")
#     eval_qa(todo, 99999999, pad_eod=False, loss_mode=True)

# #######################################################################################################

# xprint("Repeat")

# class LCG:
#     def __init__(self, seed=42):
#         self.m = 2**32  # Modulus
#         self.a = 1664525  # Multiplier
#         self.c = 1013904223  # Increment
#         self.state = seed
#     def _generate(self):
#         self.state = (self.a * self.state + self.c) % self.m
#         return self.state
#     def randint(self, min_val, max_val):
#         if min_val > max_val:
#             raise ValueError("min_val cannot be greater than max_val")
#         range_size = max_val - min_val + 1
#         return min_val + self._generate() % range_size
# lcg = LCG()
# def generate_random_number_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     if n == 1:
#         return str(generator.randint(0, 9))
#     first_digit = str(generator.randint(1, 9))
#     remaining_digits = [str(generator.randint(0, 9)) for _ in range(n - 1)]
#     return first_digit + "".join(remaining_digits)
# def generate_random_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     ccccc = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     chars = [ccccc[generator.randint(0, len(ccccc)-1)] for _ in range(n)]
#     return "".join(chars)

# for stage in range(4):
#     todo = []
#     l_max = 0
#     l_min = 1e10
#     for i in range(100):
#         l = round(pow(2,(stage+i/100)) * 100)
#         l_min = min(l, l_min)
#         l_max = max(l, l_max)
#         s = generate_random_string(l, lcg)
#         todo.append([f'\nYou must remember the secret is {s}. Repeat: the secret is', f' {s}'])
#     print(f"Len {l_min} to {l_max} : ", end="")
#     eval_qa(todo, 99999999, loss_mode=True)

# #######################################################################################################

xprint('LAMBADA')

with open(f"eval/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

eval_qa(todo, 100)

# ########################################################################################################

# xprint('MMLU')

# from datasets import load_from_disk
# mmlu_test = load_from_disk("eval/mmlu_test_dataset")

# TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
# <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>

# Assistant: The answer is'''

# CHOICES = [" A", " B", " C", " D"]

# SHUFFLE = False

# correct = 0
# total = 0
# pbar = tqdm(total=len(mmlu_test))

# choices_token = [tokenizer.encode(x) for x in CHOICES]
# assert all([len(x) == 1 for x in choices_token])
# choices_token = [x[0] for x in choices_token]

# for idx, sample in enumerate(mmlu_test):
#     question = sample["question"]
#     choices = sample["choices"]
#     subject = sample["subject"]
#     gt = sample["answer"]

#     if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
#         original_gt_text = choices[gt]
#         np.random.shuffle(choices)
#         gt = choices.index(original_gt_text)

#     all_prefix = (
#         TEMPLATE.replace("<Q>", question)
#         .replace("<|A|>", choices[0])
#         .replace("<|B|>", choices[1])
#         .replace("<|C|>", choices[2])
#         .replace("<|D|>", choices[3])
#         .replace("<SUBJECT>", subject.replace("_", " "))
#     )

#     if idx == 0:
#         print(f"Format example:")
#         print("-" * 80)
#         print(all_prefix)
#         print("-" * 80)
#         format_example = all_prefix

#     all_prefix_ids = [0] + tokenizer.encode(all_prefix.replace('\r\n','\n').strip())

#     logits = model.forward(all_prefix_ids, model.generate_zero_state(0), full_output=False)
    
#     neg_log_prob = F.log_softmax(logits, dim=-1)
#     target_prob = neg_log_prob[choices_token]
    
#     if torch.argmax(target_prob).item() == gt:
#         correct += 1
#     total += 1
#     pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
#     pbar.update(1)
# pbar.close()
# print()
