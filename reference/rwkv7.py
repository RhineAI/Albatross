########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

from typing import List
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

import torch.nn as nn
from torch.nn import functional as F

DTYPE = torch.half

########################################################################################################

HEAD_SIZE = 64

enable_print = True
def print_information(name = None, value = None):
    if not enable_print:
        return
    if name == None and value == None:
        return
    if value == None:
        print(f"--- {name} ---")
        return
    dtype_mapping = {
        torch.float16: "FP16", torch.float32: "FP32", torch.float64: "FP64", torch.bfloat16: "BF16",
        torch.int8: "INT8", torch.int32: "INT32", torch.int64: "INT64", torch.bool: "BOOL",
        float: "FP32", int: "INT32", bool: "BOOL"
    }
    if torch.is_tensor(value):
        dtype_str = dtype_mapping.get(value.dtype, str(value.dtype).replace('torch.', ''))
        print(f"{name}: {dtype_str} {list(value.shape)}")
    elif isinstance(value, (int, float, bool)):
        print(f"{name}: {dtype_mapping.get(type(value), 'UNKNOWN')}")
    else:
        print(f"{name}: {type(value).__name__} {list(value.shape) if hasattr(value, 'shape') else ''}")

import math
EXP_NEG_HALF = math.exp(-0.5)  # 0.6065306597...

def _wkv7_core(state, r, w, k, v, a, b):
    """
    Pure PyTorch implementation matching the CUDA kernel exactly.
    All internal computation in fp32. state is fp32 and modified in-place.
    Inputs r,w,k,v,a,b are fp16, output is fp16.

    state: [H, N, N] (fp32)
    r, w, k, v, a, b: [H, N] per timestep (fp16 input)

    CUDA kernel per head h, per row i (threadIdx.x):
        state[j] loaded as fp32
        for each timestep t:
            sa = sum_j(a[j] * state[j])
            state[j] = state[j] * w[j] + k[j] * v[i] + sa * b[j]   (i is the row)
            y[i] = sum_j(state[j] * r[j])
        Note: each thread i has its own state row state[i,:], and v[i] is the i-th element.
              w, k, r, a, b are shared across all threads (all rows) within a head.
    """
    # r,w,k,v,a,b: [H, N] fp32
    H, N = r.shape
    print_information('START _wkv7_core')
    print_information('H', H)
    print_information('N', N)
    print_information('r (input fp16)', r)
    print_information('w (input fp16)', w)
    print_information('k (input fp16)', k)
    print_information('v (input fp16)', v)
    print_information('a (input fp16)', a)
    print_information('b (input fp16)', b)
    print_information('state (input fp32)', state)

    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    # w transform: exp(-exp(-0.5) * w), w is already sigmoid'd from caller
    w = torch.exp(-EXP_NEG_HALF * w.float())  # [H, N]
    print_information('w (after exp transform)', w)

    # sa[h,i] = sum_j a[h,j] * state[h,i,j]  =>  sa = (state @ a.unsqueeze(-1)).squeeze(-1)  [H,N]
    # but actually sa is the same for all rows i within a head: sa[h] = sum_j a[h,j] * state[h,i,j]
    # Wait - re-reading the kernel: each thread i has state[j] = state[i][j]. sa = sum_j a[j]*state[i][j].
    # So sa is PER ROW: sa[h,i] = sum_j(a[h,j] * state[h,i,j])
    sa = torch.einsum('hn,hin->hi', a, state)  # [H, N_rows] = [H, N]
    print_information('sa (a @ state)', sa)

    # state[h,i,j] = state[h,i,j] * w[h,j] + k[h,j] * v[h,i] + sa[h,i] * b[h,j]
    state.mul_(w.unsqueeze(1))  # state[h,i,j] *= w[h,j]
    print_information('state (after *w)', state)
    state.add_(k.unsqueeze(1) * v.unsqueeze(2))  # + k[h,j] * v[h,i]
    print_information('state (after +kv)', state)
    state.add_(sa.unsqueeze(2) * b.unsqueeze(1))  # + sa[h,i] * b[h,j]
    print_information('state (after +sa*b)', state)

    # y[h,i] = sum_j(state[h,i,j] * r[h,j])
    y = torch.einsum('hin,hn->hi', state, r)  # [H, N]
    print_information('y (output)', y)
    print_information('END _wkv7_core')
    return y.to(DTYPE)

def RWKV7_ONE_OP(state, r, w, k, v, a, b):
    """Single token: r,w,k,v,a,b are [C], state is [H,N,N] fp32"""
    with torch.no_grad():
        C = r.shape[0]
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        print_information('START RWKV7_ONE_OP')
        print_information('C (channels)', C)
        print_information('H (heads)', H)
        print_information('N (head_size)', N)
        r_ = r.view(H, N)
        w_ = w.view(H, N)
        k_ = k.view(H, N)
        v_ = v.view(H, N)
        a_ = a.view(H, N)
        b_ = b.view(H, N)
        print_information('r_ (reshaped)', r_)
        print_information('w_ (reshaped)', w_)
        print_information('k_ (reshaped)', k_)
        print_information('v_ (reshaped)', v_)
        print_information('a_ (reshaped)', a_)
        print_information('b_ (reshaped)', b_)
        print_information('state (input)', state)
        y = _wkv7_core(state, r_, w_, k_, v_, a_, b_)
        print_information('y (wkv7_core output)', y)
        print_information('state (after wkv7_core)', state)
        print_information('END RWKV7_ONE_OP')
        return y.view(C)

def RWKV7_OP(state, r, w, k, v, a, b):
    """Sequence: r,w,k,v,a,b are [T,C], state is [H,N,N] fp32"""
    with torch.no_grad():
        T, C = r.shape
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        y = torch.empty((T, C), device=r.device, dtype=DTYPE)
        for t in range(T):
            y[t] = RWKV7_ONE_OP(state, r[t], w[t], k[t], v[t], a[t], b[t])
        return y

def RWKV7_BATCH_OP(state, r, w, k, v, a, b):
    """Batch: r,w,k,v,a,b are [B,T,C], state is [B,H,N,N] fp32"""
    with torch.no_grad():
        B, T, C = r.shape
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        y = torch.empty((B, T, C), device=r.device, dtype=DTYPE)
        for b_idx in range(B):
            for t in range(T):
                y[b_idx, t] = RWKV7_ONE_OP(
                    state[b_idx], r[b_idx, t], w[b_idx, t], k[b_idx, t],
                    v[b_idx, t], a[b_idx, t], b[b_idx, t])
        return y

########################################################################################################

class RWKV_x070(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = 64
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            kk = k.split('.')
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        print(args)
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0'] # actually ignored
        z['blocks.0.att.v1'] = z['blocks.0.att.a1'] # actually ignored
        z['blocks.0.att.v2'] = z['blocks.0.att.a2'] # actually ignored

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None]
        if bsz >= 1:
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False): # will modify state in-place
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    def forward_batch(self, tokens, state, full_output=False): # will modify state in-place
        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            return self.forward_batch_same_length(tokens, state, full_output)

        bsz = len(tokens)
        pos = [0] * bsz

        if full_output == False:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda") for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] for i in active]
            batch_state = [state[0][:,:,active],state[1][:,active]] # state[0]=[Layer][2][Bsz][C]    state[1]=[Layer][Bsz][H][N][N]
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)
            for k, i in enumerate(active):
                if full_output == False:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                state[0][:,:,i] = batch_state[0][:,:,k]
                state[1][:,i] = batch_state[1][:,k]
                pos[i] += step
        return out

    def forward_batch_same_length(self, tokens, state, full_output=False):
        assert type(tokens) is list
        assert len(set([len(x) for x in tokens])) == 1, 'here all sequences must have the same length'
        return self.forward_seq_batch(tokens, state, full_output)

    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            print_information('Hidden Size', x)
            print_information('Layer Number', self.n_layer)
            print_information('Embedding Weight', z['emb.weight'])


            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                if i > 0:
                    global enable_print
                    enable_print = False
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                print_information('Layer Norm 1 Weight', z[bbb+'ln1.weight'])
                print_information('Layer Norm 1 Bias', z[bbb+'ln1.bias'])
                print_information('Layer Norm 1 Input', x)
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                print_information('Layer Norm 1 Output', xx)

                print_information('Head Number', self.n_head)
                print_information('Head Size', self.head_size)
                xx, v_first = RWKV_x070_TMix_one(
                    i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias']
                )
                x = x + xx
                print_information('x (after TMix residual)', x)

                print_information('Layer Norm 2 Weight', z[bbb+'ln2.weight'])
                print_information('Layer Norm 2 Bias', z[bbb+'ln2.bias'])
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                print_information('Layer Norm 2 Output', xx)

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
                print_information('x (after CMix residual)', x)

            enable_print = True
            print_information('START Final Output')
            print_information('ln_out.weight', z['ln_out.weight'])
            print_information('ln_out.bias', z['ln_out.bias'])
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            print_information('x (after final LN)', x)
            print_information('head.weight', z['head.weight'])
            x = x @ z['head.weight']
            print_information('logits', x)
            print_information('END Final Output')
            return x
        
    
    def forward_one_alt(self, x:torch.Tensor, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x
        
    
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][torch.tensor(idxs, device=z['emb.weight'].device)]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    
    def forward_one_batch_alt(self, x:torch.Tensor, state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

########################################################################################################

def RWKV_x070_TMix_one(
        layer_id: int, H:int, N:int, x, x_prev, v_first, state,
        x_r, x_w, x_k, x_v, x_a, x_g,
        w0, w1, w2, a0, a1, a2, v0, v1, v2,
        g1, g2, k_k, k_a, r_k,
        R_, K_, V_, O_,
        ln_w, ln_b
):
    xx = x_prev[0] - x
    x_prev[0] = x
    print_information('x', x)
    print_information('x_prev', x_prev[0])
    print_information('xx', xx)

    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    g = torch.sigmoid(xg @ g1) @ g2
    print_information('xg', xg)
    print_information('g1', g1)
    print_information('g2', g2)
    print_information('g', g)

    r = xr @ R_
    print_information('x_r', x_r)
    print_information('xr', xr)
    print_information('R_', R_)
    print_information('r', r)

    w = torch.tanh(xw @ w1) @ w2
    w = torch.sigmoid(w0 + w) # !!! here we are using different w !!!
    print_information('xw', xw)
    print_information('w0', w0)
    print_information('w1', w1)
    print_information('w2', w2)
    print_information('w', w)

    k = xk @ K_
    print_information('x_k', x_k)
    print_information('xk', xk)
    print_information('K_', K_)
    print_information('k', k)

    v = xv @ V_
    print_information('x_v', x_v)
    print_information('xv', xv)
    print_information('V_', V_)
    print_information('v', v)

    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    print_information('x_a', x_a)
    print_information('xa', xa)
    print_information('a0', a0)
    print_information('a1', a1)
    print_information('a2', a2)
    print_information('a', a)

    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    print_information('k_k', k_k)
    print_information('kk (normalized)', kk)

    k = k * (1 + (a-1) * k_a) # lerp a with 1, mul k to k
    print_information('k_a', k_a)
    print_information('k (after k_a)', k)

    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    print_information('v0', v0)
    print_information('v1', v1)
    print_information('v2', v2)
    print_information('v_first', v_first)
    print_information('v (after v_first mix)', v)

    print_information('state (before wkv)', state)
    print_information('RWKV7_ONE_OP inputs: r', r)
    print_information('RWKV7_ONE_OP inputs: w', w)
    print_information('RWKV7_ONE_OP inputs: k', k)
    print_information('RWKV7_ONE_OP inputs: v', v)
    print_information('RWKV7_ONE_OP inputs: -kk', -kk)
    print_information('RWKV7_ONE_OP inputs: kk*a', kk*a)
    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kk*a) # !!! using CUDA to modify state in-place !!! (faster too)
    print_information('wkv output (xx)', xx)
    print_information('state (after wkv)', state)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)
    print_information('ln_w', ln_w)
    print_information('ln_b', ln_b)
    print_information('xx (after group_norm)', xx)

    bonus = ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    print_information('r_k', r_k)
    print_information('bonus (r*k*r_k aggregated)', bonus)
    xx = xx + bonus
    print_information('xx (after bonus)', xx)

    print_information('O_', O_)
    out = (xx * g) @ O_
    print_information('TMix output', out)
    return out, v_first


def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w) # !!! here we are using different w !!!
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a) # !!! using CUDA to modify state in-place !!!

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, v_first


def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w) # !!! here we are using different w !!!
    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kk*a) # !!! using CUDA to modify state in-place !!!

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return (xx * g) @ O_, v_first

########################################################################################################


def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    print_information('START CMix_one')
    print_information('x (CMix input)', x)
    xx = x_prev[1] - x
    x_prev[1] = x
    print_information('xx (CMix delta)', xx)
    print_information('x_k', x_k)
    k = x + xx * x_k
    print_information('k (before relu)', k)
    print_information('K_', K_)
    k = torch.relu(k @ K_) ** 2
    print_information('k (after relu^2)', k)
    print_information('V_', V_)
    out = k @ V_
    print_information('CMix output', out)
    print_information('END CMix_one')
    return out


def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    # print("Sparsity:", (k == 0).float().mean().item())
    return k @ V_


def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_
