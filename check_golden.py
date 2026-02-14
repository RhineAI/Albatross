import numpy as np
import os

DIR = r'C:\Projects\verilog-test\sim\layer0_golden'

def load_fp16(name):
    path = os.path.join(DIR, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint16).view(np.float16).astype(np.float64)

def load_fp32(name):
    path = os.path.join(DIR, f'{name}.hex')
    with open(path) as f:
        vals = [int(line.strip(), 16) for line in f if line.strip()]
    return np.array(vals, dtype=np.uint32).view(np.float32).astype(np.float64)

def stats(name, arr):
    nz = arr[arr != 0]
    print(f'{name:20s}: min={arr.min():+.6f}  max={arr.max():+.6f}  mean={arr.mean():+.6f}  nonzero={len(nz)}/{len(arr)}')

print('=== Input/Output vectors ===')
for n in ['x_in', 'ln1_out', 'tmix_out', 'x_after_att', 'ln2_out', 'cmix_out', 'x_out']:
    stats(n, load_fp16(n))

print()
print('=== State outputs ===')
stats('tmix_x_prev_out', load_fp16('tmix_x_prev_out'))
stats('cmix_x_prev_out', load_fp16('cmix_x_prev_out'))
stats('v_first_out', load_fp16('v_first_out'))
stats('att_state_out', load_fp32('att_state_out'))

print()
print('=== Residual check: x_out should = x_after_att + cmix_out ===')
x_aa = load_fp16('x_after_att')
cm = load_fp16('cmix_out')
xo = load_fp16('x_out')
expected = (x_aa.astype(np.float16) + cm.astype(np.float16)).astype(np.float64)
actual = xo
diff = np.abs(actual - expected)
print(f'  max_abs_diff = {diff.max():.8f}')
print(f'  mean_abs_diff = {diff.mean():.8f}')

print()
print('=== Value ranges for key intermediates ===')
for n in ['ln1_out', 'tmix_out', 'cmix_out']:
    v = load_fp16(n)
    absv = np.abs(v)
    print(f'{n:20s}: |min|={absv.min():.6f}  |max|={absv.max():.6f}  |mean|={absv.mean():.6f}')

print()
print('=== FP16 vs FP32 att_state range ===')
st = load_fp32('att_state_out')
print(f'  att_state: min={st.min():.8f}  max={st.max():.8f}  mean={st.mean():.8f}')
print(f'  abs range: [{np.abs(st).min():.8e}, {np.abs(st).max():.8e}]')
