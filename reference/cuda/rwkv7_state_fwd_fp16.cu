#include <stdio.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

typedef at::Half dtype;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f;
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;
#define rotator(_A,_B,_C) (two_to_neg_41*float(ro1*(_A)+ro2*(_B)+ro3*(_C)))

template <typename F>
__global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
                                F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    const int t0 = _elapsed_t[bbb];
    _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
    F state[_N_];
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];
    __shared__ F r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
    for (int _t = 0; _t < T; _t++){
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        r[i] = _r[t];
        float w0 = __expf(_w[t]);
        w0 = w0 / (w0 + 1);
        w[i] = F(exp2f(nexp_half_log2_e * w0) - 1 + rotator(t0+_t,i,(int)blockIdx.x)); 
        k[i] = _k[t];
        a[i] = _a[t];
        b[i] = _b[t];
        __syncthreads();
        F sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++){
            sa += a[j] * state[j];
        }
        F vv = F(_v[t]);
        F y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++){
            F& s = state[j];
            s += s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];
}

template <typename F>
__global__ void kernel_forward_w0_fp16_dither_one(const int B, const int C, const int H,
                                F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
    F state[_N_];
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];
    __shared__ F r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
    const int t = bbb*C + h*_N_ + i;
    __syncthreads();
    r[i] = _r[t];
    float w0 = __expf(_w[t]);
    w0 = w0 / (w0 + 1);
    w[i] = F(exp2f(nexp_half_log2_e * w0) - 1 + rotator(_elapsed_t[bbb],i,(int)blockIdx.x)); 
    k[i] = _k[t];
    a[i] = _a[t];
    b[i] = _b[t];
    __syncthreads();
    F sa = 0;
    #pragma unroll
    for (int j = 0; j < _N_; j++){
        sa += a[j] * state[j];
    }
    F vv = F(_v[t]);
    F y = 0;
    #pragma unroll
    for (int j = 0; j < _N_; j++){
        F& s = state[j];
        s += s * w[j] + k[j] * vv + sa * b[j];
        y += s * r[j];
    }
    _y[t] = F(y);
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];    
}

void cuda_forward_seq(int B, int T, int C, int H, dtype *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* elapsed_t)
{
    assert(H*_N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_seq<<<dim3(B * H), dim3(_N_), 0, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_forward_one(int B, int C, int H, dtype *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y, int* elapsed_t)
{
    assert(H*_N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_one<<<dim3(B * H), dim3(_N_), 0, stream>>>(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}



// #include <stdio.h>
// #include <assert.h>
// #include "ATen/ATen.h"
// #include <ATen/cuda/CUDAContext.h>

// typedef at::Half dtype;

// template <typename F>
// __global__ void kernel_forward(const int B, const int T, const int C, const int H,
//                                float *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                F *__restrict__ const _y)
// {
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;

//     float state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = _state[j];

//     __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

//     for (int _t = 0; _t < T; _t++)
//     {
//         const int t = bbb*T*C + h*_N_ + i + _t * C;
//         __syncthreads();
//         r[i] = float(_r[t]);
//         w[i] = __expf(-__expf(float(_w[t])));
//         k[i] = float(_k[t]);
//         a[i] = float(_a[t]);
//         b[i] = float(_b[t]);
//         __syncthreads();

//         float sa = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++)
//         {
//             sa += a[j] * state[j];
//         }

//         float vv = float(_v[t]);
//         float y = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++)
//         {
//             float& s = state[j];
//             s = s * w[j] + k[j] * vv + sa * b[j];
//             y += s * r[j];
//         }
//         _y[t] = F(y);
//     }
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = state[j];    
// }

// void cuda_forward(int B, int T, int C, int H, float *state, dtype *r, dtype* w, dtype *k, dtype *v, dtype *a, dtype *b, dtype *y)
// {
//     assert(H*_N_ == C);
//     auto stream = at::cuda::getCurrentCUDAStream();
//     kernel_forward<<<dim3(B * H), dim3(_N_), 0, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y);
// }
