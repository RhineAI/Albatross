#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <stdio.h>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>

#ifndef _N_
#define _N_ 64
#endif

typedef at::Half F;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;
#define rotator(_A,_B,_C) (two_to_neg_41*float(ro1*(_A)+ro2*(_B)+ro3*(_C)))
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))

// __global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
//                                 F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
//                                 F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
//     const int bbb = blockIdx.x / H;
//     const int h = blockIdx.x % H;
//     const int i = threadIdx.x;
//     const int t0 = _elapsed_t[bbb];
//     _state += bbb*C*_N_ + h*_N_*_N_ + i*_N_;
//     __half state[_N_];
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         state[j] = static_cast<__half>(_state[j]);
//     __shared__ __half r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];
//     for (int _t = 0; _t < T; _t++){
//         const int t = bbb*T*C + h*_N_ + i + _t * C;
//         __syncthreads();
//         r[i] = static_cast<__half>(_r[t]);
//         w[i] = __half(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(t0+_t));
//         k[i] = static_cast<__half>(_k[t]);
//         a[i] = static_cast<__half>(_a[t]);
//         b[i] = static_cast<__half>(_b[t]);
//         __syncthreads();
//         __half sa = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             sa += a[j] * state[j];
//         }
//         __half vv = static_cast<__half>(_v[t]);
//         __half y = 0;
//         #pragma unroll
//         for (int j = 0; j < _N_; j++){
//             __half& s = state[j];
//             s += s * w[j] + k[j] * vv + sa * b[j];
//             y += s * r[j];
//         }
//         _y[t] = F(y);
//     }
//     #pragma unroll
//     for (int j = 0; j < _N_; j++)
//         _state[j] = static_cast<F>(state[j]);
// }


__global__ void kernel_forward_w0_fp16_dither_seq(const int B, const int T, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        __half2 sa2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 += a[j] * state[j];
        __half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        __half vv = _v[t];
        __half2 vv2 = {vv, vv};
        __half2 y2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
        {
            __half2 &s = state[j];
            s += s * w[j] + k[j] * vv2 + sa2 * b[j];
            y2 += s * r[j];
        }
        _y[t] = y2.x + y2.y;
    }
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}

__global__ void kernel_forward_w0_fp16_dither_one(const int B, const int C, const int H,
                                                  F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                                                  F *__restrict__ const _y, const int *__restrict__ const _elapsed_t)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            // assert (row < 64);
            // assert (col < 32);
            state_smem[row][(row % 32) ^ col] = ((__half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    __half2 state[_N_ / 2];
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];

    // for (int z = 0; z < _N_; z++)
    //     assert ((reinterpret_cast<F*>(&state))[z] - (_state + i* _N_)[z] == 0);
    
    __shared__ __half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];

    const int t = bbb * C + h * _N_ + i;
    // float w0 = __expf(_w[t]);
    // w0 = w0 / (w0 + 1);
    // sigmoid = 1 / (1+exp2f(-nlog2e * x))
    ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * _w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]));
    ((F *)k)[i] = _k[t];
    ((F *)a)[i] = _a[t];
    ((F *)b)[i] = _b[t];
    ((F *)r)[i] = _r[t];
    __syncthreads();
    __half2 sa2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        sa2 += a[j] * state[j];
    __half sa = sa2.x + sa2.y;
    sa2 = {sa, sa};

    __half vv = _v[t];
    __half2 vv2 = {vv, vv};
    __half2 y2 = {0., 0.};
#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
    {
        __half2 &s = state[j];
        s += s * w[j] + k[j] * vv2 + sa2 * b[j];
        y2 += s * r[j];
    }
    _y[t] = y2.x + y2.y;

#pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
#pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++)
    {
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++)
        {
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((__half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}


union common128 {
    int4 I;
    struct {int x,y,z,w;} J;
    struct {float x,y,z,w;} F;
    struct {double x,y;} D;
    struct {half2 x,y,z,w;} G;
    struct {half a,b,c,d,e,f,g,h;} H;
    half h[8];
    int i[4];
    float f[4];
};

template <int N>
__device__ __forceinline__ void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = (unsigned int)(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"(global_ptr), "n"(N), "r"(bytes));
  } else {
    asm volatile(
#if ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"(global_ptr), "n"(N), "r"(bytes));
  }
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ static int total_nnzs = 0;
// __device__ static __half vec[16384];
// __device__ static int vec_indices[16384];

__global__ void __launch_bounds__(128, 1) spvecmatmul_kernel(
  const __half* __restrict__ vec,
  const int* __restrict__ vec_indices,
  const __half* __restrict__ mat,
  __half* __restrict__ out
  // ,int* nnz_ptr
  // ,int total_nnzs
){
  __shared__ __align__(1024) __half mat_row_smem[2][256];
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int t = threadIdx.x;
  // const int total_nnzs = *nnz_ptr;
  const int nnzs = total_nnzs;
  const int max_nnz_per_block = (nnzs + gridDim.x - 1) / gridDim.x;
  const int start_pos = bx * max_nnz_per_block;
  const int stop_pos = min(nnzs, (bx+1) * max_nnz_per_block);
  const int process_elem = stop_pos - start_pos;
  __half2 out_frag;
  *(int*)(&out_frag) = 0;
  // init
  #pragma unroll
  for(int i = 0; i < 2; i++){
    if (i < process_elem){
      int actual_pos = vec_indices[start_pos + i];
      cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * 4096 + by * 256 + t*2, true);
      cp_async_commit();
    }
  }
  // main for
  for(int i = 0; i < process_elem-2; i++){
    // take data
    cp_async_wait<1>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
    __half vec_value = vec[vec_indices[start_pos + i]];

    // store
    int actual_pos = vec_indices[start_pos + i+2];
    cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * 4096 + by * 256 + t*2, true);
    cp_async_commit();

    // compute
    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }

  // end
  if (process_elem >= 2){
    cp_async_wait<1>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[process_elem%2] + t*2);
    __half vec_value = vec[vec_indices[start_pos + process_elem - 2]];

    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }
  if (process_elem >= 1){
    cp_async_wait<0>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[(process_elem+1)%2] + t*2);
    __half vec_value = vec[vec_indices[start_pos + process_elem - 1]];

    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }
  atomicAdd((__half2*)(out + by*256 + t*2), out_frag);
}


__global__ void __launch_bounds__(128, 1) dense_to_sparse_kernel(
    const __half* __restrict__ dense_vec,
    // __half* __restrict__ vout,
    int* __restrict__ sparse_indices
    // int* __restrict__ nnz_count
) {
    constexpr int N = 16384;
    static_assert(N % (8 * 128) == 0, "N must be divisible by 1024");
    
    const int t = threadIdx.x;
    constexpr int E8 = N / (8 * 128); // 16
    __shared__ int prefix_sum[128];
    
    int thread_nnz = 0;
    for (int i = t * E8; i < (t + 1) * E8; ++i) {
        common128 z;
        z.I = ((const int4*)dense_vec)[i];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short bits = __half_as_ushort(z.h[j]);
            if (bits != 0x0000 && bits != 0x8000) {
                thread_nnz++;
            }
        }
    }

    prefix_sum[t] = thread_nnz;
    __syncthreads();

    // Sequential inclusive prefix sum by thread 0
    // if (t == 0) {
    //     for (int i = 1; i < 128; ++i) {
    //         prefix_sum[i] += prefix_sum[i - 1];
    //     }
    //     *nnz_count = prefix_sum[127];
    // }
    // __syncthreads();

    int c;
    #pragma unroll
    for(int z=1; z<128; z*=2){
      if(t >= z) {c = prefix_sum[t-z];}
      __syncthreads(); 
      if(t >= z) {prefix_sum[t] += c;}
      __syncthreads();
    }
    if (t == 0) total_nnzs = prefix_sum[127];

    // Compute exclusive prefix sum as starting offset
    int write_offset = (t == 0) ? 0 : prefix_sum[t - 1];

    for (int i = t * E8; i < (t + 1) * E8; ++i) {
        common128 z;
        z.I = ((const int4*)dense_vec)[i];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short bits = __half_as_ushort(z.h[j]);
            if (bits != 0x0000 && bits != 0x8000) {
                int idx = i * 8 + j;
                sparse_indices[write_offset] = idx;
                // vout[write_offset] = z.h[j];
                write_offset++;
            }
        }
    }
    __syncthreads();
}


__global__ void __launch_bounds__(128, 1) spvecmatmul_noindices(
  const __half* __restrict__ vec,
  const __half* __restrict__ mat,
  __half* __restrict__ out
){
  constexpr int N = 16384;
  constexpr int GRIDDIMX = 256;
  __shared__ __align__(512) __half mat_row_smem[2][256];
  __shared__ __align__(256) __half vec_slice[(N / GRIDDIMX)];
  __shared__ __align__(256) int nnz_ids[(N / GRIDDIMX)];
  __shared__ int nnz_count;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int t = threadIdx.x;
  
  constexpr int maxn_per_block = (N / GRIDDIMX);
  const int start_pos = bx * maxn_per_block;

  if (t < 32){
    *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
  }
  __syncthreads();
  if (t == 0){
    int cnt = 0;
    #pragma unroll
    for (int i=0; i<8; ++i) {
      common128 z;
      z.I = ((const int4*)vec_slice)[i];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        unsigned short bits = __half_as_ushort(z.h[j]);
        if (bits != 0x0000 && bits != 0x8000) {
          int idx = i * 8 + j;
          nnz_ids[cnt] = idx;
          cnt++;
        }
      }
    }
    nnz_count = cnt;
  }
  __syncthreads();

  __half2 out_frag;
  *(int*)(&out_frag) = 0;
  // init
  #pragma unroll
  for(int i = 0; i < 2; i++){
    if (i < nnz_count){
      int actual_pos = start_pos + nnz_ids[i];
      cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * 4096 + by * 256 + t*2, true);
      cp_async_commit();
    }
  }
  // main for
  for(int i = 0; i < nnz_count-2; i++){
    // take data
    cp_async_wait<1>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
    __half vec_value = vec_slice[nnz_ids[i]];

    // store
    int actual_pos = start_pos + nnz_ids[i+2];
    cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * 4096 + by * 256 + t*2, true);
    cp_async_commit();

    // compute
    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }

  // end
  if (nnz_count >= 2){
    cp_async_wait<1>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
    __half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }
  if (nnz_count >= 1){
    cp_async_wait<0>();
    __syncthreads();

    half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
    __half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

    out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
  }
  atomicAdd((__half2*)(out + by*256 + t*2), out_frag);
}


void cuda_forward_seq(int B, int T, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    // auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_seq<<<B * H, _N_>>>(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_forward_one(int B, int C, int H, F *state, F *r, F *w, F *k, F *v, F *a, F *b, F *y, int *elapsed_t)
{
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither_one<<<B * H, _N_, 0, stream>>>(B, C, H, state, r, w, k, v, a, b, y, elapsed_t);
}

void cuda_spmv_forward(F* vec1, F* mat, F* out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    spvecmatmul_noindices<<<dim3(256, 16, 1), dim3(128, 1, 1), 0, stream>>>((const half*)vec1, (const half*)mat, (half*)out);
}
