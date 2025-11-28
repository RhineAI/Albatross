#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <iostream>
#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#ifndef _N_
#define _N_ 64
#endif
#define BLOCKDIM 128
#define MAXNPERBLOCK 128

typedef half F;

constexpr float two_to_neg_41 = 4.547473508864641e-13f;
constexpr float nexp_half_log2_e = -0.8750387749145276f, nlog2_e = -1.4426950408889634f;
constexpr int ro1 = (int)2654435769;
#define rotator1(_A) (two_to_neg_41*float(ro1*(_A)))


template <bool Tis1=false>
__global__ void kernel_forward_w0_fp16_dither(
    const int B, const int T, const int C, const int H,
    F *__restrict__ _state, const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
    F *__restrict__ const _y, const int *__restrict__ const _elapsed_t){
    
    if constexpr (Tis1) {
        __builtin_assume(T==1);
    }
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ __align__(256) half2 state_smem[_N_][_N_ / 2];

    _state += bbb * C * _N_ + h * _N_ * _N_;
    constexpr int ldg_size = sizeof(int4) / sizeof(F);
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec = ((int4 *)_state)[j0 * _N_ + i];
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            state_smem[row][(row % 32) ^ col] = ((half2 *)&state_vec)[j1];
        }
    }
    __syncthreads();
    half2 state[_N_ / 2];
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state[j] = state_smem[i][(i % 32) ^ j];
    
    __shared__ __align__(128) half2 r[_N_ / 2], k[_N_ / 2], w[_N_ / 2], a[_N_ / 2], b[_N_ / 2];
    #pragma unroll
    for (int _t = 0; _t < T; _t++){
        const int t = bbb*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        ((F *)w)[i] = F(exp2f(nexp_half_log2_e / (1.0f + exp2f(nlog2_e * (float)_w[t]))) - 1.0f + rotator1(_elapsed_t[bbb]+_t));
        ((F *)k)[i] = _k[t];
        ((F *)a)[i] = _a[t];
        ((F *)b)[i] = _b[t];
        ((F *)r)[i] = _r[t];
        __syncthreads();
        half2 sa2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++)
            sa2 = __hfma2(a[j], state[j], sa2);
            // sa2 += a[j] * state[j];
        half sa = sa2.x + sa2.y;
        sa2 = {sa, sa};

        half vv = _v[t];
        half2 vv2 = {vv, vv};
        half2 y2 = {0., 0.};
        #pragma unroll
        for (int j = 0; j < _N_ / 2; j++){
            half2 &s = state[j];
            s = __hfma2(s, w[j], __hfma2(k[j], vv2, __hfma2(sa2, b[j], s)));
            // s += s * w[j] + k[j] * vv2 + sa2 * b[j];
            y2 = __hfma2(s, r[j], y2);
            // y2 += s * r[j];
        }
        _y[t] = y2.x + y2.y;
    }
    #pragma unroll
    for (int j = 0; j < _N_ / 2; j++)
        state_smem[i][(i % 32) ^ j] = state[j];
    __syncthreads();
    #pragma unroll
    for (int j0 = 0; j0 < _N_ / ldg_size; j0++){
        int4 state_vec;
        for (int j1 = 0; j1 < ldg_size / 2; j1++){
            int row = j0 * ldg_size + i * ldg_size / _N_;
            int col = i * ldg_size % _N_ / 2 + j1;
            ((half2 *)&state_vec)[j1] = state_smem[row][(row % 32) ^ col];
        }
        ((int4 *)_state)[j0 * _N_ + i] = state_vec;
    }
}


__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

union common128 {
    int4 I;
    struct {int x,y,z,w;} J;
    struct {float x,y,z,w;} F;
    struct {double x,y;} D;
    struct {half2 x,y,z,w;} G;
    struct {half a,b,c,d,e,f,g,h;} H;
    half h[8];
    unsigned short s[8];
    int i[4];
    float f[4];
};

template <int N>
__device__ __forceinline__ void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr, bool cond) {
    static_assert(N == 16 || N == 8 || N == 4);
    int bytes = cond ? N : 0;
    unsigned int addr = __cvta_generic_to_shared(smem_addr);
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

__global__ void __launch_bounds__(BLOCKDIM, 1) spvecmatmul_noindices(
    const int C,
    const half* __restrict__ vec,
    const half* __restrict__ mat,
    half* __restrict__ out
){
    __builtin_assume(blockDim.x == BLOCKDIM);
    __shared__ __align__(256) half mat_row_smem[2][2*BLOCKDIM];
    __shared__ __align__(256) half vec_slice[MAXNPERBLOCK];
    __shared__ __align__(256) int nnz_ids[MAXNPERBLOCK];
    __shared__ int nnz_count;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int t = threadIdx.x;
    const int warp_id = t >> 5;
    const int lane    = t & 31;
    const int start_pos = bx * MAXNPERBLOCK;

    bool vne0;
    int local_pos;
    constexpr int active_warps = MAXNPERBLOCK/32;
    __shared__ int warp_counts[active_warps], warp_prefix[active_warps];

    if (t < MAXNPERBLOCK/2){
        *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
    }
    __syncthreads();

    if (t < MAXNPERBLOCK){
        vne0 = bool(__half_as_ushort(vec_slice[t]) << 1);
        unsigned mask = __ballot_sync(0xffffffffu, vne0);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0)
            warp_counts[warp_id] = __popc(mask);
    }
    __syncthreads();

    if (t == 0) {
        int s = 0;
        #pragma unroll
        for (int w = 0; w < active_warps; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();

    if (t < MAXNPERBLOCK && vne0) {
        nnz_ids[warp_prefix[warp_id] + local_pos] = t;
    }
    __syncthreads();

    half2 out_frag;
    *(int*)(&out_frag) = 0;
    // init
    #pragma unroll
    for(int i = 0; i < 2; i++){
        if (i < nnz_count){
            int actual_pos = start_pos + nnz_ids[i];
            cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
            cp_async_commit();
        }
    }
    // main for
    for(int i = 0; i < nnz_count-2; i++){
        // take data
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
        half vec_value = vec_slice[nnz_ids[i]];

        // store
        int actual_pos = start_pos + nnz_ids[i+2];
        cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
        cp_async_commit();

        // compute
        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }

    // end
    if (nnz_count >= 2){
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    if (nnz_count >= 1){
        cp_async_wait<0>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    atomicAdd((half2*)(out + by*(2*BLOCKDIM) + t*2), out_frag);
}

#define XBLOCK 2
#define BLOCKDIMX_CMIX 512
#define NUMWARPS (BLOCKDIMX_CMIX/32)

__global__ void __launch_bounds__(BLOCKDIMX_CMIX, 1) cmix_up_kernel_2(
    const __half* __restrict__ x_0,
    const __half* __restrict__ x_1,
    const __half* __restrict__ x_k,
    const __half* __restrict__ key,
          __half* __restrict__ out,
    int indim,
    int outdim
) {
    // block processes XBLOCK outputs
    int block_x = blockIdx.x;                       // block index
    int x_base = block_x * XBLOCK;                  // first x processed by this CTA
    if (x_base >= outdim) return;

    int t = threadIdx.x;
    int lane = t & 31;
    int warp_id = t >> 5; 
    int chunk = indim / NUMWARPS;
    int r_start = warp_id * chunk;
    int r_end   = r_start + chunk;

    const __half2* h0 = reinterpret_cast<const __half2*>(x_0);
    const __half2* h1 = reinterpret_cast<const __half2*>(x_1);
    const __half2* h2 = reinterpret_cast<const __half2*>(x_k);
    const __half2* h3_x0 = reinterpret_cast<const __half2*>(key + x_base * indim);
    const __half2* h3_x1 = nullptr;
    if (XBLOCK > 1) {
        if (x_base + 1 < outdim)
            h3_x1 = reinterpret_cast<const __half2*>(key + (x_base + 1) * indim);
        else
            h3_x1 = reinterpret_cast<const __half2*>(key + x_base * indim); // won't be used
    }

    // convert r_start/end to half2 indices
    int i_start = r_start / 2;
    int i_end   = r_end / 2;

    // accumulators for each x processed by this warp (float)
    float acc_x0 = 0.0f;
    float acc_x1 = 0.0f; // used only if XBLOCK>1

    // Loop: each warp's lanes process interleaved half2s with stride = 32
    // Prefetch friendly: contiguous access across lanes -> coalesced
    for (int i = i_start + lane; i < i_end; i += 32) {
        // load half2s (coalesced across lanes)
        __half2 a0 = h0[i];
        __half2 a1 = h1[i];
        __half2 a2 = h2[i];
        __half2 b0 = h3_x0[i];

        // compute interp = a0 + (a1 - a0) * a2  using half2 intrinsics
        __half2 diff = __hsub2(a1, a0);
        __half2 interp = __hfma2(diff, a2, a0);
        __half2 prod0 = __hmul2(interp, b0);

        float2 f0 = __half22float2(prod0);
        acc_x0 += f0.x + f0.y;

        if (XBLOCK > 1) {
            __half2 b1 = h3_x1[i];
            __half2 prod1 = __hmul2(interp, b1);
            float2 f1 = __half22float2(prod1);
            acc_x1 += f1.x + f1.y;
        }
    }

    // Warp-level reduce within each warp
    float warp_sum_x0 = warp_reduce_sum(acc_x0);
    float warp_sum_x1 = XBLOCK > 1 ? warp_reduce_sum(acc_x1) : 0.0f;

    // Use shared memory to combine across NUMWARPS warps
    __shared__ float smem[NUMWARPS * XBLOCK]; // small: 16*2 = 32 floats
    int sidx = warp_id * XBLOCK + (lane == 0 ? 0 : 0); // base index for this warp
    if (lane == 0) {
        // store warp's partial sums into shared memory (one write per warp per x)
        smem[warp_id * XBLOCK + 0] = warp_sum_x0;
        if (XBLOCK > 1) smem[warp_id * XBLOCK + 1] = warp_sum_x1;
    }
    __syncthreads();

    if (t == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (int w = 0; w < NUMWARPS; ++w) {
            total0 += smem[w * XBLOCK + 0];
            if (XBLOCK > 1) total1 += smem[w * XBLOCK + 1];
        }
        float relu0 = max(total0, 0.f);
        out[x_base + 0] = __float2half_rn(relu0 * relu0);

        if (XBLOCK > 1 && x_base + 1 < outdim) {
            float relu1 = max(total1, 0.f);
            out[x_base + 1] = __float2half_rn(relu1 * relu1);
        }
    }
}

// __global__ void __launch_bounds__(BLOCKDIMX_CMIX, 1) cmix_up_kernel(
//     const __half* __restrict__ x_0,
//     const __half* __restrict__ x_1,
//     const __half* __restrict__ x_k,
//     const __half* __restrict__ key,
//           __half* __restrict__ out,
//     int indim,
//     int outdim
// ){
//     const int bx = blockIdx.x;
//     const int t = threadIdx.x;
//     __shared__ __align__(1024) half2 g3_shared[2][BLOCKDIMX_CMIX];
    
//     float acc = 0.f;
//     int lane = threadIdx.x % 32;
//     int warp = threadIdx.x / 32;

//     const auto h0 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_0) + t);
//     const auto h1 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_1) + t);
//     const auto h2 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_k) + t);
//     const auto h3 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)(key + bx * indim)) + t);

//     /* constexpr */ int N = (indim / BLOCKDIMX_CMIX) / 2;
    
//     #pragma unroll
//     for(int j=0; j<2; j++){
//         if (j<N) {
//             cp_async_gs_conditional<4>(g3_shared[j%2] + t, h3[j], true);
//             cp_async_commit();
//         }
//     }
//     #pragma unroll
//     for (int j=0; j < N; j++) {
//         __half2 a0 = *(const __half2*)(h0[j]);
//         __half2 a1 = *(const __half2*)(h1[j]);
//         __half2 a2 = *(const __half2*)(h2[j]);
//         if (j < N-1)
//             cp_async_wait<1>();
//         else
//             cp_async_wait<0>();
//         __syncthreads();
//         __half2 a3 = g3_shared[j%2][t];
//         if (j < (N-2)) {
//             cp_async_gs_conditional<4>(g3_shared[j%2] + t, h3[j+2], true);
//             cp_async_commit();
//         }
//         // a0 + (a1-a0)*a2
//         __half2 diff = __hsub2(a1, a0);
//         __half2 interp = __hfma2(diff, a2, a0);
//         __half2 prod = __hmul2(interp, a3);
//         float2 f = __half22float2(prod);
//         acc += f.x + f.y;
//     }

//     float warp_sum = warp_reduce_sum(acc);

//     __shared__ float s[BLOCKDIMX_CMIX/32];
//     if (lane == 0) s[warp] = warp_sum;
//     __syncthreads();

//     float total = 0.f;
//     if (warp == 0 && lane < (BLOCKDIMX_CMIX/32))
//         total = s[lane];

//     if (warp == 0)
//         total = warp_reduce_sum(total);

//     if (threadIdx.x == 0) {
//         float relu = max(total, 0.f);
//         out[bx] = __float2half_rn(relu * relu);
//     }
// }


__global__ void __launch_bounds__(BLOCKDIMX_CMIX, 1) cmix_up_kernel(
    const __half* __restrict__ x_0,
    const __half* __restrict__ x_1,
    const __half* __restrict__ x_k,
    const __half* __restrict__ key,
          __half* __restrict__ out,
    int indim,
    int outdim
){
    const int bx = blockIdx.x;
    const int t = threadIdx.x;
    float acc = 0.f;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    const auto h0 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_0) + t);
    const auto h1 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_1) + t);
    const auto h2 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)x_k) + t);
    const auto h3 = (const half2(*)[BLOCKDIMX_CMIX])(((const half2*)(key + bx * indim)) + t);

    int N = indim / (BLOCKDIMX_CMIX * 2);

    #pragma unroll
    for (int j=0; j < N; j++) {
        __half2 a0 = *(const __half2*)(h0[j]);
        __half2 a1 = *(const __half2*)(h1[j]);
        __half2 a2 = *(const __half2*)(h2[j]);
        __half2 a3 = *(const __half2*)(h3[j]);
        __half2 diff = __hsub2(a1, a0);
        __half2 interp = __hfma2(diff, a2, a0);
        __half2 prod = __hmul2(interp, a3);
        float2 f = __half22float2(prod);
        acc += f.x + f.y;
    }

    float warp_sum = warp_reduce_sum(acc);

    __shared__ float s[BLOCKDIMX_CMIX/32];
    if (lane == 0) s[warp] = warp_sum;
    __syncthreads();

    float total = 0.f;
    if (warp == 0 && lane < (BLOCKDIMX_CMIX/32))
        total = s[lane];

    if (warp == 0)
        total = warp_reduce_sum(total);

    if (threadIdx.x == 0) {
        float relu = max(total, 0.f);
        out[bx] = __float2half_rn(relu * relu);
    }
}


#define COPY_ZERO_X 128
template <typename T> 
__global__ __launch_bounds__(COPY_ZERO_X, 1) void copy_zero_kernel(const T* __restrict__ a, T* __restrict__ b, T* __restrict__ c, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int4* src  = (const int4*)a;
        int4*       dst  = (int4*)b;
        int4*       dst2 = (int4*)c;
        int4 v = src[i];
        dst[i] = v;
        dst2[i] = make_int4(0, 0, 0, 0);
    }
}

void copy_zero(at::Tensor &a, at::Tensor &b, at::Tensor &c){
    size_t n_uint4 = a.numel() * sizeof(F) / sizeof(int4);
    auto stream = at::cuda::getCurrentCUDAStream();
    copy_zero_kernel<<<(n_uint4+(COPY_ZERO_X-1))/COPY_ZERO_X, COPY_ZERO_X, 0, stream>>>(
        (const F*)a.data_ptr(), (F*)b.data_ptr(), (F*)c.data_ptr(), n_uint4);
}

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t){
    assert(H * _N_ == C);
    kernel_forward_w0_fp16_dither<<<B * H, _N_>>>(
        B, T, C, H, 
        (F*)state.data_ptr(), 
        (const F*)r.data_ptr(), 
        (const F*)w.data_ptr(),
        (const F*)k.data_ptr(), 
        (const F*)v.data_ptr(),
        (const F*)a.data_ptr(),
        (const F*)b.data_ptr(),
        (F*)y.data_ptr(), 
        elapsed_t.data_ptr<int>()
    );
}

void forward_one(int64_t B, int64_t C, int64_t H, at::Tensor &state, at::Tensor &r, at::Tensor &w, at::Tensor &k, at::Tensor &v, at::Tensor &a, at::Tensor &b, at::Tensor &y, at::Tensor &elapsed_t){
    assert(H * _N_ == C);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_w0_fp16_dither<1><<<B * H, _N_, 0, stream>>>(
        B, 1, C, H, 
        (F*)state.data_ptr(), 
        (const F*)r.data_ptr(), 
        (const F*)w.data_ptr(),
        (const F*)k.data_ptr(), 
        (const F*)v.data_ptr(),
        (const F*)a.data_ptr(),
        (const F*)b.data_ptr(),
        (F*)y.data_ptr(), 
        elapsed_t.data_ptr<int>()
    );
}

void spmv_forward(int64_t D, int64_t C, at::Tensor &vec1, at::Tensor &mat, at::Tensor &out) {
    assert(C % (2*BLOCKDIM) == 0);
    assert(D % MAXNPERBLOCK == 0);
    auto stream = at::cuda::getCurrentCUDAStream();
    // cudaMemsetAsync(out, 0, C*sizeof(half), stream);
    spvecmatmul_noindices<<<dim3(D/MAXNPERBLOCK, C/(2*BLOCKDIM), 1), dim3(BLOCKDIM, 1, 1), 0, stream>>>(
        C, (const F*)vec1.data_ptr(), (const F*)mat.data_ptr(), (F*)out.data_ptr()
    );
}


void cmix_up(
    int64_t indim,
    int64_t outdim,
    at::Tensor &x_0,
    at::Tensor &x_1,
    at::Tensor &x_k,
    at::Tensor &key,
    at::Tensor &out)
{
    // int indim = out.size(1);
    // int outdim = out.size(0);
    // std::cout << indim << ' ' << outdim << std::endl;
    auto stream = at::cuda::getCurrentCUDAStream();
    // (outdim + XBLOCK - 1) / XBLOCK
    cmix_up_kernel<<<outdim, BLOCKDIMX_CMIX, 0, stream>>>(
    // cmix_up_kernel_2<<<((outdim + XBLOCK - 1) / XBLOCK), BLOCKDIMX_CMIX, 0, stream>>>(
        (F*)(x_0.data_ptr()),
        (F*)(x_1.data_ptr()),
        (F*)(x_k.data_ptr()),
        (F*)(key.data_ptr()),
        (F*)(out.data_ptr()),
        indim,
        outdim
    );
}

