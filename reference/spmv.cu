#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__


#include <assert.h>
#include <stdio.h>
#define DTYPE __half
#define OUTDIM 4096
#define BLOCKDIMX 128
#define BLOCKELEM (sizeof(DTYPE)*BLOCKDIMX)
#define NUMSTAGES 3
#define PROCESSN 16

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef CUDA_NO_HALF
#include <cuda_fp16.h>

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


// #include <torch/extension.h>
// void spmv_forward(
//     torch::Tensor vec,
//     torch::Tensor vid,
//     torch::Tensor mat,
//     torch::Tensor out,
//     int nnzs
// ) {
//   spvecmatmul_kernel<<<dim3(192, 16, 1), dim3(128, 1, 1)>>>(
//     (const half*)(vec.data_ptr()),
//     (const int*) (vid.data_ptr()),
//     (const half*)(mat.data_ptr()),
//     (half*)      (out.data_ptr()),
//     nnzs
//   );
// }
// __global__ void __launch_bounds__(128, 1) spvecmatmul_kernel(__half* __restrict__ mat, __half* __restrict__ out, int* __restrict__ vec_indices, __half* __restrict__ vec_values, int m, int nnzs) {
//   __half2 out_frag;
//   __shared__ __align__(1024) __half mat_row_smem[3 * 256];
//   __half vec_value;
//   __half2 mat_row_frag;
//   const int bx = blockIdx.x;
//   const int by = blockIdx.y;
//   const int t = threadIdx.x;
//   *(uint*)(&out_frag) = 0u;
//   #pragma unroll
//   for (int stage = 0; stage < 3; stage ++){
//     int id = bx * 16 + stage;
//     if (id < nnzs) {
//         int cond = vec_indices[id];
//         cp_async_gs_conditional<4>(mat_row_smem+(2*t + stage * 256), mat+(cond * 4096 + by * 256 + 2*t), (0 <= cond && cond < m));
//         cp_async_commit();
//     }
//   }
//   for (int i1_nnz = 0; i1_nnz < (min(16, (nnzs - (bx * 16))) - 3); ++i1_nnz) {
//     int id0 = bx * 16 + i1_nnz;
//     vec_value = vec_values[id0];
//     cp_async_wait<2>();
//     __syncthreads();
//     mat_row_frag = *(__half2*)(mat_row_smem + (((i1_nnz % 3) * 256) + 2*t));
//     int id = ((bx * 16) + i1_nnz) + 3;
//     int cond = (id < nnzs)? vec_indices[id]: 0;
//     cp_async_gs_conditional<4>(mat_row_smem+(((i1_nnz % 3) * 256) + 2*t), mat+(cond * 4096 + by * 256 + 2*t), ((0 <= cond) && (cond < m)));
//     cp_async_commit();
//     out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
//   }

//   if (3 <= (nnzs - (bx * 16))) {
//     int id0 = min(((bx * 16) + 16), (nnzs)) - 3;
//     vec_value = vec_values[id0];
//     cp_async_wait<2>();
//     __syncthreads();
//     mat_row_frag = *(__half2*)(mat_row_smem + (((min(16, (nnzs - (bx * 16))) % 3) * 256) + 2*t));
//     out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
//   }
//   if (2 <= (nnzs - (bx * 16))) {
//     int id0 = min(((bx * 16) + 16), (nnzs)) - 2;
//     vec_value = vec_values[id0];
//     cp_async_wait<1>();
//     __syncthreads();
//     mat_row_frag = *(__half2*)(mat_row_smem + ((((min(16, (nnzs - (bx * 16))) + 1) % 3) * 256) + 2*t));
//     out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
//   }
//   if (1 <= (nnzs - (bx * 16))) {
//     int id0 = min(((bx * 16) + 16), (nnzs)) - 1;
//     vec_value = vec_values[id0];
//     cp_async_wait<0>();
//     __syncthreads();
//     mat_row_frag = *(__half2*)(mat_row_smem + ((((min(16, (nnzs - (bx * 16))) + 2) % 3) * 256) + 2*t));
//     out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
//   }
//   atomicAdd((__half2*)(out + by * 256 + 2*t), out_frag);
// }



// #define CHECK_CUDA(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)

// __global__ void __launch_bounds__(128, 1) dense_to_sparse_kernel(
//     const __half* __restrict__ dense_vec // ,
//     // __half* __restrict__ vout,
//     // int* __restrict__ sparse_indices,
//     // int* __restrict__ nnz_count
// ) {
//     constexpr int N = 16384;
//     static_assert(N % (8 * 128) == 0, "N must be divisible by 1024");
    
//     const int t = threadIdx.x;
//     constexpr int E8 = N / (8 * 128); // 16
//     __shared__ int prefix_sum[128];
    
//     int thread_nnz = 0;
//     for (int i = t * E8; i < (t + 1) * E8; ++i) {
//         common128 z;
//         z.I = ((const int4*)dense_vec)[i];
//         #pragma unroll
//         for (int j = 0; j < 8; ++j) {
//             unsigned short bits = __half_as_ushort(z.h[j]);
//             if (bits != 0x0000 && bits != 0x8000) {
//                 thread_nnz++;
//             }
//         }
//     }

//     prefix_sum[t] = thread_nnz;
//     __syncthreads();

//     // Sequential inclusive prefix sum by thread 0
//     if (t == 0) {
//         for (int i = 1; i < 128; ++i) {
//             prefix_sum[i] += prefix_sum[i - 1];
//         }
//         total_nnzs = prefix_sum[127];
//     }
//     __syncthreads();

//     // int c;
//     // #pragma unroll
//     // for(int z=1; z<128; z*=2){
//     //   if(t >= z) {c = prefix_sum[t-z];}
//     //   __syncthreads(); 
//     //   if(t >= z) {prefix_sum[t] += c;}
//     //   __syncthreads();
//     // }
//     // if (t == 0) total_nnzs = prefix_sum[127];

//     // Compute exclusive prefix sum as starting offset
//     int write_offset = (t == 0) ? 0 : prefix_sum[t - 1];

//     for (int i = t * E8; i < (t + 1) * E8; ++i) {
//         common128 z;
//         z.I = ((const int4*)dense_vec)[i];
//         #pragma unroll
//         for (int j = 0; j < 8; ++j) {
//             unsigned short bits = __half_as_ushort(z.h[j]);
//             if (bits != 0x0000 && bits != 0x8000) {
//                 int idx = i * 8 + j;
//                 vec_indices[write_offset] = idx;
//                 // vec[write_offset] = z.h[j];
//                 // if (t==0) {printf("%d\n", int(*(short*)(vec + write_offset)));}
//                 write_offset++;
//             }
//         }
//     }
//     __syncthreads();
// }



// #include <torch/extension.h>
// void spmv_forward(
//     torch::Tensor vecdense,
//     torch::Tensor mat,
//     torch::Tensor out
// ) {
//     auto vec_indices_t = torch::empty({16384}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
//     dense_to_sparse_kernel<<<1, 128>>>((const DTYPE*)(vecdense.data_ptr()), (int*)(vec_indices_t.data_ptr()));
//     spvecmatmul_kernel<<<dim3(192, 16, 1), dim3(128, 1, 1)>>>((const DTYPE*)(vecdense.data_ptr()), (const int*)(vec_indices_t.data_ptr()), (const DTYPE*)(mat.data_ptr()), (DTYPE*)(out.data_ptr()));
// }

#include <torch/extension.h>
void spmv_forward(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor out
) {
    spvecmatmul_noindices<<<dim3(256, 16, 1), dim3(128, 1, 1)>>>((const DTYPE*)(vec.data_ptr()), (const DTYPE*)(mat.data_ptr()), (DTYPE*)(out.data_ptr()));
}

