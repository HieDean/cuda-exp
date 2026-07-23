#include <cstdio>
#include "softmax_vx.h"

#if 0
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

template <int BLOCKSIZE>
__global__ void softmax_kernel_v0_max(const float *A, float *maxA, int n)
{
    int gblx = blockIdx.x * blockDim.x + threadIdx.x;
    int locx = threadIdx.x;
    __shared__ float smem[BLOCKSIZE];

    // load
    smem[locx] = gblx < n ? A[gblx] : -INFINITY;
    __syncthreads();

    // reduce max
    for (int stride = BLOCKSIZE >> 1; stride > 32; stride >>= 1)
    {
        if (locx < stride)
        {
            smem[locx] = fmaxf(smem[locx], smem[locx + stride]);
        }
        __syncthreads();
    }
    if (locx < 32)
    {
        smem[locx] = fmaxf(smem[locx], smem[locx + 32]);
        __syncwarp();
        smem[locx] = fmaxf(smem[locx], smem[locx + 16]);
        __syncwarp();
        smem[locx] = fmaxf(smem[locx], smem[locx + 8]);
        __syncwarp();
        smem[locx] = fmaxf(smem[locx], smem[locx + 4]);
        __syncwarp();
        smem[locx] = fmaxf(smem[locx], smem[locx + 2]);
        __syncwarp();
        smem[locx] = fmaxf(smem[locx], smem[locx + 1]);
        __syncwarp();
    }

    // max over block
    if (locx == 0)
    {
        atomicMaxFloat(maxA, smem[0]);
    }

    return;
}

template <int BLOCKSIZE>
__global__ void softmax_kernel_v0_exp_and_reduce(const float *A, float *B, float *maxA, float *sumA, int n)
{
    int gblx = blockIdx.x * blockDim.x + threadIdx.x;
    int locx = threadIdx.x;

    __shared__ float smem[BLOCKSIZE];

    // exp
    float expVal = gblx < n ? exp(A[gblx] - maxA[0]) : 0.0f;
    smem[locx] = expVal;
    if (gblx < n) {
        B[gblx] = expVal;
    }
    __syncthreads();

    // reduce
    for (int stride = BLOCKSIZE >> 1; stride > 32; stride >>= 1)
    {
        if (locx < stride)
        {
            smem[locx] += smem[locx + stride];
        }
        __syncthreads();
    }
    if (locx < 32)
    {
        smem[locx] += smem[locx + 32];
        __syncwarp();
        smem[locx] += smem[locx + 16];
        __syncwarp();
        smem[locx] += smem[locx + 8];
        __syncwarp();
        smem[locx] += smem[locx + 4];
        __syncwarp();
        smem[locx] += smem[locx + 2];
        __syncwarp();
        smem[locx] += smem[locx + 1];
        __syncwarp();
    }

    // add over block
    if (locx == 0)
    {
        atomicAdd(sumA, smem[0]);
    }

    return;
}

__global__ void softmax_kernel_v0_div(float *B, float *sumA, int n)
{
    int gblx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gblx >= n)
    {
        return;
    }

    // div
    B[gblx] /= sumA[0];

    return;
}

int softmax_v0(const float *A, float *B, int n, cudaStream_t stream)
{
    constexpr int blockDimX = 256;
    dim3 blockDims(blockDimX);

    int gridDimX = (n + blockDimX - 1) / blockDimX;
    dim3 gridDims(gridDimX);

    float *sumA, *maxA;
    cudaMallocAsync(reinterpret_cast<void **>(&sumA), sizeof(float), stream);
    cudaMemsetAsync(sumA, 0, sizeof(float), stream);

    cudaMallocAsync(reinterpret_cast<void **>(&maxA), sizeof(float), stream);
    float negInfinity = -INFINITY;
    cudaMemcpyAsync(maxA, &negInfinity, sizeof(float), cudaMemcpyHostToDevice, stream);

    softmax_kernel_v0_max<blockDimX>
        <<<gridDims, blockDims, 0, stream>>>(A, maxA, n);

    softmax_kernel_v0_exp_and_reduce<blockDimX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, maxA, sumA, n);

    softmax_kernel_v0_div
        <<<gridDims, blockDims, 0, stream>>>(B, sumA, n);

    cudaFreeAsync(sumA, stream);
    cudaFreeAsync(maxA, stream);
    cudaStreamSynchronize(stream); 
    return 0;
}
#endif
// 如果只对一行数据进行 softmax 计算, 无论是 naive softmax 还是 online softmax, 都需要全局的 max 和 sum, 所以跨 block 规约是不避免的;
// 因此我们实现更通用的 batch online softmax;
template <int BLOCKSIZE>
__global__ void softmax_kernel_v0(const float *A, float *B, int bs, int num)
{
    // BLOCKSIZE / warp_size = 8
    __shared__ float shared_sum[8];
    __shared__ float shared_max[8];

    // 用 BLOCKSIZE 个线程处理 num 个数, 先不考虑 warp 级别的优化;
    // 每个线程处理 num_element 个数;
    int num_element = (num + BLOCKSIZE - 1) / BLOCKSIZE;
    float sum = 0.0f;
    float max = -INFINITY;
    for (int ii = 0; ii < num_element; ++ii)
    {
        if (ii * BLOCKSIZE + threadIdx.x < num)
        {
            float a = A[blockIdx.x * num + ii * BLOCKSIZE + threadIdx.x];
            float _max = fmaxf(max, a);
            sum = sum * expf(max - _max) + expf(a - _max);
            max = _max;
        }
    }

    // reduce
    for (int mask = 16; mask > 0; mask >>= 1) {
        float _sum = __shfl_xor_sync(0xffffffff, sum, mask, 32);
        float _max = __shfl_xor_sync(0xffffffff, max, mask, 32);
        
        float new_max = fmaxf(max, _max);
        float exp_shift = expf(max - new_max);
        
        sum = sum * exp_shift + _sum * expf(_max - new_max);
        max = new_max;
    }

    if (threadIdx.x % 32 == 0)
    {
        shared_sum[threadIdx.x / 32] = sum;
        shared_max[threadIdx.x / 32] = max;
    }
    __syncthreads();

    // reduce again
    if (threadIdx.x < 8) {
        sum = shared_sum[threadIdx.x];
        max = shared_max[threadIdx.x];
        for (int mask = 4; mask > 0; mask >>= 1) {
            float _sum = __shfl_xor_sync(0x000000ff, sum, mask, 8);
            float _max = __shfl_xor_sync(0x000000ff, max, mask, 8);

            float new_max = fmaxf(max, _max);
            float exp_shift = expf(max - new_max);
            
            sum = sum * exp_shift + _sum * expf(_max - new_max);
            max = new_max;
        }
    }

    // broadcast
    if (threadIdx.x == 0) {
        shared_sum[0] = sum;
        shared_max[0] = max;
    }
    __syncthreads();

    sum = shared_sum[0];
    max = shared_max[0];

    // div
    for (int ii = 0; ii < num_element; ++ii)
    {
        if (ii * BLOCKSIZE + threadIdx.x < num)
        {
            float a = A[blockIdx.x * num + ii * BLOCKSIZE + threadIdx.x];
            B[blockIdx.x * num + ii * BLOCKSIZE + threadIdx.x] = expf(a - max) / sum;
        }
    }

    return;
}

int softmax_v0(const float *A, float *B, int bs, int num, cudaStream_t stream)
{
    // 为了避免跨 block 规约, 所以最起码也要一个 block 处理一行;
    constexpr int blockDimX = 256;
    dim3 blockDims(blockDimX);

    int gridDimX = bs;
    dim3 gridDims(gridDimX);

    softmax_kernel_v0<blockDimX>
        <<<gridDims, blockDims, 0, stream>>>(A, B, bs, num);

    return 0;
}