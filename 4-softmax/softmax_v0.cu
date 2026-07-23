#include <cstdio>
#include "softmax_vx.h"

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