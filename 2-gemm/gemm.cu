/**
 * simple compile:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas
 * for ncu and nsys:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas -lineinfo
 *     ncu --set full -o report -f ./gemm
 *     nsys profile --stats true ./gemm
 * for specific arch:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas -lineinfo -arch=compute_120 -code=sm_120
 */

#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "utils.h"


__global__ void gemm_v1(const float *device_a, const float *device_b, float *device_c,
                        int m, int n, int k)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
    {
        return;
    }

    float sum = 0.0;
    for (int kk = 0; kk < k; ++kk)
    {
        sum += device_a[row * k + kk] * device_b[kk * n + col];
    }
    device_c[row * n + col] = sum;

    return;
}

int device_gemm_v1(const float *device_a, const float *device_b, float *device_c,
                   int m, int n, int k, cudaStream_t stream)
{
    int blockDim_x = 32;
    int blockDim_y = 32;
    dim3 blockDims(blockDim_x, blockDim_y);

    int gridDim_x = (n + blockDim_x - 1) / blockDim_x;
    int gridDim_y = (m + blockDim_y - 1) / blockDim_y;
    dim3 gridDims(gridDim_x, gridDim_y);

    gemm_v1<<<gridDims, blockDims, 0, stream>>>(device_a, device_b, device_c, m, n, k);
    return 0;
}

__global__ void gemm_v2(const float *device_a, const float *device_b, float *device_c,
                        int m, int n, int k, int step)
{
    // const int bid = blockIdx.z * gridDim.x * gridDim.y +
    //                 blockIdx.y * gridDim.x +
    //                 blockIdx.x;
    // const int tid = threadIdx.z * blockDim.x * blockDim.y +
    //                 threadIdx.y * blockDim.x +
    //                 threadIdx.x;
    // const int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    // const int id = bid * numThreadsPerBlock + tid;
    // __syncwarp();

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ char smem[];
    float *smem_a = (float *)smem;              // [blockDim.y, step]
    float *smem_b = &smem_a[blockDim.y * step]; // [step, blockDim.x]

    if (row >= m || col >= n)
    {
        return;
    }

    if (threadIdx.x >= step || threadIdx.y >= step)
    {
        return;
    }

    float sum = 0.0;
    for (int ss = 0; ss < k; ss += step)
    {
        /**
         * 这里访问 global memory 中的 a 矩阵的时候, block 内部 x 方向的线程访问连续的地址, 所以 x 方向的访存可以合并(coalesced);
         * 访问 global memory 中的 b 矩阵的时候, block 内部 x 方向的线程访问连续的地址, 所以 x 方向的访存也可以合并(coalesced);
         * ----------------------------------------------------
         * 这里写入 shared memory 的时候, 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_a 和 smem_b 的时候,
         * 访问的是长度为 32 的连续地址, 恰好访问 32 个不同的 bank, 不存在 bank conflict;
         */
        smem_a[threadIdx.y * step + threadIdx.x] = device_a[row * k + ss + threadIdx.x];
        smem_b[threadIdx.y * blockDim.x + threadIdx.x] = device_b[(ss + threadIdx.y) * n + col];
        __syncthreads();

        /**
         * 这里访问 shared memory:
         * 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_a 的时候, 访问的是相同的地址, 即 smem_a[threadIdx.y * step + ii], 因此触发 broadcast;
         * 当 threadIdx.y 固定, threadIdx.x 从 0 到 31 访问 smem_a 的时候, 访问的是长度为 32 的连续地址, 恰好访问 32 个不同的 bank, 不存在 bank conflict;
         * PS: bank 布局: smem[0](bank0) smem[1](bank1) smem[2](bank2) ... smem[31](bank31) smem[32](bank0) ...
         */
        for (int ii = 0; ii < step; ++ii)
        {
            sum += smem_a[threadIdx.y * step + ii] * smem_b[ii * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }

    device_c[row * n + col] = sum;

    return;
}

int device_gemm_v2(const float *device_a, const float *device_b, float *device_c,
                   int m, int n, int k, cudaStream_t stream)
{
    int blockDim_x = 32;
    int blockDim_y = 32;
    int step = std::min(blockDim_x, blockDim_y); // it is better when blockDim_x == blockDim_x == step
    int smemSize = blockDim_x * step + blockDim_y * step;
    dim3 blockDims(blockDim_x, blockDim_y);

    int gridDim_x = (n + blockDim_x - 1) / blockDim_x;
    int gridDim_y = (m + blockDim_y - 1) / blockDim_y;
    dim3 gridDims(gridDim_x, gridDim_y);

    gemm_v2<<<gridDims, blockDims, smemSize * sizeof(float), stream>>>(device_a, device_b, device_c, m, n, k, step);
    return 0;
}

template <int STEP, int STRIDE>
__global__ void gemm_v3(const float *device_a, const float *device_b, float *device_c,
                        int m, int n, int k)
{
    const int tile_start_y = blockIdx.y * blockDim.y * STRIDE;
    const int tile_start_x = blockIdx.x * blockDim.x * STRIDE;
    const int side_length = STEP * STRIDE;

    // 是否超越了c的边缘
    if (tile_start_y >= m || tile_start_x >= n)
    {
        return;
    }

    extern __shared__ char smem_v3[];
    float *smem_a = (float *)smem_v3;                             // [blockDim.y * STRIDE, STEP * STRIDE]
    float *smem_b = &smem_a[blockDim.y * STRIDE * STEP * STRIDE]; // [STEP * STRIDE, blockDim.x * STRIDE]

    float sum[STRIDE * STRIDE] = {};
    for (int ss = 0; ss < k; ss += side_length)
    {
        for (int rr = 0; rr < STRIDE; ++rr)
        {
            for (int cc = 0; cc < STRIDE; ++cc)
            {
                // 我们需要使用线程id来对a b smem进行索引
                /**smem_a
                 * y: rr * blockDim.y + threadIdx.y
                 * x: cc * STEP + threadIdx.x
                 * side_length: side_length
                 * 当blockDim.x == blockDim.y == STEP的时候, smem_a和smem_b的索引可以复用
                 * smem_b
                 * y: rr * STEP + threadIdx.y
                 * x: cc * blockDim.x + threadIdx.x
                 * side_length: blockDim.x * STRIDE
                 */
                /**device_a m*k
                 * y: tile_start_y + rr * blockDim.y + threadIdx.y
                 * x: ss + cc * STEP + threadIdx.x
                 */
                /**device_b k*n
                 * y: ss + rr * STEP + threadIdx.y
                 * x: tile_start_x + cc * blockDim.x + threadIdx.x
                 */
                if (tile_start_y + rr * blockDim.y + threadIdx.y >= m || ss + cc * STEP + threadIdx.x >= k)
                {
                    smem_a[(rr * blockDim.y + threadIdx.y) * side_length + cc * STEP + threadIdx.x] = 0.0;
                }
                else
                {
                    smem_a[(rr * blockDim.y + threadIdx.y) * side_length + cc * STEP + threadIdx.x] =
                        device_a[(tile_start_y + rr * blockDim.y + threadIdx.y) * k + ss + cc * STEP + threadIdx.x];
                }

                if (ss + rr * STEP + threadIdx.y >= k || tile_start_x + cc * blockDim.x + threadIdx.x >= n)
                {
                    smem_b[(rr * STEP + threadIdx.y) * blockDim.x * STRIDE + cc * blockDim.x + threadIdx.x] = 0.0;
                }
                else
                {
                    smem_b[(rr * STEP + threadIdx.y) * blockDim.x * STRIDE + cc * blockDim.x + threadIdx.x] =
                        device_b[(ss + rr * STEP + threadIdx.y) * n + tile_start_x + cc * blockDim.x + threadIdx.x];
                }
            }
        }
        __syncthreads();

        for (int rr = 0; rr < STRIDE; ++rr)
        {
            for (int cc = 0; cc < STRIDE; ++cc)
            {
                // calc the [rr, cc] tile of c, each tile will resolved by one block.
                for (int ii = 0; ii < side_length; ++ii)
                {
                    /**smem_a
                     * y: rr * blockDim.y + threadIdx.y
                     * x: ii
                     * side_length: side_length
                     */
                    /**smem_b
                     * y: ii
                     * x: cc * blockDim.x + threadIdx.x
                     * side_length: blockDim.x * STRIDE
                     */
                    sum[rr * STRIDE + cc] +=
                        smem_a[(rr * blockDim.y + threadIdx.y) * side_length + ii] *
                        smem_b[ii * blockDim.x * STRIDE + cc * blockDim.x + threadIdx.x];
                }
            }
        }
        __syncthreads();
    }

    for (int rr = 0; rr < STRIDE; ++rr)
    {
        for (int cc = 0; cc < STRIDE; ++cc)
        {
            /**
             * y: tile_start_y + rr * blockDim.y + threadIdx.y
             * x: tile_start_x + cc * blockDim.x + threadIdx.x
             * side_length: n
             */
            if (tile_start_y + rr * blockDim.y + threadIdx.y < m &&
                tile_start_x + cc * blockDim.x + threadIdx.x < n)
            {
                device_c[(tile_start_y + rr * blockDim.y + threadIdx.y) * n +
                         tile_start_x + cc * blockDim.x + threadIdx.x] = sum[rr * STRIDE + cc];
            }
        }
    }

    return;
}

int device_gemm_v3(const float *device_a, const float *device_b, float *device_c,
                   int m, int n, int k, cudaStream_t stream)
{
    /**
     * 使用 32 * 32 个线程, 计算 c 中的 64 * 64 个点
     */
    const int step = 32;
    const int stride = 2;
    int blockDim_x = 32;
    int blockDim_y = 32;
    int smemSize = blockDim_x * stride * step * stride +
                   blockDim_y * stride * step * stride;
    dim3 blockDims(blockDim_x, blockDim_y);

    int gridDim_x = (n + blockDim_x * stride - 1) / (blockDim_x * stride);
    int gridDim_y = (m + blockDim_y * stride - 1) / (blockDim_y * stride);
    dim3 gridDims(gridDim_x, gridDim_y);

    gemm_v3<step, stride><<<gridDims, blockDims,
                            smemSize * sizeof(float), stream>>>(
        device_a, device_b, device_c, m, n, k);
    return 0;
}

int host_gemm(const float *a, const float *b, float *c, int m, int n, int k)
{
    for (int mm = 0; mm < m; ++mm)
    {
        for (int nn = 0; nn < n; ++nn)
        {
            c[mm * n + nn] = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                c[mm * n + nn] += a[mm * k + kk] * b[kk * n + nn];
            }
        }
    }
    return 0;
}

int compare(const float *a, const float *b, int length, const char *tag)
{
    int count = 0;
    std::vector<std::pair<int, std::vector<float>>> diffs;
    for (int ii = 0; ii < length; ++ii)
    {
        if (std::abs(a[ii] - b[ii]) > 1e-3)
        {
            count++;
            diffs.push_back(std::pair<int, std::vector<float>>(ii, {a[ii], b[ii]}));
        }
    }
    spdlog::info("{} diff count: {} ratio: {:.3f}",
                 tag, count, static_cast<float>(count) / static_cast<float>(length));
    // printf("%s: diff count: %d ratio: %.3f\n", tag, count, static_cast<float>(count) / static_cast<float>(length));

    for (int ii = 0; ii < std::min(static_cast<size_t>(10), diffs.size()); ++ii)
    {
        spdlog::info("{} id: {} a: {} b {}", tag, diffs[ii].first, diffs[ii].second[0], diffs[ii].second[1]);
        // printf("%s: id: %d a: %f b: %f\n", tag, diffs[ii].first, diffs[ii].second[0], diffs[ii].second[1]);
    }
    return 0;
}

int main(int argc, char **argv)
{

    int m = 512;
    int n = 512;
    int k = 512;
    if (argc == 4)
    {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
        k = std::stoi(argv[3]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /************* host *************/
    // init host mat
    std::vector<float> a, b, c;
    for (int ii = 0; ii < m * k; ++ii)
    {
        a.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < k * n; ++ii)
    {
        b.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < m * n; ++ii)
    {
        c.push_back(0.0);
    }

    // host gemm
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    host_gemm(a.data(), b.data(), c.data(), m, n, k);
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host gemm cost: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());
    // printf("host gemm cost: %ldus\n",
    //        std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /************* device *************/
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *device_a, *device_b, *device_c;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_a), a.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_a, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_b), b.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_b, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_c), c.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float milliseconds = 0;

    {
        /*** device_gemm_v1 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        device_gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        compare(c.data(), c_from_device.data(), c.size(), "device_gemm_v1 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            device_gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            device_gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("device_gemm_v1 gemm cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
        // printf("device_gemm_v1 gemm cost: %ldus\n",
        //        static_cast<int64_t>(milliseconds * 1000.0 / 10));
    }

    {
        /*** device_gemm_v2 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        device_gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        compare(c.data(), c_from_device.data(), c.size(), "device_gemm_v2 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            device_gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            device_gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("device_gemm_v2 gemm cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
        // printf("device_gemm_v2 gemm cost: %ldus\n",
        //        static_cast<int64_t>(milliseconds * 1000.0 / 10));
    }

    {
        /*** device_gemm_v3 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        device_gemm_v3(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        compare(c.data(), c_from_device.data(), c.size(), "device_gemm_v3 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            device_gemm_v3(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            device_gemm_v3(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("device_gemm_v3 gemm cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
        // printf("device_gemm_v3 gemm cost: %ldus\n",
        //        static_cast<int64_t>(milliseconds * 1000.0 / 10));
    }

    {
        /*** cublas ***/
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        float alpha = 1.0, beta = 0.0f;
        checkCudaErrors(cudaMemset(device_c, 0, c.size() * sizeof(float)));

        // first time, only for correctness check
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        compare(c.data(), c_from_device.data(), c.size(), "cublas vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);
        }

        // cublasSgemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("cublas gemm cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
        // printf("cublas gemm cost: %ldus\n",
        //        static_cast<int64_t>(milliseconds * 1000.0 / 10));

        cublasDestroy(handle);
    }

    // free
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
