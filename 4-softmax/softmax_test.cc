#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "utils.h"
#include "softmax_vx.h"

using KernelFunc = int (*)(const float *, float *, int, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, float *B, int bs, int num, cudaStream_t stream,
            int warmup, int run, std::vector<float> &b, std::string tag)
{
    // first time, only run for correctness check
    func(A, B, bs, num, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<float> b_from_device(bs * num);
    checkCudaErrors(cudaMemcpy(b_from_device.data(), B,
                               bs * num * sizeof(float), cudaMemcpyDeviceToHost));

    // check diff
    check_difference(b, b_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, bs, num, stream);
    }
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float milliseconds = 0;

    // run
    cudaEventRecord(start_gpu, stream);
    for (int ii = 0; ii < run; ++ii)
    {
        func(A, B, bs, num, stream);
    }
    cudaEventRecord(stop_gpu, stream);
    cudaEventSynchronize(stop_gpu);
    checkCudaErrors(cudaGetLastError());

    // get cost
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
    spdlog::info("{} cost: {}us", tag, static_cast<int64_t>(milliseconds * 1000.0));

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    return;
}

int main(int argc, char **argv)
{
    int bs = 256;
    int num = 4096;

    if (argc >= 3)
    {
        bs = std::stoi(argv[1]);
        num = std::stoi(argv[2]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a(bs * num), b(bs * num);
    for (int ii = 0; ii < bs * num; ++ii)
    {
        a[ii] = uniform_dist(random_engine);
    }

    // host softmax
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    for (int ii = 0; ii < bs; ++ii)
    {
        float sum = 0.0f;
        float max = -INFINITY;
        for (int jj = 0; jj < num; ++jj)
        {
            float _max = fmaxf(max, a[ii * num + jj]);
            sum = sum * expf(max - _max) + expf(a[ii * num + jj] - _max);
            max = _max;
        }
        for (int jj = 0; jj < num; ++jj)
        {
            b[ii * num + jj] = expf(a[ii * num + jj] - max) / sum;
        }
    }
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host transpose cost: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *A, *B;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), bs * num * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(A, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), bs * num * sizeof(float)));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // softmax_v0
    helper(softmax_v0, A, B, bs, num, stream, 5, 10, b, "softmax_v0");

    // softmax_v1
    helper(softmax_v1, A, B, bs, num, stream, 5, 10, b, "softmax_v1");

    {
        // 初始化 cuDNN 句柄
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        // 创建张量描述符
        cudnnTensorDescriptor_t srcDesc, dstDesc;
        cudnnCreateTensorDescriptor(&srcDesc);
        cudnnCreateTensorDescriptor(&dstDesc);

        // 假设是 NCHW 格式的 4D 张量 (N, C, H, W)
        int n = bs, c = num, h = 1, w = 1;
        cudnnSetTensor4dDescriptor(srcDesc,
                                   /*format=*/CUDNN_TENSOR_NCHW,
                                   /*dataType=*/CUDNN_DATA_FLOAT, n, c, h, w);
        // 通常输入和输出的描述符可以相同
        cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

        // 调用核心 API 执行 Softmax
        float alpha = 1.0f, beta = 0.0f;
        cudnnSoftmaxForward(handle,
                            CUDNN_SOFTMAX_ACCURATE,     // 算法：推荐使用数值稳定版本
                            CUDNN_SOFTMAX_MODE_CHANNEL, // 模式：在通道维度上计算
                            &alpha,                     // 缩放因子
                            srcDesc, A,
                            &beta, // 累加因子
                            dstDesc, B);

        // check diff
        std::vector<float> b_from_device(bs * num);
        checkCudaErrors(cudaMemcpy(b_from_device.data(), B,
                                   bs * num * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(b, b_from_device, "cudnnSoftmaxForward");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            cudnnSoftmaxForward(handle,
                                CUDNN_SOFTMAX_ACCURATE,
                                CUDNN_SOFTMAX_MODE_CHANNEL,
                                &alpha,
                                srcDesc, A,
                                &beta,
                                dstDesc, B);
        }

        // create event
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        float milliseconds = 0;

        // cudnnSoftmaxForward
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            cudnnSoftmaxForward(handle,
                                CUDNN_SOFTMAX_ACCURATE,
                                CUDNN_SOFTMAX_MODE_CHANNEL,
                                &alpha,
                                srcDesc, A,
                                &beta,
                                dstDesc, B);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("cudnnSoftmaxForward cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));

        // 清理资源
        cudnnDestroyTensorDescriptor(srcDesc);
        cudnnDestroyTensorDescriptor(dstDesc);
        cudnnDestroy(handle);
    }

    // free
    cudaFree(A);
    cudaFree(B);
    cudaStreamDestroy(stream);

    return 0;
}
