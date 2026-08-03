#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "bmm_vx.h"

using KernelFunc = int (*)(const float *, const float *, const float *, const float *,
                           float *, float *, float *, int, int, int, int, cudaStream_t);

void helper(KernelFunc func,
            const float *Q, const float *K, const float *V, const int *attnMasks,
            float *L, float *M, float *O, int bs, int num_heads, int seq_len, int head_dim, cudaStream_t stream,
            int warmup, int run,
            const std::vector<float> &l, const std::vector<float> &m, const std::vector<float> &o,
            std::string tag)
{
    // first time, only run for correctness check
    checkCudaErrors(cudaMemsetAsync(L, 0, L.size() * sizeof(float), stream));
    checkCudaErrors(cudaMemsetAsync(M, 0, M.size() * sizeof(float), stream));
    checkCudaErrors(cudaMemsetAsync(O, 0, o.size() * sizeof(float), stream));
    func(Q, K, V, attnMasks, L, M, O, bs, num_heads, seq_len, head_dim, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // check diff
    std::vector<float> l_from_device(l.size());
    checkCudaErrors(cudaMemcpy(l_from_device.data(), L,
                               l.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_difference(l, l_from_device, tag + "_l");

    std::vector<float> m_from_device(m.size());
    checkCudaErrors(cudaMemcpy(m_from_device.data(), M,
                               m.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_difference(m, m_from_device, tag + "_m");

    std::vector<float> o_from_device(o.size());
    checkCudaErrors(cudaMemcpy(o_from_device.data(), O,
                               o.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_difference(o, o_from_device, tag + "_o");

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(Q, K, V, attnMasks, L, M, O, bs, num_heads, seq_len, head_dim, stream);
    }
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    std::vector<cudaEvent_t> starts(run), stops(run);
    std::vector<float> kernelTimes(run);
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventCreate(&starts[ii]);
        cudaEventCreate(&stops[ii]);
    }

    // run
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventRecord(starts[ii], stream);
        func(Q, K, V, attnMasks, L, M, O, bs, num_heads, seq_len, head_dim, stream);
        cudaEventRecord(stops[ii], stream);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // get cost
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventElapsedTime(&kernelTimes[ii], starts[ii], stops[ii]);
    }
    auto statistics = get_statistics(kernelTimes);
    spdlog::info("{} Cost(ms): min: {} med: {} max: {} avg: {} var: {} p95: {}",
                 tag,
                 statistics.min, statistics.median, statistics.max,
                 statistics.average, statistics.variance, statistics.p95);

    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventDestroy(starts[ii]);
        cudaEventDestroy(stops[ii]);
    }
    return;
}

int host_attention(const float *q, const float *k, const float *v, const int *attn_masks,
                   float *l, float *m, float *o,
                   int bs, int num_heads, int seq_len, int head_dim)
{
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    float *attn_weights[bs * num_heads * seq_len * seq_len];
    memset(attn_weights, 0, bs * num_heads * seq_len * seq_len * sizeof(float));
    memset(o, 0, bs * num_heads * seq_len * head_dim * sizeof(float));
    for (int bn = 0; bn < bs * num_heads; ++bn)
    {
        int offset = bn * seq_len * head_dim;
        int weights_offset = bn * seq_len * seq_len;

        // attn_weight = q @ k
        for (int ii = 0; ii < seq_len; ++ii)
        {
            for (int jj = 0; jj < seq_len; ++jj)
            {
                for (int kk = 0; kk < head_dim; ++kk)
                {
                    attn_weights[weights_offset + ii * seq_len + jj] +=
                        q[offset + ii * head_dim + kk] * k[offset + jj * head_dim + kk];
                }
            }
        }

        // apply scale
        for (int ii = 0; ii < seq_len; ++ii)
        {
            for (int jj = 0; jj < seq_len; ++jj)
            {
                attn_weights[weights_offset + ii * seq_len + jj] *= scale;
            }
        }

        // apply mask
        if (attn_masks != nullptr)
        {
            for (int ii = 0; ii < seq_len; ++ii)
            {
                for (int jj = 0; jj < seq_len; ++jj)
                {
                    if (attn_masks[weights_offset + ii * seq_len + jj] == 0)
                    {
                        attn_weights[weights_offset + ii * seq_len + jj] = -1e9f;
                    }
                }
            }
        }

        // online safe softmax
        for (int ii = 0; ii < seq_len; ++ii)
        {
            float max = -INFINITY;
            float sum = 0.0f;
            for (int jj = 0; jj < seq_len; ++jj)
            {
                float new_max = fmaxf(max, attn_weights[weights_offset + ii * seq_len + jj]);
                sum = sum * expf(max - new_max) +
                      expf(attn_weights[weights_offset + ii * seq_len + jj] - new_max);
                max = new_max;
            }

            for (int jj = 0; jj < seq_len; ++jj)
            {
                attn_weights[weights_offset + ii * seq_len + jj] =
                    expf(attn_weights[weights_offset + ii * seq_len + jj] - max) / sum;
            }

            l[weights_offset + ii] = sum;
            m[weights_offset + ii] = max;
        }

        // o: attn_weight @ v
        for (int ii = 0; ii < seq_len; ++ii)
        {
            for (int jj = 0; jj < head_dim; ++jj)
            {
                for (int kk = 0; kk < seq_len; ++kk)
                {
                    o[offset + ii * head_dim + jj] +=
                        attn_weights[weights_offset + ii * seq_len + kk] * v[offset + kk * head_dim + jj];
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    int bs = 16;
    int num_heads = 32;
    int seq_len = 1024;
    int head_dim = 768;
    if (argc >= 5)
    {
        bs = std::stoi(argv[1]);
        num_heads = std::stoi(argv[2]);
        seq_len = std::stoi(argv[3]);
        head_dim = std::stoi(argv[4]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> q, k, v;
    std::vector<int> attn_masks;
    std::vector<float> l, m, o;
    for (int ii = 0; ii < bs * num_heads * seq_len * head_dim; ++ii)
    {
        q.push_back(uniform_dist(random_engine));
        k.push_back(uniform_dist(random_engine));
        v.push_back(uniform_dist(random_engine));
        o.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < bs * num_heads * seq_len; ++ii)
    {
        l.push_back(uniform_dist(random_engine));
        m.push_back(uniform_dist(random_engine));
    }

    // host attention
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    host_attention(q.data(), k.data(), v.data(), attn_masks.data(),
                   l.data(), m.data(), o.data(),
                   bs, num_heads, seq_len, head_dim);
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host_attention cost: {}ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count());

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *Q, *K, *V;
    int *attnMasks;
    float *L, *M, *O;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&Q), q.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(Q, q.data(), q.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&K), k.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(K, k.data(), k.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&V), v.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(V, v.data(), v.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&attnMasks), attn_masks.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpyAsync(attnMasks, attn_masks.data(), attn_masks.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&L), l.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(L, 0, l.size() * sizeof(float), stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&M), m.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(M, 0, m.size() * sizeof(float), stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&O), o.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(O, 0, o.size() * sizeof(float), stream));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // attention_v0
    helper(attention_v0,
           Q, K, V, attnMasks,
           L, M, O,
           bs, num_heads, seq_len, head_dim, stream,
           10, 50, attn_weights, o, "attention_v0");

    {
    }

    // free
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(attnMasks);
    cudaFree(L);
    cudaFree(M);
    cudaFree(O);
    cudaStreamDestroy(stream);

    return 0;
}
