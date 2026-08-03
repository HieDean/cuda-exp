// Adapted from NVIDIA CUDA Samples

#include <vector>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        spdlog::critical(
            "CUDA error: {} (error code {}) at {}:{}",
            cudaGetErrorString(err), static_cast<int>(err), file, line);
        std::exit(EXIT_FAILURE);
    }
}
#endif

int check_difference(const std::vector<float> &a, std::vector<float> &b, const std::string tag)
{
    assert(a.size() == b.size());

    int count = 0;
    std::vector<std::pair<int, std::vector<float>>> diffs;
    for (int ii = 0; ii < a.size(); ++ii)
    {
        if (std::abs(a[ii] - b[ii]) > 1e-3)
        {
            count++;
            diffs.push_back(std::pair<int, std::vector<float>>(ii, {a[ii], b[ii]}));
        }
    }
    spdlog::info("{} diff count: {} ratio: {:.3f}",
                 tag, count, static_cast<float>(count) / static_cast<float>(a.size()));

    for (int ii = 0; ii < std::min(static_cast<size_t>(10), diffs.size()); ++ii)
    {
        spdlog::warn("{} id: {} a: {} b {}",
                     tag, diffs[ii].first, diffs[ii].second[0], diffs[ii].second[1]);
    }
    return 0;
}

struct Statistics
{
    float min = 0.0;
    float max = 0.0;
    float average = 0.0;
    float median = 0.0;
    float variance = 0.0;
    float p95 = 0.0;
};

Statistics get_statistics(std::vector<float> values)
{
    if (values.empty())
    {
        return {};
    }

    std::sort(values.begin(), values.end());
    const size_t n = values.size();

    // 平均值
    const float sum = std::accumulate(values.begin(), values.end(), 0.0);
    const float mean = sum / static_cast<float>(n);

    // 中位数
    float median = 0.0;
    if (n % 2 == 0)
    {
        median = (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    else
    {
        median = values[n / 2];
    }

    // 方差
    float variance = 0.0;
    for (float val : values)
    {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(n);

    // P95
    const size_t p95_index = static_cast<size_t>(std::ceil(n * 0.95)) - 1;

    return {
        values.front(),   // min
        values.back(),    // max
        mean,             // average
        median,           // median
        variance,         // variance
        values[p95_index] // p95
    };
}