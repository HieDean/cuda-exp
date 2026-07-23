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
        if (std::abs(a[ii] - b[ii]) > 1e-5)
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
