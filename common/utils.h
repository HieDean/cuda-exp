// Adapted from NVIDIA CUDA Samples

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
            cudaGetErrorString(err), static_cast<int>(err), file, line
        );
        std::exit(EXIT_FAILURE);
    }
}
#endif
