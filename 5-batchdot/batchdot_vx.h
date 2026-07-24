#include <cuda_runtime.h>

int batchdot_v0(const float *A, const float *B, float *C,
                int bs, int length, cudaStream_t stream);
