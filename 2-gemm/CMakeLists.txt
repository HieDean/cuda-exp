add_executable(gemm_test
    gemm_test.cc
    gemm_v0.cu
    gemm_v1.cu
    gemm_v2.cu
    gemm_v3.cu
    gemm_v4.cu
    gemm_v5.cu
    gemm_v6.cu
    gemm_v7.cu
)
target_compile_options(gemm_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(gemm_test PRIVATE common spdlog fmt cudart cublas)
