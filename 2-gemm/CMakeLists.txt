add_executable(gemm
    gemm.cu
)

target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

target_link_libraries(gemm PRIVATE
    common
    spdlog
    fmt
    cudart
    cublas
)
