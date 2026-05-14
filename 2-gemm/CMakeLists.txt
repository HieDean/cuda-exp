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
# target_compile_options(gemm_test PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
#     $<$<COMPILE_LANGUAGE:CUDA>:-v>
# ) # 用于显示查看每个 kernel 占用的 register spill barrier 个数
# target_compile_options(gemm_test PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-maxrregcount=128>
# ) # 限制每个 kernel 最大可用寄存器个数为 128
target_link_libraries(gemm_test PRIVATE common spdlog fmt cudart cublas)
