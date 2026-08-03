add_executable(bmm_test
    bmm_test.cc
    bmm_v0.cu
    bmm_v1.cu
    bmm_v2.cu
    bmm_v3.cu
    bmm_v4.cu
    bmm_v5.cu
    bmm_v51.cu
    bmm_v52.cu
    bmm_v6.cu
    bmm_v7.cu
)
target_compile_options(bmm_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(bmm_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>
    $<$<COMPILE_LANGUAGE:CUDA>:-v>
) # 用于显示查看每个 kernel 占用的 register spill barrier 个数
# target_compile_options(bmm_test PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-maxrregcount=128>
# ) # 限制每个 kernel 最大可用寄存器个数为 128
target_link_libraries(bmm_test PRIVATE common spdlog fmt cudart cublas)
