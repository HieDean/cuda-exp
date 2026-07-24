add_executable(softmax_test
    softmax_test.cc
    softmax_v0.cu
    softmax_v1.cu
)
target_compile_options(softmax_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(softmax_test PRIVATE common spdlog fmt cudart cudnn)
