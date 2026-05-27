add_executable(elementwise_test
    elementwise_test.cc
    elementwise_v0.cu
    elementwise_v1.cu
    # elementwise_v2.cu
)
target_compile_options(elementwise_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(elementwise_test PRIVATE common spdlog fmt cudart cublas)
