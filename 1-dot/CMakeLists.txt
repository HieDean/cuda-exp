add_executable(dot_test
    dot_test.cc
    dot_v0.cu
    dot_v1.cu
    dot_v2.cu
)
target_compile_options(dot_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(dot_test PRIVATE common spdlog fmt cudart cublas)
