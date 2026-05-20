add_executable(transpose_test
    transpose_test.cc
    transpose_v0.cu
)
target_compile_options(transpose_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(transpose_test PRIVATE common spdlog fmt cudart cublas)
