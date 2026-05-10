add_executable(dot_test
    dot_test.cc
    dot_v0.cu
)
target_compile_options(dot_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(dot_test PRIVATE common spdlog fmt cudart cublas)

add_executable(dot_ncu
    dot_ncu.cc
    dot_v0.cu
)
target_compile_options(dot_ncu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(dot_ncu PRIVATE common spdlog fmt cudart cublas)