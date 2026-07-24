add_executable(batchdot_test
    batchdot_test.cc
    batchdot_v0.cu
)
target_compile_options(batchdot_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(batchdot_test PRIVATE common spdlog fmt cudart cublas)
