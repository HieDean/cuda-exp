### Dependencies
```
apt-get install libspdlog-dev
```

### How to build?
```
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=Release ..

cmake --build -j$(nproc) .
```

### Simple compile
```
nvcc -o target xxx.cu yyy.cu -lspdlog -lfmt -lcudart -lcublas -lineinfo -arch=compute_86 -code=sm_86
```

### Perf report
```bash
nsys profile --stats true ./xxx

# 列出每个 CUDA kernel 的编译资源占用
cuobjdump --dump-resource-usage ./xxx

# SASS 是 GPU 真正执行的机器指令
cuobjdump --dump-sass --function bmm_kernel_v4 ./xxx

cuobjdump --dump-sass --gpu-architecture sm_120 ./xxx
    
# PTX 是虚拟 ISA，不一定等于最终机器代码。性能分析应以 SASS 为准，但 PTX 更容易阅读
cuobjdump --dump-ptx ./xxx

# 列出嵌入的架构代码
cuobjdump --list-elf ./xxx

# 查看可用指标
ncu --query-metrics

# 查看预定义分析集合
ncu --list-sets

ncu --list-sections

#
ncu \
    --set full                       \ # basic: 基础分析
                                     \ # full: 完整分析
    --section MemoryWorkloadAnalysis \ # 指定 section
    -o report \ # 输出报告
    -f ./xxx

# 
ncu \
    --kernel-name-base function           \ # function: 函数名
                                          \ # demangled: 函数名+模板参数
                                          \ # mangled(ABI mangled symbol)
    --kernel-name 'regex:^bmm_kernel_v4$' \
    --launch-skip x                       \ # 跳过前 x 次
    --launch-count x                      \ # 采集 x 次
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \ # 被采集的硬件指标
                                                                       \ # l1tex__: L1/shared-memory 子系统
                                                                       \ # data_bank_conflicts: 数据 bank conflict 数量
                                                                       \ # pipe_lsu: Load/Store Unit 管线
                                                                       \ # mem_shared: shared memory
                                                                       \ # op_ld: load 操作
                                                                       \ # op_st: store 操作
                                                                       \ # .sum: 所有相关硬件单元和采样实例的总和
    -f ./xxx

# 分析指令和源码关联
1. 编译时包含 -lineinfo;
2. ncu \
    -k 'regex:^bmm_kernel_v4$' \
    --launch-skip 10 --launch-count 1 \
    --section SourceCounters \
    --import-source yes \
    --export bmm_v4_profile \
    -f ./xxx
3. ncu --import bmm_v4_profile.ncu-rep
```

### References

基础概念:

https://zhuanlan.zhihu.com/p/1893611197540065818

https://zhuanlan.zhihu.com/p/442304996

https://zhuanlan.zhihu.com/p/688610975

https://wcjb.github.io/posts/a3125ebc/

https://zhuanlan.zhihu.com/p/4746910252
