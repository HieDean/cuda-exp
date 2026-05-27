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
```
nsys profile --stats true ./xxx

ncu --set full -o report -f ./xxx
```

### References

基础概念:

https://zhuanlan.zhihu.com/p/1893611197540065818

https://zhuanlan.zhihu.com/p/442304996

https://zhuanlan.zhihu.com/p/688610975

https://wcjb.github.io/posts/a3125ebc/

https://zhuanlan.zhihu.com/p/4746910252
