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
