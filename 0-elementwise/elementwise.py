import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_kernel_v0(a_ptr, b_ptr, c_ptr, m, n,
               blockSizeX: tl.constexpr,
               blockSizeY: tl.constexpr, # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pidX = tl.program_id(axis=0)
    pidY = tl.program_id(axis=1)

    # This program will process inputs that are offset from the initial data.
    # Note that offsets is a list of pointers:
    x0 = pidX * blockSizeX
    y0 = pidY * blockSizeY
    xx = x0 + tl.arange(0, blockSizeX)[:, None]
    yy = y0 + tl.arange(0, blockSizeY)[None, :]

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask_xx = xx < m
    mask_yy = yy < n
    mask = mask_xx & mask_yy

    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    a = tl.load(a_ptr + xx * n + yy, mask=mask)
    b = tl.load(b_ptr + xx * n + yy, mask=mask)
    c = a * b

    # Write back to DRAM.
    tl.store(c_ptr + xx * n + yy, c, mask=mask)

def elementwise_v0(a: torch.Tensor, b: torch.Tensor):
    ###  Preallocate the output.
    c = torch.empty_like(a)

    m, n = c.shape
    blockSizeX = 32
    blockSizeY = 32

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    # gridSize = lambda meta: (triton.cdiv(m, meta['gridSizeX']), triton.cdiv(n, meta['gridSizeY']))
    gridSizeX = triton.cdiv(m, blockSizeX)
    gridSizeY = triton.cdiv(n, blockSizeY)
    gridSize = (gridSizeX, gridSizeY)

    ### NOTE: kernel 中的所有编译期常量都可以在 kernel 中被字典 `meta` 获取
    elementwise_kernel_v0[gridSize](a, b, c, m, n, blockSizeX=blockSizeX, blockSizeY=blockSizeY)

    ### kernel 是异步执行的, 虽然这里返回了 c, 但其实 kernel 还没有执行完,
    ### 后续访问 c 时会隐式同步, 但如何测量性能就需要使用`torch.cuda.synchronize()`显式同步
    return c


@triton.jit
def elementwise_kernel_v1(a_ptr, b_ptr, c_ptr, num_elements, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)

    x0 = pid * blockSize
    xx = x0 + tl.arange(0, blockSize)

    mask = xx < num_elements

    a = tl.load(a_ptr + xx, mask=mask)
    b = tl.load(b_ptr + xx, mask=mask)
    c = a * b

    tl.store(c_ptr + xx, c, mask=mask)

def elementwise_v1(a: torch.Tensor, b: torch.Tensor):
    c = torch.empty_like(a)

    blockSize = 1024
    num_elements = c.numel()
    gridSize = (triton.cdiv(num_elements, blockSize), )

    elementwise_kernel_v1[gridSize](a, b, c, num_elements, blockSize=blockSize)

    return c

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(8, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'v0', 'v1'],  # Possible values for `line_arg`.
        line_names=['Torch', 'v0', 'v1'],  # Label name for the lines.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='elementwise-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    shape = (size, size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    a = torch.rand(shape, device=device, dtype=dtype)
    b = torch.rand(shape, device=device, dtype=dtype)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: a * b)
    if provider == 'v0':
        ms = triton.testing.do_bench(lambda: elementwise_v0(a, b))
    if provider == 'v1':
        ms = triton.testing.do_bench(lambda: elementwise_v1(a, b))

    gbps = lambda ms: 3 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

def main(): 
    '''
    这里还有一些疑问:
    * benchmark 测试显示 v0 的性能比 torch 差很多, 可以理解用 2D 的形式去
      处理 elementwise 算子确实不是一种好的方式, 用 1D 的形式应该会更高效;
    * 为什么 blockSize 设置为 32*32 会比 16*16 性能好很多?
    * triton kernel 的底层调度策略目前是未知的, 我们对于 kernel 的行为做不到像 cuda kernel 那样细粒度的控制;
    * 无法像 cuda kernel 那样细粒度的控制其实是 triton 的优点, 因为用户不需要关系复杂的底层逻辑了;

    reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
    '''
    shape = [2 ** 10, 2 ** 10]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    torch.manual_seed(0)
    a = torch.rand(shape, device=device, dtype=dtype)
    b = torch.rand(shape, device=device, dtype=dtype)

    c_torch = a * b

    c_v0 = elementwise_v0(a, b)
    ### check correctness
    ### 这里访问 c_v0 时会隐式同步, 但如何测量性能就需要使用`torch.cuda.synchronize()`显示同步
    print(f'The maximum difference between torch and v0 is '
          f'{torch.max(torch.abs(c_torch - c_v0))}')
    
    c_v1 = elementwise_v1(a, b)
    print(f'The maximum difference between torch and v1 is '
          f'{torch.max(torch.abs(c_torch - c_v1))}')

    benchmark.run(print_data=True)

if __name__ == '__main__':
    main()