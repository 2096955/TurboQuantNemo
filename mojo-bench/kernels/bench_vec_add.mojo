"""Smoke test: GPU vector addition to verify Mojo Metal backend works."""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout
from memory import UnsafePointer
from time import perf_counter_ns


fn gpu_vec_add_kernel[
    layout: Layout
](a: LayoutTensor[mut=False, dtype=DType.float32, layout=layout],
  b: LayoutTensor[mut=False, dtype=DType.float32, layout=layout],
  c: LayoutTensor[mut=True, dtype=DType.float32, layout=layout]):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < a.shape[0]:
        c[tid] = a[tid] + b[tid]


fn main():
    var N = 1024
    var ctx = DeviceContext()

    print("Mojo GPU smoke test: vector add (N=", N, ")")

    # Allocate and fill host buffers
    var host_a = UnsafePointer[Float32].alloc(N)
    var host_b = UnsafePointer[Float32].alloc(N)
    var host_c = UnsafePointer[Float32].alloc(N)

    for i in range(N):
        host_a[i] = Float32(i)
        host_b[i] = Float32(i * 2)

    # Copy to device, launch kernel, copy back
    var dev_a = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_b = ctx.enqueue_create_buffer[DType.float32](N)
    var dev_c = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_copy(dev_a, host_a, N)
    ctx.enqueue_copy(dev_b, host_b, N)

    var threads_per_block = 256
    var blocks = (N + threads_per_block - 1) // threads_per_block

    ctx.enqueue_function[gpu_vec_add_kernel[Layout.row_major(N)]](
        dev_a.as_tensor(), dev_b.as_tensor(), dev_c.as_tensor(),
        grid_dim=(blocks,), block_dim=(threads_per_block,),
    )

    ctx.synchronize()
    ctx.enqueue_copy(host_c, dev_c, N)
    ctx.synchronize()

    # Verify
    var max_err: Float64 = 0.0
    for i in range(N):
        var expected = Float32(i + i * 2)
        var err = abs(Float64(host_c[i]) - Float64(expected))
        if err > max_err:
            max_err = err

    print("Max error:", max_err)
    if max_err < 1e-6:
        print("PASS")
    else:
        print("FAIL")

    host_a.free()
    host_b.free()
    host_c.free()
