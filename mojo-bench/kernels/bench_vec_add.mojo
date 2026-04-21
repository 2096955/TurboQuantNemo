"""Smoke test: GPU vector addition to verify Mojo Metal backend works."""
from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, block_dim
from layout import LayoutTensor, Layout

# Vector size and layout
comptime N = 1024
comptime layout = Layout.row_major(N)
comptime block_size = 256
comptime num_blocks = ceildiv(N, block_size)


def gpu_vec_add_kernel(
    a: LayoutTensor[DType.float32, layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, layout, MutAnyOrigin],
    c: LayoutTensor[DType.float32, layout, MutAnyOrigin],
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < N:
        c[tid] = a[tid] + b[tid]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()

        print("Mojo GPU smoke test: vector add (N=", N, ")")

        # Create host buffers and fill
        host_a = ctx.enqueue_create_host_buffer[DType.float32](N)
        host_b = ctx.enqueue_create_host_buffer[DType.float32](N)
        ctx.synchronize()

        for i in range(N):
            host_a[i] = Float32(i)
            host_b[i] = Float32(i * 2)

        # Create device buffers and copy data
        dev_a = ctx.enqueue_create_buffer[DType.float32](N)
        dev_b = ctx.enqueue_create_buffer[DType.float32](N)
        dev_c = ctx.enqueue_create_buffer[DType.float32](N)

        ctx.enqueue_copy(dst_buf=dev_a, src_buf=host_a)
        ctx.enqueue_copy(dst_buf=dev_b, src_buf=host_b)

        # Wrap device buffers in LayoutTensors
        a_tensor = LayoutTensor[DType.float32, layout](dev_a)
        b_tensor = LayoutTensor[DType.float32, layout](dev_b)
        c_tensor = LayoutTensor[DType.float32, layout](dev_c)

        # Launch kernel
        ctx.enqueue_function[gpu_vec_add_kernel, gpu_vec_add_kernel](
            a_tensor,
            b_tensor,
            c_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Copy result back
        host_c = ctx.enqueue_create_host_buffer[DType.float32](N)
        ctx.enqueue_copy(dst_buf=host_c, src_buf=dev_c)
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
