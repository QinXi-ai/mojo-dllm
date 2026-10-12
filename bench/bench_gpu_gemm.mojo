"""GPU GEMM time on LLaDA-8B's weight shapes, one shape at a time.

Usage: mojo run -I src bench/bench_gpu_gemm.mojo [tokens] [reps]

Each shape first runs for half a second to bring the GPU to its boost clock,
then `reps` times with the stream synchronized around every call;
the median is reported, since a laptop GPU's clocks and a stray process can
slow single calls. Weight bytes are synthetic (time does not depend on their
values). The per-forward line multiplies each shape by how often a LLaDA-8B
forward pass runs it: 32 layers of q, k, o (Q4_K 4096 x 4096), v (Q6_K
4096 x 4096), gate and up (Q4_K 12288 x 4096) and down (Q6_K 4096 x 12288),
plus the LM head (Q6_K 126464 x 4096) on the 32 rows a step samples.
Prints one JSON object.
"""

from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q6_K, ggml_type_name
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    dev_f32,
    dev_u8,
    gemm_gpu,
    quantize_gpu,
    staged_matrix_bytes,
)
from mojo_dllm.sys.mem import Bytes, Floats


def time_shape(
    ctx: DeviceContext, N: Int, K: Int, w_type: Int, T: Int, reps: Int
) raises -> Float64:
    var nbytes = staged_matrix_bytes(w_type, N, K)
    var host = Bytes(nbytes, zero=False)
    var p = host.u8()
    for i in range(nbytes):
        p[unsafe_offset=i] = UInt8((i * 131 + 7) % 251)
    var w = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(w, p)
    var x = Floats(T * K)
    for i in range(T * K):
        x[i] = Float32((i * 37) % 101 - 50) / 50.0
    var dx = ctx.enqueue_create_buffer[DType.float32](T * K)
    ctx.enqueue_copy(dx, x.ptr())
    var qa = GpuQActs(ctx, T, K)
    quantize_gpu(ctx, dev_f32(dx), T, K, qa)
    var out = ctx.enqueue_create_buffer[DType.float32](T * N)
    # Warm up for at least half a second: a laptop GPU idles at a low clock
    # and needs sustained work before it boosts.
    var w0 = perf_counter_ns()
    while perf_counter_ns() - w0 < 500_000_000:
        gemm_gpu(ctx, w_type, dev_u8(w), N, K, qa, T, dev_f32(out), N)
        ctx.synchronize()
    var times = List[Float64]()
    for _ in range(reps):
        var t0 = perf_counter_ns()
        gemm_gpu(ctx, w_type, dev_u8(w), N, K, qa, T, dev_f32(out), N)
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - t0) / 1e6)
    sort(times)
    return times[len(times) // 2]


def main() raises:
    var args = argv()
    var T = Int(args[1]) if len(args) > 1 else 148
    var reps = Int(args[2]) if len(args) > 2 else 30
    var ctx = DeviceContext()
    var qk = time_shape(ctx, 4096, 4096, GGML_Q4_K, T, reps)
    var v = time_shape(ctx, 4096, 4096, GGML_Q6_K, T, reps)
    var gu = time_shape(ctx, 12288, 4096, GGML_Q4_K, T, reps)
    var down = time_shape(ctx, 4096, 12288, GGML_Q6_K, T, reps)
    var head = time_shape(ctx, 126464, 4096, GGML_Q6_K, 32, reps)
    var per_forward = 32.0 * (3.0 * qk + v + 2.0 * gu + down) + head
    print(
        '{"tokens": '
        + String(T)
        + ', "reps": '
        + String(reps)
        + ', "gpu": "'
        + ctx.name()
        + '", "median_ms": {"q4k_4096x4096": '
        + String(qk)
        + ', "q6k_4096x4096": '
        + String(v)
        + ', "q4k_12288x4096": '
        + String(gu)
        + ', "q6k_4096x12288": '
        + String(down)
        + ', "q6k_126464x4096_32rows": '
        + String(head)
        + '}, "gemm_ms_per_forward": '
        + String(per_forward)
        + "}"
    )
