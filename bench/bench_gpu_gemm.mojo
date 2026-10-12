"""GPU GEMM time on a model's weight shapes, one shape at a time.

Usage: mojo run -I src bench/bench_gpu_gemm.mojo MODEL.gguf [tokens] [reps]

Each shape first runs for half a second to bring the GPU to its boost clock,
then `reps` times with the stream synchronized around every call; the median
is reported, since a laptop GPU's clocks and a stray process can slow single
calls. Weight bytes are synthetic (time does not depend on their values).

The per-forward total reads every projection's tensor type from the GGUF (a
Q4_K_M file mixes Q4_K and Q6_K layer by layer), adds the median time of that
type and shape, and counts the last layer's FFN on the 32 rows a step
samples, as the forward pass does. The LM head runs on 32 rows. Prints one
JSON object.
"""

from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_dllm.formats.gguf import (
    GGUFFile,
    GGML_Q4_K,
    GGML_Q6_K,
    ggml_type_name,
)
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


def _basename(path: String) -> String:
    var parts = path.split("/")
    return String(parts[len(parts) - 1])


def _key(t: Int, n: Int, k: Int, rows: Int) -> String:
    return (
        ggml_type_name(t)
        + " "
        + String(n)
        + "x"
        + String(k)
        + " @"
        + String(rows)
    )


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bench_gpu_gemm MODEL.gguf [tokens] [reps]")
    var g = GGUFFile(String(args[1]))
    var T = Int(args[2]) if len(args) > 2 else 148
    var reps = Int(args[3]) if len(args) > 3 else 30
    var ctx = DeviceContext()
    var n_layers = g.get_int(g.architecture() + ".block_count")
    var names: List[String] = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ]
    # Every (type, N, K, rows) the forward pass runs, timed once each.
    var keys = List[String]()
    var times = List[Float64]()
    var total = Float64(0)
    var json = String("")
    for li in range(n_layers):
        for name in names:
            var t = g.tensor("blk." + String(li) + "." + name + ".weight")
            var rows = T
            if li == n_layers - 1 and name.startswith("ffn"):
                rows = 32
            var key = _key(t.ggml_type, t.n_rows(), t.ne0, rows)
            var found = -1
            for i in range(len(keys)):
                if keys[i] == key:
                    found = i
            if found < 0:
                keys.append(key)
                times.append(
                    time_shape(ctx, t.n_rows(), t.ne0, t.ggml_type, rows, reps)
                )
                found = len(keys) - 1
            total += times[found]
    var head = g.tensor("output.weight")
    var head_ms = time_shape(
        ctx, head.n_rows(), head.ne0, head.ggml_type, 32, reps
    )
    total += head_ms
    for i in range(len(keys)):
        json += (", " if i else "") + '"' + keys[i] + '": ' + String(times[i])
    print(
        '{"model": "'
        + _basename(String(args[1]))
        + '", "tokens": '
        + String(T)
        + ', "reps": '
        + String(reps)
        + ', "gpu": "'
        + ctx.name()
        + '", "median_ms": {'
        + json
        + ', "'
        + _key(head.ggml_type, head.n_rows(), head.ne0, 32)
        + '": '
        + String(head_ms)
        + '}, "gemm_ms_per_forward": '
        + String(total)
        + "}"
    )
