"""Reproduce #73's input, distinguish launch failures from numeric differences."""

from std.math import sqrt
from std.memory import bitcast
from std.pathlib import Path
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.host.func_attribute import FuncAttribute
from mojo_dllm.gpu.kernels import _attention_kernel, dev_f32
from mojo_dllm.kernels.ops import attention
from mojo_dllm.sys.mem import Floats
from kernel import probe_kernel


def _fill(mut x: Floats, seed: Int):
    var s = UInt64(seed * 2654435761 + 1)
    for i in range(x.n):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((s >> 33) % 2001) - 1000) / 250.0


def _rel(got: Floats, want: Floats) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(want.n):
        var d = Float64(got[i]) - Float64(want[i])
        num += d * d
        den += Float64(want[i]) * Float64(want[i])
    return sqrt(num / den)


def run[BK: Int, DYNAMIC: Bool](name: String) raises:
    comptime L = 513
    comptime HD = 128
    comptime HEADS = 4
    comptime KV_HEADS = 1
    comptime BQ = 128
    comptime SHARED_BYTES = 2 * (BK * (HD + 8) + HD * (BK + 8))
    print("CASE", name, "STAGE context")
    var ctx = DeviceContext()
    print(
        "DEVICE cc",
        ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR),
        ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR),
        "static_limit",
        ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK),
        "optin_limit",
        ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN),
    )
    print(
        "CONFIG L",
        L,
        "BK",
        BK,
        "BQ",
        BQ,
        "dynamic",
        DYNAMIC,
        "shared_bytes",
        SHARED_BYTES,
    )
    var q = Floats(L * HEADS * HD)
    var k = Floats(L * KV_HEADS * HD)
    var v = Floats(L * KV_HEADS * HD)
    _fill(q, 4)
    _fill(k, 5)
    _fill(v, 6)
    for i in range(q.n):
        q[i] *= 0.1
    var want = Floats(q.n)
    attention(q.ptr(), k.ptr(), v.ptr(), want.ptr(), L, HEADS, KV_HEADS, HD, 2)
    var dq = ctx.enqueue_create_buffer[DType.float32](q.n)
    var dk = ctx.enqueue_create_buffer[DType.float32](k.n)
    var dv = ctx.enqueue_create_buffer[DType.float32](v.n)
    var baseline = ctx.enqueue_create_buffer[DType.float32](q.n)
    var candidate = ctx.enqueue_create_buffer[DType.float32](q.n)
    ctx.enqueue_copy(dq, q.ptr())
    ctx.enqueue_copy(dk, k.ptr())
    ctx.enqueue_copy(dv, v.ptr())
    ctx.synchronize()
    print("CASE", name, "STAGE baseline_launch")
    ctx.enqueue_function[_attention_kernel[128], dump_asm=Path("baseline.ptx")](
        dev_f32(dq),
        dev_f32(dk),
        dev_f32(dv),
        dev_f32(baseline),
        Int32(L),
        Int32(L),
        Int32(HEADS),
        Int32(KV_HEADS),
        Float32(1.0 / sqrt(128.0)),
        grid_dim=((L + 63) // 64, HEADS),
        block_dim=128,
    )
    ctx.synchronize()
    var original = Floats(q.n)
    ctx.enqueue_copy(original.ptr(), baseline)
    ctx.synchronize()
    var baseline_rel = _rel(original, want)
    print("BASELINE cpu_relative_error", baseline_rel)
    if not baseline_rel < 2e-3:
        raise Error("baseline CPU comparison failed")
    print("CASE", name, "STAGE candidate_launch")
    comptime if DYNAMIC:
        ctx.enqueue_function[
            probe_kernel[HD, BK, BQ, True], dump_asm=Path("candidate.ptx")
        ](
            dev_f32(dq),
            dev_f32(dk),
            dev_f32(dv),
            dev_f32(candidate),
            Int32(L),
            Int32(L),
            Int32(HEADS),
            Int32(KV_HEADS),
            Float32(1.0 / sqrt(128.0)),
            grid_dim=((L + BQ - 1) // BQ, HEADS),
            block_dim=BQ * 2,
            shared_mem_bytes=SHARED_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(SHARED_BYTES)
            ),
        )
    else:
        ctx.enqueue_function[
            probe_kernel[HD, BK, BQ, False], dump_asm=Path("candidate.ptx")
        ](
            dev_f32(dq),
            dev_f32(dk),
            dev_f32(dv),
            dev_f32(candidate),
            Int32(L),
            Int32(L),
            Int32(HEADS),
            Int32(KV_HEADS),
            Float32(1.0 / sqrt(128.0)),
            grid_dim=((L + BQ - 1) // BQ, HEADS),
            block_dim=BQ * 2,
        )
    print("CASE", name, "STAGE candidate_sync")
    ctx.synchronize()
    var actual = Floats(q.n)
    ctx.enqueue_copy(actual.ptr(), candidate)
    ctx.synchronize()
    print("CASE", name, "STAGE compare")
    var mismatch = 0
    var bit_mismatch = 0
    var nonfinite = 0
    var worst = Float64(0)
    for i in range(q.n):
        if not abs(actual[i]) <= Float32.MAX:
            nonfinite += 1
        if bitcast[DType.uint32](actual[i]) != bitcast[DType.uint32](
            original[i]
        ):
            bit_mismatch += 1
        if actual[i] != original[i]:
            if mismatch == 0:
                print(
                    "FIRST_MISMATCH index",
                    i,
                    "original",
                    original[i],
                    "candidate",
                    actual[i],
                )
            mismatch += 1
        worst = max(worst, abs(Float64(actual[i]) - Float64(original[i])))
    var rel = _rel(actual, want)
    print(
        "RESULT",
        name,
        "mismatch",
        mismatch,
        "bit_mismatch",
        bit_mismatch,
        "nonfinite",
        nonfinite,
        "max_abs",
        worst,
        "cpu_relative_error",
        rel,
    )
    if nonfinite != 0 or mismatch != 0 or not rel < 2e-3:
        raise Error("candidate numerical comparison failed")
    print("PASS", name)
