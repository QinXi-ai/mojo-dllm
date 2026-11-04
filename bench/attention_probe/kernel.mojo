"""Isolated #73 diagnostic; not used by production dispatch.

Arithmetic copied from 20f46c0 src/mojo_dllm/gpu/kernels.mojo.
Only the kernel name, template parameters and shared allocations differ.
"""
from std.math import exp
from std.memory import AddressSpace, unsafe_stack_allocation
from max.gpu import barrier, block_idx, lane_id, thread_idx, warp_id
from max.gpu.memory import external_memory
from max.gpu.primitives.warp import shuffle_xor
from mojo_dllm.gpu.kernels import H2, KPAD, VPAD, _h2, _mma_f16
from mojo_dllm.sys.mem import F32Ptr


def probe_kernel[
    HD: Int, BK: Int, BQ: Int, DYNAMIC: Bool
](
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int32,
    Lk: Int32,
    heads: Int32,
    kv_heads: Int32,
    scale: Float32,
):
    """Flash attention on the f16 tensor cores, f32 accumulation.

    Warp w owns queries 16w..16w+15 of the block's AQ. Per staged tile of AK keys it
    computes S = Q K^T in 32-key groups with m16n8k16 mmas, folds S into an online softmax,
    and adds P V with more mmas. The S accumulators of keys 16c..16c+15 are
    laid out exactly as the A operand of the P V mma, so P never leaves the
    registers. Thread (g, t) of a warp holds rows g and g + 8 of its tile.
    """
    comptime AK = BK
    comptime AQ = BQ
    comptime ATT_THREADS = BQ * 2
    comptime KC = HD // 16
    """k16 chunks of the head dimension (S = Q K^T)."""
    comptime DN = HD // 8
    """n8 tiles of the head dimension (O = P V)."""
    comptime SN = 4
    """Keep the original 32-key softmax update order, regardless of staging tile."""
    comptime KS = HD + KPAD
    comptime VS = AK + VPAD
    var sK: Pointer[
        Float16, MutUntrackedOrigin, address_space=AddressSpace.SHARED
    ]
    var sVt: Pointer[
        Float16, MutUntrackedOrigin, address_space=AddressSpace.SHARED
    ]
    comptime if DYNAMIC:
        sK = external_memory[
            Float16, address_space=AddressSpace.SHARED, alignment=16
        ]()
        sVt = sK.unsafe_offset(AK * KS)
    else:
        sK = unsafe_stack_allocation[
            AK * KS, Float16, alignment=16, address_space=AddressSpace.SHARED
        ]()
        sVt = unsafe_stack_allocation[
            HD * VS, Float16, alignment=16, address_space=AddressSpace.SHARED
        ]()
    var nq = Int(L)
    var nk = Int(Lk)
    var h = Int(block_idx.y)
    var kvh = h // (Int(heads) // Int(kv_heads))
    var stride = Int(heads) * HD
    var kv_stride = Int(kv_heads) * HD
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var r0 = Int(block_idx.x) * AQ + 16 * warp + g
    var r1 = r0 + 8
    # Q fragments for the whole head dimension, scaled, kept in registers.
    var qa = Array[H2, 4 * KC](fill=H2(0))
    comptime for c in range(KC):
        comptime for e in range(4):
            var r = r0 if (e & 1) == 0 else r1
            var col = c * 16 + 2 * t + (8 if e >= 2 else 0)
            if r < nq:
                var src = q.unsafe_offset(r * stride + h * HD + col)
                qa[c * 4 + e] = _h2(
                    src[unsafe_offset=0] * scale, src[unsafe_offset=1] * scale
                )
    var m0 = Float32.MIN
    var m1 = Float32.MIN
    var l0 = Float32(0)
    var l1 = Float32(0)
    var acc = Array[SIMD[DType.float32, 4], DN](fill=SIMD[DType.float32, 4](0))
    var kt = 0
    while kt < nk:
        barrier()
        # Stage AK keys: K row-major, V transposed, both as f16.
        comptime for i in range(AK * HD // ATT_THREADS):
            var idx = th + i * ATT_THREADS
            var j = idx // HD
            var d = idx % HD
            var kv_ = Float32(0)
            var vv = Float32(0)
            if kt + j < nk:
                var src = (kt + j) * kv_stride + kvh * HD + d
                kv_ = k[unsafe_offset=src]
                vv = v[unsafe_offset=src]
            sK[unsafe_offset=j * KS + d] = kv_.cast[DType.float16]()
            sVt[unsafe_offset=d * VS + j] = vv.cast[DType.float16]()
        barrier()
        # Larger cooperative loads reduce staging/barrier overhead. Softmax
        # and P V still advance 32 keys at a time to preserve rounding order.
        comptime for sub in range(BK // 32):
            # S = Q K^T for this warp's 16 queries and 32 keys.
            var s = Array[SIMD[DType.float32, 4], SN](
                fill=SIMD[DType.float32, 4](0)
            )
            comptime for n in range(SN):
                var krow = sK.unsafe_offset((sub * 32 + n * 8 + g) * KS + 2 * t)
                comptime for c in range(KC):
                    var b0 = krow.unsafe_offset(c * 16).unsafe_load[
                        width=2, alignment=4
                    ]()
                    var b1 = krow.unsafe_offset(c * 16 + 8).unsafe_load[
                        width=2, alignment=4
                    ]()
                    s[n] = _mma_f16(
                        qa[c * 4],
                        qa[c * 4 + 1],
                        qa[c * 4 + 2],
                        qa[c * 4 + 3],
                        b0,
                        b1,
                        s[n],
                    )
            # Online softmax over rows g (lanes 0, 1 of s[n]) and g + 8 (2, 3).
            var t0 = Float32.MIN
            var t1 = Float32.MIN
            comptime for n in range(SN):
                var sn = s[n]
                comptime for e in range(4):
                    if kt + sub * 32 + n * 8 + 2 * t + (e & 1) >= nk:
                        sn[e] = Float32.MIN
                s[n] = sn
                t0 = max(t0, max(sn[0], sn[1]))
                t1 = max(t1, max(sn[2], sn[3]))
            # The four threads of a quad hold one row between them.
            t0 = max(t0, shuffle_xor(t0, 1))
            t0 = max(t0, shuffle_xor(t0, 2))
            t1 = max(t1, shuffle_xor(t1, 1))
            t1 = max(t1, shuffle_xor(t1, 2))
            var n0 = max(m0, t0)
            var n1 = max(m1, t1)
            var c0 = exp(m0 - n0)
            var c1 = exp(m1 - n1)
            m0 = n0
            m1 = n1
            var ps0 = Float32(0)
            var ps1 = Float32(0)
            comptime for n in range(SN):
                var sn = s[n]
                comptime for e in range(4):
                    var valid = kt + sub * 32 + n * 8 + 2 * t + (e & 1) < nk
                    var mrow = m0 if e < 2 else m1
                    sn[e] = exp(sn[e] - mrow) if valid else Float32(0)
                s[n] = sn
                ps0 += sn[0] + sn[1]
                ps1 += sn[2] + sn[3]
            # l is a per-thread partial sum; the quad adds them up at the end.
            l0 = l0 * c0 + ps0
            l1 = l1 * c1 + ps1
            var corr = SIMD[DType.float32, 4](c0, c0, c1, c1)
            comptime for dn in range(DN):
                acc[dn] = acc[dn] * corr
            # O += P V: S's accumulators for keys 16c..16c+15 are P's A operand.
            comptime for c in range(2):
                var a0 = _h2(s[2 * c][0], s[2 * c][1])
                var a1 = _h2(s[2 * c][2], s[2 * c][3])
                var a2 = _h2(s[2 * c + 1][0], s[2 * c + 1][1])
                var a3 = _h2(s[2 * c + 1][2], s[2 * c + 1][3])
                comptime for dn in range(DN):
                    var vrow = sVt.unsafe_offset(
                        (dn * 8 + g) * VS + sub * 32 + c * 16 + 2 * t
                    )
                    var b0 = vrow.unsafe_load[width=2, alignment=4]()
                    var b1 = vrow.unsafe_offset(8).unsafe_load[
                        width=2, alignment=4
                    ]()
                    acc[dn] = _mma_f16(a0, a1, a2, a3, b0, b1, acc[dn])
        kt += AK
    l0 += shuffle_xor(l0, 1)
    l0 += shuffle_xor(l0, 2)
    l1 += shuffle_xor(l1, 1)
    l1 += shuffle_xor(l1, 2)
    var inv0 = 1.0 / l0
    var inv1 = 1.0 / l1
    comptime for dn in range(DN):
        var col = h * HD + dn * 8 + 2 * t
        if r0 < nq:
            o[unsafe_offset=r0 * stride + col] = acc[dn][0] * inv0
            o[unsafe_offset=r0 * stride + col + 1] = acc[dn][1] * inv0
        if r1 < nq:
            o[unsafe_offset=r1 * stride + col] = acc[dn][2] * inv1
            o[unsafe_offset=r1 * stride + col + 1] = acc[dn][3] * inv1
