# The quantized GEMM

Most of a denoising step is matrix multiplication: seven projections per
layer, 32 layers, every step, over the whole canvas. `run --verbose` prints
the split for your machine. This page explains the
kernel that does it.

## The shape of the problem

A step multiplies each weight matrix W (N rows × K columns, 4 to 6 bits per
weight) by an activation matrix X (L rows × K). L is the canvas length, a few
hundred at most. That is a skinny GEMM, and it is compute-bound: each weight
is read once per step and used L times.

## Activations: int8 with a scale per 256

Before a projection, each activation row is quantized to int8 with one f32
scale per 256 values, and the sum of each group of 16 values is kept as a
float. This is llama.cpp's Q8_K scheme, which keeps the comparison with it
fair. Integer dot products are 4× denser than f32 FMAs on AVX2.

## Weights: 8 rows per register

GGUF stores each row contiguously. A row-at-a-time dot product needs a
horizontal sum per output, and the block scales arrive one scalar at a time.
So the loader repacks every matrix once:

```text
for each group of 8 rows, for each 256-weight super-block:
    Q4_K:  d[8], dmin[8] as f32 | scales[8][8], mins[8][8] as u8 | 1024 bytes of nibbles
    Q6_K:  d[8] as f32          | scales[16][8] as i8           | 2048 bytes, one value per byte
```

Inside the weight bytes, a 32-byte load holds 4 consecutive weights of each of
the 8 rows. One `vpdpbusd` (AVX-VNNI) multiplies those 32 unsigned weights by
4 activation bytes broadcast to every lane and adds into 8 int32
accumulators: 8 rows × 4 multiply-adds in one instruction. The scale of a
sub-block is an 8-lane vector, one entry per row, applied once per 16 or 32
weights.

The zero points do not touch the inner loop. Q4_K computes
`w = d·sc·q − dmin·m`, so the second term becomes `dmin·m` times the
activation sum of the sub-block, which the quantizer already stored. Q6_K
stores `q + 32`, and the 32 comes off the same way.

## Tiling and threads

A work item is 16 output rows. It walks the tokens 4 at a time, keeping 4
int32 and 4 f32 accumulators in registers, so each loaded weight vector feeds 4
tokens. Items go to threads through an atomic counter (see
[Architecture](architecture.md#threads)).

## Checks

`tests/test_quant.mojo` compares the decoders with the `gguf` Python package's
dequantization, and both GEMMs with an f64 reference over the same dequantized
weights. Swapping the nibble halves or dropping the Q6_K zero point fails
these tests; that mutation check runs before changes to this file land.

`bench/bench_qgemm.mojo` measures the kernel alone on LLaDA's layer shapes.

## The GPU path

`src/mojo_dllm/gpu/kernels.mojo` does the same arithmetic on an NVIDIA GPU,
with a layout chosen for tensor cores instead of AVX registers.

**Weights stay in GGUF blocks.** Upload copies each Q4_K matrix as it is and
pads each 210-byte Q6_K block to 212 bytes, so every block starts on a 4-byte
boundary and the kernels can load 32-bit words.

**Activations get one scale per 32.** The GPU quantizes activations to int8
with one f32 scale per 32 values (llama.cpp's Q8_1 shape) and keeps
`scale x sum(q)` next to it. One warp quantizes one 32-value block, a value
per lane, and reduces its max and integer sum with shuffles, so loads and
stores are coalesced. A 32-weight Q4_K sub-block then meets exactly one
activation scale:

```text
sum w x = d*sc*dx * sum(q * qx)  -  dmin*m * (dx * sum(qx))
```

**One mma per sub-block.** A Q6_K thread block computes 64 weight rows by a
tile of 32, 64 or 80 tokens. For each 256-wide super-block it unpacks the
weights into shared memory as int8 (q - 32), copies the matching activations
next to them, and each of its 4 warps runs
`mma.m16n8k32.s8` per sub-block for its 16 rows and every 8 tokens. Q6_K
scales every 16 weights, so it uses `mma.m16n8k16` instead, one per scale.
The int32 result goes to f32 with the two scales and, for Q4_K, the min term.

**Q4_K: packed in shared memory, unpacked in registers.** Staging cost half
the GEMM time and did not overlap with compute
(`bench/checks/2026-10-13-gemm-cost-split.json`: with staging done once, the
GEMM took about 100 ms instead of about 220; the mma itself cost almost
nothing). So a Q4_K block copies each row's 144-byte super-block as it is, in
16-byte loads: the 16-byte header (d, dmin and the packed scales) to one
array, the 128 bytes of nibbles to another. A warp unpacks its A fragments in
registers: word i of nibble pair p holds sub-block 2p in its low nibbles and
2p + 1 in its high ones at the same position, so one shared load feeds two
mmas, and the 6-bit scales are decoded from the header per sub-block. With the
weights packed, a block holds 128 rows and runs 8 warps in the same 48 KiB,
which halves the activation copies per output. The arithmetic is unchanged, so
the logits are too (`bench/checks/2026-10-13-gemm3-logits.json`).

**Tile width per call.** Unpacking a weight tile costs the same however
many tokens then use it, so the widest tile that saves a tile wins: a
148-token canvas runs two 80-wide tiles rather than five 32-wide ones, and
each weight tile is unpacked twice instead of five times. The LM head's 32
rows keep the 32-wide tile. 80 is the widest tile whose shared memory fits
the 48 KiB a kernel gets without asking for more. Token tiles of one weight
tile run back to back, so they read it through L2.

**Shared-memory layout.** Weight rows are 272 bytes apart, so the 8 rows a
fragment load reads fall in different banks. Activation scales are stored
by sub-block, then token, then (scale, sum), so the pair of tokens a thread
folds after an mma is one 128-bit load, and the scale rows are padded to 9
and 17 floats for the same reason.
`tests/gpu/test_gpu.mojo` checks the fragment layout against a CPU matmul
on random int8 matrices, and the whole kernel against an f64 GEMM.

**Attention on f16 tensor cores.** Each warp owns 16 queries and walks the
keys 32 at a time. Q is scaled by 1/sqrt(d) and held in registers as f16
fragments; K is staged row-major and V transposed, both as f16, so every
operand load is one 32-bit word. `mma.m16n8k16` computes S = Q K^T in f32,
the four threads that share a row reduce its max with two shuffles, and the
online softmax rescales the running sums. S's accumulator for 16 keys has the
same register layout as the A operand of the next mma, so P goes from S to
P V without leaving the registers. On LLaDA-8B the logits stay as close to
the f32 CPU path as with f32 attention
(`bench/checks/2026-10-13-attention-f16-logits.json`). The kernel never
stores a score row, so the canvas length has no fixed limit.
