from std.testing import assert_equal, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile, GGML_Q4_K, ggml_row_bytes
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.kernels.packed import PackedMatrix
from mojo_dllm.sys.mem import Bytes, Floats

comptime FIX = "tests/fixtures/malformed/"


def test_base_fixture_is_valid() raises:
    var g = GGUFFile(FIX + "base.gguf")
    assert_equal(g.n_tensors(), 2)
    assert_equal(g.tensor("f32").n_elements(), 8)
    assert_equal(g.tensor("q4k").n_elements(), 256)
    assert_equal(g.tensor("q4k").ggml_type, GGML_Q4_K)
    assert_equal(len(g.get_int_array("test.array")), 3)


def test_rejects_empty() raises:
    with assert_raises(contains="file is empty or unreadable"):
        _ = GGUFFile(FIX + "empty.gguf")


def test_rejects_short_magic() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "magic-cut.gguf")


def test_rejects_header_tensor_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-12.gguf")


def test_rejects_header_kv_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-20.gguf")


def test_rejects_metadata_cut() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "metadata-cut.gguf")


def test_rejects_oversized_key() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length.gguf")


def test_rejects_key_length_that_wraps_negative() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length-negative.gguf")


def test_accepts_last_read_ending_exactly_at_eof() raises:
    var g = GGUFFile(FIX + "exact-eof.gguf")
    assert_equal(g.n_tensors(), 0)


def test_rejects_version_one() raises:
    with assert_raises(contains="unsupported GGUF version 1"):
        _ = GGUFFile(FIX + "version-1.gguf")


def test_rejects_version_four() raises:
    with assert_raises(contains="unsupported GGUF version 4"):
        _ = GGUFFile(FIX + "version-4.gguf")


def test_rejects_large_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-large.gguf")


def test_rejects_negative_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-negative.gguf")


def test_rejects_large_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-large.gguf")


def test_rejects_negative_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-negative.gguf")


def test_rejects_unknown_scalar_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "scalar-type.gguf")


def test_rejects_unknown_array_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "array-type.gguf")


def test_rejects_nested_array() raises:
    with assert_raises(
        contains="nested GGUF arrays are not supported: test.array"
    ):
        _ = GGUFFile(FIX + "nested-array.gguf")


def test_rejects_zero_dims() raises:
    with assert_raises(contains="tensor f32 has 0 dims"):
        _ = GGUFFile(FIX + "dims-0.gguf")


def test_rejects_five_dims() raises:
    with assert_raises(contains="tensor f32 has 5 dims"):
        _ = GGUFFile(FIX + "dims-5.gguf")


def test_rejects_tensor_data_cut() raises:
    with assert_raises(contains="tensor q4k extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-cut.gguf")


def test_rejects_unsupported_row_type() raises:
    with assert_raises(contains="unsupported GGUF tensor type: IQ2_XXS"):
        _ = ggml_row_bytes(16, 256)


def test_rejects_unsupported_dequant_type() raises:
    var src = Bytes(144)
    var dst = Floats(256)
    with assert_raises(contains="unsupported GGUF tensor type: Q2_K"):
        dequant_row(10, src.addr, dst.ptr(), 256)


def test_rejects_unsupported_packed_type() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="cannot repack weight type Q2_K"):
        _ = PackedMatrix(src.addr, 10, 8, 256)


def test_rejects_incomplete_dequant_block() raises:
    # Keep backing memory valid even when the length guard is mutated away.
    var src = Bytes(144)
    var dst = Floats(300)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        dequant_row(GGML_Q4_K, src.addr, dst.ptr(), 300)


def test_rejects_incomplete_packed_block() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        _ = PackedMatrix(src.addr, GGML_Q4_K, 8, 300)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
