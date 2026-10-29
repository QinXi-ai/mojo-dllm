from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.sys.mem import Bytes, Floats, alloc_bytes, free_bytes
from mojo_dllm.sys.mmap import Mapping, drop_range, populate_range
from mojo_dllm.sys.parallel import parallel_for, AtomicPtr


def test_byte_alignment_and_zero_initialization() raises:
    var blocks = List[Bytes]()
    for i in range(16):
        blocks.append(Bytes(1 + i * 17))
    for i in range(len(blocks)):
        assert_equal(blocks[i].addr % 64, 0)
        for j in range(blocks[i].size):
            assert_equal(blocks[i].u8().unsafe_load(j), 0)
    var empty = Bytes(0)
    assert_equal(empty.size, 0)
    assert_true(empty.addr != 0)


def test_huge_page_alignment() raises:
    for size in List[Int]([2 * 1024 * 1024, 2 * 1024 * 1024 + 1]):
        var b = Bytes(size)
        assert_equal(b.addr % (2 * 1024 * 1024), 0)
        for i in range(size):
            assert_equal(b.u8().unsafe_load(i), 0)


def test_negative_sizes_raise() raises:
    with assert_raises(contains="-1"):
        var addr = alloc_bytes(-1)
        free_bytes(addr)
    with assert_raises(contains="-20"):
        _ = Floats(-5)


def test_impossible_allocation_raises() raises:
    # Exceeds the virtual address space on x86-64 Linux; never touch pages.
    with assert_raises(contains="out of memory: could not allocate"):
        var addr = alloc_bytes(1 << 50, zero=False)
        free_bytes(addr)


def test_floats_initialization_roundtrip_and_zero() raises:
    var f = Floats(17)
    for i in range(f.n):
        assert_equal(f[i], Float32(0))
        f[i] = Float32(i) - 0.5
    for i in range(f.n):
        assert_equal(f[i], Float32(i) - 0.5)
    f.zero()
    for i in range(f.n):
        assert_equal(f[i], Float32(0))


def test_mapping_reads_bytes() raises:
    var m = Mapping("tests/fixtures/bad-magic.gguf")
    assert_equal(m.size, 64)
    var expected = "GGUX".as_bytes()
    for i in range(4):
        assert_equal(m.ptr().unsafe_load(i), expected[i])


def test_mapping_missing_path_names_file() raises:
    with assert_raises(
        contains="cannot open file: tests/fixtures/does-not-exist.gguf"
    ):
        _ = Mapping("tests/fixtures/does-not-exist.gguf")


def test_mapping_empty_file() raises:
    with assert_raises(
        contains=(
            "file is empty or unreadable: tests/fixtures/malformed/empty.gguf"
        )
    ):
        _ = Mapping("tests/fixtures/malformed/empty.gguf")


def test_mapping_rejects_directory() raises:
    # Depending on the filesystem, open, lseek, or mmap can reject directories.
    var rejected = False
    try:
        _ = Mapping("tests/fixtures")
    except e:
        var message = String(e)
        assert_true(
            message == "cannot open file: tests/fixtures"
            or message == "file is empty or unreadable: tests/fixtures"
            or message == "mmap failed for tests/fixtures"
        )
        rejected = True
    assert_true(rejected, "mapping a directory must fail")


def test_page_ranges() raises:
    var lo, hi = drop_range(4097, 8192)
    assert_equal(lo, 8192)
    assert_equal(hi, 12288)
    var outer_lo, outer_hi = populate_range(4097, 8192)
    assert_equal(outer_lo, 4096)
    assert_equal(outer_hi, 16384)
    var empty_lo, empty_hi = drop_range(4097, 100)
    assert_true(empty_hi <= empty_lo)
    var aligned_lo, aligned_hi = drop_range(4096, 4096)
    assert_equal(aligned_lo, 4096)
    assert_equal(aligned_hi, 8192)
    # Read-back after madvise cannot establish whether a page was dropped:
    # file-backed mappings read the same bytes again. Test the range directly.


def check_parallel(n: Int, threads: Int) raises:
    var counts = Bytes((max(n, 0) + 1) * 8)
    var ptr = AtomicPtr(unsafe_from_address=counts.addr)

    def work(i: Int) {imm}:
        _ = ptr[unsafe_offset=i].fetch_add(1)

    parallel_for(work, n, threads)
    for i in range(max(n, 0)):
        assert_equal(ptr[unsafe_offset=i].load(), Int64(1))
    assert_equal(ptr[unsafe_offset=max(n, 0)].load(), Int64(0))


def test_parallel_visits_each_item_once_and_preserves_guard() raises:
    for n in List[Int]([0, 1, 7, 64, 1000]):
        for threads in List[Int]([1, 2, 4, 64]):
            check_parallel(n, threads)


def test_parallel_nonpositive_items_and_threads() raises:
    for threads in List[Int]([-3, 0, 1, 4]):
        check_parallel(0, threads)
        check_parallel(-4, threads)
        check_parallel(7, threads)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
