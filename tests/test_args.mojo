from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import Args, parse_csv_ints


def test_values_and_booleans() raises:
    var a = Args(["mojo-dllm", "run", "--prompt", "hi", "--json"], ["json"])
    assert_equal(a.command, "run")
    assert_equal(a.get("prompt"), "hi")
    assert_true(a.has("prompt"))
    assert_true(a.has("json"))
    assert_true(not a.has("absent"))


def test_missing_value_names_flag() raises:
    with assert_raises(contains="flag --prompt needs a value"):
        _ = Args(["x", "run", "--prompt"], [])


def test_missing_required_and_defaults() raises:
    var a = Args(["x", "run"], [])
    with assert_raises(contains="missing required flag --model"):
        _ = a.get("model")
    assert_equal(a.get_or("model", "default"), "default")
    assert_equal(a.int_or("steps", 42), 42)
    assert_equal(a.float_or("temperature", 0.5), 0.5)


def test_flag_like_token_is_a_value() raises:
    var a = Args(["x", "run", "--prompt", "--json"], ["json"])
    assert_equal(a.get("prompt"), "--json")
    assert_true(not a.has("json"))


def test_numbers_and_last_duplicate_win() raises:
    var a = Args(
        ["x", "run", "--seed", "9", "--seed", "-5", "--temperature", "1e-3"], []
    )
    assert_equal(a.int_or("seed", 0), -5)
    assert_equal(a.float_or("temperature", 0), Float64(0.001))


def test_invalid_integers_raise() raises:
    for value in List[String](["abc", "", "12.5"]):
        var a = Args(["x", "run", "--steps", value], [])
        with assert_raises():
            _ = a.int_or("steps", 10)


def test_positionals_and_empty_command() raises:
    var a = Args(["x", "inspect", "model.gguf"], [])
    assert_equal(a.command, "inspect")
    assert_equal(len(a.positional), 1)
    assert_equal(a.positional[0], "model.gguf")
    var empty = Args(["x"], [])
    assert_equal(empty.command, "")


def test_unknown_flag_is_preserved() raises:
    # Characterize the current typo behavior; validation is a separate change.
    var a = Args(["x", "run", "--thread", "4"], [])
    assert_equal(a.get("thread"), "4")
    assert_equal(a.int_or("threads", 8), 8)


def test_csv_skips_empty_and_strips_whitespace() raises:
    var a = parse_csv_ints("1,2,,3")
    assert_equal(len(a), 3)
    for i in range(3):
        assert_equal(a[i], i + 1)
    var b = parse_csv_ints(" 4 , 5")
    assert_equal(len(b), 2)
    assert_equal(b[0], 4)
    assert_equal(b[1], 5)
    assert_equal(len(parse_csv_ints("")), 0)
    with assert_raises():
        _ = parse_csv_ints("a")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
