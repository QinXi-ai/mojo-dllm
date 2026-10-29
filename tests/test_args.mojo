from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import (
    Args,
    edit_distance,
    parse_csv_ints,
    suggest_flag,
    unknown_flag_error,
)


def test_values_and_booleans() raises:
    var a = Args(
        ["mojo-dllm", "run", "--prompt", "hi", "--json"], ["prompt"], ["json"]
    )
    assert_equal(a.command, "run")
    assert_equal(a.get("prompt"), "hi")
    assert_true(a.has("prompt"))
    assert_true(a.has("json"))
    assert_true(not a.has("absent"))


def test_missing_value_is_recorded_not_raised() raises:
    var a = Args(["x", "run", "--prompt"], ["prompt"], [])
    assert_equal(a.missing_value, "prompt")
    assert_true(not a.has("prompt"))
    var ok = Args(["x", "run", "--prompt", "p"], ["prompt"], [])
    assert_equal(ok.missing_value, "")


def test_help_before_a_trailing_value_flag_is_still_seen() raises:
    var a = Args(["x", "run", "--help", "--prompt"], ["prompt"], [])
    assert_true(a.has("help"))
    assert_equal(a.missing_value, "prompt")


def test_missing_required_and_defaults() raises:
    var a = Args(["x", "run"], [], [])
    with assert_raises(contains="missing required flag --model"):
        _ = a.get("model")
    assert_equal(a.get_or("model", "default"), "default")
    assert_equal(a.int_or("steps", 42), 42)
    assert_equal(a.float_or("temperature", 0.5), 0.5)


def test_flag_like_token_is_a_value() raises:
    var a = Args(["x", "run", "--prompt", "--json"], ["prompt"], ["json"])
    assert_equal(a.get("prompt"), "--json")
    assert_true(not a.has("json"))


def test_numbers_and_last_duplicate_win() raises:
    var a = Args(
        ["x", "run", "--seed", "9", "--seed", "-5", "--temperature", "1e-3"],
        ["seed", "temperature"],
        [],
    )
    assert_equal(a.int_or("seed", 0), -5)
    assert_equal(a.float_or("temperature", 0), Float64(0.001))


def test_invalid_integers_raise() raises:
    for value in List[String](["abc", "", "12.5"]):
        var a = Args(["x", "run", "--steps", value], ["steps"], [])
        with assert_raises():
            _ = a.int_or("steps", 10)


def test_positionals_and_empty_command() raises:
    var a = Args(["x", "inspect", "model.gguf"], [], [])
    assert_equal(a.command, "inspect")
    assert_equal(len(a.positional), 1)
    assert_equal(a.positional[0], "model.gguf")
    var empty = Args(["x"], [], [])
    assert_equal(empty.command, "")


def test_unknown_flags_are_collected_not_stored() raises:
    var a = Args(["x", "run", "--thread", "4"], ["threads"], [])
    assert_equal(len(a.unknown), 1)
    assert_equal(a.unknown[0], "thread")
    assert_true(not a.has("thread"))
    assert_equal(a.int_or("threads", 8), 8)


def test_help_is_recognized_in_both_spellings() raises:
    for spelling in List[String](["--help", "-h"]):
        var a = Args(["x", "run", spelling], ["model"], [])
        assert_true(a.has("help"))
        assert_equal(len(a.unknown), 0)
        assert_equal(len(a.positional), 0)


def test_help_after_a_value_flag_is_that_flags_value() raises:
    var a = Args(["x", "run", "--prompt", "--help"], ["prompt"], [])
    assert_equal(a.get("prompt"), "--help")
    assert_true(not a.has("help"))


def test_edit_distance() raises:
    assert_equal(edit_distance("steps", "steps"), 0)
    assert_equal(edit_distance("thread", "threads"), 1)
    assert_equal(edit_distance("stpes", "steps"), 2)
    assert_equal(edit_distance("", "abc"), 3)
    assert_equal(edit_distance("abc", ""), 3)
    assert_equal(edit_distance("kitten", "sitting"), 3)


def test_suggest_flag_distance_one_and_two() raises:
    var declared = List[String](["model", "steps", "threads", "device"])
    assert_equal(suggest_flag("thread", declared), "threads")
    assert_equal(suggest_flag("stpes", declared), "steps")


def test_suggest_flag_distance_three_gives_nothing() raises:
    var declared = List[String](["model", "steps", "threads", "device"])
    assert_equal(edit_distance("stepsabc", "steps"), 3)
    assert_equal(suggest_flag("stepsabc", declared), "")
    assert_equal(suggest_flag("zzzzzz", declared), "")


def test_suggest_flag_prefers_the_closest_then_the_first() raises:
    assert_equal(suggest_flag("seed", ["seedxx", "seeds"]), "seeds")
    assert_equal(suggest_flag("ab", ["ac", "ad"]), "ac")


def test_unknown_flag_error_with_suggestion() raises:
    assert_equal(
        unknown_flag_error("run", "stpes", ["model", "steps"]),
        (
            "unknown flag --stpes for 'run' (did you mean --steps?)\nsee:"
            " mojo-dllm run --help"
        ),
    )


def test_unknown_flag_error_without_suggestion() raises:
    assert_equal(
        unknown_flag_error("run", "zzzzzz", ["model", "steps"]),
        "unknown flag --zzzzzz for 'run'\nsee: mojo-dllm run --help",
    )


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
