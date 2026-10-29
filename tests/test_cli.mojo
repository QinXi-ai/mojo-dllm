from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import Args
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.sys.mmap import Mapping
from main import (
    boolean_flags,
    chat_prompt,
    check_args,
    command_spec,
    json_str,
    _csv,
    is_top_level_help,
    model_error,
    usage,
    usage_for,
)


def test_documented_booleans_are_parsed_without_values() raises:
    var flags = boolean_flags()
    var count = 0
    var doc: String
    with open("docs/cli.md", "r") as f:
        doc = f.read()
    for line in doc.split("\n"):
        if not line.startswith("| `--"):
            continue
        var cells = line.split("|")
        if String(cells[2].strip()) != "off":
            continue
        var spelling = String(cells[1].strip())
        var flag = String(spelling[byte = 3 : spelling.byte_length() - 1])
        var found = False
        for candidate in flags:
            if candidate == flag:
                found = True
        assert_true(found, "documented boolean missing: " + flag)
        var spec = command_spec("run")
        var a = Args(["x", "run", "--" + flag], spec.values, spec.booleans)
        assert_true(a.has(flag))
        assert_equal(len(a.unknown), 0)
        count += 1
    assert_equal(count, 7)
    assert_equal(len(flags), count)


def test_chat_templates_strip_user_text() raises:
    var llada = "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    var dream = (
        "<|im_start|>system\nYou are a helpful"
        " assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert_equal(chat_prompt("llada", " \thi\n"), llada)
    assert_equal(chat_prompt("dream", " \thi\n"), dream)
    assert_equal(chat_prompt("unknown", "hi"), llada)


def test_json_escapes_and_preserves_unicode() raises:
    assert_equal(json_str('"\\\n\r\t'), '"\\"\\\\\\n\\r\\t"')
    assert_equal(json_str("\x00\x08\x0b\x1f"), '"\\u0000\\u0008\\u000b\\u001f"')
    assert_equal(json_str("héllo 世界"), '"héllo 世界"')
    assert_equal(json_str(""), '""')


def test_csv_format() raises:
    assert_equal(_csv([]), "")
    assert_equal(_csv([7]), "7")
    assert_equal(_csv([1, -2, 3]), "1,-2,3")


def test_usage_lists_all_commands() raises:
    var text = usage()
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        assert_true(text.__contains__("mojo-dllm " + command + " "))


def _parse(command: String, rest: List[String]) raises -> Args:
    var spec = command_spec(command)
    var argv = List[String](["mojo-dllm", command])
    for r in rest:
        argv.append(r)
    return Args(argv, spec.values, spec.booleans)


def _check(command: String, rest: List[String]) raises -> String:
    return check_args(_parse(command, rest), command_spec(command))


def test_every_command_declares_its_flags() raises:
    var run = command_spec("run")
    assert_equal(len(run.values), 10)
    assert_equal(len(run.booleans), 7)
    var tokenize = command_spec("tokenize")
    assert_equal(len(tokenize.values), 2)
    assert_equal(len(tokenize.booleans), 1)
    var inspect = command_spec("inspect")
    assert_equal(len(inspect.values), 0)
    assert_equal(len(inspect.booleans), 0)
    assert_true(inspect.takes_path)
    var logits = command_spec("logits")
    assert_equal(len(logits.values), 6)
    assert_equal(len(logits.booleans), 0)
    with assert_raises(contains="unknown command: frobnicate"):
        _ = command_spec("frobnicate")


def test_declared_booleans_come_from_the_single_list() raises:
    var all_booleans = boolean_flags()
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        for flag in command_spec(command).booleans:
            var found = False
            for candidate in all_booleans:
                if candidate == flag:
                    found = True
            assert_true(found, command + " declares a boolean not in the list")


def test_a_flag_of_another_command_is_unknown() raises:
    var message = _check(
        "tokenize", ["--model", "m", "--prompt", "p", "--json"]
    )
    assert_equal(
        message,
        "unknown flag --json for 'tokenize'\nsee: mojo-dllm tokenize --help",
    )


def test_misspelled_flag_suggests_the_declared_one() raises:
    var message = _check("run", ["--stpes", "3"])
    assert_equal(
        message,
        (
            "unknown flag --stpes for 'run' (did you mean --steps?)\nsee:"
            " mojo-dllm run --help"
        ),
    )


def test_unknown_flag_is_reported_before_a_missing_one() raises:
    var message = _check("run", ["--stpes", "3"])
    assert_true(not message.__contains__("missing required"))


def test_missing_required_flag_prints_usage() raises:
    var message = _check("run", ["--prompt", "x"])
    assert_equal(
        message,
        "missing required flag --model for 'run'\n" + usage_for("run"),
    )
    assert_true(message.__contains__("usage:\n  mojo-dllm run --model M"))
    var second = _check("run", ["--model", "m"])
    assert_true(second.startswith("missing required flag --prompt for 'run'\n"))
    var logits = _check(
        "logits", ["--model", "m", "--tokens", "1", "--rows", "0"]
    )
    assert_true(logits.startswith("missing required flag --out for 'logits'\n"))


def test_inspect_needs_a_path_and_nothing_else() raises:
    var missing = _check("inspect", [])
    assert_equal(
        missing, "missing model path for 'inspect'\n" + usage_for("inspect")
    )
    assert_equal(_check("inspect", ["m.gguf"]), "")
    var extra = _check("run", ["--model", "m", "--prompt", "p", "stray"])
    assert_equal(
        extra,
        "unexpected argument 'stray' for 'run'\nsee: mojo-dllm run --help",
    )


def test_complete_commands_pass_the_check() raises:
    assert_equal(
        _check(
            "run", ["--model", "m", "--prompt", "p", "--json", "--steps", "4"]
        ),
        "",
    )
    assert_equal(_check("tokenize", ["--model", "m", "--prompt", "p"]), "")
    assert_equal(
        _check(
            "logits",
            ["--model", "m", "--tokens", "1", "--rows", "0", "--out", "o"],
        ),
        "",
    )


def test_usage_for_one_command() raises:
    var run = usage_for("run")
    assert_true(
        run.startswith("usage:\n  mojo-dllm run --model M --prompt TEXT")
    )
    assert_true(not run.__contains__("mojo-dllm tokenize"))
    assert_true(not run.endswith("\n"))
    var tokenize = usage_for("tokenize")
    assert_equal(
        tokenize,
        "usage:\n  mojo-dllm tokenize --model M --prompt TEXT [--no-chat]",
    )
    with assert_raises(contains="unknown command: nope"):
        _ = usage_for("nope")


def test_usage_for_names_exactly_the_declared_flags() raises:
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        var spec = command_spec(command)
        var text = usage_for(command)
        var declared = spec.declared()
        for flag in declared:
            assert_true(
                text.__contains__("--" + flag),
                command + " usage omits --" + flag,
            )
        var cleaned = (
            text.replace("\n", " ").replace("]", " ").replace("|", " ")
        )
        for token in cleaned.split(" "):
            var word = String(token)
            if not word.startswith("--"):
                continue
            var name = String(word[byte=2:])
            var found = False
            for flag in declared:
                if flag == name:
                    found = True
            assert_true(found, command + " usage names undeclared --" + name)


def test_docs_table_lists_only_declared_run_flags() raises:
    var doc: String
    with open("docs/cli.md", "r") as f:
        doc = f.read()
    var declared = command_spec("run").declared()
    var seen = 0
    for line in doc.split("\n"):
        if not line.startswith("| `--"):
            continue
        var cell = String(line.split("|")[1].strip()).replace("`", "")
        var name = String(String(cell.split(" ")[0])[byte=2:])
        var found = False
        for flag in declared:
            if flag == name:
                found = True
        assert_true(found, "docs/cli.md lists undeclared flag: " + name)
        seen += 1
    assert_equal(seen, len(declared))


def test_help_routing() raises:
    for command in List[String](["", "help", "--help", "-h"]):
        assert_true(is_top_level_help(command))
    for command in List[String](["run", "tokenize", "frobnicate"]):
        assert_true(not is_top_level_help(command))
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        assert_true(_parse(command, ["--help"]).has("help"))
        assert_true(_parse(command, ["-h"]).has("help"))
    # help wins over every other problem, because check_args runs after it.
    var a = _parse("run", ["--stpes", "--help"])
    assert_true(a.has("help"))


def _mapping_error(path: String) -> String:
    try:
        _ = Mapping(path)
    except e:
        return model_error(String(e))
    return String("opened")


def test_missing_model_file_hint() raises:
    var path = "/nonexistent/model.gguf"
    assert_equal(
        _mapping_error(path),
        "cannot open model file "
        + path
        + ": No such file or directory\nto download a model, run: ./mojo-dllm"
        " pull",
    )


def test_missing_model_file_hint_with_quote_colon_in_path() raises:
    var path = "/nonexistent/x': y"
    assert_equal(
        _mapping_error(path),
        "cannot open model file "
        + path
        + ": No such file or directory\nto download a model, run: ./mojo-dllm"
        " pull",
    )


def test_other_open_failures_get_no_pull_hint() raises:
    # ENOTDIR: a path below a regular file. EACCES takes the same branch.
    var path = "tests/fixtures/bad-magic.gguf/x"
    assert_equal(
        _mapping_error(path),
        "cannot open model file " + path + ": Not a directory",
    )
    assert_equal(
        model_error("cannot open file: m.gguf: Permission denied"),
        "cannot open model file m.gguf: Permission denied",
    )


def test_non_gguf_file_hint() raises:
    var path = "build/tests/not-a-gguf.gguf"
    with open(path, "w") as f:
        f.write("<html>404 Not Found</html>")
    var message = String("")
    try:
        _ = GGUFFile(path)
    except e:
        message = model_error(String(e))
    remove(path)
    assert_equal(
        message,
        path
        + " is not a GGUF file (bad magic).\na failed download leaves an HTML"
        " error page with a .gguf name; delete the file and run ./mojo-dllm"
        " pull again",
    )


def test_other_errors_pass_through_unchanged() raises:
    assert_equal(
        model_error("unsupported model architecture: qwen2"),
        "unsupported model architecture: qwen2",
    )
    assert_equal(
        model_error("file is empty or unreadable: m.gguf"),
        "file is empty or unreadable: m.gguf",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
