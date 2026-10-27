from std.testing import assert_equal, assert_true, TestSuite

from cli.args import Args
from main import boolean_flags, chat_prompt, json_str, _csv, usage


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
        var a = Args(["x", "run", "--" + flag], boolean_flags())
        assert_true(a.has(flag))
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
