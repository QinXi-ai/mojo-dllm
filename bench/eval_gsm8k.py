#!/usr/bin/env python3
"""Paired GSM8K evaluation: does a sampler or cache change cost accuracy?

usage:
  python3 bench/eval_gsm8k.py run --model M.gguf --arm NAME [--device gpu] [--gen-length 256]
                                  [--block-length 32] [--steps N] [--limit K] [-- extra mojo-dllm flags]
  python3 bench/eval_gsm8k.py compare ARM_A ARM_B
  python3 bench/eval_gsm8k.py --self-test

`run` answers the 200 problems in bench/eval/gsm8k-200.jsonl with `mojo-dllm
run --json` and appends one line per problem to bench/eval/runs/ARM.jsonl; a
stopped run resumes where it left off. The default is the clean baseline: one
token per denoising step (steps = generation length), greedy, LLaDA blocks of
32. Scoring takes the number after "The answer is", else the last number in
the output, and compares it with the gold answer.

`compare` pairs two arms problem by problem and writes
bench/eval/DATE-ARM_A-vs-ARM_B.json: both accuracies, the difference with a
95% confidence interval, McNemar's exact test on the discordant problems, and
the forward passes and seconds each arm spent. With n = 200 the interval is
roughly +/- 3 points, so a smaller change is "not detectable", not "zero".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "eval" / "gsm8k-200.jsonl"
RUNS = ROOT / "bench" / "eval" / "runs"
PROMPT = (
    "{question}\n\nSolve the problem step by step. End with a line of the form "
    '"The answer is N." where N is the final number.'
)
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract(text: str) -> str | None:
    """The model's final answer as a canonical number string, or None."""
    m = re.findall(r"answer is\s*:?\s*\$?\s*(" + NUMBER + ")", text, flags=re.IGNORECASE)
    if not m:
        m = re.findall(NUMBER, text)
    if not m:
        return None
    return canon(m[-1])


def canon(s: str) -> str | None:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(v)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b vs c at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_ci(b: int, c: int, n: int) -> tuple[float, float, float]:
    """Accuracy difference (B - A) and its 95% Wald interval for paired outcomes."""
    d = (c - b) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return d, d - 1.96 * se, d + 1.96 * se


def cmd_run(args: argparse.Namespace, extra: list[str]) -> int:
    problems = [json.loads(l) for l in DATA.read_text().splitlines()]
    if args.limit:
        problems = problems[: args.limit]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.arm}.jsonl"
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            done.add(json.loads(l)["index"])
    steps = args.steps or args.gen_length
    flags = ["--max-tokens", str(args.gen_length), "--block-length", str(args.block_length),
             "--steps", str(steps), "--device", args.device, "--json"] + extra
    version = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    version += "-dirty" if dirty else ""
    for k, p in enumerate(problems):
        if p["index"] in done:
            continue
        cmd = [str(ROOT / "build/mojo-dllm"), "run", "--model", args.model,
               "--prompt", PROMPT.format(question=p["question"])] + flags
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
        if r.returncode != 0 or line is None:
            raise SystemExit(f"problem {p['index']}: mojo-dllm failed (rc={r.returncode}): {r.stderr[-400:]}")
        j = json.loads(line)
        pred = extract(j["text"])
        rec = dict(index=p["index"], gold=p["answer"], pred=pred, correct=pred == canon(p["answer"]),
                   forward_passes=j["forward_passes"], generate_s=j["generate_s"],
                   stopped_early=j.get("stopped_early"), text=j["text"], flags=flags, version=version)
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{k + 1}/{len(problems)}] #{p['index']} {'ok ' if rec['correct'] else 'BAD'} "
              f"pred={pred} gold={p['answer']} passes={j['forward_passes']} {j['generate_s']:.1f}s", flush=True)
    return 0


def load_arm(name: str) -> dict[int, dict]:
    path = RUNS / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"no runs for arm {name!r} at {path}")
    return {r["index"]: r for r in map(json.loads, path.read_text().splitlines())}


def cmd_compare(a_name: str, b_name: str) -> int:
    A, B = load_arm(a_name), load_arm(b_name)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("the two arms share no problem")
    n = len(common)
    b = sum(1 for i in common if A[i]["correct"] and not B[i]["correct"])
    c = sum(1 for i in common if not A[i]["correct"] and B[i]["correct"])
    d, lo, hi = paired_ci(b, c, n)
    versions = sorted({A[i]["version"] for i in common} | {B[i]["version"] for i in common})
    doc = dict(
        date=dt.datetime.now().isoformat(timespec="seconds"), problems=n, arms=[a_name, b_name],
        versions=versions,
        accuracy={a_name: sum(A[i]["correct"] for i in common) / n, b_name: sum(B[i]["correct"] for i in common) / n},
        difference=d, ci95=[lo, hi], discordant={"a_only": b, "b_only": c}, mcnemar_p=mcnemar_exact(b, c),
        forward_passes_mean={a_name: sum(A[i]["forward_passes"] for i in common) / n,
                             b_name: sum(B[i]["forward_passes"] for i in common) / n},
        generate_s_mean={a_name: sum(A[i]["generate_s"] for i in common) / n,
                         b_name: sum(B[i]["generate_s"] for i in common) / n},
        flags={a_name: A[common[0]]["flags"], b_name: B[common[0]]["flags"]},
    )
    out = ROOT / "bench" / "eval" / f"{dt.date.today().isoformat()}-{a_name}-vs-{b_name}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("problems", "accuracy", "difference", "ci95", "discordant",
                                           "mcnemar_p", "forward_passes_mean", "generate_s_mean")}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def self_test() -> int:
    cases = [
        ("so 5 + 3 = 8. The answer is 8.", "8"),
        ("The answer is $1,250.", "1250"),
        ("the answer is: 12.50", "12.5"),
        ("first 3, then 7, finally 42", "42"),
        ("The answer is 18. Then I add 2", "18"),
        ("no number here", None),
        ("The answer is -4.", "-4"),
    ]
    bad = [(t, want, extract(t)) for t, want in cases if extract(t) != want]
    assert canon("18.00") == "18" and canon("1,000") == "1000", "canon"
    assert abs(mcnemar_exact(0, 10) - 2 / 1024) < 1e-12, "mcnemar 0/10"
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0, "mcnemar ties"
    d, lo, hi = paired_ci(2, 6, 200)
    assert abs(d - 0.02) < 1e-12 and lo < d < hi, "paired ci"
    lines = [json.loads(l) for l in DATA.read_text().splitlines()]
    assert len(lines) == 200 and all(canon(r["answer"]) is not None for r in lines), "dataset"
    if bad:
        for t, want, got in bad:
            print(f"extract({t!r}) = {got!r}, want {want!r}")
        return 1
    print(f"eval_gsm8k self-test: {len(cases)} extraction cases, statistics and dataset ok")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--device", default="gpu")
    r.add_argument("--gen-length", type=int, default=256)
    r.add_argument("--block-length", type=int, default=32)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args, extra)
    return cmd_compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
