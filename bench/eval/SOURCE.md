# GSM8K subset

`gsm8k-200.jsonl` holds 200 problems from the GSM8K test set (1,319
problems), chosen with `random.Random(20261012).sample(range(1319), 200)` and
sorted by index. Each line keeps the original index, the question, and the
final answer after `####` with thousands separators removed.

Source: `grade_school_math/data/test.jsonl` from
<https://github.com/openai/grade-school-math> (sha256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`).
Copyright (c) 2021 OpenAI, MIT License; see `LICENSE-gsm8k`.

sha256 of `gsm8k-200.jsonl` itself: `362124d2d63cd55e306e6ddfbf5080f555dc7ec52aa707edca8cd46dadb319f0`.
