# AGENTS.md

mojo-dllm runs diffusion language models (LLaDA, Dream) from GGUF files, in
Mojo. It is small on purpose and does one job: inference. This file lists
what the file tree does not show. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/development.md](docs/development.md) before you change code; both apply
to you.

## Setup

Run commands from the repository root. The project builds on Linux x86-64
only.

- `pyproject.toml` and `uv.lock` pin Mojo 1.1.0. `uv sync --frozen` installs
  it into `.venv`. The scripts call `uv run --frozen mojo`, and the
  environment variable `MOJO` replaces that command. Do not use a system
  `mojo`.
- `scripts/build.sh` builds `build/mojo-dllm` and needs no model file.
  `./mojo-dllm setup` asks before it downloads a 4.9 GB model (`--yes` skips
  the question), so run it only when the person you work for asks.
- `tests/fixtures/tiny-llada.gguf` and `tiny-dream.gguf` have no tokenizer.
  `build/mojo-dllm run` stops on them with
  `required metadata key missing: tokenizer.ggml.model`, so the tests call the
  model code without `run`.

## Verify before you say it works

- `scripts/ci-local.sh` runs the CI gates and keeps going after a failure.
  Its summary lists PASS, FAIL or SKIP for each step. A missing tool is a
  FAIL. A SKIP means the step checked nothing. Read the summary: a pipe after
  the script, such as `| tail`, can hide its exit code.
- `scripts/ci-local.sh --fast` runs the hygiene scan on staged files,
  `mojo format` and the `-Werror` build.
- `scripts/run-tests.sh` compiles each `tests/test_*.mojo` and compares the
  total of passing tests with `tests/test-count.txt` and the count of each
  file with `tests/test-counts.txt`. After you add or remove tests, run
  `UPDATE_TEST_COUNT=1 scripts/run-tests.sh` and commit both files.
- `scripts/run-gpu-tests.sh` runs `tests/gpu/` against
  `tests/gpu/test-count.txt` and `tests/gpu/test-counts.txt`, the same two
  pins. It needs an NVIDIA GPU with compute capability
  8.0 or newer and exits 1 without one. Say that you did not run it, and do
  not report it as passed.
- Launcher tests: `uv run --frozen --only-group test pytest tests/python -q`.
- `mojo format` has no check mode. The gate runs
  `uv run --frozen mojo format -q src tests bench` and fails if that changes
  a file, so run it and commit the result.

## Rules the tree does not show

- A new guard or kernel needs proof that its test bites. Copy the file, break
  the code on purpose, watch the named test fail, then restore the copy.
  `git checkout -- <path>` also discards your other uncommitted edits to that
  file.
- Do not type a benchmark number into the README or `docs/`.
  `scripts/bench_table.py --write` renders the tables from `bench/results/`,
  `bench/parity/`, `bench/quality/` and `bench/eval/`, and CI fails on drift.
- Usage errors exit with status 2; every other error exits with status 1
  (`docs/cli.md`, "Errors").
- `bench/run_bench.py` kills any process named `pytest` while it runs (set
  `"kill_pytest": false` in its config to stop that), and on a laptop it
  pauses until you plug in the charger. Compare two builds with interleaved
  runs in one session.
- Six scripts parse the JSON object that `run --json` prints:
  `bench/parity.py`, `bench/eval_gsm8k.py`, `bench/check_gpu_dispatch.py`,
  `bench/run_bench.py`, `bench/quality.py` and `scripts/gpu_profile.py`. A key
  you rename or remove breaks the ones that read it.
- The quantized GEMM and the sampler decide the numbers this project
  publishes, so `.github/CODEOWNERS` assigns `src/mojo_dllm/kernels/`,
  `src/mojo_dllm/diffusion/` and `bench/` to the maintainer. A change that
  moves performance comes with a new result file.
- `scripts/check_hygiene.sh --tracked` refuses credential patterns in tracked
  files. Keep secrets out of files, command arguments and logs.
- Git ignores `scratch/`. Put logs and experiments there.

## Pull requests

- Open pull requests against `main`. The subject is imperative and under 72
  characters, and the body says why.
- Fill in the pull request template, `.github/pull_request_template.md`.
- Open an issue, a pull request or a review comment only when the person you
  work for asks you to in this session.
- If a gate fails and you cannot fix it, report the failing step and its last
  20 lines. Do not skip, mute or delete a check.
