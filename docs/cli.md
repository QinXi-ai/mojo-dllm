# Command line

```text
mojo-dllm run      --model PATH --prompt TEXT [options]
mojo-dllm tokenize --model PATH --prompt TEXT [--no-chat]
mojo-dllm inspect  PATH
mojo-dllm logits   --model PATH --tokens IDS --rows POSITIONS --out FILE [--threads N] [--device D]
```

`mojo-dllm COMMAND --help` (or `-h`) prints that command's usage and exits 0.
`mojo-dllm`, `mojo-dllm help` and `mojo-dllm --help` print the usage of every
command. Each command declares its flags: a flag the command does not declare
stops the run (see [Errors](#errors)).

## `run`

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | LLaDA or Dream GGUF file |
| `--prompt TEXT` | required | the user message |
| `--max-tokens N` | 128 | tokens to generate (the masked canvas) |
| `--steps N` | `--max-tokens` | denoising steps; a multiple of the block count |
| `--block-length N` | 32 | LLaDA's semi-autoregressive block size; must divide `--max-tokens`. Dream ignores it |
| `--temperature T` | 0 | 0 is greedy; above 0 adds Gumbel noise as LLaDA does |
| `--remasking S` | `low_confidence` | LLaDA: `low_confidence` or `random`. Dream: `entropy` (the default), `maskgit_plus`, `topk_margin` |
| `--seed N` | 42 | seeds this program's PRNG (it does not reproduce a PyTorch run) |
| `--threads N` | all logical CPUs | worker threads (with `--device gpu`, only the sampler uses them) |
| `--device D` | `cpu` | `cpu`, or `gpu` for an NVIDIA GPU with compute capability 8.0 or newer |
| `--no-chat` | off | skip the chat template |
| `--full-logits` | off | compute logits for every position, as a generic runtime would (for benchmarking) |
| `--no-early-stop` | off | run every step even after the answer is final (the benchmark and parity harnesses use it) |
| `--visual` | off | redraw the canvas after every step: masked positions as teal blocks, tokens committed in that step in bold orange |
| `--verbose` | off | timings and memory after the text |
| `--dump-step-stats` | off | one line per denoising step |
| `--json` | off | one JSON object with timings, tokens and text |

The printed answer ends at the first end-of-text or end-of-turn token in the
canvas. Generation stops as soon as that answer is final: once the first
position after the prompt that is still masked or holds such a token holds
such a token, no later step can change the printed text, because steps only
fill masked positions. The result is the same text with fewer steps, and
`--json` reports `"stopped_early": true`. `--no-early-stop` runs every step.

## `tokenize`

Prints the token ids of the chat-formatted prompt, the same ids `run` feeds the
model. Useful to compare against `llama-tokenize`.

## `inspect`

Prints GGUF metadata, a count per tensor type, and every tensor's type and
shape. It needs no supported architecture, so it works on any GGUF file.

## `logits`

Runs one forward pass over the given token ids and writes f32 logits for the
requested positions to a file, `[positions, vocab]`, row major. The parity
tests use it; see [Correctness](correctness.md).

## Errors

An error names the thing that is wrong. Usage errors exit with status 2 and
change nothing; every other error exits with status 1.

| Message | Cause and next step |
|---|---|
| `ERROR: unknown flag --stpes for 'run' (did you mean --steps?)` | The command does not declare the flag. The parenthesis appears when a declared flag is at most 2 edits away. The next line is `see: mojo-dllm run --help` |
| `ERROR: missing required flag --model for 'run'` | A required flag is absent. The usage of the command follows |
| `ERROR: missing model path for 'inspect'` | `inspect` takes the path as an argument. The usage follows |
| `ERROR: unexpected argument 'x' for 'run'` | `run`, `tokenize` and `logits` take flags only |
| `ERROR: flag --prompt needs a value` | A value flag is the last argument |
| `ERROR: cannot open model file PATH: No such file or directory` | The path is wrong. The reason is the operating system's. The next line is `to download a model, run: ./mojo-dllm pull` |
| `ERROR: cannot open model file PATH: Permission denied` | The open failed for a reason other than a missing file, such as `Not a directory`. The operating system names the reason. No download hint follows, because `pull` would not fix it |
| `ERROR: is a directory: PATH` | The model path names a directory |
| `ERROR: PATH is not a GGUF file (bad magic).` | A failed download leaves an HTML error page with a `.gguf` name. Delete the file and run `./mojo-dllm pull` again |

The remaining errors exit with status 1 and come from the model file itself:

```text
ERROR: unsupported GGUF tensor type: IQ2_XXS
ERROR: unsupported model architecture: qwen2
ERROR: required metadata key missing: tokenizer.ggml.model
ERROR: tensor shape mismatch for blk.0.attn_q.weight: [4096, 2048], expected [4096, 4096]
ERROR: no GPU found: the GPU backend needs an NVIDIA GPU with compute capability 8.0 or newer (run with --device cpu)
```

With `--device gpu`, setting `MOJO_DLLM_GPU_PROFILE=1` makes `--verbose` split
a step into GEMM, attention and LM head time. It synchronizes the GPU after
each kernel group to measure them, so leave it off when you benchmark.
