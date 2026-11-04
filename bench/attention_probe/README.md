# Ampere attention launch diagnostic

This is a standalone diagnostic for the long-GQA test failure reported on
[PR #73](https://github.com/vladimirrott/mojo-dllm/pull/73#issuecomment-6628295161)
and reverted by [PR #149](https://github.com/vladimirrott/mojo-dllm/pull/149).
It does not restore the optimization or change production dispatch.

From this checkout, with the repository's pinned Mojo environment available:

```bash
bash scripts/diagnose-attention.sh
```

The default compiler command is `uv run --frozen mojo`. The existing `MOJO` and
`MOJO_DLLM_GPU_ARCH` overrides also work. The runner needs Bash, Git, NVIDIA's
driver tools, GNU coreutils (`timeout`, `stdbuf`, `mktemp`) and `tar`.
No model download or fixture generation is needed.

Each case is compiled into a separate executable and run in a separate process
with `MODULAR_DEBUG=device-sync-mode` and a 120-second execution timeout:

| Case | Key/query tile | Shared allocation | Launch attribute |
| --- | --- | --- | --- |
| `static68` | 128 / 128 | 68 KiB static | none, reproducing #73 |
| `static35` | 64 / 128 | 35 KiB static | none |
| `dynamic68` | 128 / 128 | 68 KiB dynamic | explicit `MAX_DYNAMIC_SHARED_SIZE_BYTES` |

`kernel.mojo` copies the arithmetic from #73 at `20f46c0`. Only the kernel name,
template parameters and allocation block differ. Keeping the diagnostic kernel
here lets the production source stay at the reverted version. The dynamic case
uses one aligned allocation with distinct K and V regions, and supplies both
`shared_mem_bytes` and the function attribute at launch.

All cases use the failing test's seeded 513-token inputs, four query heads, one
KV head and head dimension 128. Each first runs the original production kernel
and checks its relative error against CPU attention. It then launches the
candidate and reports element mismatches, bit mismatches, non-finite values,
maximum absolute difference, and relative CPU error. Success requires the same
numeric equality and CPU tolerance as the original GPU test; bit mismatches
are additional diagnostic output. Stage markers distinguish context creation,
baseline launch, candidate launch/synchronization and numerical comparison.

The final summary preserves build and execution exit codes, including failures.
A rejected kernel does not prevent the other cases from running. Exit status 1
means at least one case failed, which is expected if `static68` reproduces the
reported problem. Exit 124 for a case means its execution timed out.

The runner prints the path to `diagnostic-logs.tar.gz`. Please attach that one
file: it contains device/compiler information, the summary, complete logs, and
any PTX dumps produced before a failure. Executables are excluded. All output
stays in the printed temporary directory; the checkout is not modified by the
diagnostic. `uv` may initialize the checkout's pinned environment on first use.

## Validation before hardware reproduction

With pinned Mojo 1.1.0 (`8189361e`), all three executables compile with `-Werror`
for both `sm_80` and `sm_86`. The embedded PTX declares:

- `static68`: two 34,816-byte static arrays (69,632 bytes total).
- `static35`: 17,408- and 18,432-byte static arrays (35,840 bytes total).
- `dynamic68`: external dynamic shared memory and no static shared arrays.

The copied arithmetic was checked against #73, and Bash syntax, ShellCheck and
simulated runner failure/continuation cases passed. Actual Ampere execution and
the cause of the original failure remain to be established by this diagnostic.
