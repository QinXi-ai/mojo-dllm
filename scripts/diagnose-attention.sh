#!/usr/bin/env bash
# One-shot #73 diagnosis. Each allocation strategy gets its own executable/process.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MOJO="${MOJO:-uv run --frozen mojo}"
out="$(mktemp -d "${TMPDIR:-/tmp}/mojo-attention-probe.XXXXXX")" || exit 1
printf 'Diagnostic output: %s\n' "$out"
exec > >(tee "$out/report.log") 2>&1
printf 'Mojo attention diagnostic for #73 / #149\n'
git rev-parse HEAD 2>/dev/null || true
if ! nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version --format=csv > "$out/environment.txt" 2>&1; then
    cat "$out/environment.txt"
    echo 'ERROR: NVIDIA GPU/driver unavailable; no cases ran.'
    exit 1
fi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}" >> "$out/environment.txt"
# shellcheck disable=SC2086  # MOJO intentionally supports a command with arguments.
if ! $MOJO --version >> "$out/environment.txt" 2>&1; then
    cat "$out/environment.txt"
    echo 'ERROR: pinned compiler unavailable; no cases ran.'
    exit 1
fi
cat "$out/environment.txt"
export MODULAR_DEBUG=device-sync-mode
ulimit -c 0  # Failed launches must not create large core dumps.
printf 'case\tbuild_exit\trun_exit\n' > "$out/summary.tsv"
failed=0
for case_name in static68 static35 dynamic68; do
    mkdir -p "$out/$case_name"
    printf '\nCASE %s BUILD\n' "$case_name"
    MOJO="$MOJO" bash scripts/build.sh "bench/attention_probe/$case_name.mojo" \
        "$out/$case_name/probe" > "$out/$case_name/build.log" 2>&1
    build_rc=$?
    cat "$out/$case_name/build.log"
    if [ "$build_rc" -ne 0 ]; then
        printf '%s\t%s\tnot_run\n' "$case_name" "$build_rc" >> "$out/summary.tsv"
        failed=1
        continue
    fi
    printf 'CASE %s RUN (device-sync-mode; full log retained)\n' "$case_name"
    # Keep compiler PTX dumps with this case. A rejected kernel stays isolated.
    (cd "$out/$case_name" && timeout 120s stdbuf -oL -eL ./probe) \
        > "$out/$case_name/run.log" 2>&1
    run_rc=$?
    cat "$out/$case_name/run.log"
    if [ "$run_rc" -eq 0 ] && ! grep -qx "PASS $case_name" "$out/$case_name/run.log"; then
        echo 'ERROR: process exited 0 without completing the numeric checks.'
        run_rc=65
    fi
    printf '%s\t%s\t%s\n' "$case_name" "$build_rc" "$run_rc" >> "$out/summary.tsv"
    if [ "$run_rc" -ne 0 ]; then
        failed=1
    fi
done
printf '\nSUMMARY (nonzero static68 is useful evidence, not a reason to skip the other cases)\n'
cat "$out/summary.tsv"
if ! tar --exclude=probe -czf "$out/diagnostic-logs.tar.gz" -C "$out" \
    environment.txt summary.tsv static68 static35 dynamic68; then
    echo 'ERROR: log archive failed; the individual logs remain available.'
    failed=1
fi
printf '\nPlease return %s/diagnostic-logs.tar.gz (logs/PTX only, no binaries).\n' "$out"
printf 'Exit status: %s (1 means at least one diagnostic case failed)\n' "$failed"
exit "$failed"
