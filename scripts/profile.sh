#!/usr/bin/env bash
# Profile the Gomoku trainer with py-spy to identify performance bottlenecks.
#
# Prerequisites:
#   pip install py-spy    (or: uv pip install py-spy)
#
# Usage:
#   sudo bash scripts/profile.sh                   # default: 60 s, 100 Hz
#   sudo bash scripts/profile.sh --duration 120    # custom duration
#   sudo bash scripts/profile.sh --rate 200        # custom sample rate
#
# Outputs (written to profiling_output/):
#   flamegraph.svg   – Flame graph (open in a browser)
#   speedscope.json  – Speedscope profile (https://www.speedscope.app)
#   top.txt          – py-spy top summary snapshot
#
# All extra arguments are forwarded to the trainer, e.g.:
#   sudo bash scripts/profile.sh -- -rows 9 -columns 9 -n_in_row 4

set -euo pipefail

# ---------- configurable defaults ----------
DURATION="${DURATION:-60}"
RATE="${RATE:-100}"
OUTDIR="${OUTDIR:-profiling_output}"
# -------------------------------------------

# Parse script-level flags; everything after "--" goes to the trainer.
TRAINER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        --rate)     RATE="$2";     shift 2 ;;
        --outdir)   OUTDIR="$2";   shift 2 ;;
        --)         shift; TRAINER_ARGS+=("$@"); break ;;
        *)          TRAINER_ARGS+=("$1"); shift ;;
    esac
done

PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" || {
    echo "Error: failed to resolve Python via uv. Is uv installed?" >&2
    exit 1
}
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR}/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

echo "=== py-spy profiling ==="
echo "  Python:       ${PYTHON}"
echo "  Duration:     ${DURATION}s"
echo "  Sample rate:  ${RATE} Hz"
echo "  Output dir:   ${OUTDIR}"
echo "  Trainer args: ${TRAINER_ARGS[*]:-<none>}"
echo ""

# 1. Flame graph (SVG) — best for finding the hottest code paths
echo "[1/3] Recording flame graph (${DURATION}s) ..."
py-spy record \
    --output "${OUTDIR}/flamegraph.svg" \
    --format flamegraph \
    --duration "${DURATION}" \
    --rate "${RATE}" \
    --subprocesses \
    -- "${PYTHON}" -m gomoku.trainer "${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"}"

# 2. Speedscope JSON — interactive timeline view at https://www.speedscope.app
echo "[2/3] Recording speedscope profile (${DURATION}s) ..."
py-spy record \
    --output "${OUTDIR}/speedscope.json" \
    --format speedscope \
    --duration "${DURATION}" \
    --rate "${RATE}" \
    --subprocesses \
    -- "${PYTHON}" -m gomoku.trainer "${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"}"

# 3. Top-like snapshot — quick text summary of where time is spent
echo "[3/3] Capturing top snapshot (${DURATION}s) ..."
TOP_EXIT=0
timeout "${DURATION}" py-spy top \
    --subprocesses \
    -- "${PYTHON}" -m gomoku.trainer "${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"}" \
    > "${OUTDIR}/top.txt" 2>&1 || TOP_EXIT=$?
# Exit code 124 means timeout expired (expected); anything else is a real error.
if [[ ${TOP_EXIT} -ne 0 && ${TOP_EXIT} -ne 124 ]]; then
    echo "Warning: py-spy top exited with code ${TOP_EXIT}" >&2
fi

echo ""
echo "=== Done ==="
echo "Reports saved to ${OUTDIR}/"
echo "  flamegraph.svg   -> open in browser to visualize hot paths"
echo "  speedscope.json  -> upload to https://www.speedscope.app for interactive timeline"
echo "  top.txt          -> quick text summary of most expensive functions"
