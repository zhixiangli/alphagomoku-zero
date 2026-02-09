#!/usr/bin/env bash
# Profile the Gomoku trainer with py-spy to identify performance bottlenecks.
#
# Prerequisites:
#   pip install py-spy    (or: uv pip install py-spy)
#
# Usage:
#   bash scripts/profile.sh                        # default: 60 s, 100 Hz
#   bash scripts/profile.sh --duration 120         # custom duration
#   bash scripts/profile.sh --rate 200             # custom sample rate
#
# On macOS, py-spy needs elevated privileges. The script will automatically
# prompt for sudo when needed, preserving the resolved Python/py-spy paths.
#
# Outputs (written to profiling_output/<timestamp>/):
#   flamegraph.svg   – Flame graph (open in a browser)
#   speedscope.json  – Speedscope profile (https://www.speedscope.app)
#   raw.txt          – Collapsed stack traces with sample counts
#
# All extra arguments are forwarded to the trainer, e.g.:
#   bash scripts/profile.sh -- -rows 9 -columns 9 -n_in_row 4

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

# Resolve absolute paths *before* any sudo so they survive PATH resets.
PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" || {
    echo "Error: failed to resolve Python via uv. Is uv installed?" >&2
    exit 1
}
PYSPY="$(command -v py-spy)" || {
    echo "Error: py-spy not found. Install with: uv pip install py-spy" >&2
    exit 1
}

# On macOS, py-spy requires elevated privileges (SIP). Use sudo with the
# resolved absolute path so that the user's PATH doesn't need to be inherited.
SUDO=""
if [[ "$(uname)" == "Darwin" ]]; then
    SUDO="sudo"
    echo "Note: macOS detected — py-spy requires sudo for process inspection."
    echo "      You may be prompted for your password."
    echo ""
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR}/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

# Display trainer args safely (compatible with bash 3.2 empty arrays + set -u).
if [[ ${#TRAINER_ARGS[@]} -gt 0 ]]; then
    TRAINER_ARGS_DISPLAY="${TRAINER_ARGS[*]}"
else
    TRAINER_ARGS_DISPLAY="<none>"
fi

# Build the trainer command. Handle empty TRAINER_ARGS safely for bash 3.2.
TRAINER_CMD=("${PYTHON}" -m gomoku.trainer)
if [[ ${#TRAINER_ARGS[@]} -gt 0 ]]; then
    TRAINER_CMD+=("${TRAINER_ARGS[@]}")
fi

# Profile with each output format. Each pass launches a fresh trainer process
# so py-spy manages the child directly (avoids ptrace permission issues).
FORMATS=("flamegraph:flamegraph.svg" "speedscope:speedscope.json" "raw:raw.txt")
TOTAL=${#FORMATS[@]}
STEP=0

echo "=== py-spy profiling ==="
echo "  Python:       ${PYTHON}"
echo "  py-spy:       ${PYSPY}"
echo "  Duration:     ${DURATION}s per format (${TOTAL} formats)"
echo "  Sample rate:  ${RATE} Hz"
echo "  Output dir:   ${OUTDIR}"
echo "  Trainer args: ${TRAINER_ARGS_DISPLAY}"
echo ""

for entry in "${FORMATS[@]}"; do
    FORMAT="${entry%%:*}"
    FILENAME="${entry#*:}"
    STEP=$((STEP + 1))

    echo "[${STEP}/${TOTAL}] Recording ${FORMAT} → ${FILENAME} (${DURATION}s) ..."
    ${SUDO} "${PYSPY}" record \
        --output "${OUTDIR}/${FILENAME}" \
        --format "${FORMAT}" \
        --duration "${DURATION}" \
        --rate "${RATE}" \
        --subprocesses \
        -- "${TRAINER_CMD[@]}"
done

echo ""
echo "=== Done ==="
echo "Reports saved to ${OUTDIR}/"
echo "  flamegraph.svg   -> open in browser to visualize hot paths"
echo "  speedscope.json  -> upload to https://www.speedscope.app for interactive timeline"
echo "  raw.txt          -> collapsed stacks with sample counts (grep/sort to find hotspots)"
