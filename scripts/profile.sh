#!/usr/bin/env bash
# Profile the Gomoku trainer with py-spy.
#
# Usage:
#   bash scripts/profile.sh                        # default: 60 s, 100 Hz
#   bash scripts/profile.sh --duration 120         # custom duration
#   bash scripts/profile.sh -- -rows 9 -columns 9  # extra args for trainer

set -euo pipefail

DURATION="${DURATION:-60}"
RATE="${RATE:-100}"
OUTDIR="${OUTDIR:-profiling_output}"

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

PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

# Install py-spy if missing.
command -v py-spy &>/dev/null || uv pip install py-spy
PYSPY="$(command -v py-spy)"

OUTDIR="${OUTDIR}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTDIR}"

FORMATS=("flamegraph:flamegraph.svg" "speedscope:speedscope.json" "raw:raw.txt")

for entry in "${FORMATS[@]}"; do
    FORMAT="${entry%%:*}"
    FILENAME="${entry#*:}"
    echo "Recording ${FORMAT} → ${OUTDIR}/${FILENAME} ..."
    "${PYSPY}" record \
        --output "${OUTDIR}/${FILENAME}" \
        --format "${FORMAT}" \
        --duration "${DURATION}" \
        --rate "${RATE}" \
        --subprocesses \
        -- "${PYTHON}" -m gomoku_9_9.trainer ${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"}
done

echo "Done. Reports saved to ${OUTDIR}/"
