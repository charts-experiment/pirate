#!/bin/bash
# Waits for a run to finish ("shutting down" appears in launcher.log) and evaluates it.
# Usage: 09_evaluation/evaluate_when_done.sh RUN_FOLDER
HERE=$(cd "$(dirname "$0")/.." && pwd); D=${1:?run folder}
until grep -qE "shutting down" "$D/launcher.log" 2>/dev/null; do sleep 30; done
sleep 90
PY=python; [ -n "$CHARTS_PIRATE_ENV" ] && PY=$CHARTS_PIRATE_ENV/bin/python   # or with pirate's environment activated
"$PY" "$HERE/09_evaluation/evaluate_test_64b.py" "$D" > "$D/evaluation.log" 2>&1
echo "evaluated: $D/results.md"
