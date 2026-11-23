#!/usr/bin/env bash
# P0 (cross-generation) then P1 (DINOv3 size x resolution sweep) on DAVIS 2017 val.
# Each run appends to results/summary.csv and results/per_sequence.csv. A failing run is
# logged and the batch continues.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python
LOG=logs/p0_p1.log
mkdir -p logs
run() {
  echo "### $(date +%H:%M:%S) start $*" >> "$LOG"
  "$PY" run_davis.py "$@" 2>&1 | grep -E "^==|OOM|Error|Traceback|\] [0-9]+/30" >> "$LOG"
  echo "### $(date +%H:%M:%S) done $*" >> "$LOG"
}
# P0
run --model v1-s16 --scale 2
run --model v2r-s14 --scale 1 2
run --model v3-s16 --scale 1 2
# P1
run --model v3-s16 --scale 1.5
run --model v3-s16plus --scale 1 1.5 2
run --model v3-b16 --scale 1 1.5 2
run --model v3-l16 --scale 1 1.5 2
echo "### ALL DONE $(date +%H:%M:%S)" >> "$LOG"
