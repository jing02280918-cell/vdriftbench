#!/usr/bin/env bash
# Polls the full-run experiment process; emits a clear sentinel line only for
# things that actually matter (process died early, or the run finished),
# ignoring expected per-sample skip Tracebacks (already handled gracefully by
# pipeline.py's run_dataset try/except).
PID=266438
OUT=data/full_run/results.jsonl
TOTAL=314
while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    N=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if [ "$N" -ge "$TOTAL" ]; then
      echo "SENTINEL_RUN_FINISHED n=$N"
    else
      echo "SENTINEL_RUN_DIED_EARLY n=$N/$TOTAL"
    fi
    break
  fi
  N=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "SENTINEL_HEARTBEAT n=$N/$TOTAL $(date +%H:%M:%S)"
  sleep 300
done
