#!/usr/bin/env bash
# Watches the background full-dataset run started by main.py: prints a
# progress heartbeat every 5 minutes, and exits (with a distinct marker line)
# either when the run process exits, or when results.jsonl hasn't grown for
# longer than STALL_LIMIT seconds (e.g. a hung vLLM engine like we saw
# earlier, or a network stall against the DeepSeek API).
set -u
cd /root/vdriftbench

PID="$1"
OUTFILE="${2:-data/full_run/results.jsonl}"
TOTAL="${3:-295}"
STALL_LIMIT="${4:-1200}"

last_size=-1
last_change=$(date +%s)

while kill -0 "$PID" 2>/dev/null; do
    n=$(wc -l < "$OUTFILE" 2>/dev/null || echo 0)
    size=$(stat -c %s "$OUTFILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$size" != "$last_size" ]; then
        last_size=$size
        last_change=$now
    fi
    elapsed_since_change=$((now - last_change))
    echo "$(date -Iseconds) HEARTBEAT progress=${n}/${TOTAL} idle=${elapsed_since_change}s"
    if [ "$elapsed_since_change" -gt "$STALL_LIMIT" ]; then
        echo "STALLED: results.jsonl has not grown for ${elapsed_since_change}s (pid $PID still alive)"
        exit 1
    fi
    sleep 300
done

n=$(wc -l < "$OUTFILE" 2>/dev/null || echo 0)
if [ "$n" -ge "$TOTAL" ]; then
    echo "DONE_WATCHING: process exited, progress=${n}/${TOTAL} (complete)"
else
    echo "DONE_WATCHING: process exited EARLY, progress=${n}/${TOTAL} (check run.log for a traceback)"
fi
