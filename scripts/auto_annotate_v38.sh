#!/usr/bin/env bash
set -euo pipefail
RESULTS_FILE="${RESULTS_FILE:-data/v3.8_run/results.jsonl}"
ENRICHED="${ENRICHED:-data/dataset_100.enriched.jsonl}"
OUT_ANNOTATED="${OUT_ANNOTATED:-data/v3.8_run/dataset_100.annotated.jsonl}"
OUT_FAILED="${OUT_FAILED:-data/v3.8_run/failed_samples.jsonl}"
TARGET_COUNT="${TARGET_COUNT:-314}"
INTERVAL="${1:-60}"
PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"

echo "[v3.8 watchdog] monitoring $RESULTS_FILE, target=$TARGET_COUNT"

while true; do
    if [ -f "$RESULTS_FILE" ]; then
        COUNT=$(wc -l < "$RESULTS_FILE" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] $COUNT/$TARGET_COUNT"
        if ! pgrep -f "main.py.*v3.8_run" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] experiment exited, annotating..."
            break
        fi
        if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
            sleep 30
            break
        fi
    fi
    sleep "$INTERVAL"
done

$PYTHON scripts/annotate_results.py \
    --results "$RESULTS_FILE" \
    --enriched "$ENRICHED" \
    --out-annotated "$OUT_ANNOTATED" \
    --out-failed "$OUT_FAILED"
echo "[$(date '+%H:%M:%S')] done — $OUT_ANNOTATED"
