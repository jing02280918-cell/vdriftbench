#!/usr/bin/env bash
# auto_annotate_v33.sh — 监控 v3.3 实验完成并自动标注
set -euo pipefail

RESULTS_FILE="${RESULTS_FILE:-data/v3.3_run/results.jsonl}"
ENRICHED="${ENRICHED:-data/dataset_100.enriched.jsonl}"
OUT_ANNOTATED="${OUT_ANNOTATED:-data/v3.3_run/dataset_100.annotated.jsonl}"
OUT_FAILED="${OUT_FAILED:-data/v3.3_run/failed_samples.jsonl}"
TARGET_COUNT="${TARGET_COUNT:-314}"
INTERVAL="${1:-60}"

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"

echo "[auto_annotate_v33] 开始监控 $RESULTS_FILE"
echo "[auto_annotate_v33] 目标样本数: $TARGET_COUNT, 检查间隔: ${INTERVAL}s"

while true; do
    if [ -f "$RESULTS_FILE" ]; then
        COUNT=$(wc -l < "$RESULTS_FILE" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] 已完成: $COUNT/$TARGET_COUNT"
        
        if ! pgrep -f "main.py.*v3.3_run" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] 实验进程已退出，开始标注..."
            break
        fi
        
        if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
            echo "[$(date '+%H:%M:%S')] 结果数已达 $TARGET_COUNT，等待30秒确保写入完成..."
            sleep 30
            break
        fi
    fi
    sleep "$INTERVAL"
done

echo ""
echo "============================================"
echo "[auto_annotate_v33] 运行标注脚本..."
echo "============================================"
$PYTHON scripts/annotate_results.py \
    --results "$RESULTS_FILE" \
    --enriched "$ENRICHED" \
    --out-annotated "$OUT_ANNOTATED" \
    --out-failed "$OUT_FAILED"

echo ""
echo "[auto_annotate_v33] 标注完成！"
echo "  标注后数据集: $OUT_ANNOTATED"
echo "  失败样本清单: $OUT_FAILED"
