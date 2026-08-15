#!/usr/bin/env bash
# auto_annotate.sh — 监控 v3.2 实验完成并自动标注失败样本
# 用法: bash scripts/auto_annotate.sh [check_interval_seconds]
set -euo pipefail

RESULTS_FILE="${RESULTS_FILE:-data/v3.2_run/results.jsonl}"
ENRICHED="${ENRICHED:-data/dataset_100.enriched.jsonl}"
OUT_ANNOTATED="${OUT_ANNOTATED:-data/v3.2_run/dataset_100.annotated.jsonl}"
OUT_FAILED="${OUT_FAILED:-data/v3.2_run/failed_samples.jsonl}"
TARGET_COUNT="${TARGET_COUNT:-314}"
INTERVAL="${1:-60}"  # 默认每60秒检查一次

PYTHON="${PYTHON:-/hy-tmp/venv/bin/python}"

echo "[auto_annotate] 开始监控 $RESULTS_FILE"
echo "[auto_annotate] 目标样本数: $TARGET_COUNT, 检查间隔: ${INTERVAL}s"
echo "[auto_annotate] 实验完成后将自动运行:"
echo "  $PYTHON scripts/annotate_results.py \\"
echo "    --results $RESULTS_FILE \\"
echo "    --enriched $ENRICHED \\"
echo "    --out-annotated $OUT_ANNOTATED \\"
echo "    --out-failed $OUT_FAILED"
echo ""

while true; do
    if [ -f "$RESULTS_FILE" ]; then
        COUNT=$(wc -l < "$RESULTS_FILE" 2>/dev/null || echo 0)
        echo "[$(date '+%H:%M:%S')] 已完成: $COUNT/$TARGET_COUNT"

        # 检查实验进程是否还活着
        if ! pgrep -f "main.py.*v3.2_run" > /dev/null 2>&1; then
            echo "[$(date '+%H:%M:%S')] 实验进程已退出，开始标注..."
            break
        fi

        # 如果结果数达到目标数，再等30秒让最后一条写盘
        if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
            echo "[$(date '+%H:%M:%S')] 结果数已达 $TARGET_COUNT，等待30秒确保写入完成..."
            sleep 30
            break
        fi
    fi
    sleep "$INTERVAL"
done

# 运行标注
echo ""
echo "============================================"
echo "[auto_annotate] 运行标注脚本..."
echo "============================================"
$PYTHON scripts/annotate_results.py \
    --results "$RESULTS_FILE" \
    --enriched "$ENRICHED" \
    --out-annotated "$OUT_ANNOTATED" \
    --out-failed "$OUT_FAILED"

echo ""
echo "[auto_annotate] 标注完成！"
echo "  标注后数据集: $OUT_ANNOTATED"
echo "  失败样本清单: $OUT_FAILED"
