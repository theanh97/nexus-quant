#!/bin/bash
# R84: Daily Production Cron Job
# ================================
# Runs the full production pipeline: signal → position → alerts
# Add to crontab: 15 0 * * * /path/to/scripts/r84_daily_cron.sh
#
# Runs at 00:15 UTC daily (after midnight when Deribit closes daily candle)

QUANT_DIR="/Users/truonglys/projects/quant"
PYTHON="/usr/local/opt/python@3.14/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG_DIR="$QUANT_DIR/data/cache/deribit/real_surface/logs"
DATE=$(date -u +%Y-%m-%d)

# Ensure log directory exists
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/daily_${DATE}.log"

echo "========================================" >> "$LOG_FILE"
echo "  Daily Production Run: $DATE" >> "$LOG_FILE"
echo "  Started: $(date -u)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Step 1: Generate daily signal (R81)
echo "" >> "$LOG_FILE"
echo "  [1/3] Running R81 daily signal runner..." >> "$LOG_FILE"
cd "$QUANT_DIR"
$PYTHON scripts/r81_daily_production_runner.py >> "$LOG_FILE" 2>&1 || echo "  [1/3] WARNING: signal runner failed" >> "$LOG_FILE"
echo "  [1/3] Signal generation complete." >> "$LOG_FILE"

# Step 2: Update position tracker (R82)
echo "" >> "$LOG_FILE"
echo "  [2/3] Running R82 position tracker..." >> "$LOG_FILE"
$PYTHON scripts/r82_position_tracker.py >> "$LOG_FILE" 2>&1 || echo "  [2/3] WARNING: position tracker failed" >> "$LOG_FILE"
echo "  [2/3] Position update complete." >> "$LOG_FILE"

# Step 3: Check alerts (R83) — exit code indicates alert level
echo "" >> "$LOG_FILE"
echo "  [3/3] Running R83 alert system..." >> "$LOG_FILE"
$PYTHON scripts/r83_alert_system.py >> "$LOG_FILE" 2>&1
ALERT_EXIT=$?  # 0=OK, 1=WARNING, 2=CRITICAL
echo "  [3/3] Alert check complete (exit=$ALERT_EXIT)." >> "$LOG_FILE"

# Summary
echo "" >> "$LOG_FILE"
echo "  Completed: $(date -u)" >> "$LOG_FILE"
if [ $ALERT_EXIT -eq 2 ]; then
    echo "  STATUS: CRITICAL ALERT(S) — check alerts.jsonl" >> "$LOG_FILE"
elif [ $ALERT_EXIT -eq 1 ]; then
    echo "  STATUS: WARNING(S) — check alerts.jsonl" >> "$LOG_FILE"
else
    echo "  STATUS: ALL CLEAR" >> "$LOG_FILE"
fi
echo "========================================" >> "$LOG_FILE"
