#!/bin/bash
# NEXUS Autonomous Loop Starter
# Runs: Dashboard + Brain Loop + Research Cycle
# Usage: ./start_nexus.sh [port]

set -e
PORT=${1:-8080}
PROJ_DIR="/Users/qtmobile/Desktop/Nexus - Quant Trading "
cd "$PROJ_DIR"

echo "=== NEXUS AUTONOMOUS LOOP STARTING ==="
echo "Dashboard port: $PORT"
echo "Project: $PROJ_DIR"
echo ""

# Environment
export ZAI_API_KEY="b3893915bcea4355a46eeab30ba8db35.EExWnj8Q7bxqtvGx"
export ZAI_ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export ZAI_DEFAULT_MODEL="glm-5"

# Kill any existing processes on port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
sleep 1

# Start Dashboard in background
echo "[1/3] Starting Dashboard on port $PORT..."
python3 -m nexus_quant dashboard --artifacts artifacts --port $PORT > /tmp/nexus_dashboard.log 2>&1 &
DASH_PID=$!
echo "      Dashboard PID: $DASH_PID"
sleep 2

# Open Chrome
echo "[2/3] Opening dashboard in Chrome..."
open -a "Google Chrome" "http://localhost:$PORT" 2>/dev/null || true

# Start Brain Loop in background
echo "[3/3] Starting Brain loop (autonomous 24/7)..."
python3 -m nexus_quant brain --config configs/run_binance_nexus_alpha_v1_2023oos.json --loop > /tmp/nexus_brain.log 2>&1 &
BRAIN_PID=$!
echo "      Brain PID: $BRAIN_PID"

echo ""
echo "=== NEXUS IS RUNNING ==="
echo "Dashboard:  http://localhost:$PORT"
echo "Logs:       tail -f /tmp/nexus_dashboard.log"
echo "Brain:      tail -f /tmp/nexus_brain.log"
echo ""
echo "PIDs: dashboard=$DASH_PID brain=$BRAIN_PID"
echo "$DASH_PID $BRAIN_PID" > /tmp/nexus_pids.txt
echo ""
echo "To stop: kill \$(cat /tmp/nexus_pids.txt)"

# Keep alive — show brain log
wait $BRAIN_PID
