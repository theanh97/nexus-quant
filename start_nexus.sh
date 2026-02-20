#!/bin/bash
# NEXUS Autonomous Loop Starter — with 24/7 Watchdog
# Runs: Dashboard + Brain Loop (auto-restart on crash)
# Usage: ./start_nexus.sh [port]

PORT=${1:-8080}
PROJ_DIR="/Users/qtmobile/Desktop/Nexus - Quant Trading "
cd "$PROJ_DIR"

MAX_RESTARTS=20         # max restarts per session
RESTART_BACKOFF=30      # seconds between restarts (doubles each time, caps at 600)
MAX_BACKOFF=600

echo "=== NEXUS AUTONOMOUS LOOP v1.0.0 ==="
echo "Dashboard port: $PORT"
echo "Watchdog: auto-restart brain on crash (max $MAX_RESTARTS restarts)"
echo ""

# Environment
export ZAI_API_KEY="b3893915bcea4355a46eeab30ba8db35.EExWnj8Q7bxqtvGx"
export ZAI_ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export ZAI_DEFAULT_MODEL="glm-5"
export PYTHONUNBUFFERED=1

# Kill any existing NEXUS processes
echo "[0/3] Stopping existing NEXUS processes..."
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
pkill -f "nexus_quant dashboard" 2>/dev/null || true
pkill -f "nexus_quant brain" 2>/dev/null || true
pkill -f "nexus_quant research" 2>/dev/null || true
sleep 2

# ─── Dashboard ───────────────────────────────────────
echo "[1/3] Starting Dashboard on port $PORT..."
/usr/bin/python3 -m nexus_quant dashboard --artifacts artifacts --port $PORT \
  > /tmp/nexus_dash.log 2>&1 &
DASH_PID=$!
echo "      PID: $DASH_PID"
sleep 3

# Open Chrome
echo "[2/3] Opening Chrome..."
open -a "Google Chrome" "http://localhost:$PORT" 2>/dev/null || true

# ─── Brain Loop Watchdog ─────────────────────────────
start_brain() {
    /usr/bin/python3 -m nexus_quant brain \
        --config configs/run_binance_nexus_alpha_v1_2023oos.json \
        --loop --interval 600 \
        >> /tmp/nexus_brain.log 2>&1 &
    BRAIN_PID=$!
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Brain started PID=$BRAIN_PID" | tee -a /tmp/nexus_watchdog.log
}

echo "[3/3] Starting Brain loop with watchdog..."
start_brain
restart_count=0
current_backoff=$RESTART_BACKOFF

# Save PIDs
echo "$DASH_PID $BRAIN_PID" > /tmp/nexus_pids.txt

echo ""
echo "=== NEXUS RUNNING (24/7 mode) ==="
echo "Dashboard:    http://localhost:$PORT"
echo "Dash log:     tail -f /tmp/nexus_dash.log"
echo "Brain log:    tail -f /tmp/nexus_brain.log"
echo "Watchdog log: tail -f /tmp/nexus_watchdog.log"
echo ""
echo "Ctrl+C to stop all."
echo ""

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    echo "[WATCHDOG] Shutting down NEXUS..."
    kill $BRAIN_PID 2>/dev/null
    kill $DASH_PID 2>/dev/null
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Clean shutdown (restarts=$restart_count)" | tee -a /tmp/nexus_watchdog.log
    exit 0
}
trap cleanup SIGINT SIGTERM

# ─── Watchdog Loop ────────────────────────────────────
while true; do
    # Check if brain is still alive
    if ! kill -0 $BRAIN_PID 2>/dev/null; then
        restart_count=$((restart_count + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Brain died! Restart #$restart_count/$MAX_RESTARTS" | tee -a /tmp/nexus_watchdog.log

        if [ $restart_count -ge $MAX_RESTARTS ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Max restarts reached. Waiting 30 min before reset." | tee -a /tmp/nexus_watchdog.log
            sleep 1800
            restart_count=0
            current_backoff=$RESTART_BACKOFF
        fi

        echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Restarting brain in ${current_backoff}s..." | tee -a /tmp/nexus_watchdog.log
        sleep $current_backoff

        # Increase backoff (capped)
        current_backoff=$((current_backoff * 2))
        if [ $current_backoff -gt $MAX_BACKOFF ]; then
            current_backoff=$MAX_BACKOFF
        fi

        start_brain
        echo "$DASH_PID $BRAIN_PID" > /tmp/nexus_pids.txt
    else
        # Brain healthy — reset backoff on success
        current_backoff=$RESTART_BACKOFF
    fi

    # Check dashboard too
    if ! kill -0 $DASH_PID 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [WATCHDOG] Dashboard died! Restarting..." | tee -a /tmp/nexus_watchdog.log
        /usr/bin/python3 -m nexus_quant dashboard --artifacts artifacts --port $PORT \
          > /tmp/nexus_dash.log 2>&1 &
        DASH_PID=$!
        echo "$DASH_PID $BRAIN_PID" > /tmp/nexus_pids.txt
    fi

    sleep 30  # Check every 30 seconds
done
