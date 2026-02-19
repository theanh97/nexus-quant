#!/bin/bash
# NEXUS Autonomous Loop Starter
# Runs: Dashboard + Brain Loop + Research Cycle + Debate Engine
# Usage: ./start_nexus.sh [port]

PORT=${1:-8080}
PROJ_DIR="/Users/qtmobile/Desktop/Nexus - Quant Trading "
cd "$PROJ_DIR"

echo "=== NEXUS AUTONOMOUS LOOP STARTING ==="
echo "Dashboard port: $PORT"
echo ""

# Environment
export ZAI_API_KEY="b3893915bcea4355a46eeab30ba8db35.EExWnj8Q7bxqtvGx"
export ZAI_ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
export ZAI_DEFAULT_MODEL="glm-5"

# Kill any existing NEXUS processes
echo "[0/4] Stopping existing NEXUS processes..."
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
pkill -f "nexus_quant dashboard" 2>/dev/null || true
pkill -f "nexus_quant brain" 2>/dev/null || true
pkill -f "nexus_quant research" 2>/dev/null || true
sleep 2

# Start Dashboard
echo "[1/4] Starting Dashboard on port $PORT..."
/usr/bin/python3 -m nexus_quant dashboard --artifacts artifacts --port $PORT \
  > /tmp/nexus_dash.log 2>&1 &
DASH_PID=$!
echo "      PID: $DASH_PID"
sleep 3

# Open Chrome
echo "[2/4] Opening Chrome..."
open -a "Google Chrome" "http://localhost:$PORT" 2>/dev/null || true

# Start Brain Loop
echo "[3/4] Starting Brain loop (autonomous research 24/7)..."
/usr/bin/python3 -m nexus_quant brain \
  --config configs/run_binance_nexus_alpha_v1_2023oos.json \
  --loop > /tmp/nexus_brain.log 2>&1 &
BRAIN_PID=$!
echo "      PID: $BRAIN_PID"

# Save PIDs
echo "$DASH_PID $BRAIN_PID" > /tmp/nexus_pids.txt

echo ""
echo "=== NEXUS RUNNING ==="
echo "Dashboard:  http://localhost:$PORT"
echo "Dashboard log: tail -f /tmp/nexus_dash.log"
echo "Brain log:     tail -f /tmp/nexus_brain.log"
echo ""
echo "Tabs: Overview | Chat (⚔️Debate) | Console | Benchmark | Performance | Risk"
echo "Debate Mode: Click ⚔️ in Chat tab to get 4-model analysis + critique + synthesis"
echo ""
echo "[4/4] All processes launched. NEXUS is autonomous."
echo ""

# Monitor — keep script alive, show combined logs
tail -f /tmp/nexus_brain.log /tmp/nexus_dash.log 2>/dev/null
