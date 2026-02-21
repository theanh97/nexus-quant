#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# NEXUS — SPX PCS Project 4 Setup Script
# ═══════════════════════════════════════════════════════════════════
# Configures ALGOXPERT_DIR env var so NEXUS can find the algoxpert
# repo and its 41GB data on this machine.
#
# Usage:
#   ./scripts/setup_spx_pcs.sh                    # auto-detect
#   ./scripts/setup_spx_pcs.sh /path/to/algoxpert # manual path
#
# After running: source ~/.zshrc (or restart terminal)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Color helpers ──────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "${RED}❌  $1${NC}"; }

echo ""
echo "════════════════════════════════════════════════════════"
echo "  NEXUS SPX PCS — Project 4 Setup"
echo "════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Resolve algoxpert dir ─────────────────────────────────
if [ -n "${1:-}" ]; then
    ALGOXPERT_DIR="$1"
    echo "Using provided path: $ALGOXPERT_DIR"
elif [ -n "${ALGOXPERT_DIR:-}" ]; then
    echo "Using existing ALGOXPERT_DIR: $ALGOXPERT_DIR"
else
    # Auto-detect: look for repo in common locations
    CANDIDATES=(
        "$HOME/Desktop/algoxpert-3rd-alpha-spx"
        "$HOME/algoxpert-3rd-alpha-spx"
        "$(dirname "$PROJECT_ROOT")/algoxpert-3rd-alpha-spx"
    )
    ALGOXPERT_DIR=""
    for candidate in "${CANDIDATES[@]}"; do
        if [ -d "$candidate" ]; then
            ALGOXPERT_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$ALGOXPERT_DIR" ] || [ ! -d "$ALGOXPERT_DIR" ]; then
    err "algoxpert-3rd-alpha-spx repo not found."
    echo ""
    echo "Options:"
    echo "  1. Clone: git clone https://github.com/theanh97/algoxpert-3rd-alpha-spx.git ~/Desktop/algoxpert-3rd-alpha-spx"
    echo "  2. Provide path: ./scripts/setup_spx_pcs.sh /path/to/algoxpert-3rd-alpha-spx"
    exit 1
fi

ok "Found algoxpert repo at: $ALGOXPERT_DIR"

# ── Step 2: Verify data presence ──────────────────────────────────
DATA_DIR="$ALGOXPERT_DIR/Custom_Backtest_Framework/data"
echo ""
echo "Checking data directory: $DATA_DIR"

if [ ! -d "$DATA_DIR" ]; then
    err "Data directory not found: $DATA_DIR"
    warn "Run: cd $ALGOXPERT_DIR && ./tools/bootstrap_data.sh"
    exit 1
fi

YEARS_FOUND=()
for yr in 2020 2021 2022 2023 2024 2025; do
    if [ -d "$DATA_DIR/spxw_$yr" ]; then
        YEARS_FOUND+=("$yr")
    fi
done

if [ ${#YEARS_FOUND[@]} -eq 0 ]; then
    warn "No SPXW data found. Run: cd $ALGOXPERT_DIR && ./tools/bootstrap_data.sh"
else
    ok "Data found for years: ${YEARS_FOUND[*]}"
fi

if [ -f "$DATA_DIR/vix/vix_ohlc_1min.parquet" ]; then
    ok "VIX 1-min data present"
else
    warn "VIX data missing: $DATA_DIR/vix/vix_ohlc_1min.parquet"
fi

# ── Step 3: Export env var ─────────────────────────────────────────
echo ""
echo "Setting ALGOXPERT_DIR..."

# Detect shell config file
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    # Remove existing entry if any
    if grep -q "ALGOXPERT_DIR" "$SHELL_RC" 2>/dev/null; then
        sed -i.bak '/ALGOXPERT_DIR/d' "$SHELL_RC"
    fi
    echo "export ALGOXPERT_DIR=\"$ALGOXPERT_DIR\"" >> "$SHELL_RC"
    ok "Added to $SHELL_RC: export ALGOXPERT_DIR=\"$ALGOXPERT_DIR\""
    warn "Run: source $SHELL_RC"
else
    warn "Could not find shell config. Set manually: export ALGOXPERT_DIR=\"$ALGOXPERT_DIR\""
fi

# Also export for current session
export ALGOXPERT_DIR="$ALGOXPERT_DIR"

# ── Step 4: Verify NEXUS adapter ──────────────────────────────────
echo ""
echo "Testing NEXUS bridge adapter..."
cd "$PROJECT_ROOT"

if /usr/bin/python3 -c "
from nexus_quant.projects.spx_pcs.adapter import SPXArtifactReader
r = SPXArtifactReader()
print('Available:', r.is_available())
status = r.get_engine_status()
print('Data years:', status.get('data_years', []))
" 2>/dev/null; then
    ok "NEXUS bridge adapter working"
else
    warn "Bridge adapter test failed. Check Python path and nexus_quant installation."
fi

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup Complete"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  ALGOXPERT_DIR = $ALGOXPERT_DIR"
echo "  Data years:    ${YEARS_FOUND[*]:-none}"
echo ""
echo "  Next steps:"
echo "    1. Run doctor:    cd $ALGOXPERT_DIR/Custom_Backtest_Framework && python scripts/doctor.py"
echo "    2. Run backtest:  python scripts/run_backtest.py backtest --start 20210101 --end 20211231"
echo "    3. View results:  curl http://localhost:8080/api/spx_pcs/status"
echo "    4. Dashboard:     http://localhost:8080 → SPX PCS tab"
echo ""
