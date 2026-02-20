# ANTIGRAVITY ↔ NEXUS Sync Guide
# Copy this to Jack-Repo as .claude/CLAUDE.md (or use as agent.md)

## Identity
You are **Antigravity** — the frontend dashboard for the NEXUS Quant Trading platform.
Your backend is NEXUS at `http://localhost:8080`.

## NEXUS Backend Connection
- **Base URL**: `http://localhost:8080`
- **API prefix**: `/api/` (direct NEXUS endpoints)
- **Bridge prefix**: `/api/jackos/` (Jack's OS compatible endpoints)
- Start backend: `cd "/Users/qtmobile/Desktop/Nexus - Quant Trading " && ./start_nexus.sh 8080`

## Key API Endpoints (from NEXUS)

### Trading Data
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/metrics` | GET | Sharpe, CAGR, MDD, Calmar |
| `/api/equity` | GET | Equity curve array |
| `/api/signals/latest` | GET | Latest trading signal |
| `/api/signals/history` | GET | Signal history |
| `/api/signals/health` | GET | Signal health indicators |
| `/api/track_record` | GET | Year-by-year performance heatmap |
| `/api/benchmark` | GET | Benchmark comparison |
| `/api/portfolio_optimize` | GET | Multi-strategy portfolio optimization |

### Brain & Research
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/brain/diary` | GET | Brain diary entries |
| `/api/brain/goals` | GET | Strategic goals |
| `/api/brain/identity` | GET | System config |
| `/api/research_cycle/latest` | GET | Latest research results |
| `/api/agents/run` | GET | Run agent (ATLAS/CIPHER/ECHO/FLUX) |
| `/api/debate` | POST | Trigger debate between agents |
| `/api/debate_history` | GET | Past debate transcripts |

### System
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/system_status` | GET | Full system health |
| `/api/processes` | GET | Running processes |
| `/api/models` | GET | Available AI models |
| `/api/chat` | POST | Chat with AI (body: {message, model}) |
| `/api/control` | POST | Start/stop/restart brain/research |
| `/api/alerts` | GET | Active alerts |

### Jack's OS Bridge (compatible format)
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/jackos/system/health-trend` | GET | Terminal health dashboard |
| `/api/jackos/system/metrics` | GET | Key metrics (Sharpe, CAGR, MDD) |
| `/api/jackos/chat` | POST | Chat proxy (body: {message, model}) |
| `/api/jackos/signals` | GET | Latest trading signal |
| `/api/jackos/context/refresh` | GET | Full NEXUS context snapshot |
| `/api/jackos/terminal/sessions` | GET | Active Claude Code terminals |

## Data Format Examples

### `/api/metrics` response:
```json
{
  "summary": {
    "sharpe": 2.005,
    "cagr": 0.15,
    "max_drawdown": -0.046,
    "calmar": 3.26
  },
  "verdict": {"pass": true}
}
```

### `/api/signals/latest` response:
```json
{
  "timestamp": "2026-02-20T12:00:00Z",
  "weights": {"BTCUSDT": 0.15, "ETHUSDT": -0.08, ...},
  "regime": "normal",
  "vol_tilt_active": true,
  "signal_health": {"I460": "HEALTHY", "V1": "HEALTHY", "F144": "HEALTHY", "I415": "HEALTHY"}
}
```

### `/api/jackos/terminal/sessions` response:
```json
{
  "sessions": [
    {"terminal_id": "crypto_options", "status": "running", "phase": "Phase 137", "age_human": "5m"},
    {"terminal_id": "commodity_cta", "status": "stale", "phase": "Phase 135", "age_human": "2h"}
  ]
}
```

## NEXUS Architecture (for context)
```
nexus_quant/
├── agents/          # ATLAS, CIPHER, ECHO, FLUX + SmartRouter
├── backtest/        # BacktestEngine + costs
├── brain/           # Identity, goals, diary, reasoning
├── execution/       # Binance live trading
├── live/            # Signal generator
├── strategies/      # 50+ strategies, champion: P91b ensemble
├── web/             # FastAPI dashboard (THIS is the backend)
└── projects/        # Multi-market: crypto_perps, crypto_options, commodity_cta
```

## Sync Protocol
1. On startup: `GET /api/jackos/context/refresh` to load NEXUS state
2. Every 30s: `GET /api/jackos/system/health-trend` for health
3. On user request: `POST /api/jackos/chat` for AI interaction
4. For signals: `GET /api/jackos/signals` (poll every 60s or use SSE at `/api/stream`)

## Communication
- Vietnamese OK
- NEXUS codename: NEXUS PRIME (Claude Opus 4.6)
- Dashboard agents: ATLAS, CIPHER, ECHO, FLUX (Claude Sonnet 4.6)
- When updating UI, maintain existing NEXUS dashboard at `/nexus_quant/web/static/index.html`
- Antigravity UI is separate at `/Users/qtmobile/Desktop/Jack-Repo/jack-os-web/`

## Safety
- NEVER modify NEXUS backend code without explicit user request
- Frontend changes only affect Jack-Repo, NOT NEXUS
- All data flows are read-only unless POST endpoints are used
- Keep CORS headers in mind (NEXUS allows `*` by default)
