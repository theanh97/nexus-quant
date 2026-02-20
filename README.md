# NEXUS Quant Trading (Alpha Improvement Workforce)

This repo is a **reproducible Quant R&D + benchmark + evidence** workspace.

Core idea:
- Run strategies end-to-end: `data -> strategy -> backtest -> benchmark -> evidence ledger`.
- Improve strategies via a **verified self-learning loop** (parameter search + locked evaluation + holdout).

## Quick start (no external deps)

Run a full synthetic demo (funding/basis carry, perp-only):

```bash
python3 -m nexus_quant run --config configs/run_synthetic_funding.json
```

Run a small self-learning loop (search params, then re-evaluate):

```bash
python3 -m nexus_quant improve --config configs/run_synthetic_funding.json --trials 30
```

Run Orion autopilot (task bus: run -> research_ingest -> improve -> wisdom -> reflect -> critique -> experiments -> handoff):

```bash
python3 -m nexus_quant autopilot --config configs/run_synthetic_funding.json --bootstrap --steps 10
```

Run self-healing supervisor for 24/7 mode (auto-restart + restart cap + optional budget guard):

```bash
python3 -m nexus_quant supervisor --config configs/run_synthetic_funding.json --autopilot-interval-seconds 60 --check-interval-seconds 60 --stale-seconds 1800 --max-restarts 5 --restart-window-seconds 1800 --max-log-mb 256 --budget-safety-multiplier 1.5
```

Curate long-horizon wisdom checkpoints (ledger + memory):

```bash
python3 -m nexus_quant wisdom --artifacts artifacts --tail-events 200
```

Run deterministic reflection (analyze evidence -> update safe overrides):

```bash
python3 -m nexus_quant reflect --config configs/run_synthetic_funding.json --artifacts artifacts --tail-events 200
```

Run deterministic critique (pushback + suggested next experiments):

```bash
python3 -m nexus_quant critique --config configs/run_synthetic_funding.json --artifacts artifacts --tail-events 200
```

Monitor autopilot heartbeat:

```bash
python3 -m nexus_quant guardian --artifacts artifacts --stale-seconds 900
```

Supervisor control/status files:
- `artifacts/state/orion_supervisor_control.json` (`mode=PAUSE|PAUSED|STOP` pauses restarts)
- `artifacts/state/orion_supervisor_status.json` (live health/restart status)

Promote an accepted self-learning candidate into the main config (dry-run by default):

```bash
python3 -m nexus_quant promote --config configs/run_synthetic_funding.json --best artifacts/memory/best_params.json
```

Add user feedback to long-term memory:

```bash
python3 -m nexus_quant memory add --kind feedback --tags user,priority --content "Self-learning must be verified"
```

Ingest verified research sources (local-first) into memory:

```bash
python3 -m nexus_quant research ingest --path research/inbox --artifacts artifacts --kind source --tags research --move-to research/library
```

Artifacts are written to `artifacts/` (reports + metrics + ledger JSONL).

## Deploy online (full dashboard + API)

Use GitHub + Render deployment guide:

- `docs/DEPLOY_ONLINE.md`

This keeps realtime endpoints (`/api/stream`, `/api/log_stream`) available for multi-device access.

## Runbook

See `docs/NEXUS_QUANT_RUNBOOK.md` for:
- data cleanliness + anti-bias rules
- backtest assumptions
- benchmark v1 scorecard
- verified self-learning protocol

## Using real data (local CSV)

You can swap the data provider to `local_csv_v1` (no pandas required).

CSV expectations (one file per symbol: `{SYMBOL}.csv`):
- required columns: `timestamp`, `close`
- optional columns: `spot_close`, `funding_rate`
- optional (higher fidelity): `volume`, `spot_volume`, `mark_close`, `index_close`, `bid_close`, `ask_close`

Example config template: `configs/run_local_csv_template.json`.

## What “locked benchmark” means here

The benchmark suite is **versioned and reproducible**:
- Dataset identity: hash of provider + config (+ files if local CSV).
- Code identity: hash of key modules used for the run.
- Run identity: config hash + seed + timestamps.

The system records these into an append-only ledger (`artifacts/ledger/ledger.jsonl`).
