from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import run_one, improve_one
from .status import print_status
from .memory_cli import memory_main
from .autopilot_cli import autopilot_main
from .guardian_cli import guardian_main
from .promote_cli import promote_main
from .wisdom_cli import wisdom_main
from .research_cli import research_main
from .reflect_cli import reflect_main
from .critique_cli import critique_main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nexus_quant", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run one strategy -> backtest -> benchmark -> ledger")
    run_p.add_argument("--config", required=True, help="Path to JSON config")
    run_p.add_argument("--out", default="artifacts", help="Artifacts output dir (default: artifacts)")

    imp_p = sub.add_parser("improve", help="Self-learn params (search) -> holdout verify -> ledger")
    imp_p.add_argument("--config", required=True, help="Path to JSON config")
    imp_p.add_argument("--out", default="artifacts", help="Artifacts output dir (default: artifacts)")
    imp_p.add_argument("--trials", type=int, default=30, help="Number of candidate trials")

    dump_p = sub.add_parser("dump-config", help="Validate and pretty-print config")
    dump_p.add_argument("--config", required=True, help="Path to JSON config")

    st_p = sub.add_parser("status", help="Show ledger + last accepted params")
    st_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    st_p.add_argument("--tail", type=int, default=6, help="Number of ledger events to show")

    mem_p = sub.add_parser("memory", help="Long-term memory (feedback, decisions, notes)")
    mem_p.add_argument("memory_args", nargs=argparse.REMAINDER, help="memory subcommand and args")

    r_p = sub.add_parser("research", help="Verified research ingestion (local sources -> memory)")
    r_p.add_argument("research_args", nargs=argparse.REMAINDER, help="research subcommand and args")

    ap_p = sub.add_parser("autopilot", help="Orion autopilot (run tasks + handoff)")
    ap_p.add_argument("--config", required=True, help="Path to JSON config")
    ap_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    ap_p.add_argument("--trials", type=int, default=30, help="Self-learning trials (default: 30)")
    ap_p.add_argument("--bootstrap", action="store_true", help="Create run/improve/handoff tasks if none pending")
    ap_p.add_argument("--steps", type=int, default=10, help="Max tasks to execute this invocation")
    ap_p.add_argument("--loop", action="store_true", help="Run continuously (24/7 mode)")
    ap_p.add_argument("--interval-seconds", type=int, default=300, help="Sleep between checks when looping")
    ap_p.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0 = infinite) when looping")

    g_p = sub.add_parser("guardian", help="Guardian monitor (detect stale autopilot)")
    g_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    g_p.add_argument("--stale-seconds", type=int, default=900, help="Stale threshold for heartbeat")

    pr_p = sub.add_parser("promote", help="Promote accepted best params into a config (verified self-learning)")
    pr_p.add_argument("--config", required=True, help="Path to JSON config to update")
    pr_p.add_argument("--best", default="artifacts/memory/best_params.json", help="best_params.json path")
    pr_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    pr_p.add_argument("--apply", action="store_true", help="Actually write changes (otherwise dry-run)")

    w_p = sub.add_parser("wisdom", help="Curate long-horizon wisdom checkpoints (ledger + memory)")
    w_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    w_p.add_argument("--tail-events", type=int, default=200, help="How many ledger events to summarize (default: 200)")

    rf_p = sub.add_parser("reflect", help="Deterministic reflection (analyze evidence -> update safe overrides)")
    rf_p.add_argument("--config", required=True, help="Path to JSON config")
    rf_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    rf_p.add_argument("--tail-events", type=int, default=200, help="How many ledger events to scan (default: 200)")

    cr_p = sub.add_parser("critique", help="Deterministic critique (pushback + next experiments)")
    cr_p.add_argument("--config", required=True, help="Path to JSON config")
    cr_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    cr_p.add_argument("--tail-events", type=int, default=200, help="How many ledger events to scan (default: 200)")

    dash_p = sub.add_parser("dashboard", help="Serve the web dashboard UI")
    dash_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    dash_p.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    dash_p.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")

    sched_p = sub.add_parser("schedule", help="Critique -> generate experiment specs -> run batch")
    sched_p.add_argument("--config", required=True, help="Path to JSON config")
    sched_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    sched_p.add_argument("--workers", type=int, default=4, help="Max parallel workers (default: 4)")
    sched_p.add_argument("--trials", type=int, default=30, help="Trials per experiment (default: 30)")

    ag_p = sub.add_parser("agents", help="Run Orion agent loop (multi-agent orchestration)")
    ag_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    ag_p.add_argument("--config", required=True, help="Path to JSON config")

    risk_p = sub.add_parser("risk", help="Run VaR + regime detection on latest backtest result")
    risk_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "status":
        print_status(artifacts_dir=Path(args.artifacts), tail=int(args.tail))
        return 0
    if args.cmd == "memory":
        return memory_main(args.memory_args)
    if args.cmd == "research":
        return research_main(args.research_args)
    if args.cmd == "autopilot":
        return autopilot_main(
            config_path=Path(args.config),
            artifacts_dir=Path(args.artifacts),
            trials=int(args.trials),
            bootstrap=bool(args.bootstrap),
            steps=int(args.steps),
            loop=bool(args.loop),
            interval_seconds=int(args.interval_seconds),
            max_cycles=int(args.max_cycles),
        )
    if args.cmd == "guardian":
        return guardian_main(artifacts_dir=Path(args.artifacts), stale_seconds=int(args.stale_seconds))
    if args.cmd == "promote":
        return promote_main(config_path=Path(args.config), best_params_path=Path(args.best), artifacts_dir=Path(args.artifacts), apply=bool(args.apply))
    if args.cmd == "wisdom":
        return wisdom_main(["--artifacts", str(args.artifacts), "--tail-events", str(args.tail_events)])
    if args.cmd == "reflect":
        return reflect_main(config_path=Path(args.config), artifacts_dir=Path(args.artifacts), tail_events=int(args.tail_events))
    if args.cmd == "critique":
        return critique_main(config_path=Path(args.config), artifacts_dir=Path(args.artifacts), tail_events=int(args.tail_events))

    if args.cmd == "dashboard":
        from .web.app import serve
        serve(artifacts_dir=Path(args.artifacts), port=int(args.port), host=args.host)
        return 0

    if args.cmd == "schedule":
        from .orchestration.scheduler import ExperimentScheduler
        from .learning.critic import critique_recent
        critique = critique_recent(config_path=Path(args.config), artifacts_dir=Path(args.artifacts), tail_events=200)
        next_exps = critique.get("next_experiments") or []
        from .orchestration.priority import build_experiment_specs
        specs = build_experiment_specs(next_exps, max_experiments=int(args.workers) * 2)
        scheduler = ExperimentScheduler(Path(args.config), Path(args.artifacts), max_workers=int(args.workers))
        results = scheduler.run_batch(specs)
        report = scheduler.write_batch_report(results)
        print(f"Batch done: {report}")
        return 0

    if args.cmd == "agents":
        from .orchestration.orion import Orion, OrionConfig
        orion = Orion(OrionConfig(artifacts_dir=Path(args.artifacts), config_path=Path(args.config)))
        try:
            out = orion.run_agents()
            print(json.dumps(out, indent=2, default=str))
        finally:
            orion.close()
        return 0

    if args.cmd == "risk":
        # Find latest run metrics
        runs_dir = Path(args.artifacts) / "runs"
        if not runs_dir.exists():
            print("No runs found.")
            return 1
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
        if not run_dirs:
            print("No run directories found.")
            return 1
        latest = run_dirs[-1]
        result_path = latest / "result.json"
        if not result_path.exists():
            print(f"No result.json in {latest}")
            return 1
        result_data = json.loads(result_path.read_text())
        returns = result_data.get("returns") or []
        equity = result_data.get("equity_curve") or []
        from .risk.var import var_report
        from .risk.regime import detect_regime
        if returns:
            var = var_report(returns, equity)
            regime = detect_regime(returns)
            print(json.dumps({"var": var, "regime": regime}, indent=2))
        else:
            print("No returns data found.")
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    if args.cmd == "dump-config":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    if args.cmd == "run":
        run_one(cfg_path, out_dir=Path(args.out))
        return 0

    if args.cmd == "improve":
        improve_one(cfg_path, out_dir=Path(args.out), trials=args.trials)
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")
