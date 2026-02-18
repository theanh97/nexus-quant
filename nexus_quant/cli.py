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
