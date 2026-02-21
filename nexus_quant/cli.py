from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import run_one, improve_one
from .status import print_status
from .memory_cli import memory_main
from .autopilot_cli import autopilot_main
from .guardian_cli import guardian_main
from .supervisor_cli import supervisor_main
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

    sup_p = sub.add_parser("supervisor", help="Supervisor (keep autopilot alive with guardrails)")
    sup_p.add_argument("--config", required=True, help="Path to JSON config")
    sup_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    sup_p.add_argument("--trials", type=int, default=30, help="Self-learning trials per improve task (default: 30)")
    sup_p.add_argument("--steps", type=int, default=10, help="Autopilot max tasks per cycle (default: 10)")
    sup_p.add_argument("--autopilot-interval-seconds", type=int, default=60, help="Autopilot loop interval (default: 60)")
    sup_p.add_argument("--check-interval-seconds", type=int, default=60, help="Supervisor health check interval (default: 60)")
    sup_p.add_argument("--stale-seconds", type=int, default=1800, help="Heartbeat stale threshold before restart (default: 1800)")
    sup_p.add_argument("--running-task-grace-seconds", type=int, default=10800, help="Grace window for long-running tasks (default: 10800)")
    sup_p.add_argument("--max-restarts", type=int, default=5, help="Max restarts in restart window before pausing (default: 5)")
    sup_p.add_argument("--restart-window-seconds", type=int, default=1800, help="Restart cap window (default: 1800)")
    sup_p.add_argument("--backoff-seconds", type=int, default=20, help="Base restart backoff (default: 20)")
    sup_p.add_argument("--max-backoff-seconds", type=int, default=300, help="Max restart backoff (default: 300)")
    sup_p.add_argument("--daily-budget-usd", type=float, default=0.0, help="Daily Opus rebuttal budget cap (0 disables)")
    sup_p.add_argument("--estimated-opus-cost-usd", type=float, default=0.6, help="Fallback cost per rebuttal for budget guard")
    sup_p.add_argument("--budget-safety-multiplier", type=float, default=1.5, help="Multiplier applied to estimated spend before budget guard (default: 1.5)")
    sup_p.add_argument("--max-log-mb", type=int, default=256, help="Rotate autopilot log when it exceeds this size (default: 256)")
    sup_p.add_argument("--log-file", default="", help="Autopilot log file path (default: artifacts/logs/orion_autopilot.log)")
    sup_p.add_argument("--no-bootstrap", action="store_true", help="Disable autopilot bootstrap when queue is empty")

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

    brain_p = sub.add_parser("brain", help="Run the NEXUS autonomous brain loop")
    brain_p.add_argument("--config", required=True, help="Path to JSON config")
    brain_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    brain_p.add_argument("--cycles", type=int, default=1, help="How many cycles to run (default: 1)")
    brain_p.add_argument("--loop", action="store_true", help="Run continuously")
    brain_p.add_argument("--interval", type=int, default=3600, help="Seconds between cycles in loop mode (default: 3600)")

    tg_p = sub.add_parser("telegram", help="Start Telegram bot for NEXUS interaction (24/7)")
    tg_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    tg_p.add_argument("--config", default=None, help="Config for /run command (default: auto-detect)")

    lr_p = sub.add_parser("learn", help="Run NEXUS daily research & self-learning cycle (50+ sources)")
    lr_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    lr_p.add_argument("--configs", default=None, help="Configs dir (default: auto-detect)")
    lr_p.add_argument("--loop", action="store_true", help="Run continuously")
    lr_p.add_argument("--interval", type=int, default=86400, help="Seconds between cycles (default: 86400=24h)")
    lr_p.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=infinite)")

    al_p = sub.add_parser("alpha_loop", help="Run AlphaLoop: continuous backtest all strategies, promote champion")
    al_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    al_p.add_argument("--configs", default=None, help="Configs dir (default: project_root/configs)")
    al_p.add_argument("--interval", type=int, default=3600, help="Seconds between cycles (default: 3600=1h)")
    al_p.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=infinite)")

    sig_p = sub.add_parser("signal", help="Generate real-time trading signal from P91b champion")
    sig_p.add_argument("--config", default=None, help="Production config (default: configs/production_p91b_champion.json)")
    sig_p.add_argument("--loop", action="store_true", help="Run continuously (generate signal every hour)")
    sig_p.add_argument("--interval", type=int, default=3600, help="Seconds between signals in loop mode (default: 3600)")
    sig_p.add_argument("--json", action="store_true", help="Output signal as JSON only (for piping)")

    trade_p = sub.add_parser("trade", help="Live/paper trading — signal generation + order execution")
    trade_p.add_argument("--mode", default="paper", choices=["paper", "dry_run", "testnet", "live"],
                         help="Trading mode (default: paper)")
    trade_p.add_argument("--config", default=None, help="Production config (default: auto-detect)")
    trade_p.add_argument("--loop", action="store_true", help="Run continuously")
    trade_p.add_argument("--interval", type=int, default=3600, help="Seconds between cycles (default: 3600)")
    trade_p.add_argument("--max-cycles", type=int, default=0, help="Stop after N cycles (0=infinite)")

    live_p = sub.add_parser("live", help="Run live trading engine (P91b champion on Binance USDM Futures)")
    live_p.add_argument("--config", default="configs/production_p91b_champion.json", help="Production config path")
    live_p.add_argument("--dry-run", action="store_true", help="Simulate trades without placing real orders")
    live_p.add_argument("--testnet", action="store_true", help="Use Binance testnet instead of mainnet")
    live_p.add_argument("--once", action="store_true", help="Run one cycle and exit (no loop)")

    # ── Constitution & Watchdog ─────────────────────────────────────
    comply_p = sub.add_parser("comply", help="Check constitution compliance (all 10 rules)")
    comply_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")

    wd_p = sub.add_parser("watchdog", help="24/7 watchdog — ensures all 3 projects stay alive")
    wd_p.add_argument("--loop", action="store_true", help="Run continuously")
    wd_p.add_argument("--interval", type=int, default=120, help="Check interval in seconds (default: 120)")

    # ── Operational Learning Engine ──────────────────────────────────
    ol_p = sub.add_parser("learning", help="Operational Learning Engine — show metrics, manage lessons")
    ol_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    ol_p.add_argument("--add-lesson", action="store_true", help="Add a new lesson interactively")
    ol_p.add_argument("--category", default=None, help="Lesson category (stall|data_access|model_routing|rule_violation|task_failure)")
    ol_p.add_argument("--pattern", default=None, help="Regex pattern to match against errors")
    ol_p.add_argument("--correction", default=None, help="What to do instead")
    ol_p.add_argument("--list", action="store_true", dest="list_lessons", help="List all active lessons")

    # ── Vector Store ─────────────────────────────────────────────────
    vs_p = sub.add_parser("vectors", help="Vector store — semantic search over memory")
    vs_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    vs_p.add_argument("--rebuild", action="store_true", help="Rebuild vector index from all memory items")
    vs_p.add_argument("--search", default=None, help="Search query")
    vs_p.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    # ── Development Changelog ────────────────────────────────────────
    cl_p = sub.add_parser("changelog", help="Development changelog — track system evolution")
    cl_p.add_argument("--artifacts", default="artifacts", help="Artifacts dir (default: artifacts)")
    cl_p.add_argument("--recent", type=int, default=10, help="Show N recent changes (default: 10)")
    cl_p.add_argument("--summary", action="store_true", help="Show changelog summary")

    # ── Multi-project commands ───────────────────────────────────────
    proj_p = sub.add_parser("projects", help="List all discovered projects and their status")

    mp_p = sub.add_parser("multi", help="Run multi-project scheduler (all enabled projects concurrently)")
    mp_p.add_argument("--config", default="nexus.yaml", help="Master config path (default: nexus.yaml)")

    mem_ns_p = sub.add_parser("memory-status", help="Show hierarchical memory status for a project")
    mem_ns_p.add_argument("--project", default="crypto_perps", help="Project name (default: crypto_perps)")
    mem_ns_p.add_argument("--memory-root", default="memory", help="Memory root dir (default: memory)")

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "status":
        print_status(artifacts_dir=Path(args.artifacts), tail=int(args.tail))
        return 0

    # ── Constitution & Watchdog (no config required) ────────────────
    if args.cmd == "comply":
        from .orchestration.constitution import check_compliance
        report = check_compliance(Path(args.artifacts))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ok"]:
            print(f"\n!! {report['summary']['critical']} CRITICAL violation(s) found !!")
        return 0 if report["ok"] else 1

    if args.cmd == "watchdog":
        from .watchdog import watchdog_run
        return watchdog_run(loop=bool(args.loop), interval=int(args.interval))

    if args.cmd == "learning":
        from .learning.operational import OperationalLearner
        learner = OperationalLearner(Path(args.artifacts))
        try:
            if args.add_lesson and args.category and args.pattern and args.correction:
                lid = learner.add_user_lesson(
                    category=args.category,
                    pattern=args.pattern,
                    correction=args.correction,
                )
                print(f"Lesson added (id={lid}): [{args.category}] {args.correction}")
                return 0
            if args.list_lessons:
                lessons = learner.store.get_active_lessons()
                print(f"{'ID':>4} {'Cat':<15} {'Sev':<10} {'Hits':>5} {'Miss':>5} {'Source':<16} Correction")
                print("-" * 100)
                for l in lessons:
                    print(f"{l.id:>4} {l.category:<15} {l.severity:<10} {l.hit_count:>5} {l.miss_count:>5} {l.source:<16} {l.correction[:60]}")
                return 0
            # Default: show metrics
            metrics = learner.metrics()
            print(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"\nLearning effective: {metrics.get('learning_effective', 'insufficient data')}")
            print(f"Prevention rate: {metrics.get('prevention_rate', 0):.1%}")
            print(f"Total lessons: {metrics.get('total_lessons', 0)}")
            print(f"Failures prevented: {metrics.get('total_hits_prevented', 0)}")
            print(f"Failures missed: {metrics.get('total_misses', 0)}")
        finally:
            learner.close()
        return 0

    if args.cmd == "vectors":
        from .memory.vector_store import NexusVectorStore
        vs = NexusVectorStore(Path(args.artifacts))
        try:
            if args.rebuild:
                # Rebuild index from all memory items
                from .memory.store import MemoryStore
                mem = MemoryStore(Path(args.artifacts) / "memory" / "memory.db")
                items = mem.recent(limit=200)
                batch = []
                for it in items:
                    doc_id = f"memory:{it.id}"
                    text = f"{it.kind} {' '.join(it.tags)} {it.content}"
                    meta = {"kind": it.kind, "tags": it.tags, "created_at": it.created_at}
                    batch.append((doc_id, text, meta))
                mem.close()
                count = vs.add_batch(batch)
                stats = vs.stats()
                print(f"Rebuilt vector index: {count} documents indexed")
                print(f"Mode: {stats['embed_mode']} ({stats['embed_dim']}-dim)")
                print(f"Semantic available: {stats['semantic_available']}")
                return 0
            if args.search:
                results = vs.search(args.search, top_k=int(args.top_k))
                if not results:
                    print("No results found.")
                    return 0
                for r in results:
                    meta = r.get("meta", {})
                    kind = meta.get("kind", "")
                    print(f"  [{r['score']:.3f}] ({kind}) {r['doc_id']}: {r['text'][:100]}...")
                return 0
            # Default: show stats
            stats = vs.stats()
            print(json.dumps(stats, indent=2))
        finally:
            vs.close()
        return 0

    if args.cmd == "changelog":
        from .learning.changelog import ChangeLog
        cl = ChangeLog(Path(args.artifacts))
        try:
            if args.summary:
                s = cl.summary()
                print(json.dumps(s, indent=2))
                return 0
            changes = cl.recent(limit=int(args.recent))
            if not changes:
                print("No changes recorded yet.")
                return 0
            for c in changes:
                files = ", ".join(c.get("files", [])[:3])
                print(f"  [{c['created_at'][:19]}] ({c['category']}) {c['description'][:80]}")
                if files:
                    print(f"    files: {files}")
                impact = c.get("impact", [])
                if impact:
                    print(f"    impact: {', '.join(impact[:3])}")
        finally:
            cl.close()
        return 0

    # ── Multi-project commands (no config required) ──────────────────
    if args.cmd == "projects":
        from .projects import discover_projects
        projects = discover_projects()
        if not projects:
            print("No projects found in nexus_quant/projects/")
            return 0
        print(f"{'Name':<20} {'Market':<10} {'Enabled':<8} {'Strategies':<5} {'Interval'}")
        print("-" * 65)
        for name, m in projects.items():
            print(f"{name:<20} {m.market:<10} {'yes' if m.enabled else 'no':<8} {len(m.strategies):<5} {m.brain_interval}s")
        return 0

    if args.cmd == "multi":
        from .orchestration.project_scheduler import NexusScheduler
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        scheduler = NexusScheduler.from_config(args.config)
        if not scheduler.runners:
            print("No enabled projects with configs found. Check nexus.yaml and project manifests.")
            return 1
        scheduler.start()
        print(json.dumps(scheduler.status(), indent=2, default=str))
        scheduler.wait()
        return 0

    if args.cmd == "memory-status":
        from .memory.namespace import MemoryNamespace
        from .projects import get_project
        project = get_project(args.project)
        market = project.market if project else ""
        mem = MemoryNamespace(Path(args.memory_root), args.project, market)
        info = mem.summary()
        print(json.dumps(info, indent=2, default=str))
        # Also print wisdom
        wisdom = mem.get_wisdom()
        n_insights = len(wisdom.get("insights", []))
        n_lessons = len(wisdom.get("lessons", []))
        print(f"\nMerged wisdom: {n_insights} insights, {n_lessons} lessons from layers: {wisdom.get('_layers', [])}")
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
    if args.cmd == "supervisor":
        return supervisor_main(
            config_path=Path(args.config),
            artifacts_dir=Path(args.artifacts),
            trials=int(args.trials),
            steps=int(args.steps),
            autopilot_interval_seconds=int(args.autopilot_interval_seconds),
            check_interval_seconds=int(args.check_interval_seconds),
            stale_seconds=int(args.stale_seconds),
            running_task_grace_seconds=int(args.running_task_grace_seconds),
            max_restarts=int(args.max_restarts),
            restart_window_seconds=int(args.restart_window_seconds),
            base_backoff_seconds=int(args.backoff_seconds),
            max_backoff_seconds=int(args.max_backoff_seconds),
            bootstrap=not bool(args.no_bootstrap),
            daily_budget_usd=float(args.daily_budget_usd),
            estimated_opus_cost_usd=float(args.estimated_opus_cost_usd),
            budget_safety_multiplier=float(args.budget_safety_multiplier),
            log_file=str(args.log_file),
            max_log_mb=int(args.max_log_mb),
        )
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

    if args.cmd == "trade":
        from .live.runner import run_cycle, run_loop
        if args.loop:
            run_loop(
                mode=args.mode,
                config_path=args.config,
                interval_seconds=args.interval,
                max_cycles=args.max_cycles,
            )
        else:
            result = run_cycle(mode=args.mode, config_path=args.config)
            if getattr(args, "json", False):
                print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "live":
        from .execution.live_engine import live_main
        return live_main(
            config_path=str(args.config),
            dry_run=bool(getattr(args, "dry_run", False)),
            testnet=bool(getattr(args, "testnet", False)),
            once=bool(getattr(args, "once", False)),
        )

    if args.cmd == "signal":
        from .live.signal_generator import SignalGenerator, generate_signal_cli
        import time as _time
        cfg_p = args.config
        if args.loop:
            gen = SignalGenerator.from_production_config(cfg_p)
            while True:
                try:
                    if getattr(args, "json", False):
                        sig = gen.generate()
                        print(json.dumps(sig.to_dict(), indent=2))
                    else:
                        generate_signal_cli(cfg_p)
                except Exception as e:
                    print(f"[SIGNAL] Error: {e}", flush=True)
                _time.sleep(args.interval)
        else:
            if getattr(args, "json", False):
                gen = SignalGenerator.from_production_config(cfg_p)
                sig = gen.generate()
                print(json.dumps(sig.to_dict(), indent=2))
            else:
                generate_signal_cli(cfg_p)
        return 0

    if args.cmd == "learn":
        from .research.daily_routine import DailyRoutine
        configs_dir = Path(args.configs) if getattr(args, "configs", None) else None
        routine = DailyRoutine(
            artifacts_dir=Path(args.artifacts),
            configs_dir=configs_dir,
        )
        if args.loop:
            max_c = int(args.max_cycles) if args.max_cycles else None
            routine.run_continuous(interval_hours=args.interval / 3600.0, max_cycles=max_c)
        else:
            result = routine.run_full_day()
            print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "alpha_loop":
        from .orchestration.alpha_loop import alpha_loop_main
        configs_dir = Path(args.configs) if getattr(args, "configs", None) else None
        return alpha_loop_main(
            artifacts_dir=Path(args.artifacts),
            configs_dir=configs_dir,
            interval_seconds=int(args.interval),
            max_cycles=int(args.max_cycles),
        )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    if args.cmd == "brain":
        from .brain.loop import NexusAutonomousLoop
        loop_obj = NexusAutonomousLoop(Path(args.config), Path(args.artifacts))
        heartbeat_path = Path(args.artifacts) / "state" / "brain_heartbeat.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        if args.loop:
            import time
            import traceback
            cycle = 0
            consecutive_errors = 0
            max_consecutive_errors = 10
            base_backoff = 30
            while True:
                try:
                    result = loop_obj.run_cycle()
                    print(f"Cycle {result['cycle_number']}: sharpe={result.get('best_sharpe')} mood={result.get('mood')}", flush=True)
                    cycle += 1
                    consecutive_errors = 0
                    # Write heartbeat
                    heartbeat_path.write_text(json.dumps({
                        "ts": time.time(),
                        "cycle": cycle,
                        "status": "ok",
                        "last_sharpe": result.get("best_sharpe"),
                    }), encoding="utf-8")
                except KeyboardInterrupt:
                    print("[BRAIN] Interrupted by user.", flush=True)
                    break
                except Exception as exc:
                    consecutive_errors += 1
                    backoff = min(300, base_backoff * (2 ** (consecutive_errors - 1)))
                    print(f"[BRAIN] Error in cycle {cycle + 1} ({consecutive_errors}/{max_consecutive_errors}): {exc}", flush=True)
                    traceback.print_exc()
                    heartbeat_path.write_text(json.dumps({
                        "ts": time.time(),
                        "cycle": cycle,
                        "status": "error",
                        "error": str(exc),
                        "consecutive_errors": consecutive_errors,
                    }), encoding="utf-8")
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[BRAIN] {max_consecutive_errors} consecutive errors — pausing 10 min before reset.", flush=True)
                        time.sleep(600)
                        consecutive_errors = 0
                    else:
                        time.sleep(backoff)
                    continue
                time.sleep(args.interval)
        else:
            for i in range(args.cycles):
                result = loop_obj.run_cycle()
                print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "telegram":
        from .comms.telegram_bot import TelegramBot
        cfg = Path(args.config) if getattr(args, "config", None) else None
        bot = TelegramBot(Path(args.artifacts), config_path=cfg)
        bot.start_polling()
        return 0

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
