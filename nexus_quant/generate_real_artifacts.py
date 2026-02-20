import json
from pathlib import Path
import time
import random

def main():
    root = Path("artifacts")
    root.mkdir(exist_ok=True)
    
    # 1. Generate Tasks
    tasks_dir = root / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    tasks_db_path = tasks_dir / "tasks.json"
    MOCK_TASKS = [
        {"id": "t1", "title": "Refine Momentum Alpha", "status": "todo", "priority": "high", "category": "research", "assignee": "Model V3", "delegated_by": "NEXUS", "history": []},
        {"id": "t2", "title": "Fix P91b Drawdown issue", "status": "done", "priority": "high", "category": "practice", "assignee": "Quants", "delegated_by": "human", "history": []},
        {"id": "t3", "title": "Monitor Live Execution", "status": "in_progress", "priority": "high", "category": "procedure", "assignee": "System", "delegated_by": "atlas", "history": []},
        {"id": "t4", "title": "Train V3 Factor Model", "status": "review", "priority": "medium", "category": "research", "assignee": "GLM-5", "delegated_by": "nexus", "history": []}
    ]
    tasks_db_path.write_text(json.dumps(MOCK_TASKS, indent=2))
    
    # 2. Generate System Process Status Wrapper
    # App.py reads process stats by checking active folders, but we can mock /api/processes via modifying app.py or writing a hook.
    # Actually wait, app.py has an endpoint /api/processes but it parses `psutil`. Let's ensure psutil parses correctly.
    
    # 3. Generate Research Debate History
    debate_dir = root / "debate"
    debate_dir.mkdir(exist_ok=True)
    history_file = debate_dir / "history.json"
    MOCK_DEBATES = [
        {
            "topic": "Signal decay on momentum",
            "consensus_score": 0.85,
            "ts": "2026-02-20T17:00:00",
            "contributions": [1,2,3],
            "synthesis": "Use shorter half-life. P91b requires 14-day exponential decay."
        },
        {
            "topic": "Drawdown limits for V3",
            "consensus_score": 0.92,
            "ts": "2026-02-19T10:00:00",
            "contributions": [1,2],
            "synthesis": "Implement hard stop at -5%."
        }
    ]
    history_file.write_text(json.dumps(MOCK_DEBATES, indent=2))
    
    # 4. Generate Files for File Browser
    hand_dir = root / "handoff"
    hand_dir.mkdir(exist_ok=True)
    (hand_dir / "p91b_strategy_spec.md").write_text("# P91b V1\nChampion strategy ruleset.")
    (hand_dir / "drawdown_analysis.json").write_text(json.dumps({"max": -0.071, "trades": 540}))
    
    # 5. Generate Brain Identity & Notifications
    brain_dir = root / "brain"
    brain_dir.mkdir(exist_ok=True)
    
    (brain_dir / "identity.json").write_text(json.dumps({
        "name": "NEXUS Brain", 
        "version": "GLM-5.0", 
        "objective": "Maximize risk-adjusted returns continuously."
    }, indent=2))
    
    (brain_dir / "goals.json").write_text(json.dumps({
        "goals": ["Deploy P91b to live", "Verify OOS max_dd", "Reduce latency by 15ms"]
    }, indent=2))
    
    (brain_dir / "notifications.jsonl").write_text(
        json.dumps({"level": "INFO", "msg": "Booted Nexus brain."}) + "\n" +
        json.dumps({"level": "WARN", "msg": "Market volatility high."}) + "\n"
    )

    # 6. Generate Runs Data for metrics/equity APIs
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    run1 = runs_dir / "p91b_champion"
    run1.mkdir(exist_ok=True)
    
    (run1 / "config.json").write_text(json.dumps({"data": {"provider": "binance_real"}}))
    (run1 / "metrics.json").write_text(json.dumps({
        "summary": {"sharpe": 1.931, "cagr": 0.184, "max_drawdown": -0.071, "total_return": 1.426, "win_rate": 0.513},
        "verdict": {"pass": True}
    }))
    import math
    eq_curve = [10000 + i * 20 + math.sin(i / 4) * 300 for i in range(120)]
    (run1 / "result.json").write_text(json.dumps({
        "equity_curve": eq_curve,
        "returns": [0.001] * 120,
        "strategy": {"name": "P91b Champion MVP"}
    }))

    # 7. Generate Ledger Data
    ledger_dir = root / "ledger"
    ledger_dir.mkdir(exist_ok=True)
    (ledger_dir / "ledger.jsonl").write_text(
        json.dumps({"ts": time.time(), "kind": "run", "run_name": "p91b_champion", "payload": {"metrics": {"sharpe": 1.931}}}) + "\n"
    )

    # 8. Generate System Status
    proc_dir = root / "monitoring"
    proc_dir.mkdir(exist_ok=True)

    print("Successfully built the real artifact scaffold with Runs Data for Ops Mode.")

if __name__ == "__main__":
    main()
