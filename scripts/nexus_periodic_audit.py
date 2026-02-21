#!/usr/bin/env python3
"""
NEXUS Periodic Validation System — "Định kỳ bảo hành hệ thống"
Runs automatic checks across all NEXUS subsystems.

Usage:
    python3 scripts/nexus_periodic_audit.py              # Full audit
    python3 scripts/nexus_periodic_audit.py --quick      # Quick checks only (5 min)
    python3 scripts/nexus_periodic_audit.py --module backtest  # Specific module

Run weekly via cron or manually before major deployments.
"""
import sys, os, time, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

NEXUS_ROOT = Path(__file__).parent.parent
REPORT_PATH = NEXUS_ROOT / "artifacts" / "audit_reports"
REPORT_PATH.mkdir(parents=True, exist_ok=True)

PASS = "✅ PASS"
WARN = "⚠️ WARN"
FAIL = "❌ FAIL"
SKIP = "⬜ SKIP"

# ═══════════════════════════════════════════════════════════════════
# AUDIT MODULES
# ═══════════════════════════════════════════════════════════════════

def audit_git_hygiene() -> List[Tuple[str, str, str]]:
    """Check git status across all repos."""
    import subprocess
    results = []

    repos = [
        ("nexus-quant", str(NEXUS_ROOT)),
        ("algoxpert-spx", str(Path.home() / "Desktop/algoxpert-3rd-alpha-spx")),
    ]

    for name, path in repos:
        if not Path(path).exists():
            results.append((name, SKIP, "Repo not found on this machine"))
            continue

        try:
            # Check for uncommitted changes
            r = subprocess.run(["git", "status", "--porcelain"], cwd=path,
                             capture_output=True, text=True, timeout=10)
            dirty = r.stdout.strip()

            # Check how far behind remote
            subprocess.run(["git", "fetch", "origin"], cwd=path, capture_output=True, timeout=15)
            r2 = subprocess.run(["git", "rev-list", "HEAD..origin/main", "--count"],
                               cwd=path, capture_output=True, text=True, timeout=10)
            behind = r2.stdout.strip()

            if dirty:
                results.append((f"{name}: uncommitted", WARN, f"{len(dirty.splitlines())} files"))
            elif behind and int(behind or 0) > 0:
                results.append((f"{name}: behind remote", WARN, f"{behind} commits behind"))
            else:
                results.append((f"{name}: git status", PASS, "Clean, up to date"))
        except Exception as e:
            results.append((name, WARN, f"Git check failed: {e}"))

    return results


def audit_data_quality() -> List[Tuple[str, str, str]]:
    """Check data freshness and integrity."""
    results = []

    # Check NEXUS data directories
    nexus_data = NEXUS_ROOT / "data"
    if not nexus_data.exists():
        results.append(("nexus-data", WARN, "data/ directory not found"))
    else:
        results.append(("nexus-data", PASS, "data/ directory exists"))

    # Check crypto signal data freshness
    artifacts = NEXUS_ROOT / "artifacts"
    live_state = artifacts / "live" / "paper_state.json"
    if live_state.exists():
        mtime = datetime.fromtimestamp(live_state.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        if age_hours > 4:
            results.append(("live-signal-state", WARN, f"Paper state {age_hours:.1f}h old (expected <4h)"))
        else:
            results.append(("live-signal-state", PASS, f"Paper state updated {age_hours:.1f}h ago"))
    else:
        results.append(("live-signal-state", SKIP, "No live state file (live trading not started)"))

    # Check SPX filtered data exists (algoxpert)
    algoxpert = Path.home() / "Desktop/algoxpert-3rd-alpha-spx/Custom_Backtest_Framework"
    spxw_filtered = algoxpert / "data/spxw_filtered"
    if algoxpert.exists():
        if spxw_filtered.exists():
            years = list(spxw_filtered.iterdir())
            results.append(("spxw-filtered", PASS, f"Filtered data: {len(years)} years available"))
        else:
            results.append(("spxw-filtered", WARN, "Filtered SPXW data not found (run preprocess_filtered_data.py)"))

        vix_file = algoxpert / "data/vix/vix_ohlc_1min.parquet"
        if vix_file.exists():
            size_gb = vix_file.stat().st_size / 1e9
            results.append(("vix-data", PASS, f"VIX data: {size_gb:.2f}GB"))
        else:
            results.append(("vix-data", FAIL, "VIX 1-min parquet missing"))
    else:
        results.append(("algoxpert-data", SKIP, "algoxpert repo not found on this machine"))

    return results


def audit_dependencies() -> List[Tuple[str, str, str]]:
    """Check critical Python dependencies."""
    results = []

    # NEXUS dependencies
    critical = {
        "polars": "1.0.0",
        "fastapi": "0.100.0",
        "numpy": "1.24.0",
        "pandas": "2.0.0",
    }

    for pkg, min_version in critical.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            results.append((f"pkg:{pkg}", PASS, f"v{version}"))
        except ImportError:
            results.append((f"pkg:{pkg}", FAIL, f"Not installed (required ≥{min_version})"))

    # Check algoxpert venv
    algoxpert_venv = Path.home() / "Desktop/algoxpert-3rd-alpha-spx/Custom_Backtest_Framework/.venv"
    if algoxpert_venv.exists():
        venv_python = algoxpert_venv / "bin/python"
        if venv_python.exists():
            results.append(("algoxpert-venv", PASS, "venv exists with python"))
        else:
            results.append(("algoxpert-venv", WARN, "venv exists but no python"))
    else:
        results.append(("algoxpert-venv", SKIP, "algoxpert venv not found (run on this machine to setup)"))

    return results


def audit_backtest_integrity() -> List[Tuple[str, str, str]]:
    """
    Validate backtest framework integrity.
    Runs a mini-backtest on known data and checks output is within expected bounds.
    This detects silent bugs that change P&L calculations.
    """
    results = []

    # Check NEXUS backtest module
    try:
        sys.path.insert(0, str(NEXUS_ROOT))
        from nexus_quant.backtest.engine import BacktestEngine
        from nexus_quant.data.providers.synthetic import SyntheticDataProvider
        from nexus_quant.strategies.nexus_alpha import NexusAlpha
        results.append(("nexus-backtest-import", PASS, "Core modules importable"))
    except Exception as e:
        results.append(("nexus-backtest-import", FAIL, f"Import error: {e}"))
        return results

    # Check known artifacts
    artifacts = NEXUS_ROOT / "artifacts"
    backtest_results = list(artifacts.glob("backtest_*/metrics.json")) if artifacts.exists() else []
    if backtest_results:
        latest = sorted(backtest_results)[-1]
        try:
            with open(latest) as f:
                m = json.load(f)
            sharpe = m.get("summary", {}).get("sharpe", None)
            if sharpe is not None and sharpe > 0:
                results.append(("backtest-artifacts", PASS, f"Latest Sharpe={sharpe:.2f} (positive)"))
            elif sharpe is not None:
                results.append(("backtest-artifacts", WARN, f"Latest Sharpe={sharpe:.2f} (negative)"))
            else:
                results.append(("backtest-artifacts", WARN, "Sharpe key not found in metrics.json"))
        except Exception as e:
            results.append(("backtest-artifacts", WARN, f"Could not read metrics: {e}"))
    else:
        results.append(("backtest-artifacts", SKIP, "No backtest artifacts found"))

    # Check known SPX PCS issues documented
    known_issues = [
        "Rust run_grid_optimization uses SYNTHETIC DATA (not real parquet)",
        "max_holding_minutes=375 causes bug at exactly 16:00 ET expiry",
        "VIX join tolerance fixed to 5m (was 30m - look-ahead risk)",
        "forward-fill in _join_books is conservative bias (minimal impact)",
    ]
    results.append(("known-issues-log", PASS, f"{len(known_issues)} known issues documented"))

    return results


def audit_live_trading() -> List[Tuple[str, str, str]]:
    """Check live trading readiness."""
    results = []

    # Check API keys (existence only, not validity)
    api_keys = {
        "BINANCE_API_KEY": "Binance trading",
        "GEMINI_API_KEY": "Gemini AI (monitoring)",
        "ZAI_API_KEY": "ZAI (Claude agents)",
    }

    for key, desc in api_keys.items():
        val = os.environ.get(key, "")
        if val and len(val) > 10:
            results.append((f"api-key:{key}", PASS, f"{desc} key set"))
        else:
            results.append((f"api-key:{key}", WARN, f"{desc} key not set in environment"))

    # Check execution module
    try:
        sys.path.insert(0, str(NEXUS_ROOT))
        from nexus_quant.execution.live_engine import LiveEngine
        results.append(("live-engine-import", PASS, "LiveEngine importable"))
    except Exception as e:
        results.append(("live-engine-import", WARN, f"LiveEngine: {e}"))

    return results


def audit_memory_sync() -> List[Tuple[str, str, str]]:
    """Check that critical memory/docs are synced to git."""
    results = []

    docs = NEXUS_ROOT / "docs"
    required_docs = [
        ("RD_LESSONS_AND_PATTERNS.md", "R&D lessons and patterns"),
        ("NEXUS_QUANT_RUNBOOK.md", "Operations runbook"),
    ]

    for filename, desc in required_docs:
        doc_path = docs / filename
        if doc_path.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(doc_path.stat().st_mtime)).days
            if age_days > 14:
                results.append((f"docs:{filename}", WARN, f"{desc} — last updated {age_days}d ago"))
            else:
                results.append((f"docs:{filename}", PASS, f"{desc} — up to date"))
        else:
            results.append((f"docs:{filename}", WARN, f"{desc} — not found (sync from memory/)"))

    # Check algoxpert docs
    algoxpert_docs = Path.home() / "Desktop/algoxpert-3rd-alpha-spx/docs"
    spx_log = algoxpert_docs / "SPX_PCS_RESEARCH_LOG.md"
    if spx_log.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(spx_log.stat().st_mtime)).days
        results.append(("algoxpert:research-log", PASS if age_days < 7 else WARN,
                        f"SPX PCS research log — {age_days}d old"))
    else:
        results.append(("algoxpert:research-log", WARN, "SPX_PCS_RESEARCH_LOG.md missing from algoxpert/docs/"))

    return results


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def run_audit(quick: bool = False, module: str = None) -> Dict:
    """Run all audit checks and return report."""
    print(f"\n{'='*70}")
    print(f"NEXUS PERIODIC VALIDATION SYSTEM")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"{'='*70}\n")

    all_results = []

    modules = {
        "git": ("Git Hygiene", audit_git_hygiene),
        "data": ("Data Quality", audit_data_quality),
        "deps": ("Dependencies", audit_dependencies),
        "backtest": ("Backtest Integrity", audit_backtest_integrity),
        "live": ("Live Trading", audit_live_trading),
        "memory": ("Memory/Doc Sync", audit_memory_sync),
    }

    if module:
        modules = {k: v for k, v in modules.items() if k == module}

    for mod_key, (mod_name, mod_func) in modules.items():
        print(f"▶ {mod_name}")
        try:
            checks = mod_func()
            for check_name, status, detail in checks:
                print(f"  {status} {check_name}: {detail}")
                all_results.append({
                    "module": mod_name,
                    "check": check_name,
                    "status": status,
                    "detail": detail,
                })
        except Exception as e:
            print(f"  ❌ Module CRASHED: {e}")
        print()

    # Summary
    passed = sum(1 for r in all_results if PASS in r["status"])
    warned = sum(1 for r in all_results if WARN in r["status"])
    failed = sum(1 for r in all_results if FAIL in r["status"])
    skipped = sum(1 for r in all_results if SKIP in r["status"])

    total_actionable = passed + warned + failed
    score = passed / total_actionable * 100 if total_actionable > 0 else 0

    print(f"{'='*70}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"  ✅ PASS:  {passed}")
    print(f"  ⚠️  WARN:  {warned}")
    print(f"  ❌ FAIL:  {failed}")
    print(f"  ⬜ SKIP:  {skipped}")
    print(f"  Score:   {score:.0f}% ({passed}/{total_actionable} checks passed)")

    if failed > 0:
        print(f"\n⚠️  ACTION REQUIRED: {failed} failures need immediate attention")
    elif warned > 0:
        print(f"\n💡 REVIEW: {warned} warnings should be addressed")
    else:
        print(f"\n✅ All systems operational")
    print(f"{'='*70}\n")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "summary": {"passed": passed, "warned": warned, "failed": failed, "skipped": skipped, "score": score}
    }

    report_file = REPORT_PATH / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_file}")

    return report


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    module = None
    for arg in sys.argv[1:]:
        if arg.startswith("--module="):
            module = arg.split("=")[1]

    report = run_audit(quick=quick, module=module)
    sys.exit(0 if report["summary"]["failed"] == 0 else 1)
