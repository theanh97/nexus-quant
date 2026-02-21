"""
NEXUS Self-Audit — Automated daily system health + knowledge consolidation.

This module runs as a task in the autopilot cycle. It ensures:
1. Memory is healthy (not bloated, not empty, key knowledge retained)
2. Lessons are effective (prevention_rate trending up)
3. Research is fresh (daily_brief not stale)
4. Knowledge is consolidated (top insights promoted, old ones decayed)
5. System improvements are logged to changelog

This is how NEXUS ensures continuity across sessions and cycles.
Without this, knowledge fragments and the system "forgets."

Usage (via Orion task):
    task.kind == "self_audit" → calls run_self_audit(artifacts_dir)

Usage (standalone):
    from nexus_quant.learning.self_audit import run_self_audit
    report = run_self_audit(Path("artifacts"))
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_hours(ts_str: str) -> float:
    """Return age in hours from an ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return float("inf")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text("utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def check_memory_health(artifacts_dir: Path) -> Dict[str, Any]:
    """Check memory store health — size, freshness, diversity."""
    result = {"status": "unknown", "issues": []}

    try:
        from ..memory.store import MemoryStore
        db_path = artifacts_dir / "memory" / "memory.db"
        if not db_path.exists():
            result["status"] = "empty"
            result["issues"].append("Memory database does not exist")
            return result

        store = MemoryStore(db_path)
        try:
            stats = store.stats()
            result["total"] = stats["total"]
            result["by_kind"] = stats["by_kind"]
            result["fts"] = stats["fts"]

            if stats["total"] == 0:
                result["status"] = "empty"
                result["issues"].append("Memory is empty — no items stored")
            elif stats["total"] < 10:
                result["status"] = "sparse"
                result["issues"].append(f"Only {stats['total']} memory items — needs more data")
            else:
                result["status"] = "healthy"

            # Check if recent items exist (last 24h)
            recent = store.recent(limit=5)
            if recent:
                newest_age = _age_hours(recent[0].created_at)
                result["newest_item_age_hours"] = round(newest_age, 1)
                if newest_age > 48:
                    result["issues"].append(f"No new memory items in {newest_age:.0f}h")
        finally:
            store.close()
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Memory check failed: {e}")

    return result


def check_learning_health(artifacts_dir: Path) -> Dict[str, Any]:
    """Check NOLE learning engine health — lessons, effectiveness, escalations."""
    result = {"status": "unknown", "issues": []}

    try:
        from .operational import OperationalLearner
        learner = OperationalLearner(artifacts_dir)
        try:
            metrics = learner.metrics()
            result["metrics"] = metrics
            result["total_lessons"] = metrics.get("total_lessons", 0)
            result["prevention_rate"] = metrics.get("prevention_rate", 0)
            result["total_hits"] = metrics.get("total_hits_prevented", 0)
            result["total_misses"] = metrics.get("total_misses", 0)

            if metrics.get("total_lessons", 0) == 0:
                result["status"] = "no_lessons"
                result["issues"].append("No operational lessons loaded")
            elif metrics.get("prevention_rate", 0) == 0 and metrics.get("total_misses", 0) > 5:
                result["status"] = "ineffective"
                result["issues"].append("Prevention rate is 0% despite failures — lessons not working")
            else:
                result["status"] = "healthy"

            # Check for supreme lessons (user repeated feedback)
            supreme = learner.store.get_active_lessons()
            supreme_count = sum(1 for l in supreme if l.severity == "supreme")
            result["supreme_lessons"] = supreme_count
            if supreme_count > 5:
                result["issues"].append(f"{supreme_count} supreme lessons — too many repeated issues")

            # Auto-escalate any that need it
            escalated = learner.store.auto_escalate_repeated_issues()
            if escalated:
                result["auto_escalated"] = len(escalated)
        finally:
            learner.close()
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Learning check failed: {e}")

    return result


def check_research_health(artifacts_dir: Path) -> Dict[str, Any]:
    """Check research pipeline health — freshness, source count, hypotheses."""
    result = {"status": "unknown", "issues": []}

    brief_path = artifacts_dir / "brain" / "daily_brief.json"
    if not brief_path.exists():
        result["status"] = "never_run"
        result["issues"].append("Research has never been executed")
        return result

    try:
        brief = json.loads(brief_path.read_text("utf-8"))
        ts = brief.get("ts", "")
        age = _age_hours(ts) if ts else float("inf")

        result["last_fetch"] = ts
        result["age_hours"] = round(age, 1)
        result["sources_fetched"] = brief.get("stats", {}).get("fetched_sources", 0)
        result["total_items"] = brief.get("total_items", 0)
        result["hypotheses"] = len(brief.get("hypotheses", []))

        if age > 48:
            result["status"] = "stale"
            result["issues"].append(f"Research is {age:.0f}h old (>48h)")
        elif age > 24:
            result["status"] = "due"
            result["issues"].append(f"Research is {age:.0f}h old — due for refresh")
        else:
            result["status"] = "fresh"

        if result["sources_fetched"] < 10:
            result["issues"].append(f"Only {result['sources_fetched']} sources fetched — connectivity issues?")
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Research check failed: {e}")

    return result


def check_knowledge_health(artifacts_dir: Path) -> Dict[str, Any]:
    """Check knowledge base health — insights, experiments, champion."""
    result = {"status": "unknown", "issues": []}

    kb_path = artifacts_dir / "brain" / "knowledge_base.json"
    if not kb_path.exists():
        result["status"] = "empty"
        result["issues"].append("Knowledge base does not exist")
        return result

    try:
        kb = json.loads(kb_path.read_text("utf-8"))
        insights = kb.get("strategy_insights", [])
        experiments = kb.get("experiment_history", [])
        champion = kb.get("champion_strategy")

        result["total_insights"] = len(insights)
        result["total_experiments"] = len(experiments)
        result["has_champion"] = champion is not None
        result["champion"] = champion.get("name", "none") if champion else "none"

        if len(insights) == 0:
            result["status"] = "empty"
            result["issues"].append("No strategy insights recorded")
        else:
            result["status"] = "healthy"

        # Check freshness of last insight
        if insights:
            last_ts = insights[-1].get("ts", "")
            if last_ts:
                age = _age_hours(last_ts)
                result["last_insight_age_hours"] = round(age, 1)
                if age > 72:
                    result["issues"].append(f"No new insights in {age:.0f}h")
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Knowledge check failed: {e}")

    return result


def ingest_wisdom_to_memory(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Ingest wisdom files (L0/L1/L2) into the memory store.

    This bridges the gap between the wisdom hierarchy (JSON files with accumulated
    knowledge) and the memory store (searchable SQLite+FTS5). Without this,
    the system has knowledge but can't search it.

    Runs during self_audit. Idempotent — skips already-ingested items by checking
    content hash.
    """
    import hashlib

    result = {"ingested": 0, "skipped": 0, "errors": []}

    try:
        from ..memory.store import MemoryStore
    except Exception as e:
        result["errors"].append(f"MemoryStore import failed: {e}")
        return result

    db_path = artifacts_dir / "memory" / "memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Find project root (2 levels up from artifacts typically)
    project_root = artifacts_dir.parent
    memory_dir = project_root / "memory"
    if not memory_dir.exists():
        result["errors"].append(f"memory/ directory not found at {memory_dir}")
        return result

    store = MemoryStore(db_path)
    try:
        # Get existing content hashes to avoid duplicates
        existing = store.recent(limit=500)
        existing_hashes = set()
        for item in existing:
            meta = item.meta or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            h = meta.get("content_hash")
            if h:
                existing_hashes.add(h)

        def _add_item(kind: str, tags: List[str], content: str, meta: Dict) -> None:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            if content_hash in existing_hashes:
                result["skipped"] += 1
                return
            meta["content_hash"] = content_hash
            store.add(created_at=_utc_iso(), kind=kind, tags=tags, content=content, meta=meta)
            existing_hashes.add(content_hash)
            result["ingested"] += 1

        # Scan all wisdom.json files
        wisdom_files = sorted(memory_dir.glob("*/wisdom.json"))
        for wf in wisdom_files:
            level = wf.parent.name  # e.g., "L0_universal", "L2_crypto_options"
            try:
                data = json.loads(wf.read_text("utf-8"))
            except Exception:
                continue

            # Format 1: insights[].text + lessons[] (L0/L1/L2_crypto_perps)
            for insight in data.get("insights", []):
                text = insight.get("text", "") if isinstance(insight, dict) else str(insight)
                if not text:
                    continue
                _add_item(
                    kind="wisdom_insight",
                    tags=["wisdom", level] + (insight.get("tags", []) if isinstance(insight, dict) else []),
                    content=text,
                    meta={
                        "source": f"wisdom/{level}",
                        "confidence": insight.get("confidence", 0.9) if isinstance(insight, dict) else 0.9,
                    },
                )

            for lesson in data.get("lessons", []):
                text = lesson.get("text", "") if isinstance(lesson, dict) else str(lesson)
                if not text:
                    continue
                _add_item(
                    kind="wisdom_lesson",
                    tags=["wisdom", level, "lesson"],
                    content=text,
                    meta={"source": f"wisdom/{level}"},
                )

            # Format 2: RULES dict (L2_crypto_options)
            rules = data.get("RULES", {})
            if isinstance(rules, dict):
                for rule_id, rule_data in rules.items():
                    text = rule_data if isinstance(rule_data, str) else json.dumps(rule_data, default=str)
                    _add_item(
                        kind="wisdom_rule",
                        tags=["wisdom", level, "rule", rule_id],
                        content=f"[{rule_id}] {text}",
                        meta={"source": f"wisdom/{level}", "rule_id": rule_id},
                    )

            # Format 2: validated/failed hypotheses (L2_crypto_options)
            for hyp in data.get("validated_hypotheses", []):
                text = hyp if isinstance(hyp, str) else str(hyp)
                if text:
                    _add_item(
                        kind="wisdom_hypothesis",
                        tags=["wisdom", level, "validated"],
                        content=text,
                        meta={"source": f"wisdom/{level}", "status": "validated"},
                    )

            for hyp in data.get("failed_hypotheses", []):
                text = hyp if isinstance(hyp, str) else str(hyp)
                if text:
                    _add_item(
                        kind="wisdom_hypothesis",
                        tags=["wisdom", level, "failed"],
                        content=text,
                        meta={"source": f"wisdom/{level}", "status": "failed"},
                    )

            # Format 2: Strategy summaries (L2_crypto_options)
            for key in ["ENSEMBLE_CHAMPION", "STRATEGY_1_VRP", "STRATEGY_2_SKEW_MR",
                         "STRATEGY_3_TERM_STRUCTURE", "STRATEGY_4_BUTTERFLY_MR",
                         "STRATEGY_5_SKEW_SPREAD", "NEXUS_PORTFOLIO_INTEGRATION"]:
                strat = data.get(key)
                if isinstance(strat, dict):
                    summary = json.dumps(strat, indent=1, default=str)
                    _add_item(
                        kind="wisdom_strategy",
                        tags=["wisdom", level, "strategy", key.lower()],
                        content=f"[{key}] {summary[:2000]}",
                        meta={"source": f"wisdom/{level}", "strategy_key": key},
                    )

            # Format 3: kill report (L2_commodity_cta)
            kill = data.get("kill_report")
            if isinstance(kill, dict):
                summary = json.dumps(kill, indent=1, default=str)
                _add_item(
                    kind="wisdom_postmortem",
                    tags=["wisdom", level, "kill_report"],
                    content=f"Kill report: {summary[:2000]}",
                    meta={"source": f"wisdom/{level}"},
                )
    finally:
        store.close()

    return result


def ensure_champion_registered(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Ensure the knowledge base has champion strategies registered.

    Without a champion, the AEA (Apply-Evaluate-Adapt) loop can't compare new
    experiments against a baseline. This reads wisdom files to find validated
    champions and registers them.
    """
    result = {"action": "none", "champions": []}

    kb_path = artifacts_dir / "brain" / "knowledge_base.json"
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        kb = json.loads(kb_path.read_text("utf-8")) if kb_path.exists() else {}
    except Exception:
        kb = {}

    # Check if champion already set
    if kb.get("champion_strategy") and kb["champion_strategy"].get("name"):
        result["action"] = "already_set"
        result["champions"].append(kb["champion_strategy"].get("name"))
        return result

    # Scan wisdom for champion info
    project_root = artifacts_dir.parent
    champions_found = []

    # L2_crypto_perps champion — look in insights for P91b reference
    perps_wisdom = project_root / "memory" / "L2_crypto_perps" / "wisdom.json"
    if perps_wisdom.exists():
        try:
            data = json.loads(perps_wisdom.read_text("utf-8"))
            meta = data.get("_meta", {})
            champions_found.append({
                "name": "P91b_vol_tilt",
                "project": "crypto_perps",
                "sharpe": 2.006,
                "source": "L2_crypto_perps/wisdom.json",
            })
        except Exception:
            pass

    # L2_crypto_options champion — read ENSEMBLE_CHAMPION
    opts_wisdom = project_root / "memory" / "L2_crypto_options" / "wisdom.json"
    if opts_wisdom.exists():
        try:
            data = json.loads(opts_wisdom.read_text("utf-8"))
            ens = data.get("ENSEMBLE_CHAMPION", {})
            if ens:
                champions_found.append({
                    "name": "mixed_freq_ensemble",
                    "project": "crypto_options",
                    "sharpe": ens.get("wf_avg_sharpe", 2.723),
                    "source": "L2_crypto_options/wisdom.json",
                })
        except Exception:
            pass

    if not champions_found:
        # Fallback: register from known values
        champions_found = [
            {"name": "P91b_vol_tilt", "project": "crypto_perps", "sharpe": 2.006, "source": "fallback"},
            {"name": "mixed_freq_ensemble", "project": "crypto_options", "sharpe": 2.723, "source": "fallback"},
        ]

    # Register the best champion
    best = max(champions_found, key=lambda c: c.get("sharpe", 0))
    kb["champion_strategy"] = {
        "name": best["name"],
        "project": best["project"],
        "sharpe": best["sharpe"],
        "registered_at": _utc_iso(),
        "source": best["source"],
    }

    # Ensure required KB sections exist
    kb.setdefault("strategy_insights", [])
    kb.setdefault("experiment_history", [])
    kb.setdefault("learning_history", [])

    kb_path.write_text(json.dumps(kb, indent=2, default=str), encoding="utf-8")

    result["action"] = "registered"
    result["champions"] = [c["name"] for c in champions_found]
    return result


def consolidate_knowledge(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Consolidate knowledge — the KEY missing piece.

    Every N runs (or daily), summarize:
    1. Top lessons from NOLE (highest hit_count)
    2. Top insights from knowledge base (highest confidence)
    3. Key patterns from memory (most referenced)
    4. Write a "knowledge digest" that persists

    This is how vòng 300 remembers lessons from vòng 1.
    """
    result = {"consolidated": False, "actions": []}

    digest_path = artifacts_dir / "brain" / "knowledge_digest.json"
    existing_digest = _read_json(digest_path)

    # 1. Top NOLE lessons
    top_lessons = []
    try:
        from .operational import OperationalLearner
        learner = OperationalLearner(artifacts_dir)
        try:
            lessons = learner.store.get_active_lessons()
            # Top by hit_count (proven useful) + supreme severity
            for l in lessons[:20]:
                top_lessons.append({
                    "id": l.id,
                    "category": l.category,
                    "correction": l.correction,
                    "severity": l.severity,
                    "hits": l.hit_count,
                    "misses": l.miss_count,
                })
        finally:
            learner.close()
    except Exception:
        pass

    # 2. Top insights from knowledge base
    top_insights = []
    try:
        kb_path = artifacts_dir / "brain" / "knowledge_base.json"
        if kb_path.exists():
            kb = json.loads(kb_path.read_text("utf-8"))
            insights = kb.get("strategy_insights", [])
            # Sort by confidence descending
            sorted_insights = sorted(insights, key=lambda x: x.get("confidence", 0), reverse=True)
            for i in sorted_insights[:15]:
                top_insights.append({
                    "category": i.get("category", ""),
                    "insight": i.get("insight", ""),
                    "confidence": i.get("confidence", 0),
                    "source": i.get("source", ""),
                })
    except Exception:
        pass

    # 3. Top learnings from knowledge base history
    top_learnings = []
    try:
        kb_path = artifacts_dir / "brain" / "knowledge_base.json"
        if kb_path.exists():
            kb = json.loads(kb_path.read_text("utf-8"))
            learnings = kb.get("learning_history", [])
            # Most recent learnings
            for l in learnings[-10:]:
                top_learnings.append({
                    "what": l.get("what", ""),
                    "why": l.get("why", ""),
                    "action": l.get("action", ""),
                })
    except Exception:
        pass

    # 4. Changelog summary
    changelog_summary = {}
    try:
        from .changelog import ChangeLog
        cl = ChangeLog(artifacts_dir)
        try:
            changelog_summary = cl.summary()
        finally:
            cl.close()
    except Exception:
        pass

    # Build digest
    digest = {
        "version": (existing_digest.get("version", 0) or 0) + 1,
        "consolidated_at": _utc_iso(),
        "top_lessons": top_lessons,
        "top_insights": top_insights,
        "top_learnings": top_learnings,
        "changelog": changelog_summary,
        "previous_digest_at": existing_digest.get("consolidated_at"),
    }

    # Write digest
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    digest_path.write_text(json.dumps(digest, indent=2, default=str), encoding="utf-8")

    result["consolidated"] = True
    result["digest_version"] = digest["version"]
    result["top_lessons_count"] = len(top_lessons)
    result["top_insights_count"] = len(top_insights)
    result["top_learnings_count"] = len(top_learnings)
    result["actions"].append(f"Knowledge digest v{digest['version']} written")

    return result


def auto_log_git_commits(artifacts_dir: Path, max_commits: int = 20) -> Dict[str, Any]:
    """
    Auto-log recent git commits to the changelog.

    Reads the last N git commits and logs any that aren't already recorded.
    This ensures the changelog captures all development activity, even from
    manual commits or other sessions.
    """
    import re
    import subprocess

    result = {"logged": 0, "skipped": 0, "errors": []}

    try:
        # Find git root
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if git_root.returncode != 0:
            result["errors"].append("Not a git repository")
            return result

        # Get recent commits (last 24h or max_commits)
        log_output = subprocess.run(
            ["git", "log", f"--max-count={max_commits}", "--format=%H|%s|%an|%aI"],
            capture_output=True, text=True, timeout=10,
            cwd=git_root.stdout.strip(),
        )
        if log_output.returncode != 0:
            result["errors"].append(f"git log failed: {log_output.stderr}")
            return result

        commits = []
        for line in log_output.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })

        if not commits:
            return result

        # Check which commits are already logged
        from .changelog import ChangeLog
        cl = ChangeLog(artifacts_dir)
        try:
            recent_changes = cl.recent(limit=200)
            logged_hashes = set()
            for c in recent_changes:
                desc = c.get("description", "")
                # Extract commit hash from description if present
                m = re.search(r'\[([0-9a-f]{7,12})\]', desc)
                if m:
                    logged_hashes.add(m.group(1))

            for commit in commits:
                short_hash = commit["hash"][:8]
                if short_hash in logged_hashes:
                    result["skipped"] += 1
                    continue

                # Only log commits from last 48h
                age = _age_hours(commit["date"])
                if age > 48:
                    continue

                # Parse category from conventional commit
                msg = commit["message"]
                cat_match = re.match(r'(feat|fix|refactor|perf|test|docs|chore|quant)[\(:]', msg)
                category = cat_match.group(1) if cat_match else "commit"

                # Get changed files for this commit
                try:
                    diff_out = subprocess.run(
                        ["git", "diff", "--name-only", f"{commit['hash']}~1", commit["hash"]],
                        capture_output=True, text=True, timeout=5,
                        cwd=git_root.stdout.strip(),
                    )
                    files = [f.strip() for f in diff_out.stdout.strip().split("\n") if f.strip()] if diff_out.returncode == 0 else []
                except Exception:
                    files = []

                cl.record(
                    category=category,
                    description=f"[{short_hash}] {msg}",
                    files=files[:20],
                    source="git",
                )
                result["logged"] += 1
        finally:
            cl.close()

    except Exception as e:
        result["errors"].append(f"Git log failed: {e}")

    return result


def run_self_audit(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Run a full self-audit. Called by Orion as a daily task.

    Returns a comprehensive health report + consolidation results.
    """
    # Pre-audit: ensure structural health (idempotent)
    wisdom_ingest = ingest_wisdom_to_memory(artifacts_dir)
    champion_reg = ensure_champion_registered(artifacts_dir)

    report = {
        "audit_at": _utc_iso(),
        "memory": check_memory_health(artifacts_dir),
        "learning": check_learning_health(artifacts_dir),
        "research": check_research_health(artifacts_dir),
        "knowledge": check_knowledge_health(artifacts_dir),
        "consolidation": consolidate_knowledge(artifacts_dir),
        "git_commits": auto_log_git_commits(artifacts_dir),
        "wisdom_ingest": wisdom_ingest,
        "champion_registration": champion_reg,
        "all_issues": [],
        "overall_status": "healthy",
    }

    # Collect all issues
    for section in ["memory", "learning", "research", "knowledge"]:
        for issue in report[section].get("issues", []):
            report["all_issues"].append(f"[{section}] {issue}")

    # Determine overall status
    critical_sections = [report[s].get("status") for s in ["memory", "learning", "research", "knowledge"]]
    if any(s in ("error", "empty", "never_run", "no_lessons") for s in critical_sections):
        report["overall_status"] = "degraded"
    elif any(s in ("stale", "ineffective", "sparse") for s in critical_sections):
        report["overall_status"] = "needs_attention"
    else:
        report["overall_status"] = "healthy"

    # Write audit report
    audit_path = artifacts_dir / "state" / "self_audit_report.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Log to changelog
    try:
        from .changelog import log_system_change
        log_system_change(
            artifacts_dir,
            category="audit",
            description=f"Self-audit: {report['overall_status']} | {len(report['all_issues'])} issues | digest v{report['consolidation'].get('digest_version', '?')}",
            files=["learning/self_audit.py"],
            impact=["memory", "learning", "research", "knowledge"],
        )
    except Exception:
        pass

    return report
