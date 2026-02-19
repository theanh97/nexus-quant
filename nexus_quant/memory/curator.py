from __future__ import annotations

"""
NEXUS Memory Curator.

This module distills raw backtest run artifacts into compact, de-duplicated,
scored semantic knowledge that can be used by RAG and agentic workflows.

Constraints:
- stdlib-only (json, pathlib, datetime, math, re, collections)
"""

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


_RUN_TS_RE = re.compile(r"(?P<ts>\\d{8}T\\d{6}Z)$")
_HEX_RE = re.compile(r"^[0-9a-f]{8,}$", flags=re.IGNORECASE)


class MemoryCurator:
    """
    Curates run artifacts into semantic, long-lived knowledge.

    Files maintained (under artifacts/memory/semantic):
    - curated_knowledge.json (max 200 entries, evicts lowest-scored)
    - strategy_registry.json (one entry per strategy base name)
    - lessons_learned.jsonl (append-only log)
    """

    def __init__(self, artifacts_dir: Path) -> None:
        """
        Create a curator rooted at a given artifacts directory.

        Args:
            artifacts_dir: Path to the artifacts root (contains runs/, memory/, ...).
        """

        self.artifacts_dir = Path(artifacts_dir)
        self.runs_dir = self.artifacts_dir / "runs"
        self.semantic_dir = self.artifacts_dir / "memory" / "semantic"
        self.semantic_dir.mkdir(parents=True, exist_ok=True)

        self.curated_knowledge_path = self.semantic_dir / "curated_knowledge.json"
        self.strategy_registry_path = self.semantic_dir / "strategy_registry.json"
        self.lessons_path = self.semantic_dir / "lessons_learned.jsonl"

    # ── Public API ──────────────────────────────────────────────────────────

    def curate(self) -> dict:
        """
        Run full curation pipeline and persist outputs.

        Returns:
            Summary dict (counts, paths, key warnings).
        """

        now = datetime.now(timezone.utc)
        runs = self._scan_runs()

        registry, per_strategy = self._build_strategy_registry(runs)
        new_knowledge = self.compress_runs_to_knowledge()

        # Enforce "top-3 per strategy" by treating compress_runs_to_knowledge() output
        # as the authoritative set of run-derived memories.
        prev = self._read_json(self.curated_knowledge_path)
        prev_keys = {
            e.get("memory_key")
            for e in (prev or [])
            if isinstance(e, dict) and isinstance(e.get("memory_key"), str)
        } if isinstance(prev, list) else set()
        new_keys = {
            e.get("memory_key")
            for e in new_knowledge
            if isinstance(e, dict) and isinstance(e.get("memory_key"), str)
        }
        added = len(new_keys - prev_keys)
        removed = len(prev_keys - new_keys)

        evicted, kept = self._evict_entries(new_knowledge, max_entries=200)
        self._write_json(self.curated_knowledge_path, kept)
        self._write_json(self.strategy_registry_path, registry)

        repeat_warnings = self._detect_repeating_experiments(per_strategy)
        stagnation = self.detect_stagnation()
        stagnation_notes = self._log_stagnation_lessons(stagnation)

        return {
            "ok": True,
            "generated_at": now.isoformat(),
            "runs_scanned": len(runs),
            "strategies": len(per_strategy),
            "knowledge_entries_written": len(kept),
            "knowledge_entries_added": added,
            "knowledge_entries_removed": removed,
            "knowledge_entries_evicted": evicted,
            "paths": {
                "curated_knowledge": str(self.curated_knowledge_path),
                "strategy_registry": str(self.strategy_registry_path),
                "lessons_learned": str(self.lessons_path),
            },
            "repeat_warnings": repeat_warnings,
            "stagnation": stagnation,
            "stagnation_lessons_logged": stagnation_notes,
        }

    def get_strategy_registry(self) -> dict:
        """
        Return the strategy registry (what strategies exist and best results).

        If no registry exists yet, it is built from current run artifacts.
        """

        obj = self._read_json(self.strategy_registry_path)
        if isinstance(obj, dict) and obj:
            return obj

        runs = self._scan_runs()
        registry, _ = self._build_strategy_registry(runs)
        return registry

    def get_lessons_learned(self, limit: int = 20) -> list:
        """
        Return most recent lessons from lessons_learned.jsonl.

        Args:
            limit: Max number of items returned (most recent first).
        """

        try:
            n = int(limit)
        except Exception:
            n = 20
        n = max(1, min(n, 200))
        if not self.lessons_path.exists():
            return []

        lines = self.lessons_path.read_text("utf-8", errors="replace").splitlines()
        tail = lines[-n:]
        out: list = []
        for ln in reversed(tail):
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out

    def check_experiment_novelty(self, strategy_name: str, config: dict) -> dict:
        """
        Check whether a proposed experiment looks novel enough to run.

        Heuristics:
        - If the same config signature has been run >5x for this strategy, recommend not running.
        - If the strategy is stagnating (no improvement in last 5), recommend exploring a new direction.

        Args:
            strategy_name: Strategy base name (e.g., "binance_nexus_alpha_v1").
            config: Candidate config dict (like a run config.json payload).

        Returns:
            Dict with should_run, reason, suggestion, and context fields.
        """

        strategy = (strategy_name or "").strip()
        if not strategy:
            return {
                "should_run": False,
                "reason": "Missing strategy_name.",
                "suggestion": "Provide a strategy base name.",
            }

        sig = self._config_sig(config)
        runs = self._scan_runs()
        same_strategy = [r for r in runs if r.get("strategy") == strategy]
        sig_counts = Counter([r.get("config_sig") for r in same_strategy if r.get("config_sig")])
        seen = int(sig_counts.get(sig, 0))

        stagnation = self._detect_stagnation_for_strategy(same_strategy)
        if seen > 5:
            return {
                "should_run": False,
                "reason": f"Config already run {seen} times for strategy '{strategy}'.",
                "suggestion": "Change strategy params, data window, rebalance frequency, or risk budget to explore a new region.",
                "strategy": strategy,
                "config_sig": sig,
                "seen_count": seen,
                "stagnation": stagnation,
            }

        if stagnation.get("stagnating"):
            return {
                "should_run": True,
                "reason": f"Strategy '{strategy}' appears stagnant; novelty is valuable.",
                "suggestion": (stagnation.get("suggestion") or "Explore a new direction (features, universe, timeframe, or constraints)."),
                "strategy": strategy,
                "config_sig": sig,
                "seen_count": seen,
                "stagnation": stagnation,
            }

        return {
            "should_run": True,
            "reason": f"Config signature unseen or not overused for strategy '{strategy}'.",
            "suggestion": "Proceed; if Sharpe fails to improve, widen exploration (new features / constraints).",
            "strategy": strategy,
            "config_sig": sig,
            "seen_count": seen,
            "stagnation": stagnation,
        }

    def compress_runs_to_knowledge(self) -> list:
        """
        Convert run artifacts into semantic facts suitable for curated_knowledge.json.

        Returns:
            List of dicts with at least: insight, evidence, confidence, date.
        """

        now = datetime.now(timezone.utc)
        runs = self._scan_runs()
        if not runs:
            return []

        grouped: dict = defaultdict(list)
        for r in runs:
            grouped[r.get("strategy") or "unknown"].append(r)

        # Precompute config frequencies per strategy (for novelty).
        config_freq: dict = {}
        for strat, items in grouped.items():
            c = Counter([it.get("config_sig") for it in items if it.get("config_sig")])
            config_freq[strat] = c

        existing_keys = self._existing_memory_keys()
        out: list = []

        for strat, items in grouped.items():
            # De-dup by config_sig and keep best run per unique config.
            best_per_sig: dict = {}
            for r in items:
                sig = r.get("config_sig") or ""
                key = sig or r.get("run_id")
                if not key:
                    continue
                prev = best_per_sig.get(key)
                if prev is None or self._safe_float(r.get("sharpe")) > self._safe_float(prev.get("sharpe")):
                    best_per_sig[key] = r

            deduped = list(best_per_sig.values())
            deduped.sort(key=lambda rr: self._safe_float(rr.get("sharpe")), reverse=True)
            top = deduped[:3]

            for r in top:
                summary = r.get("metrics_summary") or {}
                sharpe = self._safe_float(r.get("sharpe"))
                calmar = self._safe_float(summary.get("calmar"))
                mdd = self._safe_float(summary.get("max_drawdown"))
                cagr = self._safe_float(summary.get("cagr"))
                turnover_max = self._safe_float(summary.get("turnover_max"))

                verdict_pass = r.get("verdict_pass")
                bias = r.get("bias_check") or {}
                bias_flags = bias.get("flags") or []

                params_hint = self._params_hint(r.get("strategy_params") or {})
                insight = self._format_insight(
                    strategy=strat,
                    sharpe=sharpe,
                    calmar=calmar,
                    cagr=cagr,
                    mdd=mdd,
                    turnover_max=turnover_max,
                    verdict_pass=verdict_pass,
                    bias_flags=bias_flags,
                    params_hint=params_hint,
                )

                confidence = self._clamp01(self._safe_float(r.get("bias_confidence"), default=0.5))
                novelty, utility, recency, score = self._score_memory(
                    r,
                    now=now,
                    config_counts=config_freq.get(strat) or Counter(),
                    existing_keys=existing_keys,
                )

                config_sig = r.get("config_sig") or ""
                memory_key = f"{strat}:{config_sig}" if config_sig else f"{strat}:{r.get('run_id')}"

                out.append(
                    {
                        "memory_key": memory_key,
                        "strategy": strat,
                        "run_id": r.get("run_id"),
                        "strategy_impl": r.get("strategy_impl"),
                        "insight": insight,
                        "evidence": {
                            "run_id": r.get("run_id"),
                            "config_sig": config_sig,
                            "metrics": {
                                "sharpe": sharpe,
                                "calmar": calmar,
                                "cagr": cagr,
                                "max_drawdown": mdd,
                                "turnover_max": turnover_max,
                            },
                            "verdict": r.get("verdict") or {},
                            "bias_check": bias,
                        },
                        "confidence": confidence,
                        "date": r.get("ts") or "",
                        "scores": {
                            "novelty": novelty,
                            "utility": utility,
                            "recency": recency,
                            "confidence": confidence,
                            "score": score,
                        },
                        "params_hint": params_hint,
                    }
                )

        return out

    def detect_stagnation(self) -> dict:
        """
        Detect learning stagnation across strategies.

        Stagnation heuristic:
        - For each strategy with >=6 runs, if best Sharpe in the last 5 runs
          does not exceed best Sharpe before those runs, flag stagnation.

        Returns:
            Dict with per-strategy stagnation and an overall summary.
        """

        runs = self._scan_runs()
        grouped: dict = defaultdict(list)
        for r in runs:
            grouped[r.get("strategy") or "unknown"].append(r)

        per: dict = {}
        stagnant_strategies: list = []
        for strat, items in grouped.items():
            report = self._detect_stagnation_for_strategy(items)
            per[strat] = report
            if report.get("stagnating"):
                stagnant_strategies.append(strat)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategies_checked": len(grouped),
            "stagnant_strategies": stagnant_strategies,
            "per_strategy": per,
        }

    def evict_low_value_memories(self, max_entries: int = 200) -> int:
        """
        Evict lowest-scored entries from curated_knowledge.json.

        Args:
            max_entries: Maximum number of entries to keep.

        Returns:
            Number of evicted entries.
        """

        try:
            n = int(max_entries)
        except Exception:
            n = 200
        n = max(1, n)

        obj = self._read_json(self.curated_knowledge_path)
        if not isinstance(obj, list):
            return 0

        evicted, kept = self._evict_entries(obj, max_entries=n)
        if evicted > 0:
            self._write_json(self.curated_knowledge_path, kept)
        return evicted

    # ── Scanning & parsing ──────────────────────────────────────────────────

    def _scan_runs(self) -> list:
        """
        Scan artifacts/runs and return parsed run records.
        """

        if not self.runs_dir.exists():
            return []

        out: list = []
        for d in self.runs_dir.iterdir():
            if not d.is_dir():
                continue
            rec = self._load_run_record(d)
            if rec is not None:
                out.append(rec)
        out.sort(key=lambda r: r.get("_ts_epoch", 0))
        return out

    def _load_run_record(self, run_dir: Path) -> dict | None:
        """
        Load a run directory into a normalized run record dict.
        """

        config = self._read_json(run_dir / "config.json") or {}
        metrics = self._read_json(run_dir / "metrics.json") or {}

        if not isinstance(config, dict):
            config = {}
        if not isinstance(metrics, dict):
            metrics = {}

        summary = metrics.get("summary")
        if not isinstance(summary, dict):
            summary = metrics

        run_name = config.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            run_name = self._base_strategy_name(run_dir.name)

        ts_dt = self._parse_run_timestamp(run_dir)
        ts_iso = ts_dt.isoformat()

        sharpe = self._safe_float(summary.get("sharpe"))
        verdict_obj = metrics.get("verdict") or {}
        if not isinstance(verdict_obj, dict):
            verdict_obj = {}
        verdict_pass = verdict_obj.get("pass")
        if verdict_pass not in (True, False):
            verdict_pass = None

        bias = metrics.get("bias_check") or {}
        if not isinstance(bias, dict):
            bias = {}
        bias_conf = self._safe_float(bias.get("confidence"), default=0.5)

        strat_obj = config.get("strategy") or {}
        if not isinstance(strat_obj, dict):
            strat_obj = {}
        strategy_impl = strat_obj.get("name") if isinstance(strat_obj.get("name"), str) else None
        strategy_params = strat_obj.get("params") or {}
        if not isinstance(strategy_params, dict):
            strategy_params = {}

        cfg_sig = self._config_sig(config)
        run_id = run_dir.name
        cfg_fp, code_fp = self._parse_run_dir_fingerprints(run_dir.name)

        return {
            "run_id": run_id,
            "strategy": run_name,
            "strategy_impl": strategy_impl,
            "strategy_params": strategy_params,
            "config_sig": cfg_sig,
            "config_fingerprint": cfg_fp,
            "code_fingerprint": code_fp,
            "metrics_summary": summary,
            "metrics_raw": metrics,
            "sharpe": sharpe,
            "verdict": verdict_obj,
            "verdict_pass": verdict_pass,
            "bias_check": bias,
            "bias_confidence": bias_conf,
            "ts": ts_iso,
            "_ts_epoch": int(ts_dt.timestamp()),
            "_run_dir": str(run_dir),
        }

    def _parse_run_timestamp(self, run_dir: Path) -> datetime:
        """
        Parse run timestamp from run directory name suffix, falling back to mtime.
        """

        name = run_dir.name
        m = _RUN_TS_RE.search(name)
        if m:
            raw = m.group("ts")
            try:
                dt = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass

        try:
            return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _base_strategy_name(self, run_dir_name: str) -> str:
        """
        Return strategy base name from a run directory name.
        """

        s = (run_dir_name or "").strip()
        if not s:
            return "unknown"
        return s.split(".", 1)[0]

    def _parse_run_dir_fingerprints(self, run_dir_name: str) -> tuple[str | None, str | None]:
        """
        Extract config/code fingerprints from a run directory name if present.

        Expected pattern: <run_name>.<cfg_fp>.<code_fp>.<timestamp>
        """

        parts = (run_dir_name or "").split(".")
        if len(parts) < 4:
            return None, None
        cfg_fp = parts[1]
        code_fp = parts[2]
        if not _HEX_RE.match(cfg_fp or ""):
            cfg_fp = None
        if not _HEX_RE.match(code_fp or ""):
            code_fp = None
        return cfg_fp, code_fp

    # ── Registry building ───────────────────────────────────────────────────

    def _build_strategy_registry(self, runs: list) -> tuple[dict, dict]:
        """
        Build a strategy registry and an internal per-strategy run grouping.
        """

        per_strategy: dict = defaultdict(list)
        for r in runs:
            per_strategy[r.get("strategy") or "unknown"].append(r)

        registry: dict = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategies": {},
        }

        for strat, items in per_strategy.items():
            items_sorted = sorted(items, key=lambda rr: rr.get("_ts_epoch", 0))
            times_run = len(items_sorted)
            last = items_sorted[-1] if items_sorted else {}
            best = None
            for r in items_sorted:
                if best is None or self._safe_float(r.get("sharpe")) > self._safe_float(best.get("sharpe")):
                    best = r

            best_sharpe = self._safe_float((best or {}).get("sharpe"), default=None)
            best_params_hint = self._params_hint((best or {}).get("strategy_params") or {})

            verdict_pass = (best or {}).get("verdict_pass")
            verdict = "UNKNOWN"
            if verdict_pass is True:
                verdict = "PASS"
            elif verdict_pass is False:
                verdict = "FAIL"

            lessons = self._lessons_from_run(best or {})
            # Add repeat-config warning summary (counts only; full warning goes to lessons_learned.jsonl).
            cfg_counts = Counter([it.get("config_sig") for it in items_sorted if it.get("config_sig")])
            most_common = cfg_counts.most_common(1)
            if most_common:
                sig, cnt = most_common[0]
                if cnt > 5:
                    lessons.append(f"WARNING: repeated config_sig={sig} run {cnt} times (experiment repetition).")

            registry["strategies"][strat] = {
                "name": strat,
                "best_sharpe": best_sharpe,
                "best_params_hint": best_params_hint,
                "times_run": times_run,
                "last_run": last.get("ts"),
                "verdict": verdict,
                "lessons_learned": lessons,
            }

        return registry, per_strategy

    def _lessons_from_run(self, run: dict) -> list:
        """
        Extract short, human-readable lessons from a single run record.
        """

        if not run:
            return []

        lessons: list = []

        verdict = run.get("verdict") or {}
        if isinstance(verdict, dict):
            reasons = verdict.get("reasons") or []
            if isinstance(reasons, list) and reasons:
                reasons_s = ", ".join([str(x) for x in reasons[:5]])
                lessons.append(f"Verdict reasons: {reasons_s}.")

        bias = run.get("bias_check") or {}
        if isinstance(bias, dict):
            flags = bias.get("flags") or []
            if isinstance(flags, list) and flags:
                flags_s = ", ".join([str(x) for x in flags[:5]])
                lessons.append(f"Bias flags: {flags_s}.")
            if bias.get("likely_overfit") is True:
                lessons.append("Bias check indicates likely overfit; improve robustness (WF, constraints, holdouts).")

        return lessons

    # ── Knowledge persistence ───────────────────────────────────────────────

    def _merge_curated_knowledge(self, new_entries: list) -> tuple[list, int, int]:
        """
        Merge new knowledge entries into curated_knowledge.json (by memory_key).
        """

        existing = self._read_json(self.curated_knowledge_path)
        if not isinstance(existing, list):
            existing = []

        by_key: dict = {}
        for e in existing:
            if not isinstance(e, dict):
                continue
            k = e.get("memory_key")
            if isinstance(k, str) and k:
                by_key[k] = e

        added = 0
        updated = 0
        for e in new_entries:
            if not isinstance(e, dict):
                continue
            k = e.get("memory_key")
            if not isinstance(k, str) or not k:
                continue
            prev = by_key.get(k)
            if prev is None:
                by_key[k] = e
                added += 1
                continue

            prev_score = self._safe_float(((prev.get("scores") or {}).get("score")), default=0.0)
            new_score = self._safe_float(((e.get("scores") or {}).get("score")), default=0.0)
            # Prefer higher score; if tie, prefer newer evidence.
            if (new_score > prev_score + 1e-9) or (new_score == prev_score and (e.get("date") or "") > (prev.get("date") or "")):
                by_key[k] = e
                updated += 1

        merged = list(by_key.values())
        return merged, added, updated

    def _evict_entries(self, entries: list, *, max_entries: int) -> tuple[int, list]:
        """
        Evict lowest-scored entries from an in-memory list.
        """

        try:
            n = int(max_entries)
        except Exception:
            n = 200
        n = max(1, n)

        clean: list = [e for e in entries if isinstance(e, dict)]
        if len(clean) <= n:
            # Still sort for stable top-first ordering.
            clean.sort(key=self._entry_score, reverse=True)
            return 0, clean

        clean.sort(key=self._entry_score, reverse=True)
        kept = clean[:n]
        evicted = len(clean) - len(kept)
        return evicted, kept

    def _entry_score(self, entry: dict) -> float:
        """
        Extract overall score from a curated_knowledge entry.
        """

        if not isinstance(entry, dict):
            return 0.0
        scores = entry.get("scores") or {}
        if not isinstance(scores, dict):
            return 0.0
        return float(self._safe_float(scores.get("score"), default=0.0))

    def _existing_memory_keys(self) -> set:
        """
        Read existing curated_knowledge.json and return a set of memory keys.
        """

        obj = self._read_json(self.curated_knowledge_path)
        if not isinstance(obj, list):
            return set()
        keys: set = set()
        for e in obj:
            if not isinstance(e, dict):
                continue
            k = e.get("memory_key")
            if isinstance(k, str) and k:
                keys.add(k)
        return keys

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_memory(self, run: dict, *, now: datetime, config_counts: Counter, existing_keys: set) -> tuple[float, float, float, float]:
        """
        Compute novelty/utility/recency/confidence-derived score for a run.
        """

        sharpe = self._safe_float(run.get("sharpe"), default=None)
        verdict_pass = run.get("verdict_pass")
        bias = run.get("bias_check") or {}
        likely_overfit = bool(isinstance(bias, dict) and bias.get("likely_overfit") is True)
        confidence = self._clamp01(self._safe_float(run.get("bias_confidence"), default=0.5))

        # Utility: Sharpe-based sigmoid with penalties for failing verdict or likely overfit.
        utility = self._sigmoid(sharpe if sharpe is not None else -3.0)
        if verdict_pass is False:
            utility *= 0.65
        if likely_overfit:
            utility *= 0.75
        utility = self._clamp01(utility)

        # Recency: exponential decay with ~30-day time constant.
        ts_iso = run.get("ts") or ""
        ts_dt = self._parse_iso(ts_iso) or now
        age = max(timedelta(0), now - ts_dt)
        age_days = age.total_seconds() / 86400.0
        recency = math.exp(-age_days / 30.0)
        recency = self._clamp01(recency)

        # Novelty: penalize repeated configs within strategy and existing curated knowledge.
        sig = run.get("config_sig") or ""
        cnt = int(config_counts.get(sig, 0)) if sig else 1
        novelty = 1.0 / max(1, cnt)
        mem_key = f"{run.get('strategy') or 'unknown'}:{sig}" if sig else ""
        if mem_key and mem_key in existing_keys:
            novelty *= 0.5
        novelty = self._clamp01(novelty)

        score = (0.45 * utility + 0.30 * novelty + 0.25 * recency) * (0.5 + 0.5 * confidence)
        score = self._clamp01(score)
        return novelty, utility, recency, score

    def _sigmoid(self, x: float | None) -> float:
        """
        Numerically-stable logistic sigmoid.
        """

        if x is None:
            return 0.0
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v >= 12.0:
            return 0.999994
        if v <= -12.0:
            return 0.000006
        return 1.0 / (1.0 + math.exp(-v))

    # ── Insight formatting ──────────────────────────────────────────────────

    def _format_insight(
        self,
        *,
        strategy: str,
        sharpe: float | None,
        calmar: float | None,
        cagr: float | None,
        mdd: float | None,
        turnover_max: float | None,
        verdict_pass: bool | None,
        bias_flags: list,
        params_hint: dict,
    ) -> str:
        """
        Build a compact, human-readable insight string for a run.
        """

        def fnum(v: float | None, fmt: str) -> str:
            if v is None:
                return "n/a"
            try:
                return format(float(v), fmt)
            except Exception:
                return "n/a"

        sharpe_s = fnum(sharpe, ".3f")
        calmar_s = fnum(calmar, ".3f")
        cagr_s = fnum(cagr, ".3f")
        mdd_s = fnum(mdd, ".3f")
        tmax_s = fnum(turnover_max, ".3f")
        verdict_s = "UNKNOWN" if verdict_pass is None else ("PASS" if verdict_pass else "FAIL")

        flags = []
        if isinstance(bias_flags, list):
            flags = [str(x) for x in bias_flags[:4]]

        # params_hint is already compact (subset). Represent deterministically.
        params_s = ""
        if isinstance(params_hint, dict) and params_hint:
            try:
                params_s = json.dumps(params_hint, sort_keys=True, ensure_ascii=False)
            except Exception:
                params_s = str(params_hint)

        parts = [
            f"{strategy}: Sharpe={sharpe_s}, Calmar={calmar_s}, CAGR={cagr_s}, MDD={mdd_s}, TurnoverMax={tmax_s}, Verdict={verdict_s}",
        ]
        if flags:
            parts.append(f"BiasFlags={','.join(flags)}")
        if params_s:
            parts.append(f"Params={params_s}")
        return " | ".join(parts)

    def _params_hint(self, params: dict, *, max_keys: int = 10) -> dict:
        """
        Create a compact params hint dict for registry/knowledge entries.
        """

        if not isinstance(params, dict) or not params:
            return {}
        keys = sorted([k for k in params.keys() if isinstance(k, str)])
        selected = keys[: max(1, int(max_keys))]
        out: dict = {}
        for k in selected:
            v = params.get(k)
            # Avoid huge nested blobs in hints.
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)[:120]
        return out

    # ── Repeat detection & lesson logging ───────────────────────────────────

    def _detect_repeating_experiments(self, per_strategy: dict) -> list:
        """
        Detect repeated runs with identical config (>5x) and log WARNING lessons.
        """

        warnings: list = []
        for strat, items in (per_strategy or {}).items():
            cfg_sigs = [it.get("config_sig") for it in items if it.get("config_sig")]
            counts = Counter(cfg_sigs)
            for sig, cnt in counts.items():
                if cnt <= 5:
                    continue
                msg = f"Repeating experiments: strategy '{strat}' ran the same config_sig={sig} {cnt} times."
                evidence = {"strategy": strat, "config_sig": sig, "count": cnt}
                ok = self._append_lesson(
                    level="WARNING",
                    kind="repeat_experiment",
                    strategy=strat,
                    message=msg,
                    evidence=evidence,
                )
                if ok:
                    warnings.append(evidence)
        return warnings

    def _log_stagnation_lessons(self, stagnation_report: dict) -> int:
        """
        Log stagnation findings as INFO lessons (deduped).
        """

        per = (stagnation_report or {}).get("per_strategy") or {}
        if not isinstance(per, dict):
            return 0

        logged = 0
        for strat, rep in per.items():
            if not isinstance(rep, dict) or not rep.get("stagnating"):
                continue
            msg = rep.get("message") or f"Stagnation detected for strategy '{strat}'."
            evidence = rep.get("evidence") or {}
            ok = self._append_lesson(
                level="INFO",
                kind="stagnation",
                strategy=strat,
                message=str(msg),
                evidence=evidence,
            )
            if ok:
                logged += 1
        return logged

    def _append_lesson(self, *, level: str, kind: str, strategy: str, message: str, evidence: dict | None = None) -> bool:
        """
        Append a lesson to lessons_learned.jsonl (append-only), avoiding duplicates by key.
        """

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "ts": now,
            "level": str(level or "INFO").upper(),
            "kind": str(kind or "note"),
            "strategy": str(strategy or ""),
            "message": str(message or ""),
            "evidence": evidence or {},
        }
        key_src = json.dumps(
            {"kind": payload["kind"], "strategy": payload["strategy"], "message": payload["message"]},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload["key"] = self._fnv1a_64(key_src)

        existing_keys = self._load_recent_lesson_keys(limit_lines=600)
        if payload["key"] in existing_keys:
            return False

        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lessons_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    def _load_recent_lesson_keys(self, *, limit_lines: int = 600) -> set:
        """
        Load a set of recent lesson keys from lessons_learned.jsonl (tail-only).
        """

        if not self.lessons_path.exists():
            return set()

        try:
            n = int(limit_lines)
        except Exception:
            n = 600
        n = max(50, min(n, 5000))

        lines = self.lessons_path.read_text("utf-8", errors="replace").splitlines()[-n:]
        keys: set = set()
        for ln in lines:
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict) and isinstance(obj.get("key"), str):
                    keys.add(obj["key"])
            except Exception:
                continue
        return keys

    # ── Stagnation internals ────────────────────────────────────────────────

    def _detect_stagnation_for_strategy(self, items: list) -> dict:
        """
        Detect stagnation for one strategy given its run records.
        """

        runs = [it for it in (items or []) if isinstance(it, dict)]
        runs.sort(key=lambda rr: rr.get("_ts_epoch", 0))
        sharpe_series = [self._safe_float(r.get("sharpe"), default=None) for r in runs]
        # Filter None values but keep alignment with runs if needed.
        indexed = [(runs[i], sharpe_series[i]) for i in range(len(runs)) if sharpe_series[i] is not None]

        if len(indexed) < 6:
            return {
                "stagnating": False,
                "reason": "insufficient_history",
                "runs_considered": len(indexed),
            }

        # Split into before / last-5 by time order.
        before = indexed[:-5]
        last5 = indexed[-5:]
        best_before = max([s for _, s in before])
        best_last5 = max([s for _, s in last5])

        if best_last5 > best_before + 1e-9:
            return {
                "stagnating": False,
                "reason": "improving",
                "runs_considered": len(indexed),
                "best_before": best_before,
                "best_last5": best_last5,
            }

        strat = (runs[-1].get("strategy") if runs else None) or "unknown"
        msg = (
            f"Stagnation: strategy '{strat}' last 5 runs did not beat prior best Sharpe "
            f"(best_before={best_before:.3f}, best_last5={best_last5:.3f})."
        )
        suggestion = (
            "Explore a new direction: change feature set, universe, timeframe, constraints, or strategy family; "
            "consider ablations to find limiting factor."
        )
        return {
            "stagnating": True,
            "reason": "no_improvement_last5",
            "runs_considered": len(indexed),
            "best_before": best_before,
            "best_last5": best_last5,
            "message": msg,
            "suggestion": suggestion,
            "evidence": {
                "best_before": best_before,
                "best_last5": best_last5,
                "last5_run_ids": [r.get("run_id") for r, _ in last5],
            },
        }

    # ── Config signature ────────────────────────────────────────────────────

    def _config_sig(self, config: dict) -> str:
        """
        Compute a stable config signature (FNV-1a over canonical JSON).
        """

        if not isinstance(config, dict) or not config:
            return "0" * 16

        # Drop keys that should not define novelty.
        filtered = {}
        for k, v in config.items():
            if k in {"created_at", "ts", "timestamp"}:
                continue
            if k == "run_name":
                continue
            filtered[k] = v

        canon = self._canonical_json(filtered)
        return self._fnv1a_64(canon)

    def _canonical_json(self, obj: object) -> str:
        """
        Deterministic JSON serialization (stable ordering, compact separators).
        """

        try:
            return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return json.dumps(str(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _fnv1a_64(self, text: str) -> str:
        """
        Compute a stable 64-bit FNV-1a hash hex digest for a given string.
        """

        h = 0xCBF29CE484222325
        for b in (text or "").encode("utf-8", errors="replace"):
            h ^= b
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"{h:016x}"

    # ── JSON I/O utilities ──────────────────────────────────────────────────

    def _read_json(self, path: Path) -> object | None:
        """
        Read JSON from disk, returning None on failure.
        """

        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return None

    def _write_json(self, path: Path, obj: object) -> None:
        """
        Write JSON to disk (pretty-printed, UTF-8).
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf-8")

    # ── Parsing helpers ─────────────────────────────────────────────────────

    def _safe_float(self, v: object, *, default: float | None = 0.0) -> float | None:
        """
        Convert a value to float, returning default on failure.
        """

        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    def _clamp01(self, x: float | None) -> float:
        """
        Clamp a number into [0, 1].
        """

        if x is None:
            return 0.0
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _parse_iso(self, s: str) -> datetime | None:
        """
        Parse an ISO-8601 datetime string into an aware datetime (UTC if missing).
        """

        raw = (s or "").strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
