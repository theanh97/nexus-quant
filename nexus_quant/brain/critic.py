from __future__ import annotations

try:
    import hashlib
except Exception:  # pragma: no cover
    hashlib = None  # type: ignore

try:
    import json
except Exception:  # pragma: no cover
    json = None  # type: ignore

try:
    import re
except Exception:  # pragma: no cover
    re = None  # type: ignore

try:
    from dataclasses import dataclass
except Exception:  # pragma: no cover
    dataclass = None  # type: ignore

try:
    from datetime import datetime, timezone
except Exception:  # pragma: no cover
    datetime = None  # type: ignore
    timezone = None  # type: ignore

try:
    from pathlib import Path
except Exception:  # pragma: no cover
    Path = None  # type: ignore

try:
    from typing import Any, Dict, List, Optional, Tuple
except Exception:  # pragma: no cover
    Any = object  # type: ignore
    Dict = dict  # type: ignore
    List = list  # type: ignore
    Optional = object  # type: ignore
    Tuple = tuple  # type: ignore


_DEFAULT_MAX_REPEAT = 5
_DEFAULT_STAGNATION_WINDOW = 10
_DEFAULT_STAGNATION_SHARPE_IMPROVEMENT = 0.1

_IGNORE_KEYS_FOR_HASH = {
    "seed",
    "run_name",
    "notes",
    "comment",
    "created_at",
    "timestamp",
    "ts",
    "artifacts_dir",
    "output_dir",
}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _read_json_dict(path: "Path") -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)  # type: ignore[attr-defined]
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _strip_ignored_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            if str(k) in _IGNORE_KEYS_FOR_HASH:
                continue
            out[str(k)] = _strip_ignored_keys(obj[k])
        return out
    if isinstance(obj, list):
        return [_strip_ignored_keys(v) for v in obj]
    return obj


def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(  # type: ignore[attr-defined]
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    except Exception:
        try:
            return str(obj)
        except Exception:
            return repr(obj)


def _sha256_text(s: str) -> str:
    try:
        h = hashlib.sha256()  # type: ignore[union-attr]
        h.update(s.encode("utf-8", errors="replace"))
        return h.hexdigest()
    except Exception:
        # Extremely defensive fallback.
        return str(abs(hash(s)))


def _parse_run_ts(run_id: str) -> Optional[float]:
    """
    Parse the run timestamp from the standard run_id suffix:
    ...YYYYMMDDTHHMMSSZ
    """
    try:
        ts_part = run_id.split(".")[-1]
        if not ts_part:
            return None
        if re is not None:
            if not re.fullmatch(r"\d{8}T\d{6}Z", ts_part):
                return None
        # Use datetime if available; otherwise fall back to lexicographic ordering later.
        if datetime is None or timezone is None:
            return None
        dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)  # type: ignore[union-attr]
        return dt.timestamp()
    except Exception:
        return None


def _run_sort_key(p: "Path") -> Tuple[int, float, str]:
    """
    Sort newest last:
    - prefer parsed timestamp (if present)
    - fallback to directory mtime
    """
    try:
        ts = _parse_run_ts(p.name)
        if ts is not None:
            return (0, float(ts), p.name)
    except Exception:
        pass
    try:
        return (1, float(p.stat().st_mtime), p.name)
    except Exception:
        return (2, 0.0, p.name)


if dataclass is not None:

    @dataclass
    class RunRecord:
        run_id: str
        run_name: str
        strategy: str
        signature: str
        signature_similar: str
        sharpe: Optional[float]
        ts: float
        config: Dict[str, Any]

else:

    class RunRecord:  # type: ignore[no-redef]
        def __init__(
            self,
            run_id: str,
            run_name: str,
            strategy: str,
            signature: str,
            signature_similar: str,
            sharpe: Optional[float],
            ts: float,
            config: Dict[str, Any],
        ) -> None:
            self.run_id = run_id
            self.run_name = run_name
            self.strategy = strategy
            self.signature = signature
            self.signature_similar = signature_similar
            self.sharpe = sharpe
            self.ts = ts
            self.config = config


class BrainCritic:
    """
    Lightweight self-critique + novelty gate.

    Reads `artifacts/runs/*/config.json` and `metrics.json` to:
    - deduplicate repeated runs via a stable signature hash
    - detect stagnation from recent Sharpe behavior
    - suggest pivots when exploration is too narrow
    """

    def __init__(
        self,
        artifacts_dir: "Path",
        *,
        runs_subdir: str = "runs",
        max_repeat: int = _DEFAULT_MAX_REPEAT,
        stagnation_window: int = _DEFAULT_STAGNATION_WINDOW,
        stagnation_sharpe_improvement: float = _DEFAULT_STAGNATION_SHARPE_IMPROVEMENT,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.runs_dir = self.artifacts_dir / runs_subdir
        self.max_repeat = int(max_repeat)
        self.stagnation_window = int(stagnation_window)
        self.stagnation_sharpe_improvement = float(stagnation_sharpe_improvement)

    def signature(self, strategy_name: str, config: Dict[str, Any]) -> str:
        base = _strip_ignored_keys(config or {})
        canon = _canonical_json(base)
        raw = f"{strategy_name}\n{canon}"
        return _sha256_text(raw)

    def signature_similar(self, strategy_name: str, config: Dict[str, Any]) -> str:
        """
        Coarser signature used to detect repeated "same experiment class" runs:
        ignores obvious metadata plus `strategy.params` + `self_learn`.
        """
        base = _strip_ignored_keys(config or {})
        try:
            if isinstance(base, dict):
                base = dict(base)
                base.pop("self_learn", None)
                s = base.get("strategy")
                if isinstance(s, dict):
                    s = dict(s)
                    s.pop("params", None)
                    base["strategy"] = s
        except Exception:
            pass
        canon = _canonical_json(base)
        raw = f"{strategy_name}\n{canon}"
        return _sha256_text(raw)

    def _scan_runs(self, *, limit: Optional[int] = None) -> List[RunRecord]:
        runs: List[RunRecord] = []
        try:
            if not self.runs_dir.exists():
                return []
        except Exception:
            return []

        try:
            dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        except Exception:
            return []

        # Sort ascending; most-recent is at the end.
        dirs.sort(key=_run_sort_key)
        if limit is not None:
            try:
                dirs = dirs[-int(limit) :]
            except Exception:
                pass

        for d in dirs:
            cfgp = d / "config.json"
            mp = d / "metrics.json"
            cfg = _read_json_dict(cfgp) or {}
            metrics = _read_json_dict(mp) or {}

            run_name = ""
            strat = ""
            try:
                run_name = str(cfg.get("run_name") or "")
            except Exception:
                run_name = ""
            if not run_name:
                try:
                    run_name = d.name.split(".")[0]
                except Exception:
                    run_name = ""

            try:
                strat = str(((cfg.get("strategy") or {}) or {}).get("name") or "")
            except Exception:
                strat = ""
            if not strat:
                # Best-effort fallback to run_id prefix.
                try:
                    strat = d.name.split(".")[0]
                except Exception:
                    strat = ""

            sharpe = None
            try:
                summary = metrics.get("summary") or {}
                if isinstance(summary, dict):
                    sharpe = _safe_float(summary.get("sharpe"))
            except Exception:
                sharpe = None

            # Signatures are anchored on the *experiment name* (run_name),
            # since that is what uniquely identifies the "proposal" in run IDs.
            sig = self.signature(run_name, cfg)
            sig_sim = self.signature_similar(run_name, cfg)

            ts = None
            try:
                ts = _parse_run_ts(d.name)
            except Exception:
                ts = None
            if ts is None:
                try:
                    ts = float(d.stat().st_mtime)
                except Exception:
                    ts = 0.0

            runs.append(
                RunRecord(
                    run_id=d.name,
                    run_name=run_name,
                    strategy=strat,
                    signature=sig,
                    signature_similar=sig_sim,
                    sharpe=sharpe,
                    ts=float(ts or 0.0),
                    config=cfg,
                )
            )

        return runs

    def check_novelty(self, strategy_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a dict with:
        - should_run: bool
        - reason: str
        - already_run: int (# prior runs with same signature)
        """
        exp_name = str(strategy_name or "")
        sig = self.signature(exp_name, config or {})
        sig_sim = self.signature_similar(exp_name, config or {})
        runs = self._scan_runs()

        already_run = 0
        already_run_similar = 0
        already_run_name = 0
        recent_same: List[str] = []
        recent_name: List[str] = []
        try:
            for r in runs:
                if r.run_name == exp_name:
                    already_run_name += 1
                    recent_name.append(r.run_id)
                if r.signature == sig:
                    already_run += 1
                    recent_same.append(r.run_id)
                if r.signature_similar == sig_sim:
                    already_run_similar += 1
            recent_same = recent_same[-5:]
            recent_name = recent_name[-5:]
        except Exception:
            pass

        should_run = True
        reason = "novel_signature"
        if already_run > 0:
            reason = f"signature_seen_before:{already_run}"
        if already_run_similar > 0 and already_run_similar != already_run:
            reason = f"similar_signature_seen_before:{already_run_similar},exact:{already_run}"
        if already_run_name > 0 and already_run_name != already_run:
            reason = f"run_name_seen_before:{already_run_name},similar:{already_run_similar},exact:{already_run}"
        if (
            already_run_name > int(self.max_repeat)
            or already_run_similar > int(self.max_repeat)
            or already_run > int(self.max_repeat)
        ):
            should_run = False
            reason = (
                f"repeat_blocked:run_name={already_run_name},similar={already_run_similar},"
                f"exact={already_run}>max_repeat={self.max_repeat}"
            )

        return {
            "should_run": bool(should_run),
            "reason": str(reason),
            "strategy": exp_name,
            "signature": sig,
            "signature_short": sig[:16],
            "signature_similar": sig_sim,
            "signature_similar_short": sig_sim[:16],
            "already_run": int(already_run),
            "already_run_similar": int(already_run_similar),
            "already_run_name": int(already_run_name),
            "max_repeat": int(self.max_repeat),
            "recent_same_run_ids": recent_same,
            "recent_run_ids_for_name": recent_name,
            "total_runs": int(len(runs)),
        }

    def self_critique(self) -> Dict[str, Any]:
        """
        Evaluate:
        - stagnation: last N runs show < threshold Sharpe improvement
        - repetition: top repeated signatures / dominant strategies
        - pivots: 3 suggested next directions
        """
        runs = self._scan_runs()
        total_runs = len(runs)

        # --- Stagnation ---
        window = max(1, int(self.stagnation_window))
        recent = sorted(runs, key=lambda r: float(r.ts))[-window:]
        sharpes = [r.sharpe for r in recent if isinstance(r.sharpe, (int, float))]
        sharpe_improvement = None
        stagnation = False
        if len(sharpes) >= 2:
            try:
                sharpe_improvement = float(max(sharpes) - min(sharpes))
                stagnation = sharpe_improvement < float(self.stagnation_sharpe_improvement)
            except Exception:
                sharpe_improvement = None
                stagnation = False

        # --- Repetition ---
        sig_counts: Dict[str, int] = {}
        sig_sim_counts: Dict[str, int] = {}
        run_name_counts: Dict[str, int] = {}
        strat_counts: Dict[str, int] = {}
        for r in runs:
            sig_counts[r.signature] = sig_counts.get(r.signature, 0) + 1
            sig_sim_counts[r.signature_similar] = sig_sim_counts.get(r.signature_similar, 0) + 1
            run_name_counts[r.run_name] = run_name_counts.get(r.run_name, 0) + 1
            strat_counts[r.strategy] = strat_counts.get(r.strategy, 0) + 1

        top_sigs = sorted(sig_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top_sigs_sim = sorted(sig_sim_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top_run_names = sorted(run_name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        top_strats = sorted(strat_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

        repetition_flag = False
        try:
            repetition_flag = bool(
                (top_sigs and top_sigs[0][1] > int(self.max_repeat))
                or (top_sigs_sim and top_sigs_sim[0][1] > int(self.max_repeat))
                or (top_strats and top_strats[0][1] > int(self.max_repeat))
            )
        except Exception:
            repetition_flag = False

        pivots = self.suggest_pivots(runs)

        return {
            "total_runs": int(total_runs),
            "stagnation": {
                "flag": bool(stagnation),
                "window": int(window),
                "sharpe_improvement": sharpe_improvement,
                "threshold": float(self.stagnation_sharpe_improvement),
                "recent_sharpes": [float(s) for s in sharpes][-window:],
            },
            "repetition": {
                "flag": bool(repetition_flag),
                "max_repeat": int(self.max_repeat),
                "top_signatures": [{"signature_short": s[:16], "count": int(c)} for s, c in top_sigs],
                "top_signatures_similar": [{"signature_short": s[:16], "count": int(c)} for s, c in top_sigs_sim],
                "top_run_names": [{"run_name": str(n), "count": int(c)} for n, c in top_run_names],
                "top_strategies": [{"strategy": str(n), "count": int(c)} for n, c in top_strats],
            },
            "pivots": list(pivots),
        }

    def suggest_pivots(self, runs: Optional[List[RunRecord]] = None) -> List[str]:
        """
        Return 3 unexplored strategy ideas, based on observed gaps in run history.
        """
        runs = runs if runs is not None else self._scan_runs()
        seen_strats = {str(r.strategy) for r in runs if getattr(r, "strategy", None)}

        # Inspect data coverage (especially 2022 bear market weakness).
        covers_2022 = False
        try:
            for r in runs:
                data = (r.config.get("data") or {}) if isinstance(r.config, dict) else {}
                start = str(data.get("start") or "")
                end = str(data.get("end") or "")
                # crude year parse, but robust to ISO strings
                try:
                    sy = int(start[:4])
                    ey = int(end[:4])
                except Exception:
                    continue
                if sy <= 2022 <= ey:
                    covers_2022 = True
                    break
        except Exception:
            covers_2022 = False

        # Keep the list stdlib-only (no importing strategy registry).
        known_ideas = [
            (
                "tsmom_v1",
                "Run `tsmom_v1` with volatility targeting and a regime filter; validate specifically on 2022 bear data.",
            ),
            (
                "combined_carry_mom_v1",
                "Try `combined_carry_mom_v1` (carry+momentum blend) to diversify away from single-factor alpha.",
            ),
            (
                "ensemble_v1",
                "Build `ensemble_v1` combining uncorrelated sub-strategies with regime-adaptive weights (reduce 2022 drawdowns).",
            ),
            (
                "nexus_alpha_v2",
                "A/B test `nexus_alpha_v2` vs v1 with explicit bear-regime risk throttles and stricter turnover constraints.",
            ),
            (
                "mean_reversion_xs_v1",
                "Expand mean-reversion: add volatility-band entries + funding filter; test crash periods separately.",
            ),
        ]

        pivots: List[str] = []

        # Prioritize 2022 gap if not covered.
        if not covers_2022:
            pivots.append(
                "Add a 2021–2023 backtest window (must include 2022) and make strategy selection regime-adaptive (bear/sideways/trend)."
            )

        for name, idea in known_ideas:
            if name not in seen_strats:
                pivots.append(idea)
            if len(pivots) >= 3:
                break

        # Fallback: fill to 3 with generic but actionable pivots.
        while len(pivots) < 3:
            if "regime" not in " ".join(pivots).lower():
                pivots.append("Introduce a volatility/regime detector and switch between trend and mean-reversion allocations.")
            elif "oos" not in " ".join(pivots).lower():
                pivots.append("Add explicit OOS validation (walk-forward or 3-way split) to reduce overfitting from repeated trials.")
            else:
                pivots.append("Broaden symbol universe + timeframes; enforce novelty constraints per (strategy, params, data_range).")

        return pivots[:3]
