from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .schema import MarketDataset


def validate_dataset(dataset: MarketDataset, expected_step_seconds: Optional[int] = None) -> Dict[str, Any]:
    """
    Data quality gate focused on bias prevention and basic sanity.

    This is intentionally strict. If a dataset fails hard checks, backtests are not trustworthy.
    """
    timeline = dataset.timeline
    issues: List[Dict[str, Any]] = []

    def add(level: str, code: str, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        issues.append({"level": level, "code": code, "message": msg, "extra": extra or {}})

    # Timeline checks
    if len(timeline) < 2:
        add("error", "timeline_too_short", "Timeline must contain at least 2 timestamps.")
    if any(t2 <= t1 for t1, t2 in zip(timeline, timeline[1:])):
        add("error", "timeline_not_strictly_increasing", "Timeline must be strictly increasing (sorted, no duplicates).")

    timeline_set = set(timeline)

    if expected_step_seconds is not None and expected_step_seconds > 0 and len(timeline) >= 3:
        steps = [timeline[i] - timeline[i - 1] for i in range(1, len(timeline))]
        bad = sum(1 for s in steps if s != expected_step_seconds)
        frac_bad = bad / float(len(steps)) if steps else 0.0
        if frac_bad > 0.05:
            add(
                "error",
                "timeline_step_excessive_gaps",
                "Timeline step is irregular for more than 5% of bars.",
                {"expected_step_seconds": expected_step_seconds, "fraction_bad": round(frac_bad, 6)},
            )
        elif frac_bad > 0.01:
            add(
                "warn",
                "timeline_step_irregular",
                "Timeline step is irregular for more than 1% of bars.",
                {"expected_step_seconds": expected_step_seconds, "fraction_bad": round(frac_bad, 6)},
            )

    # Series length checks
    n = len(timeline)
    semantic_profile: Dict[str, Dict[str, float]] = {}

    def _semantic_stats(xs: List[float]) -> Dict[str, float]:
        if len(xs) < 2:
            return {
                "n": float(len(xs)),
                "zero_return_fraction": 0.0,
                "extreme_return_fraction": 0.0,
                "unique_price_fraction": 0.0,
                "return_mean": 0.0,
                "return_std": 0.0,
            }

        returns: List[float] = []
        zero = 0
        extreme = 0
        try:
            prev = float(xs[0])
        except Exception:
            prev = 0.0
        for cur_raw in xs[1:]:
            try:
                cur = float(cur_raw)
            except Exception:
                prev = 0.0
                continue
            if not math.isfinite(prev) or not math.isfinite(cur) or prev <= 0.0:
                prev = cur
                continue
            r = (cur / prev) - 1.0
            if math.isfinite(r):
                returns.append(r)
                if abs(r) <= 1e-12:
                    zero += 1
                if abs(r) >= 0.50:
                    extreme += 1
            prev = cur

        uniq_vals = set()
        for v in xs:
            try:
                uniq_vals.add(round(float(v), 8))
            except Exception:
                continue
        uniq = len(uniq_vals)
        n_ret = len(returns)
        if n_ret <= 0:
            return {
                "n": float(len(xs)),
                "zero_return_fraction": 1.0,
                "extreme_return_fraction": 0.0,
                "unique_price_fraction": float(uniq) / float(max(1, len(xs))),
                "return_mean": 0.0,
                "return_std": 0.0,
            }

        mean_r = sum(returns) / float(n_ret)
        var_r = sum((r - mean_r) ** 2 for r in returns) / float(max(1, n_ret - 1))
        std_r = math.sqrt(max(0.0, var_r))
        return {
            "n": float(len(xs)),
            "zero_return_fraction": float(zero) / float(n_ret),
            "extreme_return_fraction": float(extreme) / float(n_ret),
            "unique_price_fraction": float(uniq) / float(max(1, len(xs))),
            "return_mean": mean_r,
            "return_std": std_r,
        }

    for sym in dataset.symbols:
        closes = dataset.perp_close.get(sym)
        if closes is None or len(closes) != n:
            add("error", "perp_close_length_mismatch", f"perp_close[{sym}] length must match timeline.", {"len": len(closes or [])})
            continue
        if any((c is None) or (not math.isfinite(float(c))) or (float(c) <= 0.0) for c in closes):
            add("error", "perp_close_non_positive", f"perp_close[{sym}] contains non-positive values.")
        stats = _semantic_stats(closes)
        semantic_profile[sym] = stats
        if stats["zero_return_fraction"] > 0.95:
            add(
                "error",
                "perp_close_stale_series",
                f"perp_close[{sym}] is stale (>95% zero-return bars).",
                {
                    "zero_return_fraction": round(stats["zero_return_fraction"], 6),
                    "unique_price_fraction": round(stats["unique_price_fraction"], 6),
                },
            )
        elif stats["zero_return_fraction"] > 0.50:
            add(
                "warn",
                "perp_close_many_zero_returns",
                f"perp_close[{sym}] has many zero-return bars (>50%).",
                {
                    "zero_return_fraction": round(stats["zero_return_fraction"], 6),
                    "unique_price_fraction": round(stats["unique_price_fraction"], 6),
                },
            )
        if stats["extreme_return_fraction"] > 0.01:
            add(
                "error",
                "perp_close_jump_anomaly",
                f"perp_close[{sym}] has too many extreme jumps (|r|>=50%).",
                {"extreme_return_fraction": round(stats["extreme_return_fraction"], 6)},
            )
        elif stats["extreme_return_fraction"] > 0.002:
            add(
                "warn",
                "perp_close_jump_outliers",
                f"perp_close[{sym}] has notable extreme jumps (|r|>=50%).",
                {"extreme_return_fraction": round(stats["extreme_return_fraction"], 6)},
            )
        if stats["unique_price_fraction"] < 0.02:
            add(
                "error",
                "perp_close_low_unique_ratio",
                f"perp_close[{sym}] has too few unique price points.",
                {"unique_price_fraction": round(stats["unique_price_fraction"], 6)},
            )

        if dataset.spot_close is not None:
            spots = dataset.spot_close.get(sym)
            if spots is None or len(spots) != n:
                add("warn", "spot_close_length_mismatch", f"spot_close[{sym}] length mismatch; basis may be unreliable.")
            else:
                # Spot may be missing (0), so only flag negative values.
                if any((x is None) or (float(x) < 0.0) for x in spots):
                    add("error", "spot_close_negative", f"spot_close[{sym}] contains negative values.")

        # Optional extended fields (warn on length mismatch, error on negative values).
        def _check_opt_series(
            series: Optional[Dict[str, List[float]]],
            *,
            code_len: str,
            code_neg: str,
            allow_zero: bool,
        ) -> None:
            if series is None:
                return
            xs = series.get(sym)
            if xs is None or len(xs) != n:
                add("warn", code_len, f"{code_len}[{sym}] length mismatch.", {"len": len(xs or [])})
                return
            if allow_zero:
                bad = any((x is None) or (float(x) < 0.0) for x in xs)
            else:
                bad = any((x is None) or (float(x) <= 0.0) for x in xs)
            if bad:
                add("error", code_neg, f"{code_neg}[{sym}] contains invalid (negative/zero) values.")

        _check_opt_series(dataset.perp_mark_close, code_len="perp_mark_length_mismatch", code_neg="perp_mark_invalid", allow_zero=True)
        _check_opt_series(dataset.perp_index_close, code_len="perp_index_length_mismatch", code_neg="perp_index_invalid", allow_zero=True)
        _check_opt_series(dataset.perp_volume, code_len="perp_volume_length_mismatch", code_neg="perp_volume_negative", allow_zero=True)
        _check_opt_series(dataset.spot_volume, code_len="spot_volume_length_mismatch", code_neg="spot_volume_negative", allow_zero=True)

        # Bid/ask sanity (if present).
        if dataset.bid_close is not None or dataset.ask_close is not None:
            bid = (dataset.bid_close or {}).get(sym)
            ask = (dataset.ask_close or {}).get(sym)
            if bid is None or ask is None or len(bid) != n or len(ask) != n:
                add("warn", "bidask_length_mismatch", f"bid/ask length mismatch for {sym}.")
            else:
                bad = 0
                for b, a in zip(bid, ask):
                    bb = float(b or 0.0)
                    aa = float(a or 0.0)
                    if bb < 0.0 or aa < 0.0:
                        bad += 1
                    # 0 indicates missing; only check ordering when both are non-zero.
                    if bb > 0.0 and aa > 0.0 and bb > aa:
                        bad += 1
                if bad > 0:
                    add("error", "bid_gt_ask", f"bid_close[{sym}] > ask_close[{sym}] (or negative) on {bad} bars.")

        # Funding times should be on timeline (event at bar close).
        f = dataset.funding.get(sym, {})
        if f:
            bad_ts = [t for t in f.keys() if t not in timeline_set]
            if bad_ts:
                add("warn", "funding_off_timeline", f"funding[{sym}] has events not on timeline; they will be ignored.", {"count": len(bad_ts)})

    ok = not any(x["level"] == "error" for x in issues)
    return {
        "ok": ok,
        "provider": dataset.provider,
        "fingerprint": dataset.fingerprint,
        "symbols": list(dataset.symbols),
        "bars": n,
        "expected_step_seconds": expected_step_seconds,
        "semantic_profile": semantic_profile,
        "issues": issues,
    }
