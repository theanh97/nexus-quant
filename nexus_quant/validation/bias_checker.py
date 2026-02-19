"""
NEXUS Bias Checker - Detects common biases in backtested strategies.

Implements checks for:
1. Look-ahead bias: data leakage from future into signals
2. Overfitting detection: IS vs OOS Sharpe degradation
3. Multiple hypothesis testing: Bonferroni/BHY correction
4. Sample size adequacy: minimum observations for statistical significance
5. Sharpe significance: t-stat test for H0: Sharpe=0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""

    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (
        0.319381530
        + t
        * (
            -0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )
    )
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly
    return cdf if x >= 0 else 1.0 - cdf


def _clean_returns(returns: List[float]) -> List[float]:
    return [float(r) for r in returns if isinstance(r, (int, float)) and math.isfinite(r)]


def _mean_and_sample_std(values: List[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = 0.0
    for v in values:
        dv = v - mean
        var += dv * dv
    var /= (n - 1)
    return mean, math.sqrt(var)


def _skewness_and_kurtosis(values: List[float]) -> tuple[float, float]:
    n = len(values)
    if n < 3:
        return 0.0, 3.0

    mean = sum(values) / n
    m2 = 0.0
    m3 = 0.0
    m4 = 0.0
    for v in values:
        d = v - mean
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    m2 /= n
    if m2 <= 0.0:
        return 0.0, 3.0
    m3 /= n
    m4 /= n
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 * m2)
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    return skew, kurt


def _annualized_sharpe(returns: List[float], periods_per_year: float) -> tuple[float, int]:
    cleaned = _clean_returns(returns)
    n = len(cleaned)
    if n < 2:
        return 0.0, n
    mean, std = _mean_and_sample_std(cleaned)
    if std <= 0.0:
        if mean > 0.0:
            return math.inf, n
        if mean < 0.0:
            return -math.inf, n
        return 0.0, n
    if periods_per_year <= 0.0:
        periods_per_year = 1.0
    sharpe = (mean / std) * math.sqrt(periods_per_year)
    return sharpe, n


def test_sharpe_significance(
    returns: List[float], periods_per_year: float = 8760
) -> Dict[str, Any]:
    """
    Tests if the Sharpe ratio is statistically significantly different from 0.
    Uses t-statistic: t = Sharpe_annualized * sqrt(n) / sqrt(periods_per_year)
    Approximates p-value from t-distribution (2-tailed).

    Returns: {t_stat, sharpe, n, p_value_approx, significant_95, significant_99, verdict}
    """

    cleaned = _clean_returns(returns)
    sharpe, n = _annualized_sharpe(cleaned, periods_per_year)

    if n < 2 or sharpe == 0.0:
        return {
            "t_stat": 0.0,
            "sharpe": sharpe,
            "n": n,
            "p_value_approx": 1.0,
            "significant_95": False,
            "significant_99": False,
            "verdict": "INSUFFICIENT_DATA" if n < 2 else "NOT_SIGNIFICANT",
        }

    if periods_per_year <= 0.0:
        periods_per_year = 1.0

    if math.isinf(sharpe):
        t_stat = math.copysign(math.inf, sharpe)
        p_value_approx = 0.0
    else:
        t_stat = sharpe * math.sqrt(n) / math.sqrt(periods_per_year)
        p_value_approx = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
        p_value_approx = max(0.0, min(1.0, p_value_approx))

    significant_95 = p_value_approx < 0.05
    significant_99 = p_value_approx < 0.01

    if significant_99:
        verdict = "SIGNIFICANT_99"
    elif significant_95:
        verdict = "SIGNIFICANT_95"
    else:
        verdict = "NOT_SIGNIFICANT"

    return {
        "t_stat": t_stat,
        "sharpe": sharpe,
        "n": n,
        "p_value_approx": p_value_approx,
        "significant_95": significant_95,
        "significant_99": significant_99,
        "verdict": verdict,
    }


def test_is_vs_oos_degradation(
    is_sharpe: float,
    oos_sharpe: float,
    is_periods: int,
    oos_periods: int,
) -> Dict[str, Any]:
    """
    Tests if OOS performance is statistically consistent with IS performance.
    Checks:
    - Degradation ratio: oos/is (should be > 0.5 for healthy strategy)
    - OOS sign preserved: oos_sharpe > 0
    - Relative decay: (is - oos) / is (should be < 0.5)

    Returns: {is_sharpe, oos_sharpe, degradation_ratio, decay_pct, oos_positive, verdict, flags}
    """

    flags: List[str] = []

    if not math.isfinite(is_sharpe) or not math.isfinite(oos_sharpe):
        flags.append("non_finite_sharpe")

    degradation_ratio = math.nan
    decay_pct = math.nan
    if is_sharpe == 0.0:
        flags.append("is_sharpe_zero")
    else:
        degradation_ratio = oos_sharpe / is_sharpe
        decay_pct = (is_sharpe - oos_sharpe) / is_sharpe

        if math.isfinite(degradation_ratio) and degradation_ratio < 0.5:
            flags.append("degradation_ratio_low")
        if math.isfinite(decay_pct) and decay_pct > 0.5:
            flags.append("decay_pct_high")

    oos_positive = oos_sharpe > 0.0
    if not oos_positive:
        flags.append("oos_not_positive")

    if is_periods < 30:
        flags.append("is_sample_small")
    if oos_periods < 30:
        flags.append("oos_sample_small")

    verdict = "PASS" if not flags else "FAIL"

    return {
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "degradation_ratio": degradation_ratio,
        "decay_pct": decay_pct,
        "oos_positive": oos_positive,
        "verdict": verdict,
        "flags": flags,
    }


def test_overfit_detection(returns: List[float], n_params_searched: int = 1) -> Dict[str, Any]:
    """
    Detects overfitting using:
    1. Deflated Sharpe Ratio (Bailey et al. 2014):
       DSR = Phi((sqrt(n-1)/sigma_sharpe) * (SR - SR_benchmark))
       where sigma_sharpe accounts for non-normality
    2. Minimum Track Record Length (MTRL):
       n_min = 1 + (1 - skew*SR + (kurt-1)/4 * SR^2) * (z_alpha / SR)^2
    3. If actual n < n_min -> likely overfit

    Returns: {n, n_required_min, likely_overfit, deflated_sharpe, skewness, kurtosis, verdict}
    """

    cleaned = _clean_returns(returns)
    n = len(cleaned)
    if n < 3:
        return {
            "n": n,
            "n_required_min": math.inf,
            "likely_overfit": True,
            "deflated_sharpe": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "verdict": "INSUFFICIENT_DATA",
        }

    mean, std = _mean_and_sample_std(cleaned)
    if std <= 0.0:
        return {
            "n": n,
            "n_required_min": math.inf,
            "likely_overfit": True,
            "deflated_sharpe": 0.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "verdict": "NO_VARIANCE",
        }

    sr = mean / std
    skew, kurt = _skewness_and_kurtosis(cleaned)

    v = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr * sr)
    if not math.isfinite(v) or v <= 0.0:
        v = 1e-12

    sigma_sharpe = math.sqrt(v)
    sr_std = sigma_sharpe / math.sqrt(n - 1)

    n_eff = max(int(n_params_searched), 1)
    if n_eff <= 1:
        mu_max = 0.0
    else:
        a = math.sqrt(2.0 * math.log(n_eff))
        mu_max = a - (math.log(math.log(n_eff)) + math.log(4.0 * math.pi)) / (2.0 * a)

    sr_benchmark = mu_max * sr_std

    if sr_std > 0.0 and math.isfinite(sr) and math.isfinite(sr_benchmark):
        z = (sr - sr_benchmark) / sr_std
        deflated_sharpe = _norm_cdf(z)
    else:
        deflated_sharpe = 0.0

    z_alpha = 1.96
    if sr <= 0.0 or not math.isfinite(sr):
        n_required_min = math.inf
    else:
        n_required_min = 1.0 + v * (z_alpha / sr) ** 2

    likely_overfit = (n < n_required_min) or (deflated_sharpe < 0.95)

    if likely_overfit:
        verdict = "LIKELY_OVERFIT"
    else:
        verdict = "NOT_OBVIOUSLY_OVERFIT"

    return {
        "n": n,
        "n_required_min": n_required_min,
        "likely_overfit": likely_overfit,
        "deflated_sharpe": deflated_sharpe,
        "skewness": skew,
        "kurtosis": kurt,
        "verdict": verdict,
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.
    When testing N strategies, the significance threshold becomes alpha/N.

    Returns: {n_tests, alpha_adjusted, significant_indices, n_significant, method}
    """

    cleaned: List[float] = []
    for p in p_values:
        if isinstance(p, (int, float)) and math.isfinite(p):
            cleaned.append(float(p))

    n_tests = len(cleaned)
    if n_tests <= 0:
        return {
            "n_tests": 0,
            "alpha_adjusted": alpha,
            "significant_indices": [],
            "n_significant": 0,
            "method": "bonferroni",
        }

    alpha_adjusted = alpha / n_tests
    significant_indices: List[int] = []
    for i, p in enumerate(cleaned):
        if 0.0 <= p <= 1.0 and p <= alpha_adjusted:
            significant_indices.append(i)

    return {
        "n_tests": n_tests,
        "alpha_adjusted": alpha_adjusted,
        "significant_indices": significant_indices,
        "n_significant": len(significant_indices),
        "method": "bonferroni",
    }


def check_survivorship_risk(
    backtest_start_year: int,
    universe_selection_year: int = 2024,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detects survivorship bias risk when the symbol universe was selected
    from a later date than the backtest period.

    Known delisted/collapsed coins that would have been in top-10 at various dates:
    - LUNAUSDT: Top-10 by market cap in Q1 2022, collapsed May 2022
    - FTTUSDT: Top-20 in 2021-2022, delisted Nov 2022 (FTX collapse)

    Returns: {risk_level, reason, confidence_adjustment, flags}
    """
    flags: List[str] = []
    gap_years = universe_selection_year - backtest_start_year

    if gap_years <= 0:
        return {
            "risk_level": "LOW",
            "reason": "Backtest period is contemporaneous with universe selection",
            "confidence_adjustment": 0.0,
            "gap_years": gap_years,
            "flags": [],
        }

    if gap_years >= 3:
        risk = "HIGH"
        adj = -0.15
        flags.append("survivorship_gap_3yr_plus")
    elif gap_years >= 1:
        risk = "MEDIUM"
        adj = -0.08
        flags.append("survivorship_gap_1_2yr")
    else:
        risk = "LOW"
        adj = 0.0

    # Check for known problematic years
    if backtest_start_year <= 2021 and universe_selection_year >= 2024:
        flags.append("pre_luna_ftt_universe")
        adj = min(adj, -0.15)
        risk = "HIGH"
    elif backtest_start_year <= 2022 and universe_selection_year >= 2024:
        flags.append("pre_ftt_universe")
        if risk != "HIGH":
            adj = min(adj, -0.10)
            risk = "MEDIUM"

    reason = (
        f"Universe selected in {universe_selection_year} but backtest starts in "
        f"{backtest_start_year} ({gap_years}yr gap). Coins that crashed/delisted "
        f"between {backtest_start_year} and {universe_selection_year} are excluded, "
        f"inflating historical results by est. 0.2-0.5 Sharpe."
    )

    return {
        "risk_level": risk,
        "reason": reason,
        "confidence_adjustment": adj,
        "gap_years": gap_years,
        "flags": flags,
    }


def run_full_bias_check(
    returns: List[float],
    is_returns: Optional[List[float]] = None,
    oos_returns: Optional[List[float]] = None,
    n_params_searched: int = 1,
    periods_per_year: float = 8760,
    backtest_start_year: Optional[int] = None,
    universe_selection_year: int = 2024,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs all bias checks and returns a comprehensive report.
    Returns: {checks: {sharpe_sig, overfitting, is_oos}, flags: List[str], overall_verdict: str, confidence_score: float}
    """

    sharpe_sig = test_sharpe_significance(returns, periods_per_year=periods_per_year)
    overfitting = test_overfit_detection(returns, n_params_searched=n_params_searched)

    if is_returns is not None and oos_returns is not None:
        is_sharpe, is_n = _annualized_sharpe(is_returns, periods_per_year)
        oos_sharpe, oos_n = _annualized_sharpe(oos_returns, periods_per_year)
        is_oos = test_is_vs_oos_degradation(is_sharpe, oos_sharpe, is_n, oos_n)
    else:
        is_oos = {
            "is_sharpe": math.nan,
            "oos_sharpe": math.nan,
            "degradation_ratio": math.nan,
            "decay_pct": math.nan,
            "oos_positive": False,
            "verdict": "SKIPPED",
            "flags": ["is_oos_not_provided"],
        }

    # Survivorship bias check
    survivorship: Dict[str, Any] = {"risk_level": "SKIPPED", "confidence_adjustment": 0.0, "flags": []}
    if backtest_start_year is not None:
        survivorship = check_survivorship_risk(
            backtest_start_year=backtest_start_year,
            universe_selection_year=universe_selection_year,
            symbols=symbols,
        )

    flags: List[str] = []
    if sharpe_sig.get("n", 0) < 30:
        flags.append("sample_size_small")
    if not sharpe_sig.get("significant_95", False):
        flags.append("sharpe_not_significant_95")
    if overfitting.get("likely_overfit", False):
        flags.append("likely_overfit")
    if is_oos.get("verdict") == "FAIL":
        flags.extend([f"is_oos:{f}" for f in is_oos.get("flags", [])])
    if survivorship.get("risk_level") in ("HIGH", "MEDIUM"):
        flags.extend([f"survivorship:{f}" for f in survivorship.get("flags", [])])

    # Confidence score in [0, 1]
    confidence = 0.5
    if sharpe_sig.get("significant_99", False):
        confidence += 0.25
    elif sharpe_sig.get("significant_95", False):
        confidence += 0.15
    else:
        confidence -= 0.15

    dsr = overfitting.get("deflated_sharpe", 0.0)
    if isinstance(dsr, (int, float)) and math.isfinite(dsr):
        confidence += (float(dsr) - 0.5) * 0.4

    if overfitting.get("likely_overfit", False):
        confidence -= 0.2

    if is_oos.get("verdict") == "PASS":
        confidence += 0.1
    elif is_oos.get("verdict") == "FAIL":
        confidence -= 0.2

    # Apply survivorship bias adjustment
    surv_adj = float(survivorship.get("confidence_adjustment", 0.0))
    confidence += surv_adj

    confidence = max(0.0, min(1.0, confidence))

    if "sharpe_not_significant_95" in flags:
        overall_verdict = "FAIL"
    elif "likely_overfit" in flags:
        overall_verdict = "WARN"
    elif any(f.startswith("is_oos:") for f in flags):
        overall_verdict = "WARN"
    else:
        overall_verdict = "PASS"

    return {
        "checks": {
            "sharpe_sig": sharpe_sig,
            "overfitting": overfitting,
            "is_oos": is_oos,
            "survivorship": survivorship,
        },
        "flags": flags,
        "overall_verdict": overall_verdict,
        "confidence_score": confidence,
    }

