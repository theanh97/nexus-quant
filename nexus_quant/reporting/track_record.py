"""
NEXUS Platform Track Record
==============================
Unified performance tracking across all NEXUS projects.

Records:
- Per-year Sharpe ratios for each project
- Max drawdown, CAGR, Calmar ratio
- Comparison vs benchmarks (SPY, BTC, SG CTA, Gold)
- Cross-project correlation (diversification value)
- Capital allocation suggestions

Usage:
    from nexus_quant.reporting import NexusTrackRecord
    tr = NexusTrackRecord.load()
    print(tr.summary())
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmarks import BENCHMARK_DATA, compute_benchmark_sharpe


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class YearMetrics:
    """Performance metrics for a single year."""
    year: int
    sharpe: Optional[float] = None
    cagr_pct: Optional[float] = None           # Annualised return (%)
    max_drawdown_pct: Optional[float] = None   # Max drawdown (%)
    calmar: Optional[float] = None             # CAGR / MaxDD
    n_trades: Optional[int] = None
    status: str = "live"  # "live", "backtest", "oos", "pending"


@dataclass
class ProjectRecord:
    """Full track record for one NEXUS project."""
    name: str
    market: str                         # "crypto", "commodity", "fx", "equity"
    asset_class: str                    # "perpetual_futures", "commodity_futures", etc.
    description: str
    champion_config: str                # path to champion config file
    target_sharpe: float = 0.8
    status: str = "development"         # "production", "validated", "development", "planned"

    # Per-year metrics
    years: Dict[int, YearMetrics] = field(default_factory=dict)

    # Aggregate stats (computed from years)
    avg_sharpe: Optional[float] = None
    min_sharpe: Optional[float] = None
    max_sharpe: Optional[float] = None
    avg_cagr_pct: Optional[float] = None
    avg_max_dd_pct: Optional[float] = None

    # Meta
    inception_date: str = ""
    last_updated: str = ""
    notes: str = ""

    def compute_aggregates(self) -> None:
        """Recompute summary stats from year metrics."""
        sharpes = [m.sharpe for m in self.years.values()
                   if m.sharpe is not None and m.status in ("live", "backtest", "oos")]
        if sharpes:
            self.avg_sharpe = round(sum(sharpes) / len(sharpes), 3)
            self.min_sharpe = round(min(sharpes), 3)
            self.max_sharpe = round(max(sharpes), 3)

        cagrs = [m.cagr_pct for m in self.years.values()
                 if m.cagr_pct is not None]
        if cagrs:
            self.avg_cagr_pct = round(sum(cagrs) / len(cagrs), 2)

        dds = [m.max_drawdown_pct for m in self.years.values()
               if m.max_drawdown_pct is not None]
        if dds:
            self.avg_max_dd_pct = round(sum(dds) / len(dds), 2)

        self.last_updated = datetime.utcnow().strftime("%Y-%m-%d")

    def meets_target(self) -> bool:
        return (self.min_sharpe is not None and self.min_sharpe >= self.target_sharpe * 0.5
                and self.avg_sharpe is not None and self.avg_sharpe >= self.target_sharpe)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert int keys to str for JSON
        d["years"] = {str(k): v for k, v in d["years"].items()}
        return d


@dataclass
class BenchmarkRecord:
    """Annual performance for a market benchmark."""
    name: str
    years: Dict[int, Optional[float]]   # year -> Sharpe ratio
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "years": {str(k): v for k, v in self.years.items()},
            "description": self.description,
        }


# ── Main track record class ───────────────────────────────────────────────────


class NexusTrackRecord:
    """
    Master track record for the NEXUS Quant Trading Platform.
    Aggregates performance across all projects and benchmarks.
    """

    RECORD_PATH = Path("artifacts/track_record.json")

    def __init__(self) -> None:
        self.projects: Dict[str, ProjectRecord] = {}
        self.benchmarks: Dict[str, BenchmarkRecord] = {}
        self.created: str = datetime.utcnow().strftime("%Y-%m-%d")
        self.platform_version: str = "1.0.0"

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def build_from_memory(cls) -> "NexusTrackRecord":
        """
        Construct the track record from known research results in memory.
        Populated from memory/MEMORY.md Phase history and validation results.
        """
        tr = cls()
        tr._add_crypto_perps()
        tr._add_crypto_options()
        tr._add_commodity_cta()
        tr._add_fx_majors()
        tr._add_benchmarks()
        return tr

    @classmethod
    def load(cls, path: Optional[str] = None) -> "NexusTrackRecord":
        """Load from saved JSON, or build from memory if file missing."""
        p = Path(path) if path else cls.RECORD_PATH
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                return cls._from_dict(data)
            except Exception:
                pass
        return cls.build_from_memory()

    def save(self, path: Optional[str] = None) -> None:
        """Persist to JSON."""
        p = Path(path) if path else self.RECORD_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    # ── Project loaders (from known memory) ───────────────────────────────────

    def _add_crypto_perps(self) -> None:
        """
        crypto_perps champion: P91b + Vol Tilt (Phase 113-129)
        All years are OOS (out-of-sample) validated backtest results.
        Source: MEMORY.md Phase history + validation outputs
        """
        pr = ProjectRecord(
            name="crypto_perps",
            market="crypto",
            asset_class="perpetual_futures",
            description=(
                "Cross-sectional momentum + funding carry on Binance USDM perpetuals. "
                "Ensemble: V1(27.47%) + I460_bw168_k4(19.67%) + I415_bw216_k4(32.47%) "
                "+ F144_k2(20.39%). Vol tilt overlay r=0.65 when vol_mom_z > threshold."
            ),
            champion_config="configs/production_p91b_champion.json",
            target_sharpe=1.0,
            status="production",
            inception_date="2021-01-01",
            notes=(
                "Phase 129 (vol regime overlay): WF validated. "
                "Phase 135 cross-asset correlation confirmed < 0.2 with commodity CTA."
            ),
        )

        # Per-year validated results (from MEMORY.md)
        # status="oos" = out-of-sample validated backtest (not paper trading yet)
        pr.years = {
            2021: YearMetrics(year=2021, sharpe=2.926, cagr_pct=None, max_drawdown_pct=-4.6,  status="oos"),
            2022: YearMetrics(year=2022, sharpe=1.427, cagr_pct=None, max_drawdown_pct=-2.0,  status="oos"),
            2023: YearMetrics(year=2023, sharpe=1.439, cagr_pct=None, max_drawdown_pct=-2.7,  status="oos"),
            2024: YearMetrics(year=2024, sharpe=2.198, cagr_pct=None, max_drawdown_pct=-2.7,  status="oos"),
            2025: YearMetrics(year=2025, sharpe=2.038, cagr_pct=None, max_drawdown_pct=-1.6,  status="oos"),
            2026: YearMetrics(year=2026, sharpe=1.512, cagr_pct=None, max_drawdown_pct=-0.5,  status="oos"),
        }

        pr.compute_aggregates()
        self.projects["crypto_perps"] = pr

    def _add_crypto_options(self) -> None:
        """
        crypto_options: VRP (Variance Risk Premium) strategy — Phase 2 validated.
        Proper gamma/theta model: PnL = 0.5*(IV²-RV²)*dt per bar.
        Source: Phase 2 WF backtest (2021-2025), synthetic Deribit-calibrated IV.
        """
        pr = ProjectRecord(
            name="crypto_options",
            market="crypto",
            asset_class="options",
            description=(
                "Variance Risk Premium: delta-hedged short straddle on BTC/ETH options (Deribit). "
                "Carry-style: always short vol at 1.5x leverage. Exit only on extreme vol spike "
                "(VRP z-score < -2.0). Proper gamma/theta P&L model: 0.5*(IV²-RV²)*dt. "
                "Weekly rebalancing to control costs."
            ),
            champion_config="configs/crypto_options_vrp.json",
            target_sharpe=0.8,
            status="validated",
            inception_date="2026-02-20",
            notes=(
                "Phase 2: Proper options P&L model (gamma/theta). "
                "All 5 WF years positive. exit_z=-2.0 key insight: z=-1.5 causes overtrading. "
                "Skew MR + Term Structure: failed WF — need real Deribit IV chain data. "
                "Estimated corr vs crypto_perps: 0.2-0.35 (both lose in crashes)."
            ),
        )

        pr.years = {
            2021: YearMetrics(year=2021, sharpe=1.273, cagr_pct=6.8, max_drawdown_pct=-3.8, status="oos"),
            2022: YearMetrics(year=2022, sharpe=1.446, cagr_pct=5.8, max_drawdown_pct=-3.7, status="oos"),
            2023: YearMetrics(year=2023, sharpe=1.944, cagr_pct=3.4, max_drawdown_pct=-1.1, status="oos"),
            2024: YearMetrics(year=2024, sharpe=1.344, cagr_pct=4.3, max_drawdown_pct=-2.7, status="oos"),
            2025: YearMetrics(year=2025, sharpe=1.595, cagr_pct=5.9, max_drawdown_pct=-2.8, status="oos"),
        }

        pr.compute_aggregates()
        self.projects["crypto_options"] = pr

    def _add_commodity_cta(self) -> None:
        """
        commodity_cta: Phase 135 — infrastructure complete, backtest pending.
        Results will be updated after real Yahoo data download.
        """
        pr = ProjectRecord(
            name="commodity_cta",
            market="commodity",
            asset_class="commodity_futures",
            description=(
                "CTA Trend+Carry+Momentum+Value ensemble on 13 commodity futures "
                "(Energy/Metals/Grains/Softs). Static weights: Trend(40%)+Carry(30%)"
                "+MomValue(30%). Vol tilt overlay. Target: Sharpe > 0.8."
            ),
            champion_config="configs/cta_ensemble.json",
            target_sharpe=0.8,
            status="development",
            inception_date="2026-02-20",
            notes=(
                "Phase 135: Infrastructure complete (data provider + 4 strategies). "
                "Backtest pending real data download from Yahoo Finance. "
                "Validated on synthetic data: pipeline OK, all features computed. "
                "Expected live Sharpe: 0.8-1.2 based on academic CTA literature."
            ),
        )
        # No years yet — awaiting real backtest
        pr.compute_aggregates()
        self.projects["commodity_cta"] = pr

    def _add_fx_majors(self) -> None:
        """fx_majors: Template only, not yet active."""
        pr = ProjectRecord(
            name="fx_majors",
            market="fx",
            asset_class="spot_fx",
            description="FX major pairs strategy (EURUSD, GBPUSD, JPYUSD, etc.). Template only.",
            champion_config="",
            target_sharpe=0.8,
            status="planned",
            inception_date="",
            notes="Template project. No strategy implemented yet.",
        )
        pr.compute_aggregates()
        self.projects["fx_majors"] = pr

    def _add_benchmarks(self) -> None:
        """Add standard market benchmarks for comparison."""
        years = [2021, 2022, 2023, 2024, 2025]
        for bm_name, bm_data in BENCHMARK_DATA.items():
            sharpes = {yr: bm_data["annual_sharpe"].get(yr) for yr in years}
            self.benchmarks[bm_name] = BenchmarkRecord(
                name=bm_name,
                years=sharpes,
                description=f"Annual Sharpe ratio (estimated). Annual vol: {bm_data['vol_estimate']:.0f}%",
            )

    # ── Summary & reporting ────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable performance comparison table."""
        years = [2021, 2022, 2023, 2024, 2025, 2026]
        lines = []
        lines.append("=" * 90)
        lines.append("NEXUS QUANT TRADING PLATFORM — TRACK RECORD")
        lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 90)
        lines.append("")

        # Header
        col_w = 8
        header = f"{'Project/Benchmark':<28s}"
        for y in years:
            header += f"{y:>{col_w}d}"
        header += f"{'AVG':>{col_w}}{'MIN':>{col_w}}{'Status':>12}"
        lines.append(header)
        lines.append("-" * 90)

        # NEXUS Projects
        lines.append("── NEXUS PROJECTS (Sharpe ratio, OOS backtest) ──")
        for proj_name, proj in self.projects.items():
            if proj.status == "planned":
                continue
            row = f"{proj_name:<28s}"
            for y in years:
                yr_m = proj.years.get(y)
                if yr_m and yr_m.sharpe is not None:
                    row += f"{yr_m.sharpe:>{col_w}.3f}"
                else:
                    row += f"{'—':>{col_w}}"
            avg_s = f"{proj.avg_sharpe:.3f}" if proj.avg_sharpe else "—"
            min_s = f"{proj.min_sharpe:.3f}" if proj.min_sharpe else "—"
            row += f"{avg_s:>{col_w}}{min_s:>{col_w}}{proj.status:>12}"
            lines.append(row)

        lines.append("")
        lines.append("── BENCHMARKS (Sharpe ratio, annual returns) ──")
        bm_display = {
            "SPY (S&P 500)": "SPY (S&P 500)",
            "BTC Buy-Hold": "BTC Buy-Hold",
            "SG CTA Index": "SG CTA Index",
            "Gold (GC)": "Gold (GC)",
            "60/40 Portfolio": "60/40 Portfolio",
        }
        bm_years = [2021, 2022, 2023, 2024, 2025]
        for bm_key, bm_label in bm_display.items():
            bm = self.benchmarks.get(bm_key)
            if not bm:
                continue
            row = f"{bm_label:<28s}"
            for y in years:
                s = bm.years.get(y) if bm else None
                if s is not None:
                    row += f"{s:>{col_w}.2f}"
                else:
                    row += f"{'—':>{col_w}}"
            lines.append(row)

        lines.append("")
        lines.append("=" * 90)
        lines.append("TARGETS vs BENCHMARKS")
        lines.append("-" * 90)
        for proj_name, proj in self.projects.items():
            if proj.status == "planned" or not proj.avg_sharpe:
                continue
            sg_cta_avg = _safe_avg([self.benchmarks.get("SG CTA Index", BenchmarkRecord("", {})).years.get(y)
                                    for y in [2021, 2022, 2023, 2024]])
            spy_avg = _safe_avg([self.benchmarks.get("SPY (S&P 500)", BenchmarkRecord("", {})).years.get(y)
                                 for y in [2021, 2022, 2023, 2024]])
            min_str = f"{proj.min_sharpe:.3f}" if proj.min_sharpe is not None else "—"
            sg_str = f"{sg_cta_avg:.2f}" if sg_cta_avg is not None else "—"
            outcome = "BEATS TARGET" if proj.meets_target() else "PENDING"
            lines.append(
                f"  {proj_name:<22}: AVG={proj.avg_sharpe:.3f}  MIN={min_str}  "
                f"Target={proj.target_sharpe}  vs_SG_CTA={sg_str}  [{outcome}]"
            )
        lines.append("=" * 90)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": "NEXUS Quant Trading",
            "version": self.platform_version,
            "created": self.created,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d"),
            "projects": {k: v.to_dict() for k, v in self.projects.items()},
            "benchmarks": {k: v.to_dict() for k, v in self.benchmarks.items()},
            "meta": {
                "crypto_perps_status": "OOS validated — 5yr avg Sharpe 2.005",
                "crypto_options_status": "VRP validated — 5yr avg Sharpe 1.520, min 1.273",
                "commodity_cta_status": "Infrastructure complete — backtest pending real data",
                "target": "Each project > Sharpe 0.8 min, diversified across markets",
            },
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "NexusTrackRecord":
        tr = cls()
        tr.created = data.get("created", "")
        tr.platform_version = data.get("version", "1.0.0")
        for pname, pdata in data.get("projects", {}).items():
            years = {}
            for yr_str, ym_dict in pdata.get("years", {}).items():
                yr = int(yr_str)
                years[yr] = YearMetrics(**ym_dict)
            pdata["years"] = years
            tr.projects[pname] = ProjectRecord(**pdata)
        for bname, bdata in data.get("benchmarks", {}).items():
            bdata["years"] = {int(k): v for k, v in bdata.get("years", {}).items()}
            tr.benchmarks[bname] = BenchmarkRecord(**bdata)
        return tr

    def api_dict(self) -> Dict[str, Any]:
        """Compact dict for the web dashboard API."""
        years = [2021, 2022, 2023, 2024, 2025, 2026]
        projects_out = []
        for proj_name, proj in self.projects.items():
            year_sharpes = {}
            year_mdd = {}
            for y in years:
                ym = proj.years.get(y)
                year_sharpes[y] = round(ym.sharpe, 3) if ym and ym.sharpe is not None else None
                year_mdd[y] = round(ym.max_drawdown_pct, 2) if ym and ym.max_drawdown_pct is not None else None
            projects_out.append({
                "name": proj_name,
                "market": proj.market,
                "status": proj.status,
                "description": proj.description,
                "avg_sharpe": proj.avg_sharpe,
                "min_sharpe": proj.min_sharpe,
                "target_sharpe": proj.target_sharpe,
                "meets_target": proj.meets_target(),
                "year_sharpe": year_sharpes,
                "year_mdd": year_mdd,
                "champion_config": proj.champion_config,
                "notes": proj.notes,
            })

        benchmarks_out = []
        bm_order = ["SPY (S&P 500)", "BTC Buy-Hold", "SG CTA Index", "Gold (GC)", "60/40 Portfolio"]
        for bm_name in bm_order:
            bm = self.benchmarks.get(bm_name)
            if not bm:
                continue
            year_sharpes = {y: round(bm.years.get(y), 2) if bm.years.get(y) is not None else None
                            for y in years}
            benchmarks_out.append({
                "name": bm_name,
                "year_sharpe": year_sharpes,
                "description": bm.description,
            })

        return {
            "generated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "projects": projects_out,
            "benchmarks": benchmarks_out,
            "years": years,
            "meta": {
                "note": "All Sharpe ratios are annualised. NEXUS = OOS backtest. Benchmarks = estimated from returns.",
                "crypto_perps_best": "AVG=2.005, MIN=1.427 (5-yr OOS)",
                "crypto_options_best": "VRP AVG=1.520, MIN=1.273 (5-yr OOS, synthetic IV)",
                "target": "Each project MIN Sharpe > target_sharpe",
            },
        }


def _safe_avg(vals: list) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None
