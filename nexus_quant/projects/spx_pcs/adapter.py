"""
SPX PCS Bridge Adapter
======================

Đọc artifacts từ algoxpert engine và convert sang NEXUS format.
Không cần copy data — chỉ đọc JSON/CSV kết quả từ algoxpert artifacts/.

Environment:
  ALGOXPERT_DIR: path tới algoxpert-3rd-alpha-spx repo
                 (default: Desktop/algoxpert-3rd-alpha-spx)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_algoxpert_dir() -> Path:
    """Resolve path to algoxpert repo from env var or default locations."""
    env_dir = os.environ.get("ALGOXPERT_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    # Default: sibling on Desktop
    home = Path.home()
    candidates = [
        home / "Desktop" / "algoxpert-3rd-alpha-spx",
        home / "algoxpert-3rd-alpha-spx",
        Path(__file__).parent.parent.parent.parent.parent / "algoxpert-3rd-alpha-spx",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Cannot find algoxpert-3rd-alpha-spx repo. "
        "Set env var ALGOXPERT_DIR=/path/to/algoxpert-3rd-alpha-spx"
    )


class SPXArtifactReader:
    """
    Đọc backtest artifacts từ algoxpert engine.

    Artifacts location:
      {algoxpert_dir}/Custom_Backtest_Framework/artifacts/runtime/
      {algoxpert_dir}/Custom_Backtest_Framework/artifacts/optimization/
    """

    def __init__(self, algoxpert_dir: Optional[Path] = None) -> None:
        if algoxpert_dir is None:
            algoxpert_dir = _resolve_algoxpert_dir()
        self.algoxpert_dir = Path(algoxpert_dir)
        self.framework_dir = self.algoxpert_dir / "Custom_Backtest_Framework"
        self.runtime_dir = self.framework_dir / "artifacts" / "runtime"
        self.optimization_dir = self.framework_dir / "artifacts" / "optimization"

    def is_available(self) -> bool:
        """Check if algoxpert repo is accessible."""
        return self.algoxpert_dir.exists() and self.runtime_dir.exists()

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Return list of all pcs_summary_*.json files sorted by date."""
        if not self.runtime_dir.exists():
            return []
        summaries = []
        for f in sorted(self.runtime_dir.glob("pcs_summary_*.json")):
            try:
                with open(f) as fp:
                    d = json.load(fp)
                    d["_filename"] = f.name
                    d["_path"] = str(f)
                    summaries.append(d)
            except Exception:
                continue
        return summaries

    def get_latest_summary(self) -> Dict[str, Any]:
        """Return the most recent backtest summary."""
        summaries = self.get_all_summaries()
        if not summaries:
            return {
                "status": "no_results",
                "message": "No algoxpert backtest results found. Run a backtest first.",
                "algoxpert_dir": str(self.algoxpert_dir),
                "project": "spx_pcs",
            }

        # Sort by end date in filename (pcs_summary_YYYYMMDD_YYYYMMDD.json)
        latest = summaries[-1]
        return self._format_summary(latest)

    def get_year_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Aggregate backtest results by year.
        Returns: {year: {sharpe, cagr, max_drawdown, trades, win_rate}}
        """
        summaries = self.get_all_summaries()
        year_data: Dict[int, List[Dict]] = {}

        for s in summaries:
            period = s.get("period", {})
            start_str = period.get("start", "")
            if len(start_str) >= 4:
                year = int(start_str[:4])
                year_data.setdefault(year, []).append(s)

        result = {}
        for year, year_summaries in year_data.items():
            # Use summary with most trades as representative
            best = max(year_summaries, key=lambda x: x.get("trades", 0))
            result[year] = {
                "sharpe": best.get("sharpe_annual"),
                "cagr_pct": best.get("cagr_annual"),
                "max_drawdown_pct": best.get("max_drawdown_pct"),
                "trades": best.get("trades", 0),
                "win_rate_pct": best.get("win_rate_pct"),
                "days": best.get("days", 0),
            }
        return result

    def get_optimization_results(self) -> List[Dict[str, Any]]:
        """Return list of optimization grid search results."""
        if not self.optimization_dir.exists():
            return []
        results = []
        for f in sorted(self.optimization_dir.glob("grid_*.csv")):
            results.append({
                "filename": f.name,
                "path": str(f),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "size_bytes": f.stat().st_size,
            })
        return results

    def get_engine_status(self) -> Dict[str, Any]:
        """Return status info about algoxpert engine availability."""
        available = self.is_available()
        rust_built = (
            self.framework_dir / "spx_core" / "target" / "release"
        ).exists() if available else False

        data_dir = self.framework_dir / "data"
        data_years = []
        if data_dir.exists():
            data_years = [
                d.name.replace("spxw_", "")
                for d in sorted(data_dir.iterdir())
                if d.is_dir() and d.name.startswith("spxw_")
            ]

        return {
            "available": available,
            "algoxpert_dir": str(self.algoxpert_dir),
            "rust_built": rust_built,
            "data_years": data_years,
            "data_size_note": "41GB SPXW parquet (not copied into NEXUS)",
        }

    def _format_summary(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert algoxpert summary format → NEXUS-friendly dict."""
        period = raw.get("period", {})
        return {
            "project": "spx_pcs",
            "status": "development",
            "period": {
                "start": period.get("start"),
                "end": period.get("end"),
                "days": raw.get("days"),
            },
            "performance": {
                "sharpe_annual": raw.get("sharpe_annual"),
                "sortino_annual": raw.get("sortino_annual"),
                "cagr_annual_pct": raw.get("cagr_annual"),
                "max_drawdown_pct": raw.get("max_drawdown_pct"),
                "total_return_pct": raw.get("total_return_pct"),
                "win_rate_pct": raw.get("win_rate_pct"),
                "profit_factor": raw.get("profit_factor"),
                "calmar": raw.get("calmar"),
            },
            "trade_stats": {
                "total_trades": raw.get("trades"),
                "wins": raw.get("wins"),
                "avg_holding_min": raw.get("avg_holding_min"),
                "avg_credit_entry": raw.get("avg_credit_entry"),
                "avg_debit_exit": raw.get("avg_debit_exit"),
                "total_fees_usd": raw.get("total_fees_usd"),
            },
            "params": raw.get("params", {}),
            "source_file": raw.get("_filename"),
            "engine": "algoxpert-3rd-alpha-spx (Python + Rust)",
            "note": "Results from algoxpert bridge adapter. WF validation pending.",
        }
