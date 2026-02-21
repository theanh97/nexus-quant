"""
SPX PCS Runner — Bridge to algoxpert engine
============================================

Wrapper để trigger algoxpert backtest/optimization từ NEXUS.
Sử dụng subprocess để gọi algoxpert CLI.

Usage:
    runner = SPXRunner()
    result = runner.run_backtest("20240101", "20241231")
    result = runner.run_optimization("configs/optimize_config.json")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .adapter import _resolve_algoxpert_dir


class SPXRunner:
    """
    Trigger algoxpert engine từ NEXUS.
    
    Gọi: python scripts/run_backtest.py backtest --start YYYYMMDD --end YYYYMMDD
    trong thư mục Custom_Backtest_Framework của algoxpert.
    """

    def __init__(self, algoxpert_dir: Optional[Path] = None) -> None:
        if algoxpert_dir is None:
            algoxpert_dir = _resolve_algoxpert_dir()
        self.algoxpert_dir = Path(algoxpert_dir)
        self.framework_dir = self.algoxpert_dir / "Custom_Backtest_Framework"
        self.python = sys.executable

    def _run_cmd(self, cmd: list, timeout: int = 300) -> Dict[str, Any]:
        """Run command in framework_dir and return result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.framework_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-5000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_backtest(
        self,
        start: str,
        end: str,
        config: Optional[str] = None,
        strategy: Optional[str] = None,
        require_real_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Run algoxpert backtest.
        
        Args:
            start: YYYYMMDD format
            end: YYYYMMDD format
            config: path to config JSON (relative to framework_dir)
            strategy: strategy preset name (e.g., 'baseline', 'mr_bounce_dynamic')
            require_real_data: if True, fail if only synthetic data
        """
        cmd = [
            self.python, "scripts/run_backtest.py", "backtest",
            "--start", start,
            "--end", end,
        ]
        if config:
            cmd += ["--config", config]
        if strategy:
            cmd += ["--strategy", strategy]
        if require_real_data:
            cmd.append("--require-real-data")

        result = self._run_cmd(cmd, timeout=600)
        result["type"] = "backtest"
        result["period"] = {"start": start, "end": end}
        result["timestamp"] = datetime.utcnow().isoformat()
        return result

    def run_optimization(
        self,
        config: str = "configs/optimize_config.json",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run algoxpert grid optimization."""
        cmd = [
            self.python, "scripts/run_backtest.py", "optimize",
            "--config", config,
        ]
        if start:
            cmd += ["--start", start]
        if end:
            cmd += ["--end", end]

        result = self._run_cmd(cmd, timeout=3600)
        result["type"] = "optimization"
        result["timestamp"] = datetime.utcnow().isoformat()
        return result

    def run_doctor(self) -> Dict[str, Any]:
        """Run algoxpert doctor (health check)."""
        cmd = [self.python, "scripts/doctor.py"]
        result = self._run_cmd(cmd, timeout=60)
        result["type"] = "doctor"
        return result

    def run_institutional_gate(self) -> Dict[str, Any]:
        """Run algoxpert institutional validation gate."""
        cmd = [
            self.python, "scripts/run_backtest.py", "institutional-gate"
        ]
        result = self._run_cmd(cmd, timeout=120)
        result["type"] = "institutional_gate"
        return result

    def get_available_strategies(self) -> list:
        """Return list of available strategy presets from algoxpert."""
        cmd = [
            self.python, "scripts/run_backtest.py", "list-strategies"
        ]
        result = self._run_cmd(cmd, timeout=30)
        strategies = []
        if result.get("success") and result.get("stdout"):
            for line in result["stdout"].splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    strategies.append(line)
        return strategies or [
            "baseline", "mr_bounce_dynamic", "ma_crossover",
            "rsi_oversold", "bb_oversold", "combo_2of3_mr_rsi_bb",
            "combo_3of3_mr_ma_bb", "trend_retake",
        ]
