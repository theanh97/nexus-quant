"""
report_cli.py — NEXUS Quant self-contained HTML performance report generator.

Usage:
    python3 nexus_quant/report_cli.py --artifacts artifacts --out /tmp/nexus_report.html

Requirements: stdlib only (json, pathlib, math, statistics, datetime, argparse, html).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def _load_run_records(artifacts_dir: Path) -> list[dict]:
    """Return one record per run directory that contains metrics.json + result.json."""
    records: list[dict] = []
    runs_dir = artifacts_dir / "runs"
    if not runs_dir.exists():
        return records

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        result_path = run_dir / "result.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not result_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
            result = json.loads(result_path.read_text())
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
        except Exception:
            continue

        summary = metrics.get("summary", {})
        verdict_raw = metrics.get("verdict", {})
        verdict_pass = verdict_raw.get("pass", True) if isinstance(verdict_raw, dict) else bool(verdict_raw)

        strategy_name = (
            config.get("strategy", {}).get("name", "")
            or result.get("strategy", "")
            or run_dir.name
        )
        run_name = config.get("run_name", run_dir.name)

        records.append({
            "run_dir": run_dir.name,
            "run_name": run_name,
            "strategy": strategy_name,
            "sharpe": summary.get("sharpe", float("nan")),
            "calmar": summary.get("calmar", float("nan")),
            "mdd": summary.get("max_drawdown", float("nan")),
            "cagr": summary.get("cagr", float("nan")),
            "win_rate": summary.get("win_rate", float("nan")),
            "volatility": summary.get("volatility", float("nan")),
            "verdict": "PASS" if verdict_pass else "FAIL",
            "equity_curve": result.get("equity_curve", []),
            "returns": result.get("returns", []),
        })
    return records


def _load_ledger_events(artifacts_dir: Path, n: int = 10) -> list[dict]:
    """Return the last n ledger events."""
    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    if not ledger_path.exists():
        return []
    events: list[dict] = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events[-n:]


def _load_wisdom(artifacts_dir: Path) -> dict:
    """Load wisdom/latest.json if present."""
    candidates = [
        artifacts_dir / "wisdom" / "latest.json",
        artifacts_dir / "memory" / "best_params.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Statistics helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _safe(x: float, decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}"


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def _downsample(seq: list, max_points: int = 500) -> list:
    """Uniformly downsample a sequence to at most max_points entries."""
    n = len(seq)
    if n <= max_points:
        return seq
    step = n / max_points
    return [seq[int(i * step)] for i in range(max_points)]


def _compute_risk_stats(returns: list[float]) -> dict:
    """Compute VaR95, VaR99, CVaR95, Skewness, Kurtosis, AnnVol from a returns list."""
    data = [r for r in returns if not math.isnan(r)]
    if len(data) < 4:
        return {}
    n = len(data)
    sorted_r = sorted(data)

    # VaR (parametric-free, historical)
    var95_idx = max(0, int(math.floor(0.05 * n)) - 1)
    var99_idx = max(0, int(math.floor(0.01 * n)) - 1)
    var95 = -sorted_r[var95_idx]
    var99 = -sorted_r[var99_idx]

    # CVaR95 (expected shortfall)
    cutoff95 = int(math.floor(0.05 * n))
    tail = sorted_r[:max(1, cutoff95)]
    cvar95 = -statistics.mean(tail)

    # Mean, std
    mean_r = statistics.mean(data)
    try:
        std_r = statistics.stdev(data)
    except statistics.StatisticsError:
        std_r = 0.0

    # Annualised vol (hourly bars assumed → 8760 bars/year)
    ann_vol = std_r * math.sqrt(8760)

    # Skewness
    if std_r == 0:
        skew = 0.0
    else:
        skew = sum((x - mean_r) ** 3 for x in data) / (n * std_r ** 3)

    # Excess kurtosis
    if std_r == 0:
        kurt = 0.0
    else:
        kurt = sum((x - mean_r) ** 4 for x in data) / (n * std_r ** 4) - 3.0

    return {
        "var95": var95,
        "var99": var99,
        "cvar95": cvar95,
        "skewness": skew,
        "kurtosis": kurt,
        "ann_vol": ann_vol,
    }


def _histogram(data: list[float], bins: int = 20) -> tuple[list[float], list[int]]:
    """Return (bin_centres, counts) for a histogram."""
    clean = [x for x in data if not math.isnan(x)]
    if not clean:
        return [], []
    lo, hi = min(clean), max(clean)
    if lo == hi:
        return [lo], [len(clean)]
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in clean:
        idx = min(int((x - lo) / width), bins - 1)
        counts[idx] += 1
    centres = [lo + (i + 0.5) * width for i in range(bins)]
    return centres, counts


# ---------------------------------------------------------------------------
# HTML / CSS / JS templates
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --border: #30363d;
  --accent: #58a6ff;
  --accent2: #3fb950;
  --warn: #f85149;
  --text: #c9d1d9;
  --muted: #8b949e;
  --header-h: 70px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }

/* ── Header ── */
header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 32px;
  height: var(--header-h);
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
header h1 {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.3px;
}
header .ts { color: var(--muted); font-size: 0.8rem; }

/* ── Layout ── */
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 28px 24px 60px;
  display: grid;
  gap: 24px;
}

/* ── Card ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 24px;
  overflow: hidden;
}
.card-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-title .badge {
  background: rgba(88,166,255,0.12);
  color: var(--accent);
  border-radius: 4px;
  padding: 1px 8px;
  font-size: 0.7rem;
  font-weight: 700;
}

/* ── Two-column grid for charts ── */
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
@media (max-width: 860px) {
  .grid-2 { grid-template-columns: 1fr; }
}

/* ── Tables ── */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.87rem;
}
thead tr { border-bottom: 1px solid var(--border); }
th {
  text-align: left;
  color: var(--muted);
  font-weight: 600;
  padding: 8px 10px;
  white-space: nowrap;
}
td { padding: 9px 10px; border-bottom: 1px solid rgba(48,54,61,0.6); }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(88,166,255,0.04); }
.pass { color: var(--accent2); font-weight: 700; }
.fail { color: var(--warn); font-weight: 700; }
.mono { font-family: 'Cascadia Code', 'Fira Mono', monospace; font-size: 0.82rem; }
.right { text-align: right; }

/* ── Chart containers ── */
.chart-wrap {
  position: relative;
  width: 100%;
}

/* ── KPI strip ── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}
.kpi {
  background: rgba(88,166,255,0.06);
  border: 1px solid rgba(88,166,255,0.18);
  border-radius: 8px;
  padding: 14px 18px;
}
.kpi .label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 1.5rem; font-weight: 700; color: var(--accent); margin-top: 4px; }
.kpi .sub { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

/* ── Footer ── */
footer {
  border-top: 1px solid var(--border);
  padding: 18px 32px;
  text-align: center;
  color: var(--muted);
  font-size: 0.8rem;
  background: var(--bg2);
}
"""

_CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"

# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(str(s))


def _fmt_num(x, decimals=4):
    if isinstance(x, float) and math.isnan(x):
        return "—"
    return f"{x:.{decimals}f}"


def _strategy_table_html(records: list[dict]) -> str:
    if not records:
        return "<p style='color:var(--muted)'>No run records found.</p>"

    rows = ""
    for r in records:
        verdict_cls = "pass" if r["verdict"] == "PASS" else "fail"
        rows += f"""
        <tr>
          <td><span class='mono'>{_esc(r['strategy'])}</span></td>
          <td class='right'>{_fmt_num(r['sharpe'], 3)}</td>
          <td class='right'>{_fmt_num(r['calmar'], 2)}</td>
          <td class='right'>{_pct(r['mdd'])}</td>
          <td class='right'>{_pct(r['cagr'])}</td>
          <td class='right'>{_pct(r['win_rate'])}</td>
          <td class='right {verdict_cls}'>{_esc(r['verdict'])}</td>
        </tr>"""

    return f"""
    <table>
      <thead>
        <tr>
          <th>Strategy</th>
          <th class='right'>Sharpe</th>
          <th class='right'>Calmar</th>
          <th class='right'>MDD</th>
          <th class='right'>CAGR</th>
          <th class='right'>Win Rate</th>
          <th class='right'>Verdict</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _risk_table_html(stats: dict) -> str:
    if not stats:
        return "<p style='color:var(--muted)'>Insufficient return data.</p>"
    rows = [
        ("VaR 95%", _pct(stats.get("var95", float("nan")))),
        ("VaR 99%", _pct(stats.get("var99", float("nan")))),
        ("CVaR 95%", _pct(stats.get("cvar95", float("nan")))),
        ("Skewness", _fmt_num(stats.get("skewness", float("nan")), 4)),
        ("Kurtosis (excess)", _fmt_num(stats.get("kurtosis", float("nan")), 4)),
        ("Ann. Volatility", _pct(stats.get("ann_vol", float("nan")))),
    ]
    trs = "".join(f"<tr><td>{_esc(k)}</td><td class='right mono'>{_esc(v)}</td></tr>" for k, v in rows)
    return f"""
    <table>
      <thead><tr><th>Metric</th><th class='right'>Value</th></tr></thead>
      <tbody>{trs}</tbody>
    </table>"""


def _ledger_table_html(events: list[dict]) -> str:
    if not events:
        return "<p style='color:var(--muted)'>No ledger events found.</p>"
    rows = ""
    for ev in reversed(events):
        ts = ev.get("ts", "—")
        kind = ev.get("kind", "—")
        run_name = ev.get("run_name", "—")
        payload = ev.get("payload", {})
        sharpe = "—"
        verdict = "—"
        if isinstance(payload, dict):
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                summary = metrics.get("summary", {})
                sharpe = _fmt_num(summary.get("sharpe", float("nan")), 3) if summary else "—"
                vrd = metrics.get("verdict", {})
                if isinstance(vrd, dict):
                    verdict = "PASS" if vrd.get("pass", True) else "FAIL"
        verdict_cls = "pass" if verdict == "PASS" else ("fail" if verdict == "FAIL" else "")
        rows += f"""
        <tr>
          <td class='mono'>{_esc(str(ts)[:19])}</td>
          <td>{_esc(kind)}</td>
          <td class='mono' style='font-size:0.78rem'>{_esc(str(run_name)[:40])}</td>
          <td class='right'>{sharpe}</td>
          <td class='right {verdict_cls}'>{verdict}</td>
        </tr>"""

    return f"""
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Kind</th>
          <th>Run Name</th>
          <th class='right'>Sharpe</th>
          <th class='right'>Verdict</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _kpi_strip_html(best: dict) -> str:
    if not best:
        return ""
    kpis = [
        ("Sharpe", _fmt_num(best.get("sharpe", float("nan")), 2), "best run"),
        ("CAGR", _pct(best.get("cagr", float("nan"))), "annualised"),
        ("Max DD", _pct(best.get("mdd", float("nan"))), "max drawdown"),
        ("Win Rate", _pct(best.get("win_rate", float("nan"))), "of trades"),
        ("Calmar", _fmt_num(best.get("calmar", float("nan")), 2), "cagr/mdd"),
    ]
    items = "".join(
        f"<div class='kpi'><div class='label'>{_esc(label)}</div>"
        f"<div class='value'>{_esc(value)}</div>"
        f"<div class='sub'>{_esc(sub)}</div></div>"
        for label, value, sub in kpis
    )
    return f"<div class='kpi-strip'>{items}</div>"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_report(artifacts_dir: Path, out_path: Path) -> Path:
    artifacts_dir = artifacts_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    records = _load_run_records(artifacts_dir)
    ledger_events = _load_ledger_events(artifacts_dir, n=10)
    _wisdom = _load_wisdom(artifacts_dir)

    # Pick the best run by Sharpe for equity curve + risk stats
    valid = [r for r in records if not math.isnan(r["sharpe"])]
    best = max(valid, key=lambda r: r["sharpe"]) if valid else (records[0] if records else {})

    # Equity curve (downsampled)
    eq_raw = best.get("equity_curve", []) if best else []
    eq_ds = _downsample([float(v) for v in eq_raw], 500)
    eq_labels = list(range(len(eq_ds)))

    # Returns (for histogram + risk stats)
    rets_raw = best.get("returns", []) if best else []
    rets = [float(v) for v in rets_raw if v is not None]

    # Risk stats
    risk_stats = _compute_risk_stats(rets)

    # Histogram
    hist_centres, hist_counts = _histogram(rets, bins=20)
    hist_labels = [f"{c:.5f}" for c in hist_centres]

    # Timestamp
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Best strategy name
    best_name = _esc(best.get("strategy", "N/A")) if best else "N/A"
    best_run_name = _esc(best.get("run_name", "N/A")) if best else "N/A"

    # JSON blobs for JS
    eq_json = json.dumps(eq_ds)
    eq_labels_json = json.dumps(eq_labels)
    hist_labels_json = json.dumps(hist_labels)
    hist_counts_json = json.dumps(hist_counts)

    # KPI strip
    kpi_html = _kpi_strip_html(best) if best else ""

    # Section HTMLs
    sec1_html = _strategy_table_html(records)
    risk_html = _risk_table_html(risk_stats)
    ledger_html = _ledger_table_html(ledger_events)

    # -----------------------------------------------------------------------
    # Compose full HTML
    # -----------------------------------------------------------------------
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NEXUS Quant — Performance Report</title>
  <script src="{_CHARTJS_CDN}"></script>
  <style>{_CSS}</style>
</head>
<body>

<header>
  <h1>&#9650; NEXUS Quant &mdash; Performance Report</h1>
  <span class="ts">Generated: {ts_str}</span>
</header>

<main>

  <!-- KPI Strip -->
  {kpi_html}

  <!-- Section 1: Strategy Comparison -->
  <div class="card">
    <div class="card-title">
      01 &nbsp; Strategy Comparison
      <span class="badge">{len(records)} runs</span>
    </div>
    {sec1_html}
  </div>

  <!-- Section 2 + 3: Charts side-by-side -->
  <div class="grid-2">

    <!-- Equity Curve -->
    <div class="card">
      <div class="card-title">02 &nbsp; Best Strategy Equity Curve</div>
      <p style="color:var(--muted);font-size:0.8rem;margin-bottom:14px;">
        Strategy: <strong style="color:var(--text)">{best_name}</strong>
        &nbsp;&bull;&nbsp; Run: <span class="mono" style="font-size:0.78rem">{best_run_name}</span>
      </p>
      <div class="chart-wrap" style="height:280px">
        <canvas id="equityChart"></canvas>
      </div>
    </div>

    <!-- Return Distribution -->
    <div class="card">
      <div class="card-title">03 &nbsp; Return Distribution</div>
      <p style="color:var(--muted);font-size:0.8rem;margin-bottom:14px;">
        Hourly returns histogram &mdash; {len(rets):,} observations, 20 bins
      </p>
      <div class="chart-wrap" style="height:280px">
        <canvas id="histChart"></canvas>
      </div>
    </div>

  </div>

  <!-- Section 4: Risk Stats -->
  <div class="card">
    <div class="card-title">04 &nbsp; Risk Statistics</div>
    {risk_html}
  </div>

  <!-- Section 5: Ledger Events -->
  <div class="card">
    <div class="card-title">
      05 &nbsp; Recent Ledger Events
      <span class="badge">last {min(10, len(ledger_events))}</span>
    </div>
    {ledger_html}
  </div>

</main>

<footer>Generated by NEXUS Quant v1.0 &nbsp;&bull;&nbsp; {ts_str}</footer>

<script>
// ── Embedded data ──
const EQ_LABELS = {eq_labels_json};
const EQ_DATA   = {eq_json};
const HIST_LABELS = {hist_labels_json};
const HIST_COUNTS = {hist_counts_json};

// ── Shared Chart.js defaults ──
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ── Equity Curve ──
(function() {{
  const ctx = document.getElementById('equityChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: EQ_LABELS,
      datasets: [{{
        label: 'Portfolio NAV',
        data: EQ_DATA,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.2,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#161b22',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {{
            label: ctx => ' NAV: ' + ctx.parsed.y.toFixed(6)
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ maxTicksLimit: 8, color: '#8b949e' }},
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          title: {{ display: true, text: 'Bar Index', color: '#8b949e' }}
        }},
        y: {{
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          title: {{ display: true, text: 'NAV (normalised)', color: '#8b949e' }}
        }}
      }}
    }}
  }});
}})();

// ── Return Histogram ──
(function() {{
  const ctx = document.getElementById('histChart').getContext('2d');

  // Colour bars: negative = red, positive = green
  const colors = HIST_LABELS.map(l => parseFloat(l) < 0
    ? 'rgba(248,81,73,0.7)'
    : 'rgba(63,185,80,0.7)');
  const borders = HIST_LABELS.map(l => parseFloat(l) < 0 ? '#f85149' : '#3fb950');

  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: HIST_LABELS,
      datasets: [{{
        label: 'Count',
        data: HIST_COUNTS,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#161b22',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {{
            label: ctx => ' Count: ' + ctx.parsed.y
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{
            maxTicksLimit: 10,
            color: '#8b949e',
            callback: function(val, idx) {{
              return parseFloat(HIST_LABELS[idx]).toFixed(4);
            }}
          }},
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          title: {{ display: true, text: 'Return', color: '#8b949e' }}
        }},
        y: {{
          ticks: {{ color: '#8b949e' }},
          grid: {{ color: 'rgba(48,54,61,0.5)' }},
          title: {{ display: true, text: 'Frequency', color: '#8b949e' }}
        }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""

    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML performance report for NEXUS Quant."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="Path to the artifacts directory (default: ./artifacts)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/nexus_report.html"),
        help="Output path for the HTML report (default: /tmp/nexus_report.html)",
    )
    args = parser.parse_args()

    out = generate_report(artifacts_dir=args.artifacts, out_path=args.out)
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
