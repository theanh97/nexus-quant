#!/usr/bin/env python3
"""
R85: Production Monitoring Dashboard
========================================

Generates a self-contained HTML dashboard showing:
  1. Equity curve (cumulative P&L)
  2. Drawdown chart
  3. BF z-score history with trade markers
  4. Current signal & health panel
  5. Trade log table
  6. Recent daily P&L table

Usage:
  python3 scripts/r85_monitoring_dashboard.py              # Generate & open
  python3 scripts/r85_monitoring_dashboard.py --no-open    # Generate only
  python3 scripts/r85_monitoring_dashboard.py --serve      # Generate & serve on localhost:8085
"""
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
OUTPUT_PATH = DATA_DIR / "dashboard.html"

NO_OPEN = "--no-open" in sys.argv
SERVE_MODE = "--serve" in sys.argv


def load_json(path):
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_jsonl(path, max_lines=1000):
    """Load JSONL file, return list of dicts."""
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
                    if len(entries) >= max_lines:
                        break
    except Exception:
        pass
    return entries


def generate_dashboard():
    """Generate the full HTML dashboard."""
    # Load data
    state = load_json(DATA_DIR / "position_state.json")
    signal = load_json(DATA_DIR / "latest_signal.json")
    alerts = load_jsonl(DATA_DIR / "alerts.jsonl")

    equity_curve = state.get("equity_curve", [])
    trade_log = state.get("trade_log", [])
    daily_log = state.get("daily_log", [])
    cumulative = state.get("cumulative_pnl", {})
    current_pos = state.get("current_position", {})
    stats = state.get("stats", {})

    # Compute drawdown series from equity curve
    drawdown = []
    peak = 0
    for pt in equity_curve:
        val = pt["cum_pnl_pct"]
        if val > peak:
            peak = val
        dd = val - peak
        drawdown.append({"date": pt["date"], "dd_pct": round(dd, 4)})

    # Extract z-score series from daily_log
    z_scores = [{"date": d["date"], "z": d["z_score"], "pos": d["position"]} for d in daily_log]

    # Chart data: subsample equity if too large (>500 points)
    eq_dates = [p["date"] for p in equity_curve]
    eq_vals = [p["cum_pnl_pct"] for p in equity_curve]
    dd_dates = [p["date"] for p in drawdown]
    dd_vals = [p["dd_pct"] for p in drawdown]

    step = max(1, len(eq_dates) // 500)
    eq_dates_s = eq_dates[::step]
    eq_vals_s = eq_vals[::step]
    dd_dates_s = dd_dates[::step]
    dd_vals_s = dd_vals[::step]

    # Trade markers for equity chart
    trade_markers = []
    eq_date_set = {p["date"]: p["cum_pnl_pct"] for p in equity_curve}
    for t in trade_log:
        if t["date"] in eq_date_set:
            trade_markers.append({
                "x": t["date"],
                "y": round(eq_date_set[t["date"]], 4),
                "type": t["type"],
                "z": round(t["z_score"], 3),
                "dir": "SHORT" if t["to"] < 0 else "LONG",
            })

    # Format signal info
    bf_sig = signal.get("bf_signal", {})
    vrp_sig = signal.get("vrp_signal", {})
    health = signal.get("health", {})
    perf = signal.get("performance", {})
    live = signal.get("live", {})

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEXUS Crypto Options — Production Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0a0e17;
    color: #c8d6e5;
    padding: 16px;
  }}
  .header {{
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 20px;
  }}
  .header h1 {{
    font-size: 24px;
    color: #00d2ff;
    letter-spacing: 2px;
  }}
  .header .subtitle {{
    font-size: 12px;
    color: #576574;
    margin-top: 4px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    max-width: 1400px;
    margin: 0 auto;
  }}
  .card {{
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 16px;
  }}
  .card.full {{ grid-column: 1 / -1; }}
  .card h2 {{
    font-size: 14px;
    color: #00d2ff;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
  }}
  .metric {{
    background: #0d1320;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 10px;
    text-align: center;
  }}
  .metric .label {{
    font-size: 10px;
    color: #576574;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .metric .value {{
    font-size: 20px;
    font-weight: bold;
    margin-top: 4px;
  }}
  .metric .value.green {{ color: #00d97e; }}
  .metric .value.red {{ color: #e74c3c; }}
  .metric .value.blue {{ color: #00d2ff; }}
  .metric .value.yellow {{ color: #f1c40f; }}
  .metric .value.neutral {{ color: #c8d6e5; }}
  .chart-container {{
    position: relative;
    height: 280px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th {{
    text-align: left;
    color: #576574;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 6px 8px;
    border-bottom: 1px solid #1e2a3a;
  }}
  td {{
    padding: 5px 8px;
    border-bottom: 1px solid #0d1320;
    font-variant-numeric: tabular-nums;
  }}
  tr:hover {{ background: #0d1320; }}
  .pos-long {{ color: #00d97e; }}
  .pos-short {{ color: #e74c3c; }}
  .pos-flat {{ color: #576574; }}
  .alert-warn {{ color: #f1c40f; }}
  .alert-crit {{ color: #e74c3c; }}
  .alert-info {{ color: #00d2ff; }}
  .signal-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
  }}
  .signal-badge.hold {{ background: #1e2a3a; color: #576574; }}
  .signal-badge.long {{ background: #0d3320; color: #00d97e; }}
  .signal-badge.short {{ background: #330d0d; color: #e74c3c; }}
  .signal-badge.strong {{ background: #0d2833; color: #00d2ff; }}
  .signal-badge.warning {{ background: #332d0d; color: #f1c40f; }}
  .signal-badge.critical {{ background: #330d0d; color: #e74c3c; }}
  .footer {{
    text-align: center;
    margin-top: 20px;
    font-size: 10px;
    color: #333d4a;
  }}
  @media (max-width: 900px) {{
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>NEXUS — Production Dashboard</h1>
  <div class="subtitle">BTC Options Vol Trading System &nbsp;|&nbsp; Generated: {now_str}</div>
</div>

<!-- Signal & Health Panel -->
<div class="grid">
<div class="card full">
  <h2>Current Status</h2>
  <div class="metric-grid">
    <div class="metric">
      <div class="label">BF Signal</div>
      <div class="value {'green' if bf_sig.get('signal') == 'LONG' else 'red' if bf_sig.get('signal') == 'SHORT' else 'neutral'}">{bf_sig.get('signal', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">BF Z-Score</div>
      <div class="value blue">{bf_sig.get('z_score', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Position</div>
      <div class="value {'green' if current_pos.get('bf_direction', 0) > 0 else 'red' if current_pos.get('bf_direction', 0) < 0 else 'neutral'}">{'LONG BF' if current_pos.get('bf_direction', 0) > 0 else 'SHORT BF' if current_pos.get('bf_direction', 0) < 0 else 'FLAT'}</div>
    </div>
    <div class="metric">
      <div class="label">Health</div>
      <div class="value {'green' if health.get('status') == 'STRONG' else 'yellow' if health.get('status') == 'MODERATE' else 'red'}">{health.get('score', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Total P&L</div>
      <div class="value {'green' if cumulative.get('total_pnl_pct', 0) >= 0 else 'red'}">{cumulative.get('total_pnl_pct', 0):+.2f}%</div>
    </div>
    <div class="metric">
      <div class="label">Max DD</div>
      <div class="value red">{cumulative.get('max_dd_pct', 0):.2f}%</div>
    </div>
    <div class="metric">
      <div class="label">VRP Spread</div>
      <div class="value {'green' if vrp_sig.get('vrp_spread', 0) > 0 else 'red'}">{vrp_sig.get('vrp_spread', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">BTC Price</div>
      <div class="value neutral">${live.get('price', 0):,.0f}</div>
    </div>
    <div class="metric">
      <div class="label">IV (ATM)</div>
      <div class="value blue">{live.get('iv_pct', 'N/A')}%</div>
    </div>
    <div class="metric">
      <div class="label">Win Rate</div>
      <div class="value {'green' if stats.get('hit_rate_pct', 0) >= 55 else 'yellow'}">{stats.get('hit_rate_pct', 'N/A')}%</div>
    </div>
  </div>
</div>

<!-- Equity Curve -->
<div class="card full">
  <h2>Equity Curve — Cumulative P&L (%)</h2>
  <div class="chart-container">
    <canvas id="equityChart"></canvas>
  </div>
</div>

<!-- Drawdown -->
<div class="card">
  <h2>Drawdown (%)</h2>
  <div class="chart-container">
    <canvas id="drawdownChart"></canvas>
  </div>
</div>

<!-- Performance Breakdown -->
<div class="card">
  <h2>Performance Summary</h2>
  <div class="metric-grid">
    <div class="metric">
      <div class="label">Sharpe (90d)</div>
      <div class="value green">{perf.get('recent_90d', {}).get('sharpe', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Sharpe (180d)</div>
      <div class="value green">{perf.get('recent_180d', {}).get('sharpe', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Sharpe (Full)</div>
      <div class="value blue">{perf.get('full_sample', {}).get('sharpe', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Ann Return</div>
      <div class="value green">{perf.get('full_sample', {}).get('ann_ret_pct', 'N/A')}%</div>
    </div>
    <div class="metric">
      <div class="label">Total Trades</div>
      <div class="value neutral">{stats.get('n_trades', 'N/A')}</div>
    </div>
    <div class="metric">
      <div class="label">Trades/Year</div>
      <div class="value neutral">{stats.get('trades_per_year', 'N/A')}</div>
    </div>
  </div>
</div>

<!-- Trade Log -->
<div class="card full">
  <h2>Trade Log (Last 15)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Date</th>
        <th>Type</th>
        <th>Direction</th>
        <th>Z-Score</th>
        <th>BF Value</th>
      </tr>
    </thead>
    <tbody>
"""
    # Add last 15 trades (reversed for most recent first)
    for t in reversed(trade_log[-15:]):
        direction = "LONG" if t["to"] > 0 else "SHORT" if t["to"] < 0 else "FLAT"
        dir_class = "pos-long" if t["to"] > 0 else "pos-short" if t["to"] < 0 else "pos-flat"
        html += f"""      <tr>
        <td>{t['trade_num']}</td>
        <td>{t['date']}</td>
        <td>{t['type']}</td>
        <td class="{dir_class}">{direction}</td>
        <td>{t['z_score']:+.3f}</td>
        <td>{t['bf_value']:.5f}</td>
      </tr>
"""
    html += """    </tbody>
  </table>
</div>

<!-- Daily P&L -->
<div class="card full">
  <h2>Daily P&L (Last 30 Days, bps)</h2>
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>BF P&L</th>
        <th>VRP P&L</th>
        <th>Portfolio</th>
        <th>Cumulative %</th>
        <th>Position</th>
        <th>Z-Score</th>
      </tr>
    </thead>
    <tbody>
"""
    for d in reversed(daily_log[-30:]):
        pos_str = "LONG" if d["position"] > 0 else "SHORT" if d["position"] < 0 else "FLAT"
        pos_class = "pos-long" if d["position"] > 0 else "pos-short" if d["position"] < 0 else "pos-flat"
        pnl_class = "pos-long" if d["port_pnl"] >= 0 else "pos-short"
        html += f"""      <tr>
        <td>{d['date']}</td>
        <td class="{'pos-long' if d['bf_pnl'] >= 0 else 'pos-short'}">{d['bf_pnl']:+.2f}</td>
        <td class="{'pos-long' if d['vrp_pnl'] >= 0 else 'pos-short'}">{d['vrp_pnl']:+.2f}</td>
        <td class="{pnl_class}">{d['port_pnl']:+.2f}</td>
        <td>{d['cum_pnl_pct']:.4f}%</td>
        <td class="{pos_class}">{pos_str}</td>
        <td>{d['z_score']:+.3f}</td>
      </tr>
"""
    html += """    </tbody>
  </table>
</div>

<!-- Alerts -->
<div class="card full">
  <h2>Recent Alerts</h2>
"""
    if not alerts:
        html += '  <p style="color: #576574; font-size: 12px;">No alerts recorded.</p>\n'
    else:
        html += """  <table>
    <thead>
      <tr>
        <th>Timestamp</th>
        <th>Level</th>
        <th>Check</th>
        <th>Message</th>
      </tr>
    </thead>
    <tbody>
"""
        for a in reversed(alerts[-20:]):
            level = a.get("level", "INFO")
            level_class = "alert-crit" if level == "CRITICAL" else "alert-warn" if level == "WARNING" else "alert-info"
            html += f"""      <tr>
        <td>{a.get('timestamp', 'N/A')}</td>
        <td class="{level_class}">{level}</td>
        <td>{a.get('check', 'N/A')}</td>
        <td>{a.get('message', 'N/A')}</td>
      </tr>
"""
        html += """    </tbody>
  </table>
"""
    html += "</div>\n</div>\n"

    # JavaScript for charts
    html += f"""
<div class="footer">
  NEXUS Crypto Options — R85 Monitoring Dashboard — {now_str}
</div>

<script>
const eqDates = {json.dumps(eq_dates_s)};
const eqVals = {json.dumps(eq_vals_s)};
const ddDates = {json.dumps(dd_dates_s)};
const ddVals = {json.dumps(dd_vals_s)};
const tradeMarkers = {json.dumps(trade_markers)};

// Color scheme
const CYAN = '#00d2ff';
const GREEN = '#00d97e';
const RED = '#e74c3c';
const YELLOW = '#f1c40f';
const GRID = '#1e2a3a';
const BG = '#111827';

// Equity curve
const eqCtx = document.getElementById('equityChart').getContext('2d');
new Chart(eqCtx, {{
  type: 'line',
  data: {{
    labels: eqDates,
    datasets: [
      {{
        label: 'Cumulative P&L (%)',
        data: eqVals,
        borderColor: CYAN,
        backgroundColor: 'rgba(0, 210, 255, 0.05)',
        fill: true,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
      }},
      {{
        label: 'Trades',
        data: tradeMarkers.map(t => ({{ x: t.x, y: t.y }})),
        type: 'scatter',
        pointRadius: 5,
        pointStyle: tradeMarkers.map(t => t.dir === 'LONG' ? 'triangle' : 'rectRot'),
        pointBackgroundColor: tradeMarkers.map(t => t.dir === 'LONG' ? GREEN : RED),
        pointBorderColor: tradeMarkers.map(t => t.dir === 'LONG' ? GREEN : RED),
        showLine: false,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: true, labels: {{ color: '#576574', font: {{ size: 10 }} }} }},
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            if (ctx.datasetIndex === 1) {{
              const t = tradeMarkers[ctx.dataIndex];
              return t ? `${{t.type}} ${{t.dir}} z=${{t.z}} @ ${{t.y}}%` : '';
            }}
            return `P&L: ${{ctx.parsed.y.toFixed(4)}}%`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'category',
        ticks: {{ color: '#576574', maxTicksLimit: 10, font: {{ size: 9 }} }},
        grid: {{ color: GRID }}
      }},
      y: {{
        ticks: {{ color: '#576574', font: {{ size: 9 }}, callback: v => v.toFixed(1) + '%' }},
        grid: {{ color: GRID }}
      }}
    }}
  }}
}});

// Drawdown chart
const ddCtx = document.getElementById('drawdownChart').getContext('2d');
new Chart(ddCtx, {{
  type: 'line',
  data: {{
    labels: ddDates,
    datasets: [{{
      label: 'Drawdown (%)',
      data: ddVals,
      borderColor: RED,
      backgroundColor: 'rgba(231, 76, 60, 0.1)',
      fill: true,
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
    }},
    scales: {{
      x: {{
        type: 'category',
        ticks: {{ color: '#576574', maxTicksLimit: 8, font: {{ size: 9 }} }},
        grid: {{ color: GRID }}
      }},
      y: {{
        ticks: {{ color: '#576574', font: {{ size: 9 }}, callback: v => v.toFixed(2) + '%' }},
        grid: {{ color: GRID }}
      }}
    }}
  }}
}});
</script>

</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("  R85: Production Monitoring Dashboard")
    print("=" * 60)

    html = generate_dashboard()

    # Write HTML
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"\n  Dashboard generated: {OUTPUT_PATH}")
    print(f"  Size: {len(html):,} bytes")

    if SERVE_MODE:
        import http.server
        import socketserver
        port = 8085
        os.chdir(OUTPUT_PATH.parent)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"\n  Serving on http://localhost:{port}/dashboard.html")
            print("  Press Ctrl+C to stop.")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Server stopped.")
    elif not NO_OPEN:
        webbrowser.open(f"file://{OUTPUT_PATH}")
        print("  Opened in browser.")

    print("=" * 60)


if __name__ == "__main__":
    main()
