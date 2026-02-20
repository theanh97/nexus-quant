#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text("utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _latest_run_dir(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime)
    return dirs[-1]


def _summary_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    s = metrics.get("summary") or metrics
    return {
        "sharpe": s.get("sharpe"),
        "calmar": s.get("calmar"),
        "cagr": s.get("cagr"),
        "max_drawdown": s.get("max_drawdown"),
        "win_rate": s.get("win_rate"),
        "total_return": s.get("total_return"),
        "periods": s.get("periods"),
        "num_trades": s.get("num_trades"),
    }


def _fmt_num(v: Any, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "N/A"


def _fmt_pct(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v) * 100:.{nd}f}%"
    except Exception:
        return "N/A"


def _collect(artifacts_dir: Path) -> Dict[str, Any]:
    runs_dir = artifacts_dir / "runs"
    latest_dir = _latest_run_dir(runs_dir)
    latest_metrics = _read_json(latest_dir / "metrics.json") if latest_dir else {}
    latest_summary = _summary_from_metrics(latest_metrics)

    run_rows: List[Dict[str, Any]] = []
    if runs_dir.exists():
        for d in sorted([x for x in runs_dir.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
            m = _read_json(d / "metrics.json")
            s = _summary_from_metrics(m)
            run_rows.append(
                {
                    "run_id": d.name,
                    "sharpe": s.get("sharpe"),
                    "calmar": s.get("calmar"),
                    "cagr": s.get("cagr"),
                    "max_drawdown": s.get("max_drawdown"),
                    "win_rate": s.get("win_rate"),
                    "ts": dt.datetime.fromtimestamp(d.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
                }
            )

    ledger_rows = _read_jsonl(artifacts_dir / "ledger" / "ledger.jsonl")
    runs = [r for r in ledger_rows if r.get("kind") == "run"]
    learns = [r for r in ledger_rows if r.get("kind") == "self_learn"]
    accepted = [r for r in learns if (r.get("payload") or {}).get("accepted")]

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts_dir": str(artifacts_dir),
        "latest_run_dir": str(latest_dir) if latest_dir else "",
        "latest_summary": latest_summary,
        "latest_run_name": (latest_dir.name.split(".")[0] if latest_dir else ""),
        "recent_runs": run_rows[:10],
        "ledger": {
            "total_events": len(ledger_rows),
            "run_events": len(runs),
            "self_learn_events": len(learns),
            "accepted_candidates": len(accepted),
            "accept_rate": (len(accepted) / len(learns)) if learns else 0.0,
            "last_run_ts": (runs[-1].get("ts") if runs else None),
        },
    }


def _build_markdown_vi(data: Dict[str, Any]) -> str:
    s = data.get("latest_summary") or {}
    l = data.get("ledger") or {}
    gen = data.get("generated_at_utc") or ""
    run_name = data.get("latest_run_name") or "N/A"
    rows = data.get("recent_runs") or []

    lines: List[str] = []
    lines.append("# NEXUS Demo Pack (Tiếng Việt)")
    lines.append("")
    lines.append(f"- Thời điểm tạo: `{gen}`")
    lines.append(f"- Run gần nhất: `{run_name}`")
    lines.append("")
    lines.append("## 1) Snapshot hiệu quả hiện tại")
    lines.append(f"- Sharpe: **{_fmt_num(s.get('sharpe'))}**")
    lines.append(f"- Calmar: **{_fmt_num(s.get('calmar'))}**")
    lines.append(f"- CAGR: **{_fmt_pct(s.get('cagr'))}**")
    lines.append(f"- Max Drawdown: **{_fmt_pct(s.get('max_drawdown'))}**")
    lines.append(f"- Win Rate: **{_fmt_pct(s.get('win_rate'))}**")
    lines.append(f"- Total Return: **{_fmt_pct(s.get('total_return'))}**")
    lines.append("")
    lines.append("## 2) Bằng chứng hệ thống hoạt động liên tục")
    lines.append(f"- Run events đã ghi nhận: **{l.get('run_events', 0)}**")
    lines.append(f"- Self-learning events: **{l.get('self_learn_events', 0)}**")
    lines.append(f"- Candidate được chấp nhận: **{l.get('accepted_candidates', 0)}**")
    lines.append(f"- Tỷ lệ accepted: **{_fmt_pct(l.get('accept_rate'), 2)}**")
    lines.append("")
    lines.append("## 3) Demo flow 10–15 phút (khuyến nghị)")
    lines.append("1. Mở tab `Overview`: giải thích KPI và equity curve.")
    lines.append("2. Mở tab `Console`: cho thấy process monitor + realtime logs.")
    lines.append("3. Mở tab `Chat`: hỏi NEXUS về chiến lược và risk.")
    lines.append("4. Mở tab `Performance` + `Risk`: chứng minh kết quả và quản trị rủi ro.")
    lines.append("5. Kết thúc bằng tab `Pitch`: tóm tắt giá trị business.")
    lines.append("")
    lines.append("## 4) Talking points (để thuyết trình)")
    lines.append("- NEXUS là vòng lặp tự động: data -> strategy -> backtest -> benchmark -> evidence ledger.")
    lines.append("- Mọi kết quả có dấu vết trong artifacts + ledger, không trình diễn “ảo”.")
    lines.append("- Dashboard realtime cho phép quan sát trực tiếp trạng thái hệ thống.")
    lines.append("- Có self-learning nhưng vẫn có cơ chế kiểm định và guardrails.")
    lines.append("")
    lines.append("## 5) Recent runs (top 10)")
    lines.append("| Run | Sharpe | Calmar | CAGR | MDD | Win Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r.get('run_id','')[:42]}` | {_fmt_num(r.get('sharpe'))} | {_fmt_num(r.get('calmar'))} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | {_fmt_pct(r.get('win_rate'))} |"
        )
    lines.append("")
    lines.append("## 6) Command trước giờ demo")
    lines.append("```bash")
    lines.append("python3 scripts/build_demo_pack.py --artifacts artifacts")
    lines.append("python3 -m nexus_quant dashboard --artifacts artifacts --host 0.0.0.0 --port 8080")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _build_markdown_en(data: Dict[str, Any]) -> str:
    s = data.get("latest_summary") or {}
    l = data.get("ledger") or {}
    gen = data.get("generated_at_utc") or ""
    run_name = data.get("latest_run_name") or "N/A"
    rows = data.get("recent_runs") or []

    lines: List[str] = []
    lines.append("# NEXUS Demo Pack (English)")
    lines.append("")
    lines.append(f"- Generated at: `{gen}`")
    lines.append(f"- Latest run: `{run_name}`")
    lines.append("")
    lines.append("## 1) Current effectiveness snapshot")
    lines.append(f"- Sharpe: **{_fmt_num(s.get('sharpe'))}**")
    lines.append(f"- Calmar: **{_fmt_num(s.get('calmar'))}**")
    lines.append(f"- CAGR: **{_fmt_pct(s.get('cagr'))}**")
    lines.append(f"- Max Drawdown: **{_fmt_pct(s.get('max_drawdown'))}**")
    lines.append(f"- Win Rate: **{_fmt_pct(s.get('win_rate'))}**")
    lines.append(f"- Total Return: **{_fmt_pct(s.get('total_return'))}**")
    lines.append("")
    lines.append("## 2) Evidence of autonomous operation")
    lines.append(f"- Recorded run events: **{l.get('run_events', 0)}**")
    lines.append(f"- Self-learning events: **{l.get('self_learn_events', 0)}**")
    lines.append(f"- Accepted candidates: **{l.get('accepted_candidates', 0)}**")
    lines.append(f"- Acceptance rate: **{_fmt_pct(l.get('accept_rate'), 2)}**")
    lines.append("")
    lines.append("## 3) Recommended 10–15 minute demo flow")
    lines.append("1. `Overview`: explain KPI strip and equity curve.")
    lines.append("2. `Console`: show process monitor + realtime logs.")
    lines.append("3. `Chat`: ask NEXUS strategy/risk questions live.")
    lines.append("4. `Performance` + `Risk`: prove outcomes and risk controls.")
    lines.append("5. End at `Pitch`: summarize business value.")
    lines.append("")
    lines.append("## 4) Core talking points")
    lines.append("- NEXUS is an autonomous loop: data -> strategy -> backtest -> benchmark -> evidence ledger.")
    lines.append("- Every claim is backed by artifacts and ledger records.")
    lines.append("- Realtime dashboard demonstrates actual system state, not slides.")
    lines.append("- Self-learning is constrained by validation and guardrails.")
    lines.append("")
    lines.append("## 5) Recent runs (top 10)")
    lines.append("| Run | Sharpe | Calmar | CAGR | MDD | Win Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r.get('run_id','')[:42]}` | {_fmt_num(r.get('sharpe'))} | {_fmt_num(r.get('calmar'))} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | {_fmt_pct(r.get('win_rate'))} |"
        )
    lines.append("")
    lines.append("## 6) Pre-demo command")
    lines.append("```bash")
    lines.append("python3 scripts/build_demo_pack.py --artifacts artifacts")
    lines.append("python3 -m nexus_quant dashboard --artifacts artifacts --host 0.0.0.0 --port 8080")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_demo_pack(artifacts_dir: Path) -> Tuple[Path, Path, Path]:
    data = _collect(artifacts_dir)
    out_dir = artifacts_dir / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "demo_summary.json"
    md_vi_path = out_dir / "DEMO_SUMMARY.vi.md"
    md_en_path = out_dir / "DEMO_SUMMARY.en.md"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_vi_path.write_text(_build_markdown_vi(data), encoding="utf-8")
    md_en_path.write_text(_build_markdown_en(data), encoding="utf-8")
    return json_path, md_vi_path, md_en_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts)
    json_path, md_vi_path, md_en_path = build_demo_pack(artifacts_dir)
    print(f"Demo pack generated:")
    print(f"- {json_path}")
    print(f"- {md_vi_path}")
    print(f"- {md_en_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
