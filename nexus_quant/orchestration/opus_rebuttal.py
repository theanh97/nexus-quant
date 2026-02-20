from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class OpusRebuttalConfig:
    enabled: bool = True
    model: str = "opus"
    effort: str = "medium"
    budget_usd: float = 0.6
    timeout_seconds: int = 180


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    # Accept raw JSON output.
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Accept markdown fenced JSON block.
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def run_opus_rebuttal(
    *,
    topic: str,
    context: Dict[str, Any],
    cfg: OpusRebuttalConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or OpusRebuttalConfig()
    if not cfg.enabled:
        return {"ok": False, "reason": "disabled"}

    if str(os.environ.get("NX_ENABLE_OPUS_REBUTTAL", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return {"ok": False, "reason": "disabled_by_env"}

    prompt = (
        "You are a hostile but practical reviewer for an autonomous quant research system.\n"
        "Challenge assumptions, highlight hidden failure modes, and propose concrete fixes.\n\n"
        f"Topic: {topic}\n"
        "Context (JSON):\n"
        f"{json.dumps(context, ensure_ascii=True, sort_keys=True)}\n\n"
        "Return STRICT JSON only with keys:\n"
        "{\n"
        '  "verdict":"KEEP"|"REVISE"|"ROLLBACK",\n'
        '  "critical_issues":[{"issue":str,"severity_0_10":number,"fix":str}],\n'
        '  "value_checks":[str],\n'
        '  "next_actions":[str]\n'
        "}\n"
    )

    cmd = [
        "claude",
        "-p",
        "--model",
        cfg.model,
        "--effort",
        cfg.effort,
        "--max-budget-usd",
        str(cfg.budget_usd),
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(cfg.timeout_seconds),
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}

    raw_out = (proc.stdout or "").strip()
    raw_err = (proc.stderr or "").strip()
    parsed = _extract_json(raw_out)
    parsed_ok = isinstance(parsed, dict) and bool(str(parsed.get("verdict") or "").strip())
    return {
        "ok": proc.returncode == 0 and bool(parsed_ok),
        "returncode": int(proc.returncode),
        "parsed": parsed,
        "parsed_ok": bool(parsed_ok),
        "raw_output": raw_out,
        "raw_error": raw_err,
        "model": str(cfg.model),
        "effort": str(cfg.effort),
        "budget_usd": float(cfg.budget_usd),
        "command": cmd,
    }


def persist_opus_rebuttal(
    *,
    out_json_path: Path,
    out_text_path: Path,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_text_path.write_text(str(payload.get("raw_output") or ""), encoding="utf-8")
    return {"json_path": str(out_json_path), "text_path": str(out_text_path)}
