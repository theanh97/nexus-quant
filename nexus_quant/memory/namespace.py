"""
NEXUS Hierarchical Memory Namespace.

Memory layers:
  L0: Universal    — applies to ALL projects (math, risk mgmt, R&D methodology)
  L1: Market-class — applies to one market type (crypto, fx, options)
  L2: Project      — applies to one specific project (crypto_perps, fx_majors)
  L3: Ephemeral    — current run data (not persisted long-term)

Cross-pollination: insights validated across projects get promoted L2→L1→L0.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.memory.namespace")


class MemoryNamespace:
    """Hierarchical memory store with L0/L1/L2 layers."""

    def __init__(self, memory_root: Path, project_name: str = "", market: str = ""):
        self.memory_root = Path(memory_root)
        self.project_name = project_name
        self.market = market

        # Layer paths
        self.l0_dir = self.memory_root / "L0_universal"
        self.l1_dir = self.memory_root / f"L1_{market}" if market else None
        self.l2_dir = self.memory_root / f"L2_{project_name}" if project_name else None

        # Ensure dirs exist
        self.l0_dir.mkdir(parents=True, exist_ok=True)
        if self.l1_dir:
            self.l1_dir.mkdir(parents=True, exist_ok=True)
        if self.l2_dir:
            self.l2_dir.mkdir(parents=True, exist_ok=True)

    # ── Read (merge all layers) ─────────────────────────────────────────

    def get_wisdom(self) -> Dict[str, Any]:
        """Get merged wisdom from all applicable layers (L0 + L1 + L2)."""
        merged: Dict[str, Any] = {"_layers": [], "insights": [], "lessons": []}

        for layer_name, layer_dir in self._active_layers():
            wisdom = self._read_json(layer_dir / "wisdom.json")
            if wisdom:
                merged["_layers"].append(layer_name)
                merged["insights"].extend(wisdom.get("insights", []))
                merged["lessons"].extend(wisdom.get("lessons", []))
                # Merge other keys (layer-specific)
                for k, v in wisdom.items():
                    if k not in ("insights", "lessons", "_meta"):
                        merged.setdefault(k, v)

        return merged

    def get_lessons(self) -> List[Dict[str, Any]]:
        """Get all lessons from all layers, tagged with source layer."""
        all_lessons: List[Dict[str, Any]] = []
        for layer_name, layer_dir in self._active_layers():
            lessons_path = layer_dir / "lessons.jsonl"
            if lessons_path.exists():
                for line in lessons_path.read_text(encoding="utf-8").splitlines():
                    try:
                        entry = json.loads(line)
                        entry["_layer"] = layer_name
                        all_lessons.append(entry)
                    except Exception:
                        continue
        return all_lessons

    # ── Write (to specific layer) ───────────────────────────────────────

    def add_insight(self, text: str, layer: str = "L2", tags: List[str] = None,
                    source_project: str = "", confidence: float = 0.5) -> None:
        """Add an insight to a specific memory layer."""
        layer_dir = self._get_layer_dir(layer)
        if not layer_dir:
            logger.warning("Cannot write to layer %s — dir not configured", layer)
            return

        wisdom = self._read_json(layer_dir / "wisdom.json") or {
            "insights": [], "lessons": [], "_meta": {}
        }

        insight = {
            "text": text,
            "tags": tags or [],
            "source_project": source_project or self.project_name,
            "confidence": confidence,
            "ts": time.time(),
        }
        wisdom["insights"].append(insight)
        wisdom["_meta"]["last_updated"] = time.time()
        wisdom["_meta"]["count"] = len(wisdom["insights"])

        self._write_json(layer_dir / "wisdom.json", wisdom)
        logger.info("Added insight to %s: %s", layer, text[:80])

    def add_lesson(self, lesson: Dict[str, Any], layer: str = "L2") -> None:
        """Append a lesson to lessons.jsonl in a specific layer."""
        layer_dir = self._get_layer_dir(layer)
        if not layer_dir:
            return

        lesson.setdefault("ts", time.time())
        lesson.setdefault("source_project", self.project_name)

        lessons_path = layer_dir / "lessons.jsonl"
        with open(lessons_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(lesson, sort_keys=True) + "\n")

    # ── Cross-pollination ───────────────────────────────────────────────

    def promote_insight(self, text: str, from_layer: str = "L2", to_layer: str = "L1",
                        reason: str = "") -> None:
        """Promote an insight from a lower layer to a higher one."""
        self.add_insight(
            text=text,
            layer=to_layer,
            tags=["promoted", f"from_{from_layer}"],
            source_project=self.project_name,
            confidence=0.7,
        )
        logger.info("Promoted insight %s→%s: %s (reason: %s)",
                     from_layer, to_layer, text[:60], reason)

    def scan_for_promotions(self) -> List[Dict[str, Any]]:
        """
        Scan L2 insights and identify candidates for promotion to L1/L0.
        Criteria: high confidence + validated across multiple runs.
        Returns list of promotion candidates (caller decides whether to promote).
        """
        candidates: List[Dict[str, Any]] = []
        if not self.l2_dir:
            return candidates

        wisdom = self._read_json(self.l2_dir / "wisdom.json")
        if not wisdom:
            return candidates

        for insight in wisdom.get("insights", []):
            conf = float(insight.get("confidence", 0))
            tags = insight.get("tags", [])
            if conf >= 0.8 and "promoted" not in tags:
                candidates.append({
                    "text": insight["text"],
                    "confidence": conf,
                    "source_project": insight.get("source_project", ""),
                    "suggested_target": "L1" if self.l1_dir else "L0",
                })

        return candidates

    # ── Project-specific storage ────────────────────────────────────────

    def get_project_state(self) -> Dict[str, Any]:
        """Read project-level state (ledger summary, best params, etc.)."""
        if not self.l2_dir:
            return {}
        return self._read_json(self.l2_dir / "state.json") or {}

    def save_project_state(self, state: Dict[str, Any]) -> None:
        """Save project-level state."""
        if not self.l2_dir:
            return
        state["_updated"] = time.time()
        self._write_json(self.l2_dir / "state.json", state)

    # ── Layer info ──────────────────────────────────────────────────────

    def _active_layers(self) -> List[tuple]:
        """Return (name, dir) for all active layers, from L0 (broadest) to L2 (most specific)."""
        layers = [("L0", self.l0_dir)]
        if self.l1_dir and self.l1_dir.exists():
            layers.append(("L1", self.l1_dir))
        if self.l2_dir and self.l2_dir.exists():
            layers.append(("L2", self.l2_dir))
        return layers

    def _get_layer_dir(self, layer: str) -> Optional[Path]:
        if layer == "L0":
            return self.l0_dir
        if layer == "L1":
            return self.l1_dir
        if layer == "L2":
            return self.l2_dir
        return None

    def summary(self) -> Dict[str, Any]:
        """Summary of all memory layers."""
        info: Dict[str, Any] = {"project": self.project_name, "market": self.market, "layers": {}}
        for name, layer_dir in self._active_layers():
            wisdom = self._read_json(layer_dir / "wisdom.json")
            lessons_count = 0
            lp = layer_dir / "lessons.jsonl"
            if lp.exists():
                lessons_count = sum(1 for _ in lp.read_text(encoding="utf-8").splitlines() if _.strip())
            info["layers"][name] = {
                "dir": str(layer_dir),
                "insights": len((wisdom or {}).get("insights", [])),
                "lessons": lessons_count,
            }
        return info

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
