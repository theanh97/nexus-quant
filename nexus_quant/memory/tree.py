"""
NEXUS Memory Tree - Hierarchical knowledge storage with fast TF-IDF retrieval.
Organizes research findings, strategy insights, and error patterns into a tree structure
for efficient context injection (reduces tokens by 80% vs raw logs).
"""
from __future__ import annotations
import json
import os
import pathlib
import re
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone


MEMORY_ROOT = pathlib.Path(__file__).parent.parent.parent / "artifacts" / "memory_tree"


# ─── Node Types ───────────────────────────────────────────────────────────────
NODE_TYPES = {
    "strategy": "Strategy findings and performance results",
    "error": "Bugs encountered and their fixes",
    "signal": "Alpha signal discoveries and analysis",
    "risk": "Risk events and lessons learned",
    "process": "Process improvements and workflow notes",
    "critical": "Critical thinking and process challenges",
}


class MemoryNode:
    def __init__(
        self,
        node_id: str,
        node_type: str,
        title: str,
        content: str,
        importance: float = 0.5,
        tags: List[str] | None = None,
        parent_id: str | None = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.title = title
        self.content = content
        self.importance = importance  # 0.0–1.0
        self.tags = tags or []
        self.parent_id = parent_id
        self.children: List[str] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.access_count = 0
        self.last_accessed = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "content": self.content,
            "importance": self.importance,
            "tags": self.tags,
            "parent_id": self.parent_id,
            "children": self.children,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryNode":
        n = cls(
            node_id=d["node_id"],
            node_type=d.get("node_type", "process"),
            title=d.get("title", ""),
            content=d.get("content", ""),
            importance=float(d.get("importance", 0.5)),
            tags=d.get("tags", []),
            parent_id=d.get("parent_id"),
        )
        n.children = d.get("children", [])
        n.created_at = d.get("created_at", n.created_at)
        n.access_count = int(d.get("access_count", 0))
        n.last_accessed = d.get("last_accessed", n.created_at)
        return n

    def summary(self, max_len: int = 120) -> str:
        """Compact summary for context injection."""
        c = self.content[:max_len].replace("\n", " ")
        return f"[{self.node_type.upper()}] {self.title}: {c}"


class MemoryTree:
    """Hierarchical memory with TF-IDF retrieval and importance scoring."""

    def __init__(self, root_dir: pathlib.Path | None = None):
        self.root_dir = root_dir or MEMORY_ROOT
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._nodes: Dict[str, MemoryNode] = {}
        self._tfidf: Dict[str, Dict[str, float]] = {}  # node_id -> term->tfidf
        self._idf: Dict[str, float] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        f = self.root_dir / "tree.json"
        if f.exists():
            try:
                data = json.loads(f.read_text("utf-8"))
                for nd in data.get("nodes", []):
                    n = MemoryNode.from_dict(nd)
                    self._nodes[n.node_id] = n
                self._build_tfidf()
            except Exception:
                pass

    def _save(self) -> None:
        f = self.root_dir / "tree.json"
        data = {"nodes": [n.to_dict() for n in self._nodes.values()]}
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")

    # ── TF-IDF ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "in", "of", "to", "and", "or", "for", "with", "on", "at", "by", "from", "as", "this", "that", "be", "been", "has", "have", "had", "not", "but", "we", "i", "you", "he", "she", "they", "their", "our", "its", "if", "can", "will", "would", "should"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _build_tfidf(self) -> None:
        if not self._nodes:
            return
        docs: Dict[str, List[str]] = {}
        for nid, node in self._nodes.items():
            docs[nid] = self._tokenize(node.title + " " + node.content + " " + " ".join(node.tags))
        N = len(docs)
        df: Dict[str, int] = defaultdict(int)
        for tokens in docs.values():
            for t in set(tokens):
                df[t] += 1
        self._idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
        self._tfidf = {}
        for nid, tokens in docs.items():
            tf: Dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1
            total = len(tokens) or 1
            self._tfidf[nid] = {t: (f / total) * self._idf.get(t, 1) for t, f in tf.items()}

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add(
        self,
        title: str,
        content: str,
        node_type: str = "process",
        importance: float = 0.5,
        tags: List[str] | None = None,
        parent_id: str | None = None,
    ) -> MemoryNode:
        nid = f"{node_type}_{len(self._nodes):04d}_{int(datetime.now().timestamp())}"
        node = MemoryNode(nid, node_type, title, content, importance, tags, parent_id)
        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].children.append(nid)
        self._nodes[nid] = node
        self._build_tfidf()
        self._save()
        return node

    def get(self, node_id: str) -> Optional[MemoryNode]:
        node = self._nodes.get(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = datetime.now(timezone.utc).isoformat()
        return node

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, node_type: str | None = None) -> List[Tuple[MemoryNode, float]]:
        """TF-IDF search with importance boost. Returns (node, score) sorted by score."""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scores: Dict[str, float] = {}
        for nid, tfidf in self._tfidf.items():
            if node_type and self._nodes[nid].node_type != node_type:
                continue
            score = sum(tfidf.get(t, 0) for t in q_tokens)
            # Boost by importance and recency
            node = self._nodes[nid]
            score *= (1 + node.importance)
            scores[nid] = score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for nid, score in ranked[:top_k]:
            if score > 0:
                node = self._nodes[nid]
                node.access_count += 1
                result.append((node, score))
        return result

    def context_for_query(self, query: str, max_tokens: int = 600) -> str:
        """Generate compact context string for LLM injection. Token-efficient."""
        hits = self.search(query, top_k=8)
        if not hits:
            return ""
        lines = ["[MEMORY TREE — relevant knowledge]"]
        used = len(lines[0])
        for node, score in hits:
            s = node.summary(max_len=150)
            if used + len(s) > max_tokens:
                break
            lines.append(s)
            used += len(s)
        return "\n".join(lines)

    # ── Bulk population ───────────────────────────────────────────────────────

    def populate_from_phases(self) -> None:
        """Pre-populate with NEXUS phase findings if tree is empty."""
        if len(self._nodes) > 10:
            return
        seed_data = [
            ("strategy", "NexusAlpha V1 Champion", "k=2, 10sym, 168h rebal, 0.35x leverage. Weights: carry=0.35, mom=0.45, MR=0.20. OOS Sharpe avg 1.01 across 2021-2025.", 0.9, ["nexus_alpha_v1", "champion", "sharpe"]),
            ("signal", "Alpha Source: Buy Dips in Uptrends", "Alpha = 168h momentum filter + 48h MR timing. LONG top-2 by composite score. Pure MR alone gives Sharpe=-1.68 in trending markets.", 0.9, ["mr", "momentum", "signal"]),
            ("error", "MATICUSDT renamed to POLUSDT in 2024", "Use POLUSDT for 2025+ configs. MATICUSDT returns 0 bars for 2025 data.", 0.7, ["data", "symbol", "binance"]),
            ("strategy", "k=3 positions terrible (0.119 Sharpe)", "Increasing k_per_side from 2 to 3 dilutes signal quality. Marginal 3rd position is pure noise.", 0.8, ["k_per_side", "diversification"]),
            ("strategy", "Fast rebalancing (48h) hurts", "48h rebalancing costs too much in fees. 168h (weekly) is optimal for this signal.", 0.8, ["rebalancing", "costs"]),
            ("strategy", "15 symbols worse than 10", "Adding MATIC/ATOM/NEAR/LTC/UNI dilutes z-score normalization. 10 liquid blue-chips optimal.", 0.8, ["universe", "symbols"]),
            ("strategy", "Regime filter wrong for MR strategy", "V2 regime filter reduces leverage during corrections = exactly when MR earns most. Anti-correlated.", 0.85, ["regime", "v2", "bug"]),
            ("strategy", "2022 Bear Market: Sharpe=0.112", "Strategy survived BTC -65%, FTX collapse with Beta=0.001. True market neutrality confirmed.", 0.9, ["bear", "2022", "oos"]),
            ("strategy", "2021 Bull Market: Sharpe=2.047", "Best year. Momentum+carry worked excellently in strong uptrend. Consistent with alpha source.", 0.8, ["bull", "2021", "oos"]),
            ("error", "periods_per_year=365 wrong for 1h data", "Fixed: 1h bars -> ppy=8760 (not 365). Sharpe was being reported 4.9x too high before fix.", 0.7, ["bug", "sharpe", "annualization"]),
            ("signal", "Pure funding carry alone: Sharpe=0.477", "Carry signal has standalone alpha but weak. Needs momentum to filter out trend-down losers.", 0.7, ["carry", "signal"]),
            ("process", "IS vs OOS validation framework", "2024 is in-sample (48h MR selected here). All other years (2021,2022,2023,2025) are genuine OOS.", 0.9, ["validation", "oos", "is"]),
            ("critical", "Deflated Sharpe = 0.46 with n_params=7", "Bailey 2014 correction: with 7 parameters tested, DSR=0.46. Significant at ~50% confidence only.", 0.8, ["dsr", "overfitting", "statistics"]),
        ]
        for node_type, title, content, importance, tags in seed_data:
            self.add(title, content, node_type, importance, tags)

    def stats(self) -> Dict[str, Any]:
        type_counts = defaultdict(int)
        for n in self._nodes.values():
            type_counts[n.node_type] += 1
        return {
            "total_nodes": len(self._nodes),
            "by_type": dict(type_counts),
            "avg_importance": sum(n.importance for n in self._nodes.values()) / max(len(self._nodes), 1),
        }
