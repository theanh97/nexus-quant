"""
NEXUS RAG-lite: Semantic search over memory items using TF-IDF cosine similarity.
Pure stdlib implementation - no external dependencies.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Small, opinionated English stop-word list (stdlib-only project).
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "etc",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "out",
    "over",
    "she",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "up",
    "us",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
    "yours",
}


def _tokenize(text: str) -> List[str]:
    """
    Lowercase, split on non-alphanumeric, filter stop-words.
    """
    out: List[str] = []
    for tok in _TOKEN_SPLIT_RE.split((text or "").lower()):
        if not tok:
            continue
        if tok in _STOP_WORDS:
            continue
        # Avoid excessive noise from 1-char tokens (but keep years, ids, etc.).
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(tok)
    return out


class TFIDFIndex:
    def __init__(self) -> None:
        self._documents: Dict[str, Dict[str, int]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._df: Dict[str, int] = {}
        self._num_docs: int = 0

        self._dirty: bool = True
        self._idf: Dict[str, float] = {}
        self._doc_norms: Dict[str, float] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        doc_id = str(doc_id)
        if doc_id in self._documents:
            self._remove_document(doc_id)

        tokens = _tokenize(text)
        counts: Dict[str, int] = dict(Counter(tokens))
        self._documents[doc_id] = counts
        self._doc_lengths[doc_id] = len(tokens)
        self._num_docs += 1

        for term in counts.keys():
            self._df[term] = int(self._df.get(term, 0)) + 1

        self._dirty = True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q = (query or "").strip()
        try:
            top_k_i = int(top_k)
        except Exception:
            top_k_i = 5
        if top_k_i <= 0 or not q:
            return []
        if self._num_docs <= 0:
            return []

        self._ensure_idf()

        q_tokens = _tokenize(q)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        q_len = len(q_tokens)

        q_weights: Dict[str, float] = {}
        q_sumsq = 0.0
        for term, c in q_counts.items():
            tf = c / q_len
            idf = self._idf.get(term)
            if idf is None:
                idf = self._idf_value(self._df.get(term, 0))
            w = tf * idf
            q_weights[term] = w
            q_sumsq += w * w

        q_norm = math.sqrt(q_sumsq)
        if q_norm <= 0.0:
            return []

        scored: List[Tuple[str, float]] = []
        for doc_id, term_counts in self._documents.items():
            doc_len = self._doc_lengths.get(doc_id, 0)
            if doc_len <= 0:
                continue
            doc_norm = self._doc_norms.get(doc_id, 0.0)
            if doc_norm <= 0.0:
                continue

            dot = 0.0
            for term, q_w in q_weights.items():
                c = term_counts.get(term, 0)
                if c <= 0:
                    continue
                idf = self._idf.get(term)
                if idf is None:
                    idf = self._idf_value(self._df.get(term, 0))
                dot += q_w * ((c / doc_len) * idf)

            if dot <= 0.0:
                continue
            score = dot / (doc_norm * q_norm)
            if score > 0.0:
                scored.append((doc_id, float(score)))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k_i]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: Path) -> None:
        path = Path(path)
        obj = json.loads(path.read_text("utf-8"))

        # Allow wrapper formats (e.g., NexusRAG stores {"tfidf": {...}, "docs": {...}}).
        if isinstance(obj, dict) and "tfidf" in obj and isinstance(obj["tfidf"], dict):
            obj = obj["tfidf"]

        self._load_from_dict(obj)

    def _remove_document(self, doc_id: str) -> None:
        counts = self._documents.pop(doc_id, None)
        if counts is None:
            return
        self._doc_lengths.pop(doc_id, None)
        self._num_docs = max(0, self._num_docs - 1)

        for term in counts.keys():
            new_df = int(self._df.get(term, 0)) - 1
            if new_df <= 0:
                self._df.pop(term, None)
            else:
                self._df[term] = new_df
        self._dirty = True

    def _idf_value(self, df: int) -> float:
        # Requirement: IDF = log(total_docs / docs_containing_term + 1)
        # We use a small smoothing term (+1 in denominator) to avoid division-by-zero.
        return math.log((self._num_docs / (int(df) + 1)) + 1.0)

    def _ensure_idf(self) -> None:
        if not self._dirty:
            return

        self._idf = {term: self._idf_value(df) for term, df in self._df.items()}

        doc_norms: Dict[str, float] = {}
        for doc_id, term_counts in self._documents.items():
            doc_len = self._doc_lengths.get(doc_id, 0)
            if doc_len <= 0:
                doc_norms[doc_id] = 0.0
                continue
            s = 0.0
            for term, c in term_counts.items():
                idf = self._idf.get(term)
                if idf is None:
                    idf = self._idf_value(self._df.get(term, 0))
                w = (c / doc_len) * idf
                s += w * w
            doc_norms[doc_id] = math.sqrt(s)

        self._doc_norms = doc_norms
        self._dirty = False

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "format": "nexus.tfidf",
            "version": 1,
            "num_docs": self._num_docs,
            "df": dict(self._df),
            "documents": {
                doc_id: {"length": int(self._doc_lengths.get(doc_id, 0)), "terms": dict(term_counts)}
                for doc_id, term_counts in self._documents.items()
            },
        }

    def _load_from_dict(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            raise ValueError("Invalid TF-IDF index JSON: expected object")

        documents_obj = obj.get("documents")
        if not isinstance(documents_obj, dict):
            raise ValueError("Invalid TF-IDF index JSON: missing 'documents'")

        documents: Dict[str, Dict[str, int]] = {}
        doc_lengths: Dict[str, int] = {}
        for doc_id, d in documents_obj.items():
            if not isinstance(doc_id, str) or not isinstance(d, dict):
                continue
            terms_obj = d.get("terms") or {}
            if not isinstance(terms_obj, dict):
                terms_obj = {}
            term_counts: Dict[str, int] = {}
            for term, c in terms_obj.items():
                if not isinstance(term, str):
                    continue
                try:
                    ci = int(c)
                except Exception:
                    continue
                if ci > 0:
                    term_counts[term] = ci
            documents[doc_id] = term_counts

            try:
                l = int(d.get("length", 0))
            except Exception:
                l = 0
            if l <= 0:
                l = sum(term_counts.values())
            doc_lengths[doc_id] = l

        df_obj = obj.get("df") or {}
        df: Dict[str, int] = {}
        if isinstance(df_obj, dict):
            for term, c in df_obj.items():
                if not isinstance(term, str):
                    continue
                try:
                    ci = int(c)
                except Exception:
                    continue
                if ci > 0:
                    df[term] = ci

        try:
            num_docs = int(obj.get("num_docs", len(documents)))
        except Exception:
            num_docs = len(documents)

        self._documents = documents
        self._doc_lengths = doc_lengths
        self._df = df
        self._num_docs = max(0, num_docs)
        self._dirty = True
        self._ensure_idf()


class NexusRAG:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.index_path = self.artifacts_dir / "brain" / "rag_index.json"

        self.index = TFIDFIndex()
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._load_if_exists()

    def index_memory_store(self) -> int:
        """
        Index all rows in artifacts/memory/memory.db (memory_items).
        """
        db_path = self.artifacts_dir / "memory" / "memory.db"
        if not db_path.exists():
            return 0

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            try:
                cur.execute(
                    "SELECT id, created_at, kind, tags_json, content, meta_json, run_id FROM memory_items ORDER BY id ASC"
                )
            except Exception:
                return 0

            n = 0
            for r in cur.fetchall():
                try:
                    item_id = int(r["id"])
                except Exception:
                    continue
                content = str(r["content"] or "").strip()
                if not content:
                    continue

                kind = str(r["kind"] or "")
                created_at = str(r["created_at"] or "")
                run_id = str(r["run_id"]) if r["run_id"] is not None else None

                try:
                    tags = json.loads(r["tags_json"] or "[]") or []
                    if not isinstance(tags, list):
                        tags = []
                    tags = [str(t) for t in tags]
                except Exception:
                    tags = []

                try:
                    meta = json.loads(r["meta_json"] or "{}") or {}
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}

                doc_id = f"memory:{item_id}"
                index_text = " ".join(
                    p
                    for p in [
                        kind,
                        " ".join(tags),
                        content,
                        json.dumps(meta, sort_keys=True),
                    ]
                    if p and str(p).strip()
                )
                self.index.add_document(doc_id, index_text)
                self._docs[doc_id] = {
                    "source": "memory",
                    "id": item_id,
                    "created_at": created_at,
                    "kind": kind,
                    "tags": tags,
                    "run_id": run_id,
                    "content": content,
                    "meta": meta,
                }
                n += 1
            return n
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    def index_diary(self) -> int:
        """
        Index all JSON diary entries in artifacts/brain/diary/*.json.
        """
        diary_dir = self.artifacts_dir / "brain" / "diary"
        if not diary_dir.exists():
            return 0

        n = 0
        for p in sorted(diary_dir.glob("*.json")):
            try:
                entry = json.loads(p.read_text("utf-8"))
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue

            date = str(entry.get("date") or "")
            mood = str(entry.get("mood") or "")
            best_strategy = entry.get("best_strategy")
            best_sharpe = entry.get("best_sharpe")
            goals_progress = str(entry.get("goals_progress") or "")
            raw_summary = str(entry.get("raw_summary") or "")

            key_learnings = entry.get("key_learnings") or []
            if not isinstance(key_learnings, list):
                key_learnings = []
            next_plans = entry.get("next_plans") or []
            if not isinstance(next_plans, list):
                next_plans = []
            anomalies = entry.get("anomalies") or []
            if not isinstance(anomalies, list):
                anomalies = []

            parts: List[str] = []
            for v in [
                date,
                mood,
                str(best_strategy or ""),
                str(best_sharpe or ""),
                goals_progress,
                raw_summary,
            ]:
                if v and v.strip():
                    parts.append(v.strip())
            parts.extend(str(x) for x in key_learnings if x is not None and str(x).strip())
            parts.extend(str(x) for x in next_plans if x is not None and str(x).strip())
            parts.extend(str(x) for x in anomalies if x is not None and str(x).strip())

            index_text = " ".join(parts).strip()
            if not index_text:
                continue

            doc_id = f"diary:{p.stem}"
            self.index.add_document(doc_id, index_text)
            self._docs[doc_id] = {
                "source": "diary",
                "file": str(p),
                "date": date,
                "mood": mood,
                "best_strategy": best_strategy,
                "best_sharpe": best_sharpe,
                "key_learnings": key_learnings,
                "next_plans": next_plans,
                "anomalies": anomalies,
                "goals_progress": goals_progress,
                "raw_summary": raw_summary,
                "content": index_text,
            }
            n += 1

        return n

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        out: List[Dict] = []
        for doc_id, score in self.index.search(query, top_k=top_k):
            meta = dict(self._docs.get(doc_id, {}))
            item: Dict[str, Any] = {"doc_id": doc_id, "score": float(score)}
            item.update(meta)
            item.setdefault("content", "")
            out.append(item)
        return out

    def rebuild(self) -> Dict[str, int]:
        self.index = TFIDFIndex()
        self._docs = {}

        mem_n = self.index_memory_store()
        diary_n = self.index_diary()
        self._save()
        return {"memory": int(mem_n), "diary": int(diary_n)}

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "format": "nexus.rag_index",
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tfidf": self.index._to_dict(),
            "docs": self._docs,
        }
        self.index_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

    def _load_if_exists(self) -> None:
        if not self.index_path.exists():
            return
        try:
            obj = json.loads(self.index_path.read_text("utf-8"))
        except Exception:
            return

        if isinstance(obj, dict):
            docs_obj = obj.get("docs")
            if isinstance(docs_obj, dict):
                self._docs = docs_obj

        try:
            self.index.load(self.index_path)
        except Exception:
            # Best-effort: ignore corrupted/unknown index formats.
            self.index = TFIDFIndex()
            return
