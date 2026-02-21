"""
NEXUS Vector Store — Semantic search with local embeddings.

Architecture:
- SQLite storage (portable, no external DB)
- fastembed for neural embeddings (if installed, ~150MB)
- TF-IDF fallback (zero dependencies, always works)
- numpy for cosine similarity (already in project)

Graceful degradation:
  fastembed installed → semantic embeddings (384-dim, understands meaning)
  fastembed missing   → TF-IDF vectors (sparse, keyword-based)
  Either way, the system works on ANY machine.

Usage:
    from nexus_quant.memory.vector_store import NexusVectorStore

    vs = NexusVectorStore(artifacts_dir)
    vs.add("memory:1", "strategy Sharpe improved after adding vol filter", {"kind": "insight"})
    vs.add("memory:2", "API timeout on Deribit options feed", {"kind": "error"})

    results = vs.search("volatility filter performance", top_k=5)
    # => [{"doc_id": "memory:1", "score": 0.85, "text": "...", "meta": {...}}, ...]
"""
from __future__ import annotations

import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Embedding backend detection ─────────────────────────────────────

_FASTEMBED_AVAILABLE = False
_EMBED_MODEL = None
_EMBED_DIM = 0

try:
    from fastembed import TextEmbedding as _FastTextEmbedding
    _FASTEMBED_AVAILABLE = True
    _EMBED_DIM = 384  # bge-small-en-v1.5 default
except ImportError:
    pass


def _get_embedder():
    """Lazy-load the embedding model (only when first needed)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    if _FASTEMBED_AVAILABLE:
        _EMBED_MODEL = _FastTextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL


def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """
    Embed texts using fastembed if available.
    Returns (N, D) numpy array or None if fastembed not available.
    """
    if not _FASTEMBED_AVAILABLE or not texts:
        return None
    model = _get_embedder()
    if model is None:
        return None
    embeddings = list(model.embed(texts))
    return np.array(embeddings, dtype=np.float32)


# ── SQLite blob helpers ─────────────────────────────────────────────

def _vec_to_blob(vec: np.ndarray) -> bytes:
    """Convert float32 numpy vector to bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
    """Convert SQLite BLOB back to numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ── TF-IDF fallback embedder ───────────────────────────────────────

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "no", "not", "of", "on", "or", "our", "out",
    "she", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "too", "up",
    "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.split(text.lower()) if t and len(t) >= 2 and t not in _STOP]


class _TFIDFEmbedder:
    """
    Sparse TF-IDF embedder as fallback when fastembed is not available.
    Produces dense vectors by hashing tokens into a fixed-dimension space.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._idf: Dict[str, float] = {}
        self._num_docs = 0

    def _hash_token(self, token: str) -> int:
        """Stable hash of token to dimension index."""
        h = 5381
        for c in token:
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        return h % self.dim

    def fit_documents(self, texts: List[str]) -> None:
        """Build IDF from a corpus of texts."""
        df: Dict[str, int] = {}
        for text in texts:
            seen = set(_tokenize(text))
            for token in seen:
                df[token] = df.get(token, 0) + 1
        self._num_docs = len(texts)
        n = max(1, self._num_docs)
        self._idf = {t: math.log((n / (c + 1)) + 1.0) for t, c in df.items()}

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a fixed-dim dense vector."""
        tokens = _tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        counts = Counter(tokens)
        total = len(tokens)
        vec = np.zeros(self.dim, dtype=np.float32)

        for token, count in counts.items():
            tf = count / total
            idf = self._idf.get(token, math.log(max(1, self._num_docs) + 1.0))
            idx = self._hash_token(token)
            vec[idx] += tf * idf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


# ── NexusVectorStore ────────────────────────────────────────────────

class NexusVectorStore:
    """
    Unified vector store with automatic backend selection.

    - fastembed available → 384-dim semantic embeddings
    - fastembed missing → 256-dim TF-IDF hashed vectors
    - Both use SQLite + numpy cosine similarity

    The store auto-detects its mode and is fully portable across machines.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.db_path = self.artifacts_dir / "memory" / "vectors.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        # Detect embedding mode
        self.semantic_mode = _FASTEMBED_AVAILABLE
        self.embed_dim = _EMBED_DIM if self.semantic_mode else 256

        # TF-IDF fallback embedder
        self._tfidf: Optional[_TFIDFEmbedder] = None
        if not self.semantic_mode:
            self._tfidf = _TFIDFEmbedder(dim=self.embed_dim)
            self._rebuild_tfidf_idf()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @property
    def mode(self) -> str:
        return "semantic" if self.semantic_mode else "tfidf"

    def add(self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add or update a document with its embedding."""
        if not text or not text.strip():
            return

        # Compute embedding
        embedding = self._embed_one(text)
        if embedding is None:
            return

        meta_json = json.dumps(meta or {}, sort_keys=True)
        blob = _vec_to_blob(embedding)
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

        cur = self._conn.cursor()
        # Upsert
        existing = cur.execute("SELECT id FROM vector_docs WHERE doc_id=?", (doc_id,)).fetchone()
        if existing:
            cur.execute(
                "UPDATE vector_docs SET text=?, meta_json=?, embedding=?, embed_mode=?, updated_at=? WHERE doc_id=?",
                (text[:2000], meta_json, blob, self.mode, now, doc_id),
            )
        else:
            cur.execute(
                "INSERT INTO vector_docs (doc_id, text, meta_json, embedding, embed_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (doc_id, text[:2000], meta_json, blob, self.mode, now, now),
            )
        self._conn.commit()

    def add_batch(self, items: List[Tuple[str, str, Optional[Dict[str, Any]]]]) -> int:
        """
        Batch add documents. More efficient than individual adds for fastembed.

        items: [(doc_id, text, meta), ...]
        Returns: number of documents added.
        """
        if not items:
            return 0

        texts = [text for _, text, _ in items if text and text.strip()]
        if not texts:
            return 0

        # Batch embed
        if self.semantic_mode:
            embeddings = _embed_texts(texts)
            if embeddings is None:
                # Fallback to individual
                for doc_id, text, meta in items:
                    self.add(doc_id, text, meta)
                return len(items)
        else:
            embeddings = self._tfidf.embed_batch(texts) if self._tfidf else None
            if embeddings is None:
                return 0

        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        cur = self._conn.cursor()
        count = 0
        text_idx = 0

        for doc_id, text, meta in items:
            if not text or not text.strip():
                continue
            embedding = embeddings[text_idx]
            text_idx += 1

            meta_json = json.dumps(meta or {}, sort_keys=True)
            blob = _vec_to_blob(embedding)

            existing = cur.execute("SELECT id FROM vector_docs WHERE doc_id=?", (doc_id,)).fetchone()
            if existing:
                cur.execute(
                    "UPDATE vector_docs SET text=?, meta_json=?, embedding=?, embed_mode=?, updated_at=? WHERE doc_id=?",
                    (text[:2000], meta_json, blob, self.mode, now, doc_id),
                )
            else:
                cur.execute(
                    "INSERT INTO vector_docs (doc_id, text, meta_json, embedding, embed_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                    (doc_id, text[:2000], meta_json, blob, self.mode, now, now),
                )
            count += 1

        self._conn.commit()
        return count

    def search(self, query: str, top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Semantic search. Returns top-K most similar documents.

        Returns: [{"doc_id": str, "score": float, "text": str, "meta": dict}, ...]
        """
        if not query or not query.strip():
            return []

        query_vec = self._embed_one(query)
        if query_vec is None:
            return []

        # Load all vectors from DB (for small-medium stores, this is fast enough)
        cur = self._conn.cursor()
        rows = cur.execute("SELECT doc_id, text, meta_json, embedding FROM vector_docs").fetchall()

        if not rows:
            return []

        # Build matrix for vectorized cosine similarity
        doc_ids = []
        texts = []
        metas = []
        vectors = []

        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            vec = _blob_to_vec(blob, self.embed_dim)
            if len(vec) != len(query_vec):
                continue  # Dimension mismatch (mode changed)

            meta = {}
            try:
                meta = json.loads(r["meta_json"] or "{}")
            except Exception:
                pass

            # Apply meta filter if specified
            if filter_meta:
                skip = False
                for k, v in filter_meta.items():
                    if meta.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            doc_ids.append(str(r["doc_id"]))
            texts.append(str(r["text"] or ""))
            metas.append(meta)
            vectors.append(vec)

        if not vectors:
            return []

        # Vectorized cosine similarity with numpy
        matrix = np.array(vectors, dtype=np.float32)
        q = query_vec.astype(np.float32)

        # Normalize
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q_normalized = q / q_norm

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix_normalized = matrix / norms

        # Cosine similarity = dot product of normalized vectors
        scores = matrix_normalized @ q_normalized

        # Get top-K indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append({
                "doc_id": doc_ids[idx],
                "score": round(score, 4),
                "text": texts[idx],
                "meta": metas[idx],
            })

        return results

    def count(self) -> int:
        cur = self._conn.cursor()
        return int(cur.execute("SELECT COUNT(1) FROM vector_docs").fetchone()[0])

    def stats(self) -> Dict[str, Any]:
        cur = self._conn.cursor()
        total = int(cur.execute("SELECT COUNT(1) FROM vector_docs").fetchone()[0])
        by_mode = {}
        for r in cur.execute("SELECT embed_mode, COUNT(1) AS c FROM vector_docs GROUP BY embed_mode").fetchall():
            by_mode[str(r["embed_mode"])] = int(r["c"])
        return {
            "total_docs": total,
            "embed_mode": self.mode,
            "embed_dim": self.embed_dim,
            "by_mode": by_mode,
            "semantic_available": self.semantic_mode,
        }

    def _embed_one(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text using the best available backend."""
        if self.semantic_mode:
            result = _embed_texts([text])
            if result is not None and len(result) > 0:
                return result[0]
            return None
        elif self._tfidf:
            return self._tfidf.embed(text)
        return None

    def _rebuild_tfidf_idf(self) -> None:
        """Rebuild TF-IDF IDF weights from existing documents."""
        if self._tfidf is None:
            return
        cur = self._conn.cursor()
        try:
            rows = cur.execute("SELECT text FROM vector_docs").fetchall()
            texts = [str(r["text"] or "") for r in rows if r["text"]]
            if texts:
                self._tfidf.fit_documents(texts)
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vector_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                embedding BLOB NOT NULL,
                embed_mode TEXT NOT NULL DEFAULT 'tfidf',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_doc_id ON vector_docs(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_mode ON vector_docs(embed_mode)")
        self._conn.commit()


# ── SmartContextBuilder ─────────────────────────────────────────────

class SmartContextBuilder:
    """
    Builds optimal context for LLM queries by retrieving only relevant items.

    Instead of dumping ALL memory into context (wasteful), this:
    1. Takes the current query/task description
    2. Searches the vector store for relevant items
    3. Returns a compact context string within a token budget
    4. Prioritizes by relevance score + recency

    This is how you minimize token costs while maximizing context quality.
    """

    def __init__(self, vector_store: NexusVectorStore, max_tokens: int = 2000) -> None:
        self.vs = vector_store
        self.max_tokens = max_tokens

    def build_context(
        self,
        query: str,
        top_k: int = 10,
        include_categories: Optional[List[str]] = None,
        max_chars: int = 0,
    ) -> Dict[str, Any]:
        """
        Build a context string from the most relevant items.

        Returns:
            {
                "context": str,      # The context string (ready for LLM)
                "items_used": int,   # How many items were included
                "total_chars": int,  # Total characters
                "est_tokens": int,   # Estimated tokens (~4 chars/token)
                "relevance_scores": [float, ...],  # Scores of included items
            }
        """
        if max_chars <= 0:
            max_chars = self.max_tokens * 4  # ~4 chars per token

        # Search for relevant items
        filter_meta = None
        if include_categories:
            # We'll filter manually since SQLite filter is single-key
            pass

        results = self.vs.search(query, top_k=top_k)

        # Filter by category if specified
        if include_categories:
            results = [r for r in results if r.get("meta", {}).get("kind") in include_categories]

        # Build context string within budget
        parts = []
        total_chars = 0
        scores = []

        for r in results:
            text = r.get("text", "").strip()
            if not text:
                continue

            meta = r.get("meta", {})
            kind = meta.get("kind", "")
            score = r.get("score", 0)

            # Format: [kind] (score) text
            entry = f"[{kind}] {text}" if kind else text

            # Check budget
            if total_chars + len(entry) + 2 > max_chars:
                # Truncate last entry to fit
                remaining = max_chars - total_chars - 2
                if remaining > 50:
                    parts.append(entry[:remaining] + "...")
                    scores.append(score)
                break

            parts.append(entry)
            total_chars += len(entry) + 1  # +1 for newline
            scores.append(score)

        context = "\n".join(parts)
        est_tokens = len(context) // 4  # rough estimate

        return {
            "context": context,
            "items_used": len(parts),
            "total_chars": len(context),
            "est_tokens": est_tokens,
            "relevance_scores": scores,
        }

    def build_agent_context(
        self,
        task_description: str,
        wisdom_query: str = "",
        max_wisdom_tokens: int = 800,
        max_memory_tokens: int = 600,
        max_lesson_tokens: int = 400,
    ) -> Dict[str, Any]:
        """
        Build a complete agent context with three sections:
        1. Relevant wisdom (strategies, metrics, insights)
        2. Relevant memory (past decisions, feedback)
        3. Relevant operational lessons

        This replaces dumping ALL wisdom + ALL memory into context.
        Total token budget = max_wisdom + max_memory + max_lesson tokens.
        """
        combined_query = f"{task_description} {wisdom_query}".strip()

        # Section 1: Wisdom context
        wisdom_ctx = self.build_context(
            query=combined_query,
            top_k=8,
            include_categories=["wisdom", "insight", "run", "self_learn", "critique", "reflection"],
            max_chars=max_wisdom_tokens * 4,
        )

        # Section 2: Memory context
        memory_ctx = self.build_context(
            query=combined_query,
            top_k=6,
            include_categories=["feedback", "agent_output", "handoff", "policy_review"],
            max_chars=max_memory_tokens * 4,
        )

        # Section 3: Lessons context
        lesson_ctx = self.build_context(
            query=combined_query,
            top_k=5,
            include_categories=["lesson", "preflight_warning", "rules_reminder"],
            max_chars=max_lesson_tokens * 4,
        )

        total_tokens = wisdom_ctx["est_tokens"] + memory_ctx["est_tokens"] + lesson_ctx["est_tokens"]

        return {
            "wisdom": wisdom_ctx,
            "memory": memory_ctx,
            "lessons": lesson_ctx,
            "total_est_tokens": total_tokens,
            "query": combined_query[:200],
        }
