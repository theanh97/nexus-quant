"""
NEXUS Multi-Model Debate Engine
Calls Claude, GPT-5.2 (Codex), GLM-5, MiniMax on the same question,
then has each model critique the others, then synthesizes a final verdict.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DebateContribution:
    model_name: str
    role: str          # "analysis" | "critique" | "synthesis"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tokens_used: int = 0


@dataclass
class DebateRound:
    topic: str
    round_id: str
    contributions: List[DebateContribution]
    synthesis: str
    consensus_score: float        # 0-1
    action_items: List[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DebateEngine:
    DEFAULT_MODELS = ["glm-5", "claude-sonnet-4-6", "codex", "minimax-2.5"]

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.debate_dir = self.artifacts_dir / "debate"
        self.debate_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.debate_dir / "history.jsonl"

    # ------------------------------------------------------------------
    # Low-level model caller
    # ------------------------------------------------------------------

    def _call_model(self, model: str, system: str, prompt: str) -> tuple:
        """
        Returns (response_text, tokens_used).
        Never raises — on error returns descriptive string and 0 tokens.
        """
        try:
            # ---- Anthropic / ZAI family --------------------------------
            if model in ("glm-5", "claude-sonnet-4-6", "claude-opus-4-6"):
                import anthropic

                api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
                base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL", "https://api.anthropic.com")
                base_url = base_url.rstrip("/")

                model_id = {
                    "glm-5": "glm-4",
                    "claude-sonnet-4-6": "claude-sonnet-4-6",
                    "claude-opus-4-6": "claude-opus-4-6",
                }.get(model, "glm-4")

                client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
                msg = client.messages.create(
                    model=model_id,
                    max_tokens=800,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text, msg.usage.output_tokens

            # ---- Codex / GPT-5.2 family --------------------------------
            elif model in ("codex", "gpt-5.2"):
                import subprocess

                codex_bin = "/Applications/Codex.app/Contents/Resources/codex"
                combined = f"SYSTEM: {system}\n\nUSER: {prompt}"
                result = subprocess.run(
                    [
                        codex_bin,
                        "exec",
                        "-c",
                        'sandbox_permissions=["disk-full-read-access","disk-write-access"]',
                        "-",
                    ],
                    input=combined.encode(),
                    capture_output=True,
                    timeout=45,
                )
                text = result.stdout.decode(errors="replace").strip() or "(no response)"
                return text, len(text.split())

            # ---- MiniMax family ----------------------------------------
            elif model in ("minimax-2.5", "minimax"):
                try:
                    from minimax import Minimax

                    client = Minimax(
                        api_key=os.environ.get("MINIMAX_API_KEY", ""),
                        group_id=os.environ.get("MINIMAX_GROUP_ID", ""),
                    )
                    resp = client.chat.completions.create(
                        model="abab6.5s-chat",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    text = resp.choices[0].message.content
                    return text, len(text.split())
                except Exception as e:
                    return f"(MiniMax unavailable: {e})", 0

            # ---- Unknown model -----------------------------------------
            else:
                return "(model unavailable)", 0

        except Exception as e:
            return f"(failed: {e})", 0

    # ------------------------------------------------------------------
    # Main debate orchestration
    # ------------------------------------------------------------------

    def run_debate(self, topic: str, context: str = "", models: list = None) -> DebateRound:
        if models is None:
            models = list(self.DEFAULT_MODELS)

        round_id = str(uuid.uuid4())[:8]
        contributions: List[DebateContribution] = []

        # ==============================================================
        # ROUND 1 — Independent analysis
        # ==============================================================
        round1_responses: dict = {}  # model -> text

        r1_system = (
            "You are a quantitative trading expert. Be concise (max 200 words). "
            "Give your honest independent analysis."
        )

        for model in models:
            prompt = (
                f"Topic: {topic}\n\n"
                f"Context: {context}\n\n"
                "Give your analysis. Be specific, critical, and honest."
            )
            try:
                text, tokens = self._call_model(model, r1_system, prompt)
            except Exception as e:
                text, tokens = f"(failed: {e})", 0

            round1_responses[model] = text
            contributions.append(
                DebateContribution(
                    model_name=model,
                    role="analysis",
                    content=text,
                    tokens_used=tokens,
                )
            )

        # ==============================================================
        # ROUND 2 — Cross-critique
        # ==============================================================
        r2_system = (
            "You are a critical reviewer. Identify agreements and disagreements. Max 150 words."
        )

        for model in models:
            your_view = round1_responses.get(model, "")

            # Summarise what the OTHER models said
            other_lines = []
            for other_model, other_text in round1_responses.items():
                if other_model == model:
                    continue
                snippet = other_text[:100].replace("\n", " ")
                other_lines.append(f"  {other_model}: {snippet}...")
            other_views = "\n".join(other_lines) if other_lines else "(no other views)"

            prompt = (
                f"Topic: {topic}\n\n"
                f"Other models said:\n{other_views}\n\n"
                f"Your round 1 view: {your_view}\n\n"
                "Critique: What did others get right/wrong? "
                "What are you most confident about?"
            )
            try:
                text, tokens = self._call_model(model, r2_system, prompt)
            except Exception as e:
                text, tokens = f"(failed: {e})", 0

            contributions.append(
                DebateContribution(
                    model_name=model,
                    role="critique",
                    content=text,
                    tokens_used=tokens,
                )
            )

        # ==============================================================
        # ROUND 3 — Synthesis (by glm-5 acting as meta-analyst)
        # ==============================================================
        # Build full transcript
        transcript_lines = [f"=== DEBATE TOPIC ===\n{topic}\n"]
        if context:
            transcript_lines.append(f"=== CONTEXT ===\n{context}\n")

        transcript_lines.append("=== ROUND 1 — INDEPENDENT ANALYSIS ===")
        for c in contributions:
            if c.role == "analysis":
                transcript_lines.append(f"\n[{c.model_name}]\n{c.content}")

        transcript_lines.append("\n=== ROUND 2 — CROSS-CRITIQUE ===")
        for c in contributions:
            if c.role == "critique":
                transcript_lines.append(f"\n[{c.model_name}]\n{c.content}")

        full_transcript = "\n".join(transcript_lines)

        synthesis_system = (
            "You are a meta-analyst synthesizing multiple expert opinions. "
            "Be decisive. Max 250 words."
        )
        synthesis_prompt = (
            f"{full_transcript}\n\n"
            "=== SYNTHESIS REQUEST ===\n"
            "Please provide:\n"
            "1. CONSENSUS POINTS: What all models agreed on.\n"
            "2. KEY DISAGREEMENTS: Where models diverged and why.\n"
            "3. RECOMMENDED ACTION: The single best course of action.\n"
            "4. ACTION: (list specific actionable steps, one per line starting with 'ACTION:')\n"
            "5. CONFIDENCE: A score between 0.0 and 1.0 reflecting overall consensus strength. "
            "Format: 'confidence: 0.X'\n"
        )

        try:
            synthesis_text, synthesis_tokens = self._call_model(
                "glm-5", synthesis_system, synthesis_prompt
            )
        except Exception as e:
            synthesis_text = f"(synthesis failed: {e})"
            synthesis_tokens = 0

        contributions.append(
            DebateContribution(
                model_name="glm-5",
                role="synthesis",
                content=synthesis_text,
                tokens_used=synthesis_tokens,
            )
        )

        # ------------------------------------------------------------------
        # Parse action items
        # ------------------------------------------------------------------
        action_items: List[str] = []
        for line in synthesis_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("ACTION:"):
                item = stripped[len("ACTION:"):].strip()
                if item:
                    action_items.append(item)
            else:
                # Also capture numbered list items (e.g. "1. Do something")
                m = re.match(r"^\d+\.\s+(.+)$", stripped)
                if m:
                    candidate = m.group(1).strip()
                    # Only include if it looks like an action (not just a label)
                    if len(candidate) > 10 and not candidate.lower().startswith("consensus"):
                        action_items.append(candidate)

        # ------------------------------------------------------------------
        # Parse consensus score
        # ------------------------------------------------------------------
        consensus_score = 0.5
        score_match = re.search(
            r"confidence[:\s]+([0-9]\.[0-9]+|[01])", synthesis_text, re.IGNORECASE
        )
        if score_match:
            try:
                consensus_score = float(score_match.group(1))
                consensus_score = max(0.0, min(1.0, consensus_score))
            except ValueError:
                pass

        # ------------------------------------------------------------------
        # Build and persist DebateRound
        # ------------------------------------------------------------------
        debate_round = DebateRound(
            topic=topic,
            round_id=round_id,
            contributions=contributions,
            synthesis=synthesis_text,
            consensus_score=consensus_score,
            action_items=action_items,
            metadata={
                "models_used": models,
                "context_length": len(context),
                "total_contributions": len(contributions),
            },
        )

        self._save_round(debate_round)
        return debate_round

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_round(self, debate_round: DebateRound) -> None:
        record = {
            "round_id": debate_round.round_id,
            "topic": debate_round.topic,
            "contributions": [asdict(c) for c in debate_round.contributions],
            "synthesis": debate_round.synthesis,
            "consensus_score": debate_round.consensus_score,
            "action_items": debate_round.action_items,
            "created_at": debate_round.created_at,
            "metadata": debate_round.metadata,
        }
        try:
            with open(self.history_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Don't let persistence failure break the caller

    def get_history(self, n: int = 20) -> list:
        """Return last n debate rounds as a list of dicts."""
        if not self.history_file.exists():
            return []
        try:
            lines = self.history_file.read_text(encoding="utf-8").splitlines()
            tail = lines[-n:] if len(lines) > n else lines
            records = []
            for line in tail:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return records
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    def quick_debate(self, topic: str, context: str = "") -> dict:
        """Run a full debate and return a simplified summary dict."""
        debate_round = self.run_debate(topic=topic, context=context)

        simple_contributions = []
        for c in debate_round.contributions:
            simple_contributions.append(
                {
                    "model": c.model_name,
                    "round": c.role,
                    "content": c.content,
                }
            )

        return {
            "topic": debate_round.topic,
            "round_id": debate_round.round_id,
            "contributions": simple_contributions,
            "synthesis": debate_round.synthesis,
            "action_items": debate_round.action_items,
            "consensus_score": debate_round.consensus_score,
            "created_at": debate_round.created_at,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[DebateEngine] = None


def get_engine(artifacts_dir: str = "artifacts") -> DebateEngine:
    global _engine
    if _engine is None:
        _engine = DebateEngine(artifacts_dir=artifacts_dir)
    return _engine
