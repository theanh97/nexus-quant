#!/usr/bin/env python3
"""
Quick test: verify all model integrations work.
Tests Gemini, GLM-5, Claude, Codex connectivity.
"""
import os, sys, time

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

from nexus_quant.brain.reasoning import _call_llm


def test_model(name, model_id):
    """Test a single model and report result."""
    print(f"\n{'='*50}")
    print(f"Testing: {name} (model_id={model_id})")
    print(f"{'='*50}")
    t0 = time.time()
    try:
        result = _call_llm(
            system_prompt="You are a quant trading expert. Be concise.",
            user_message="In one sentence, what is the Sharpe ratio?",
            max_tokens=100,
            model=model_id,
        )
        elapsed = time.time() - t0
        ok = not result.startswith("[") or "error" not in result.lower()
        status = "OK" if ok else "WARN"
        print(f"  Status: {status} ({elapsed:.1f}s)")
        print(f"  Response: {result[:200]}")
        return ok
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Status: FAIL ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        return False


def main():
    print("NEXUS Model Connectivity Test")
    print("=" * 60)
    print(f"GEMINI_API_KEY: {'SET' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}")
    print(f"ZAI_API_KEY: {'SET' if os.environ.get('ZAI_API_KEY') else 'NOT SET'}")
    print(f"MINIMAX_API_KEY: {'SET' if os.environ.get('MINIMAX_API_KEY') else 'NOT SET'}")

    models = [
        ("Gemini 2.5 Flash (FREE)", "gemini-2.5-flash"),
        ("Gemini 2.5 Pro (FREE)", "gemini-2.5-pro"),
        ("GLM-5 (ZAI)", "glm-5"),
        ("Claude Sonnet 4.6 (ZAI)", "claude-sonnet-4-6"),
        ("Codex / GPT-5.2", "codex"),
    ]

    results = {}
    for name, model_id in models:
        results[name] = test_model(name, model_id)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon}] {name}")

    ok_count = sum(1 for v in results.values() if v)
    print(f"\n  {ok_count}/{len(results)} models operational")


if __name__ == "__main__":
    main()
