"""
CLI entry point for the CHARITH full stack agent.

Usage:
    python scripts/play_full_stack.py --game ls20 --model gemma3:12b
"""

import argparse
import sys
import time

sys.path.insert(0, "src")

import arc_agi  # noqa: E402

from charith.full_stack.charith_full_stack_agent import CharithFullStackAgent  # noqa: E402
from charith.full_stack.llm_reasoner import LLMReasoner  # noqa: E402
from charith.llm_agent.ollama_client import OllamaClient  # noqa: E402


class _OllamaAdapter:
    """Adapts existing OllamaClient.query() → generate() for LLMReasoner."""

    def __init__(self, client: OllamaClient):
        self._client = client

    def generate(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
        return self._client.query(system_prompt, user_prompt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 with CHARITH full stack agent")
    parser.add_argument("--game", default="ls20", help="ARC game id")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model name")
    parser.add_argument("--num-actions", type=int, default=8, help="Action space size")
    args = parser.parse_args()

    print(f"[charith-full-stack] game={args.game} model={args.model}")

    arcade = arc_agi.Arcade()
    env = arcade.make(args.game)
    env.reset()

    ollama = OllamaClient(model=args.model)
    llm = LLMReasoner(_OllamaAdapter(ollama))

    agent = CharithFullStackAgent(env, llm, num_actions=args.num_actions)

    start = time.time()
    result = agent.play_level()
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Level completed: {result.completed}")
    print(f"  Attempts: {len(result.attempts)}")
    print(f"  Total actions: {result.total_actions}")
    print(f"  Total LLM calls: {result.total_llm_calls}")
    print(f"  Active expansions: {result.final_table_stats['active_expansions']}")
    print(f"  Wall time: {elapsed:.1f}s")
    for i, a in enumerate(result.attempts):
        print(
            f"  Attempt {i+1}: phase_reached={a.phase_reached} "
            f"reason={a.reason} "
            f"confirmed={a.hypotheses_confirmed}/{a.hypotheses_generated}"
        )
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
