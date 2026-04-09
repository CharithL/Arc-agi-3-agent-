"""
CLI entry point for the CHARITH full stack agent.

Usage:
    python scripts/play_full_stack.py --game ls20 --model gemma3:12b
"""

import argparse
import io
import sys
import time

import numpy as np

# Force stdout/stderr to UTF-8 so verbose output containing arrows, emoji, etc.
# doesn't crash on Windows cp1252 consoles.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

import arc_agi  # noqa: E402
from arcengine import GameAction  # noqa: E402

from charith.full_stack.charith_full_stack_agent import CharithFullStackAgent  # noqa: E402
from charith.full_stack.llm_reasoner import LLMReasoner  # noqa: E402
from charith.llm_agent.ollama_client import OllamaClient  # noqa: E402


class _OllamaAdapter:
    """Adapts existing OllamaClient.query() -> generate() for LLMReasoner."""

    def __init__(self, client: OllamaClient, verbose: bool = False):
        self._client = client
        self._verbose = verbose
        self._call_count = 0

    def generate(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
        self._call_count += 1
        if self._verbose:
            print(f"\n[VERBOSE LLM #{self._call_count}] available={self._client.available} model={self._client.model}")
            print(f"[VERBOSE LLM #{self._call_count}] system_prompt ({len(system_prompt)} chars):")
            print(system_prompt[:500] + ("..." if len(system_prompt) > 500 else ""))
            print(f"[VERBOSE LLM #{self._call_count}] user_prompt ({len(user_prompt)} chars):")
            print(user_prompt[:800] + ("..." if len(user_prompt) > 800 else ""))
        response = self._client.query(system_prompt, user_prompt)
        if self._verbose:
            retry_mode = getattr(self._client, "_last_retry_mode", "?")
            print(f"[VERBOSE LLM #{self._call_count}] retry_mode={retry_mode} "
                  f"raw response ({len(response) if response else 0} chars):")
            print(repr(response)[:1500])
        return response


_ACTION_MAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def _frame_grid(frame):
    if frame is None or not hasattr(frame, "frame") or not frame.frame:
        return None
    g = frame.frame[0]
    return np.asarray(g) if g is not None else None


class _RealEnvAdapter:
    """
    Wraps arc_agi LocalEnvironmentWrapper to expose the mock-env interface
    expected by Explorer/Verifier/Executor.
    In verbose mode, prints pixel-diff magnitude per step.
    """

    def __init__(self, env, verbose: bool = False):
        self._env = env
        self._latest = None
        self._levels_completed = 0
        self._verbose = verbose
        self._step_count = 0

    def reset(self):
        self._latest = self._env.reset()
        if self._latest is not None and hasattr(self._latest, "levels_completed"):
            self._levels_completed = self._latest.levels_completed
        if self._verbose:
            g = _frame_grid(self._latest)
            shape = g.shape if g is not None else None
            print(f"[VERBOSE env] reset -> grid shape={shape}")
        return self._latest

    def get_observation(self):
        return self._latest

    def step(self, action_id: int):
        self._step_count += 1
        before_grid = _frame_grid(self._latest)

        game_action = _ACTION_MAP.get(action_id, GameAction.ACTION1)
        frame = self._env.step(game_action)
        self._latest = frame

        if self._verbose:
            after_grid = _frame_grid(frame)
            if before_grid is not None and after_grid is not None \
                    and before_grid.shape == after_grid.shape:
                diff = int(np.sum(np.abs(before_grid.astype(int) - after_grid.astype(int))))
                changed_cells = int(np.sum(before_grid != after_grid))
                print(f"[VERBOSE env] step#{self._step_count} action={action_id} "
                      f"pixel_diff_abs_sum={diff} changed_cells={changed_cells}")
            else:
                print(f"[VERBOSE env] step#{self._step_count} action={action_id} "
                      f"(shape mismatch or None) before={before_grid is not None} "
                      f"after={after_grid is not None}")

        reward = 0.0
        done = False
        if frame is not None:
            if hasattr(frame, "levels_completed"):
                new_levels = frame.levels_completed
                if new_levels > self._levels_completed:
                    reward = 1.0
                    self._levels_completed = new_levels
                    # Level advance IS the success signal for M1 — don't wait
                    # for full-game terminal state.
                    done = True
                    if self._verbose:
                        print(f"[VERBOSE env] LEVEL COMPLETED (levels_completed -> {new_levels})")
            if hasattr(frame, "state"):
                state_val = str(frame.state)
                if ("FINISHED" in state_val or "WON" in state_val or "LOST" in state_val) \
                        and "NOT_FINISHED" not in state_val:
                    done = True

        return frame, reward, done, {}


def _install_verbose_hooks():
    """
    Monkey-patch Explorer/Hypothesizer/Verifier to log at key points.
    Keeps production code untouched; diagnostic lives only in this script.
    """
    from charith.alfa_loop import explorer as _exp
    from charith.alfa_loop import hypothesizer as _hyp
    from charith.alfa_loop import verifier as _ver

    # Phase 1: wrap Explorer.explore to print changes per action
    _orig_explore = _exp.Explorer.explore

    def _v_explore(self, num_actions=8):
        evidence = _orig_explore(self, num_actions=num_actions)
        print(f"\n[VERBOSE Phase 1] explored {len(evidence)} actions:")
        for e in evidence:
            print(f"  action={e.action} reward={e.reward} done={e.done} "
                  f"changes={e.changes!r} desc={e.description!r}")
        return evidence

    _exp.Explorer.explore = _v_explore

    # Phase 2: wrap Hypothesizer.generate to print raw LLM response + parsed hyps
    _orig_gen = _hyp.Hypothesizer.generate

    def _v_generate(self, evidence, active_expansions):
        # Capture raw LLM result by hooking reason_json once
        captured = {}
        orig_reason = self.llm.reason_json

        def _spy(system, user):
            r = orig_reason(system, user)
            captured["raw"] = r
            captured["user_prompt"] = user
            return r

        self.llm.reason_json = _spy
        try:
            hyps, goal = _orig_gen(self, evidence, active_expansions)
        finally:
            self.llm.reason_json = orig_reason

        print(f"\n[VERBOSE Phase 2] raw LLM response: {captured.get('raw')!r}")
        print(f"[VERBOSE Phase 2] goal_guess: {goal!r}")
        print(f"[VERBOSE Phase 2] parsed {len(hyps)} hypotheses:")
        for i, h in enumerate(hyps):
            print(f"  [{i}] rule={h.rule!r} test_action={h.test_action} "
                  f"status={h.status} expected={h.expected}")
        return hyps, goal

    _hyp.Hypothesizer.generate = _v_generate

    # Phase 3: wrap Verifier.verify to print expected vs actual per hypothesis
    _orig_verify = _ver.Verifier.verify

    def _v_verify(self, hypotheses):
        print(f"\n[VERBOSE Phase 3] verifying {len(hypotheses)} hypotheses")
        result = _orig_verify(self, hypotheses)
        for i, h in enumerate(result):
            print(f"  [{i}] action={h.test_action} status={h.status} "
                  f"score={h.match_score} expected={h.expected} "
                  f"actual_summary={h.actual_summary!r}")
        return result

    _ver.Verifier.verify = _v_verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 with CHARITH full stack agent")
    parser.add_argument("--game", default="ls20", help="ARC game id")
    parser.add_argument("--model", default="gemma3:12b", help="Ollama model name")
    parser.add_argument("--num-actions", type=int, default=8, help="Action space size")
    parser.add_argument("--verbose", action="store_true", help="Print per-phase diagnostics")
    args = parser.parse_args()

    print(f"[charith-full-stack] game={args.game} model={args.model} verbose={args.verbose}")

    if args.verbose:
        _install_verbose_hooks()

    arcade = arc_agi.Arcade()
    raw_env = arcade.make(args.game)
    env = _RealEnvAdapter(raw_env, verbose=args.verbose)
    env.reset()

    ollama = OllamaClient(model=args.model, temperature=0.3, max_tokens=2000)
    if args.verbose:
        print(f"[VERBOSE ollama] available={ollama.available} model={ollama.model} max_tokens={ollama.max_tokens}")
    llm = LLMReasoner(_OllamaAdapter(ollama, verbose=args.verbose))

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
