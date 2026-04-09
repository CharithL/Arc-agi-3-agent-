"""
Play the CHARITH full-stack agent on every ARC-AGI-3 public game.

Usage:
    python scripts/play_all_full_stack.py --model gemma4:latest --max-levels 5

For each game, plays up to --max-levels levels, logs per-game results to
logs/all_full_stack/<game>.log, and prints an aggregate summary at the end.

Does NOT submit a scorecard — the Arcade prints a scorecard guid at the
start of every game session. If you want to submit to the ARC Prize
leaderboard, review the aggregate report first and submit manually.
"""

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Force UTF-8 stdout/stderr so Windows cp1252 doesn't choke on arrows/emoji.
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


_ACTION_MAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


class _OllamaAdapter:
    def __init__(self, client: OllamaClient):
        self._client = client

    def generate(self, system_prompt: str, user_prompt: str, **_kwargs) -> str:
        return self._client.query(system_prompt, user_prompt)


class _RealEnvAdapter:
    def __init__(self, env):
        self._env = env
        self._latest = None
        self._levels_completed = 0

    def reset(self):
        self._latest = self._env.reset()
        if self._latest is not None and hasattr(self._latest, "levels_completed"):
            self._levels_completed = self._latest.levels_completed
        return self._latest

    def get_observation(self):
        return self._latest

    def step(self, action_id: int):
        game_action = _ACTION_MAP.get(action_id, GameAction.ACTION1)
        frame = self._env.step(game_action)
        self._latest = frame

        reward = 0.0
        done = False
        if frame is not None:
            if hasattr(frame, "levels_completed"):
                new_levels = frame.levels_completed
                if new_levels > self._levels_completed:
                    reward = 1.0
                    self._levels_completed = new_levels
                    done = True
            if hasattr(frame, "state"):
                state_val = str(frame.state)
                if ("FINISHED" in state_val or "WON" in state_val or "LOST" in state_val) \
                        and "NOT_FINISHED" not in state_val:
                    done = True

        return frame, reward, done, {}


def _list_games() -> list:
    """Enumerate game ids from environment_files/ (on-disk public set)."""
    env_dir = Path("environment_files")
    if not env_dir.is_dir():
        return []
    return sorted([p.name for p in env_dir.iterdir() if p.is_dir()])


def _play_one(game_id: str, model: str, max_levels: int, num_actions: int):
    """
    Play a single game and return (GameResult, error_str_or_None).
    Each game gets its own Arcade + env + agent so state doesn't leak.
    """
    try:
        arcade = arc_agi.Arcade()
        raw_env = arcade.make(game_id)
        env = _RealEnvAdapter(raw_env)
        env.reset()

        ollama = OllamaClient(model=model, temperature=0.3, max_tokens=2000)
        llm = LLMReasoner(_OllamaAdapter(ollama))
        agent = CharithFullStackAgent(env, llm, num_actions=num_actions)

        result = agent.play_game(game_id=game_id, max_levels=max_levels)
        return result, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"{type(e).__name__}: {e}\n{tb}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CHARITH full stack on all public games")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--max-levels", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=8)
    parser.add_argument("--games", default="",
                        help="Comma-separated subset (default: all games in environment_files/)")
    parser.add_argument("--log-dir", default="logs/all_full_stack")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = _list_games()

    if not games:
        print("ERROR: no games found. Populate environment_files/ or pass --games.")
        return 2

    print(f"[play_all] model={args.model} max_levels={args.max_levels} games={len(games)}")

    rows = []
    total_levels_completed = 0
    total_levels_attempted = 0
    total_actions = 0
    total_llm_calls = 0
    grand_start = time.time()

    for i, game_id in enumerate(games, 1):
        print(f"\n[{i}/{len(games)}] >>> {game_id}")
        t0 = time.time()
        result, err = _play_one(game_id, args.model, args.max_levels, args.num_actions)
        elapsed = time.time() - t0

        log_path = log_dir / f"{game_id}.json"
        if err is not None:
            print(f"  ERROR after {elapsed:.1f}s: {err.splitlines()[0]}")
            log_path.write_text(json.dumps({"game_id": game_id, "error": err}, indent=2))
            rows.append({
                "game_id": game_id,
                "levels_completed": 0,
                "levels_attempted": 0,
                "total_actions": 0,
                "total_llm_calls": 0,
                "wall_time_sec": elapsed,
                "stopped_reason": "error",
                "error": err.splitlines()[0],
            })
            continue

        print(f"  levels: {result.levels_completed}/{result.levels_attempted}  "
              f"actions={result.total_actions}  llm_calls={result.total_llm_calls}  "
              f"time={result.wall_time_sec:.1f}s  stopped={result.stopped_reason}")

        log_path.write_text(json.dumps({
            "game_id": result.game_id,
            "levels_completed": result.levels_completed,
            "levels_attempted": result.levels_attempted,
            "total_actions": result.total_actions,
            "total_llm_calls": result.total_llm_calls,
            "wall_time_sec": result.wall_time_sec,
            "stopped_reason": result.stopped_reason,
        }, indent=2))

        total_levels_completed += result.levels_completed
        total_levels_attempted += result.levels_attempted
        total_actions += result.total_actions
        total_llm_calls += result.total_llm_calls
        rows.append({
            "game_id": game_id,
            "levels_completed": result.levels_completed,
            "levels_attempted": result.levels_attempted,
            "total_actions": result.total_actions,
            "total_llm_calls": result.total_llm_calls,
            "wall_time_sec": result.wall_time_sec,
            "stopped_reason": result.stopped_reason,
        })

    grand_elapsed = time.time() - grand_start

    print("\n" + "=" * 72)
    print("  AGGREGATE REPORT")
    print("=" * 72)
    print(f"  Games played: {len(games)}")
    print(f"  Levels completed: {total_levels_completed} / {total_levels_attempted}")
    print(f"  Total actions: {total_actions}")
    print(f"  Total LLM calls: {total_llm_calls}")
    print(f"  Wall time: {grand_elapsed:.1f}s")
    print("-" * 72)
    print(f"  {'game':<12} {'levels':>8} {'acts':>6} {'llm':>5} {'time':>7}  reason")
    for r in rows:
        print(f"  {r['game_id']:<12} "
              f"{r['levels_completed']}/{r['levels_attempted']:<4} "
              f"{r['total_actions']:>6} "
              f"{r['total_llm_calls']:>5} "
              f"{r['wall_time_sec']:>6.1f}s  "
              f"{r['stopped_reason']}")

    # JSON aggregate for later analysis
    agg_path = log_dir / "_aggregate.json"
    agg_path.write_text(json.dumps({
        "model": args.model,
        "max_levels": args.max_levels,
        "games": len(games),
        "total_levels_completed": total_levels_completed,
        "total_levels_attempted": total_levels_attempted,
        "total_actions": total_actions,
        "total_llm_calls": total_llm_calls,
        "wall_time_sec": grand_elapsed,
        "rows": rows,
    }, indent=2))
    print(f"\n  Aggregate JSON: {agg_path}")

    return 0 if total_levels_completed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
