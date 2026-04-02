"""
Run CHARITH agent on multiple real ARC-AGI-3 games and compare results.
Picks 6 diverse games (different tags, different IDs).

Usage: uv run python scripts/diagnose_multi.py
"""
import sys
sys.path.insert(0, 'src')

from charith.agent import CHARITHAgent
import time

# Pick 6 diverse games covering different tags
GAMES = [
    "ls20",   # keyboard
    "bp35",   # keyboard_click
    "vc33",   # click
    "lf52",   # click
    "ft09",   # no tags
    "tr87",   # keyboard
]

MAX_ACTIONS = 100

print(f"{'Game':>8} | {'Tag':>15} | {'Acts':>4} | {'Rules':>5} | {'Acc':>6} | "
      f"{'Preds':>5} | {'Lvls':>4} | {'Grid':>9} | {'Colors':>8} | "
      f"{'Objects':>7} | {'Ctrl':>4} | {'Avail':>10} | {'TPS':>5}")
print("-" * 130)

for game_id in GAMES:
    try:
        agent = CHARITHAgent()
        start = time.perf_counter()
        scorecard = agent.play_game(game_id, max_actions=MAX_ACTIONS)
        elapsed = time.perf_counter() - start
        tps = agent._total_actions / max(elapsed, 0.001)

        # Get grid info from last state
        grid_shape = "?"
        colors = "?"
        n_objects = "?"
        n_ctrl = "?"
        avail = "?"

        if agent._prev_grid is not None:
            import numpy as np
            grid_shape = f"{agent._prev_grid.shape[0]}x{agent._prev_grid.shape[1]}"
            colors = f"{len(np.unique(agent._prev_grid))}"

        if agent._prev_percept is not None:
            n_objects = str(agent._prev_percept.object_count)
            n_ctrl = str(len(agent._controllable_ids))

        avail_str = str(agent._available_actions) if hasattr(agent, '_available_actions') else "?"

        # Get tag from scorecard
        tag = "?"
        if isinstance(scorecard, dict) and 'tags_scores' in scorecard:
            tags = scorecard.get('tags_scores', [])
            if tags:
                tag = tags[0].get('id', '?') if isinstance(tags[0], dict) else str(tags[0])
        elif hasattr(scorecard, 'tags_scores') and scorecard.tags_scores:
            tag = scorecard.tags_scores[0].id if hasattr(scorecard.tags_scores[0], 'id') else str(scorecard.tags_scores[0])

        acc = agent.world_model.get_accuracy()
        print(f"{game_id:>8} | {tag:>15} | {agent._total_actions:>4} | "
              f"{agent.world_model.get_rule_count():>5} | {acc:>5.1%} | "
              f"{agent.world_model._total_predictions:>5} | {agent._levels_completed:>4} | "
              f"{grid_shape:>9} | {colors:>8} | {n_objects:>7} | {n_ctrl:>4} | "
              f"{avail_str:>10} | {tps:>5.0f}")

    except Exception as e:
        print(f"{game_id:>8} | ERROR: {e}")

print("\nDone.")
