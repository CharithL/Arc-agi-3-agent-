"""
Evaluate the LLM agent on ALL 25 public ARC-AGI-3 games.

Collects per-game data:
- C1 expansions, hypothesis confidence, working actions found
- Minimum goal distance achieved, controllable detected
- Final hypothesis, level completions

Outputs summary table sorted by "most promising" games.
Saves detailed logs to logs/{game_id}_llm.json.

Usage:
  set PYTHONPATH=src
  py -3.12 scripts/evaluate_all_games.py              # uses Gemma 4 (free)
  py -3.12 scripts/evaluate_all_games.py claude        # uses Claude API ($)
  py -3.12 scripts/evaluate_all_games.py gemma4 50     # custom model + actions
"""
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, 'src')

from charith.llm_agent.agent import LLMAgent

# ---- Config ----
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gemma4'
MAX_ACTIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# Map "claude" shorthand
if MODEL == 'claude':
    MODEL = 'auto'  # auto picks Claude API if ANTHROPIC_API_KEY is set

# ---- Get game list ----
try:
    import arc_agi
    arcade = arc_agi.Arcade()
    games = arcade.get_environments()
    GAME_IDS = [g.game_id for g in games]
    print(f"Found {len(GAME_IDS)} games")
except Exception as e:
    print(f"ERROR getting game list: {e}")
    sys.exit(1)

# ---- Evaluate each game ----
results = {}
Path("logs").mkdir(exist_ok=True)

print(f"\nEvaluating {len(GAME_IDS)} games with {MODEL} ({MAX_ACTIONS} actions each)")
print("=" * 90)
print(f"{'Game':>8} | {'C1+':>3} | {'Conf':>5} | {'WorkAct':>7} | {'MinDist':>7} | "
      f"{'Ctrl':>4} | {'Lvls':>4} | {'Hypothesis':>35}")
print("-" * 90)

for i, game_id in enumerate(GAME_IDS):
    short = game_id.split('-')[0]
    t0 = time.time()

    try:
        agent = LLMAgent(model=MODEL, temperature=0.3)
        result = agent.play_game(game_id, max_actions=MAX_ACTIONS)
        elapsed = time.time() - t0

        # Collect metrics
        n_c1 = len(agent.c1c2.c1_expansions)

        # Highest confidence
        max_conf = 'none'
        if agent.c1c2.hypotheses:
            confs = []
            for h in agent.c1c2.hypotheses:
                c = h.confidence if hasattr(h, 'confidence') else h.get('confidence', 'low')
                confs.append(c)
            if 'high' in confs:
                max_conf = 'high'
            elif 'med' in confs:
                max_conf = 'med'
            elif confs:
                max_conf = confs[-1]

        # Working actions (actions that produced real effects)
        n_working = 0
        for a, counter in agent.context.action_effect_map.items():
            has_real = any('no movement' not in e.lower() and 'wall' not in e.lower()
                         for e in counter.keys())
            if has_real:
                n_working += 1

        # Minimum goal distance
        min_dist = agent.translator._prev_goal_distance
        if min_dist is None:
            min_dist = -1

        # Controllable detected?
        ctrl_detected = len(agent._controllable_ids) > 0

        # Final hypothesis
        final_hyp = ''
        if agent.c1c2.hypotheses:
            h = agent.c1c2.hypotheses[-1]
            final_hyp = h.text if hasattr(h, 'text') else h.get('text', '')

        # Levels completed
        levels = result.get('levels_completed', 0) if isinstance(result, dict) else 0

        game_data = {
            'game_id': game_id,
            'short': short,
            'c1_expansions': n_c1,
            'max_confidence': max_conf,
            'working_actions': n_working,
            'min_goal_dist': float(min_dist) if min_dist >= 0 else None,
            'controllable_detected': ctrl_detected,
            'levels_completed': levels,
            'final_hypothesis': final_hyp[:200],
            'actions_taken': result.get('actions_taken', MAX_ACTIONS) if isinstance(result, dict) else MAX_ACTIONS,
            'elapsed_seconds': round(elapsed, 1),
            'action_effects': {
                str(k): {eff: cnt for eff, cnt in v.items()}
                for k, v in agent.context.action_effect_map.items()
            },
            'c1_expansion_names': [e.name if hasattr(e, 'name') else str(e) for e in agent.c1c2.c1_expansions],
        }
        results[short] = game_data

        # Save per-game log
        with open(f"logs/{short}_llm.json", 'w') as f:
            json.dump(game_data, f, indent=2, default=str)

        # Print row
        dist_str = f"{min_dist:.1f}" if min_dist >= 0 else "  ?"
        hyp_short = final_hyp[:35] if final_hyp else '(none)'
        lvl_str = f"{levels}" if levels > 0 else " 0"
        print(f"{short:>8} | {n_c1:>3} | {max_conf:>5} | {n_working:>7} | {dist_str:>7} | "
              f"{'Y' if ctrl_detected else 'N':>4} | {lvl_str:>4} | {hyp_short:>35}")

    except Exception as e:
        err_msg = str(e)[:60]
        print(f"{short:>8} | ERROR: {err_msg}")
        results[short] = {'game_id': game_id, 'error': str(e)[:200]}

# ---- Summary ----
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

# Sort by "most promising": levels > 0 first, then by min_dist (lower = closer)
# then by working_actions (more = better), then by max_confidence
def sort_key(item):
    d = item[1]
    if 'error' in d:
        return (0, 999, 0, 0)
    lvls = d.get('levels_completed', 0)
    dist = d.get('min_goal_dist') or 999
    n_work = d.get('working_actions', 0)
    conf_score = {'high': 3, 'med': 2, 'low': 1}.get(d.get('max_confidence', ''), 0)
    return (-lvls, dist, -n_work, -conf_score)

sorted_results = sorted(results.items(), key=sort_key)

print(f"\nTop 10 most promising games:")
for rank, (short, d) in enumerate(sorted_results[:10], 1):
    if 'error' in d:
        print(f"  {rank:2d}. {short:>8}: ERROR")
        continue
    lvls = d.get('levels_completed', 0)
    dist = d.get('min_goal_dist')
    dist_str = f"{dist:.1f}" if dist is not None else "?"
    n_work = d.get('working_actions', 0)
    conf = d.get('max_confidence', '?')
    ctrl = 'Y' if d.get('controllable_detected') else 'N'
    hyp = d.get('final_hypothesis', '')[:50]
    print(f"  {rank:2d}. {short:>8}: dist={dist_str:>6} work={n_work} conf={conf:>4} ctrl={ctrl} lvls={lvls} | {hyp}")

# Total stats
total = len(results)
errors = sum(1 for d in results.values() if 'error' in d)
with_ctrl = sum(1 for d in results.values() if d.get('controllable_detected'))
with_levels = sum(1 for d in results.values() if d.get('levels_completed', 0) > 0)
print(f"\nTotal: {total} games, {errors} errors, {with_ctrl} with controllable detected, {with_levels} with levels completed")

# Save full results
with open("results/all_games_evaluation.json", 'w') as f:
    json.dump({
        'config': {'model': MODEL, 'max_actions': MAX_ACTIONS},
        'results': results,
        'sorted_top10': [short for short, _ in sorted_results[:10]],
    }, f, indent=2, default=str)

print(f"\nResults saved to results/all_games_evaluation.json")
print(f"Per-game logs saved to logs/{{game_id}}_llm.json")
