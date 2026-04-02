"""
Diagnostic script: Play a real ARC-AGI-3 game and log what the agent
perceives, predicts, and decides at each tick.

Usage:
    uv run python scripts/diagnose_game.py ls20 50
    uv run python scripts/diagnose_game.py bp35 100
"""
import sys
import time
import numpy as np

sys.path.insert(0, 'src')

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.perception.object_tracker import ObjectTracker
from charith.world_model.model import WorldModel, PredictionError
from charith.action.thompson import ThompsonSampler
from charith.action.action_space import N_ACTIONS
from charith.metacognition.goal_discovery import GoalDiscovery
from charith.memory.sequences import ActionSequenceMemory

# ---------- Parse args ----------
game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 50

# ---------- Setup ----------
import arc_agi
from arcengine import GameAction

ACTION_MAP = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2,
    2: GameAction.ACTION3, 3: GameAction.ACTION4,
    4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}

arc = arc_agi.Arcade()
env = arc.make(game_id)
perception = CoreKnowledgePerception()
tracker = ObjectTracker()
world_model = WorldModel()
goal_disc = GoalDiscovery()
explorer = ThompsonSampler(n_actions=N_ACTIONS, info_gain_weight=0.5)
seq_mem = ActionSequenceMemory(n_actions=N_ACTIONS)

# ---------- Reset ----------
frame = env.reset()
grid = frame.frame[0].astype(int)
available = frame.available_actions
prev_grid = None
prev_percept = None
prev_action = 0
controllable = set()

print(f"{'='*70}")
print(f"CHARITH Diagnostic — Game: {game_id}")
print(f"Grid: {grid.shape}, Colors: {sorted(np.unique(grid))}")
print(f"Available actions: {available}")
print(f"Win levels: {frame.win_levels}")
print(f"{'='*70}\n")

start = time.perf_counter()

for tick in range(max_actions):
    # ---- Perceive ----
    percept = perception.perceive(grid)

    # ---- Track ----
    matched_pairs = []
    if prev_percept is not None:
        matched_pairs = tracker.match(prev_percept.objects, percept.objects)
        curr_map = {o.object_id: o for o in percept.objects}
        for _, curr_id in matched_pairs:
            if curr_id in curr_map:
                perception.agency.record_object_displacement(
                    curr_id, curr_map[curr_id].centroid)

    # ---- Controllable ----
    controllable = perception.agency.detect_controllable_objects(
        percept.objects, perception.agency._contingencies)

    # ---- Context ----
    context = world_model.extract_context(percept, controllable)

    # ---- Predict ----
    predicted = None
    if prev_percept is not None:
        predicted = world_model.predict(prev_action, context)

    # ---- Actual effects ----
    actual_effects = []
    if prev_percept is not None:
        actual_effects = world_model.compute_effects(
            prev_percept, percept, matched_pairs)

    # ---- Update ----
    if prev_percept is not None:
        world_model.update(prev_action, context, actual_effects, tick)

    # ---- Select action (constrained to available) ----
    avail_mapped = [a - 1 for a in available if 1 <= a <= 7]  # GameAction 1-7 -> index 0-6
    if not avail_mapped:
        avail_mapped = list(range(min(N_ACTIONS, 4)))

    action = explorer.select_action(
        context_hash=hash(grid.tobytes()),
        available_actions=avail_mapped,
        prev_action=prev_action if tick > 0 else None,
        sequence_memory=seq_mem if tick > 0 else None,
    )

    # ---- Log every 5 ticks ----
    if tick % 5 == 0 or tick < 3:
        n_objects = percept.object_count
        n_moved = sum(1 for e in actual_effects
                      if e.displacement != (0, 0))
        n_appeared = sum(1 for e in actual_effects if e.appeared)
        n_disappeared = sum(1 for e in actual_effects if e.disappeared)
        pred_str = "None" if predicted is None else f"{len(predicted)} effects"

        print(f"[Tick {tick:3d}] "
              f"Objects: {n_objects:3d} | "
              f"Controllable: {len(controllable)} | "
              f"Moved: {n_moved} App: {n_appeared} Dis: {n_disappeared} | "
              f"Predicted: {pred_str} | "
              f"Rules: {world_model.get_rule_count():3d} | "
              f"Action: {action} -> GameAction.ACTION{action+1}")

    # ---- Act ----
    game_action = ACTION_MAP.get(action, GameAction.ACTION1)
    result = env.step(game_action)

    # ---- Process result ----
    new_grid = result.frame[0].astype(int) if result.frame else grid
    new_levels = result.levels_completed
    state = str(result.state)
    available = result.available_actions

    # Reward
    reward = goal_disc.update(
        new_grid, action, score=float(new_levels),
        level_complete=(new_levels > 0 and prev_percept is not None),
    )
    explorer.update(hash(grid.tobytes()), action, reward)
    if tick > 0:
        seq_mem.update(prev_action, action, reward)

    # Contingency
    if prev_grid is not None:
        perception.agency.record_action_contingency(prev_action, prev_grid, new_grid)

    # State change
    changed_cells = int(np.sum(grid != new_grid))
    if changed_cells > 0 and tick % 5 == 0:
        print(f"         State change: {changed_cells} cells changed")

    # Level transition
    if new_levels > 0:
        print(f"\n  *** LEVEL COMPLETE at tick {tick}! ***\n")

    if 'NOT_FINISHED' not in state:
        print(f"\n  Game over: {state}")
        break

    # Update state
    prev_grid = grid.copy()
    prev_percept = percept
    prev_action = action
    grid = new_grid

elapsed = time.perf_counter() - start

# ---------- Summary ----------
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Game:              {game_id}")
print(f"Ticks:             {tick + 1}")
print(f"Time:              {elapsed:.2f}s ({(tick+1)/elapsed:.0f} ticks/sec)")
print(f"Rules learned:     {world_model.get_rule_count()}")
print(f"Pred accuracy:     {world_model.get_accuracy():.1%}")
print(f"Unique contexts:   {len(set(str(r.context_features) for rules in world_model._rules.values() for r in rules))}")
print(f"Controllable obj:  {len(controllable)}")
print(f"Grid shape:        {grid.shape}")
print(f"Color range:       {sorted(np.unique(grid))}")

# Action distribution
summary = explorer.get_action_summary()
print(f"\nAction distribution:")
for a, s in sorted(summary.items()):
    if s['times_taken'] > 0:
        mean_r = s.get('mean_reward', s.get('mean', 0.0))
        print(f"  Action {a}: {s['times_taken']:3d} times, mean={mean_r:.3f}")

# Best goal hypothesis
best = goal_disc.get_best_hypothesis()
if best:
    print(f"\nBest goal hypothesis: {best.description} (confidence={best.confidence:.2f})")

# Show rule examples
print(f"\nSample rules (first 5):")
count = 0
for action_idx, rules in world_model._rules.items():
    for rule in rules[:2]:
        effs = ", ".join(
            f"color={e.object_color} d={e.displacement}"
            for e in rule.effects[:3]
        )
        print(f"  Action {action_idx}: ctx={dict(list(rule.context_features.items())[:3])} -> [{effs}] (conf={rule.confidence:.2f})")
        count += 1
        if count >= 5:
            break
    if count >= 5:
        break
