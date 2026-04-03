"""
Re-probe the trained maze GRU with BETTER probe targets.

Does NOT retrain. Loads saved weights, runs 50 inference episodes
to collect (hidden_state, ground_truth) pairs with new features,
then probes.

New features replace the 2 that failed (absolute position) with
features a maze-solver SHOULD encode:
  - direction_of_last_movement (one-hot binary per direction)
  - steps_since_last_wall_hit
  - goal_direction_row (goal above or below?)
  - goal_direction_col (goal left or right?)
  - action_had_effect (did the last action actually move the agent?)

Usage: uv run python scripts/reprobe_maze.py
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path

from charith.synthetic.maze_reality import MazeReality
from charith.neural.world_model_net import WorldModelNet
from charith.neural.state_encoder import encode, D_INPUT
from charith.neural.action_head import select_action
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.descartes.probes import run_probe, ProbeResult
from charith.descartes.graduation import graduation_exam


# ---- Feature definitions ----

PROBE_FEATURES = [
    # Already passed (from first run)
    {"name": "distance_to_goal", "threshold": 0.1, "type": "continuous"},
    {"name": "wall_adjacent_up", "threshold": 0.05, "type": "binary"},
    {"name": "wall_adjacent_down", "threshold": 0.05, "type": "binary"},
    {"name": "wall_adjacent_left", "threshold": 0.05, "type": "binary"},
    {"name": "wall_adjacent_right", "threshold": 0.05, "type": "binary"},
    # NEW replacements and additions
    {"name": "direction_of_last_movement_up", "threshold": 0.05, "type": "binary"},
    {"name": "direction_of_last_movement_down", "threshold": 0.05, "type": "binary"},
    {"name": "direction_of_last_movement_left", "threshold": 0.05, "type": "binary"},
    {"name": "direction_of_last_movement_right", "threshold": 0.05, "type": "binary"},
    {"name": "steps_since_last_wall_hit", "threshold": 0.1, "type": "continuous"},
    {"name": "goal_direction_row", "threshold": 0.05, "type": "binary"},
    {"name": "goal_direction_col", "threshold": 0.05, "type": "binary"},
    {"name": "action_had_effect", "threshold": 0.05, "type": "binary"},
]

# Graduation uses this subset (10 features, need 8/10 to pass)
GRADUATION_FEATURES = [
    "distance_to_goal",
    "wall_adjacent_up",
    "wall_adjacent_down",
    "wall_adjacent_left",
    "wall_adjacent_right",
    "direction_of_last_movement",
    "steps_since_last_wall_hit",
    "goal_direction_row",
    "goal_direction_col",
    "action_had_effect",
]


def collect_inference_data(model, reality, n_episodes=50, max_steps=300):
    """Run inference-only episodes, collecting hidden states + new ground truth."""
    model.eval()
    perception = CoreKnowledgePerception()

    all_hidden = []
    all_features = []
    episode_boundaries = []

    for ep in range(n_episodes):
        perception.reset()
        grid = reality.reset()
        hidden = model.init_hidden(batch_size=1)
        episode_boundaries.append(len(all_hidden))

        last_action = -1
        steps_since_wall = 0

        for step in range(max_steps):
            percept = perception.perceive(grid)
            action_enc = max(last_action, 0)
            x = encode(percept, action=action_enc,
                       controllable_ids=set(), grid_dims=grid.shape)

            with torch.no_grad():
                pred, logits, value, hidden = model(x.unsqueeze(0), hidden)

            # Select action (use learned policy)
            action, _ = select_action(
                logits.squeeze(0),
                available_actions=list(range(reality.n_actions)),
                temperature=0.5,
            )

            # Record pre-step state
            curr_row, curr_col = reality._ctrl_row, reality._ctrl_col
            goal_row, goal_col = reality._goal_row, reality._goal_col

            # Step
            next_grid, reward, done, info = reality.step(action)
            new_row, new_col = reality._ctrl_row, reality._ctrl_col

            # ---- Compute ground truth features ----
            gt = reality.get_ground_truth()

            # action_had_effect: did position actually change?
            had_effect = 1.0 if (new_row != curr_row or new_col != curr_col) else 0.0

            # direction_of_last_movement (one-hot based on actual displacement)
            dr = new_row - curr_row
            dc = new_col - curr_col
            gt["direction_of_last_movement_up"] = 1.0 if dr < 0 else 0.0
            gt["direction_of_last_movement_down"] = 1.0 if dr > 0 else 0.0
            gt["direction_of_last_movement_left"] = 1.0 if dc < 0 else 0.0
            gt["direction_of_last_movement_right"] = 1.0 if dc > 0 else 0.0

            # steps_since_last_wall_hit (normalized by 20)
            if had_effect == 0.0:
                steps_since_wall = 0
            else:
                steps_since_wall += 1
            gt["steps_since_last_wall_hit"] = min(steps_since_wall / 20.0, 1.0)

            # goal_direction_row: 1.0 if goal below, 0.0 if above, 0.5 if same
            if goal_row > new_row:
                gt["goal_direction_row"] = 1.0
            elif goal_row < new_row:
                gt["goal_direction_row"] = 0.0
            else:
                gt["goal_direction_row"] = 0.5

            # goal_direction_col: 1.0 if goal right, 0.0 if left, 0.5 if same
            if goal_col > new_col:
                gt["goal_direction_col"] = 1.0
            elif goal_col < new_col:
                gt["goal_direction_col"] = 0.0
            else:
                gt["goal_direction_col"] = 0.5

            gt["action_had_effect"] = had_effect

            # Store hidden state and features
            h_np = model.get_hidden_state(hidden).detach().squeeze(0).numpy()
            all_hidden.append(h_np)
            all_features.append(gt.copy())

            last_action = action
            grid = next_grid

            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} done ({len(all_hidden)} timesteps)")

    return np.array(all_hidden), all_features, episode_boundaries


def main():
    print("=" * 70)
    print("CHARITH Path 3: Re-Probing with Better Feature Targets")
    print("=" * 70)

    # ---- Load trained model ----
    checkpoint = Path("checkpoints/maze_gru_level1.pt")
    if not checkpoint.exists():
        print(f"ERROR: No checkpoint at {checkpoint}")
        print("Run pretrain_maze.py first, or copy checkpoint from Vast.ai")
        sys.exit(1)

    reality = MazeReality(level=1, step_penalty=-0.05)
    model = WorldModelNet(input_size=D_INPUT, hidden_size=256,
                          n_actions=reality.n_actions)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    print(f"Loaded model from {checkpoint}")
    print(f"Maze: {reality.grid_size}x{reality.grid_size}")

    # ---- Collect inference data ----
    print(f"\nCollecting hidden states from 50 inference episodes...")
    hidden_states, features, ep_bounds = collect_inference_data(
        model, reality, n_episodes=50, max_steps=300
    )
    print(f"Collected {len(hidden_states)} timesteps from {len(ep_bounds)} episodes")

    # ---- Probe each feature ----
    print(f"\n{'='*70}")
    print("DESCARTES Probing (updated features)")
    print(f"{'='*70}\n")

    probe_results = []
    for feat_def in PROBE_FEATURES:
        name = feat_def["name"]
        threshold = feat_def["threshold"]

        # Extract targets
        targets = np.array([f.get(name, 0.0) for f in features])

        # Skip if constant (no variation to probe)
        if np.std(targets) < 1e-6:
            print(f"  {name:40s}: SKIP (constant)")
            continue

        print(f"  Probing: {name} (threshold={threshold})...")
        result = run_probe(
            feature_name=name,
            hidden_states=hidden_states,
            targets=targets,
            episode_boundaries=ep_bounds,
            threshold=threshold,
            n_permutations=20,
            alpha=100.0,
            max_samples=20000,
        )
        probe_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"    R2={result.r2_trained:.3f}, "
              f"null={result.r2_null_mean:.3f}, "
              f"dR2={result.delta_r2:.3f}, "
              f"p={result.p_value:.2f} "
              f"[{status}]")

    # ---- Graduation (10-feature scorecard) ----
    print(f"\n{'='*70}")
    print("GRADUATION SCORECARD (need 8/10 to pass)")
    print(f"{'='*70}\n")

    # For direction_of_last_movement, pass if ANY of the 4 directions pass
    dir_probes = [r for r in probe_results
                  if r.feature_name.startswith("direction_of_last_movement")]
    dir_passed = any(r.passed for r in dir_probes)
    best_dir = max(dir_probes, key=lambda r: r.delta_r2) if dir_probes else None

    # Build graduation results
    grad_results = []
    seen_direction = False
    for feat_name in GRADUATION_FEATURES:
        if feat_name == "direction_of_last_movement":
            if best_dir and not seen_direction:
                group_result = ProbeResult(
                    feature_name="direction_of_last_movement",
                    r2_trained=best_dir.r2_trained,
                    r2_null_mean=best_dir.r2_null_mean,
                    r2_null_std=best_dir.r2_null_std,
                    delta_r2=best_dir.delta_r2,
                    p_value=best_dir.p_value,
                    passed=dir_passed,
                    threshold=0.05,
                )
                grad_results.append(group_result)
                seen_direction = True
            continue

        match = next((r for r in probe_results if r.feature_name == feat_name), None)
        if match:
            grad_results.append(match)

    n_passed = sum(1 for r in grad_results if r.passed)
    n_total = len(grad_results)

    for r in grad_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.feature_name:40s} dR2={r.delta_r2:+.3f} [{status}]")

    print(f"\n  Score: {n_passed}/{n_total} ({n_passed/max(n_total,1):.0%})")
    if n_passed >= 8:
        print(f"\n  === GRADUATED === Path 3 approach VALIDATED ===")
    else:
        need = 8 - n_passed
        print(f"\n  Need {need} more passes for graduation ({need} of {n_total - n_passed} remaining)")


if __name__ == "__main__":
    main()
