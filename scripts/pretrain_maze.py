"""
First milestone experiment: Train GRU on maze_reality, probe for distance_to_goal.

Success criterion: delta_R^2 > 0.1 for distance_to_goal feature.
This single result validates the entire Path 3 approach.

Usage: uv run python scripts/pretrain_maze.py
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path

from charith.synthetic.maze_reality import MazeReality
from charith.neural.world_model_net import WorldModelNet
from charith.neural.state_encoder import D_INPUT
from charith.neural.training import train_on_reality, collect_probing_dataset
from charith.descartes.probes import run_probe
from charith.descartes.mandatory_features import MAZE_FEATURES
from charith.descartes.graduation import graduation_exam


def main():
    print("=" * 70)
    print("CHARITH Path 3: Maze Reality Pre-Training + DESCARTES Validation")
    print("=" * 70)

    # ---- Configuration ----
    N_EPISODES = 300       # Enough to learn maze navigation
    LEVEL = 1              # Start with 8x8 mazes (simplest)
    HIDDEN_SIZE = 256
    LR = 3e-4

    # ---- Initialize ----
    print(f"\nConfig: {N_EPISODES} episodes, level={LEVEL} (8x8), hidden={HIDDEN_SIZE}")
    print(f"Target: distance_to_goal probe delta_R^2 > 0.1\n")

    reality = MazeReality(level=LEVEL)
    model = WorldModelNet(input_size=D_INPUT, hidden_size=HIDDEN_SIZE,
                          n_actions=reality.n_actions)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Maze grid size: {reality.grid_size}x{reality.grid_size}")

    # ---- Train ----
    print(f"\n{'='*70}")
    print("PHASE 1: Training GRU on Maze Reality")
    print(f"{'='*70}\n")

    stats, episode_data = train_on_reality(
        model=model,
        reality=reality,
        n_episodes=N_EPISODES,
        lr=LR,
        pred_loss_weight=1.0,
        rl_loss_weight=0.5,
        max_steps_per_episode=200,
        collect_probing_data=True,
        verbose=True,
    )

    print(f"\n--- Training Summary ---")
    print(f"  Episodes: {stats.episodes}")
    print(f"  Total steps: {stats.total_steps}")
    print(f"  Mean reward: {stats.mean_reward:+.3f}")
    print(f"  Solve rate: {stats.solve_rate:.1%}")
    print(f"  Mean pred loss: {stats.mean_pred_loss:.4f}")

    # ---- Probe ----
    print(f"\n{'='*70}")
    print("PHASE 2: DESCARTES Probing on Hidden States")
    print(f"{'='*70}\n")

    probe_results = []
    for feature in MAZE_FEATURES:
        print(f"Probing: {feature.name} (threshold={feature.threshold})...")

        h_states, targets, ep_bounds = collect_probing_dataset(
            episode_data, feature.name
        )

        if len(h_states) < 100:
            print(f"  SKIP: insufficient data ({len(h_states)} samples)")
            continue

        result = run_probe(
            feature_name=feature.name,
            hidden_states=h_states,
            targets=targets,
            episode_boundaries=ep_bounds,
            threshold=feature.threshold,
            n_permutations=50,  # Fewer for speed; 100 for paper
        )

        probe_results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  R^2_trained = {result.r2_trained:.3f}")
        print(f"  R^2_null    = {result.r2_null_mean:.3f} +/- {result.r2_null_std:.3f}")
        print(f"  delta_R^2   = {result.delta_r2:.3f}")
        print(f"  p-value     = {result.p_value:.3f}")
        print(f"  [{status}]\n")

    # ---- Graduation ----
    print(f"{'='*70}")
    print("PHASE 3: Graduation Exam")
    print(f"{'='*70}\n")

    if probe_results:
        exam = graduation_exam(probe_results)
        print(f"  Score: {exam['n_passed']}/{exam['n_total']} features passed ({exam['score']:.0%})")
        print(f"  Result: {'GRADUATED' if exam['passed'] else 'FAILED'}")
        if exam['passed_features']:
            print(f"  Passed: {', '.join(exam['passed_features'])}")
        if exam['failed_features']:
            print(f"  Failed: {', '.join(exam['failed_features'])}")
    else:
        print("  No probe results to evaluate.")

    # ---- Key Result ----
    print(f"\n{'='*70}")
    print("KEY RESULT: distance_to_goal")
    print(f"{'='*70}")
    dtg = next((r for r in probe_results if r.feature_name == 'distance_to_goal'), None)
    if dtg:
        print(f"\n  delta_R^2 = {dtg.delta_r2:.3f}")
        if dtg.delta_r2 > 0.1:
            print(f"  STATUS: SUCCESS -- The GRU encodes distance-to-goal!")
            print(f"  The synthetic reality forced spatial goal representations.")
            print(f"  Path 3 approach is VALIDATED.")
        else:
            print(f"  STATUS: NOT YET -- delta_R^2 below 0.1 threshold")
            print(f"  Options: train more episodes, increase hidden size, or simplify maze")
    else:
        print("  distance_to_goal probe not run (insufficient data)")

    # ---- Save model ----
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / "maze_gru_level1.pt")
    print(f"\nModel saved to checkpoints/maze_gru_level1.pt")


if __name__ == "__main__":
    main()
