"""
THE EXPERIMENT THAT MATTERS: Does Maze Pre-Training Transfer to Real ARC-AGI-3?

Protocol:
1. Train GRU on maze reality (30 min)
2. Deploy on 5 real keyboard games
3. Compare: pre-trained agent vs random actions
4. Metric: average prediction error over first 50 steps

Usage on Vast.ai:
    git pull && uv sync && uv run python scripts/experiment_transfer.py
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import time
import json
from pathlib import Path

from charith.synthetic.maze_reality import MazeReality
from charith.neural.world_model_net import WorldModelNet
from charith.neural.state_encoder import encode, D_INPUT
from charith.neural.action_head import select_action
from charith.neural.training import train_on_reality
from charith.perception.core_knowledge import CoreKnowledgePerception

MAZE_EPISODES = 1500
MAZE_LEVEL = 1
REAL_GAMES = ['ls20', 'tr87', 'g50t', 'wa30', 're86']
EVAL_STEPS = 50
HIDDEN_SIZE = 256


def evaluate_on_real_game(model, game_id, n_steps=50):
    """Play a real game and measure prediction MSE (lower = better)."""
    import arc_agi
    from arcengine import GameAction

    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)
    frame = env.reset()
    grid = np.array(frame.frame[0])
    avail = frame.available_actions

    perception = CoreKnowledgePerception()
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    pred_errors = []
    prev_encoding = None

    for step in range(n_steps):
        percept = perception.perceive(grid)
        action = avail[step % len(avail)]
        x = encode(percept, action=action - 1,
                   controllable_ids=set(), grid_dims=grid.shape)

        with torch.no_grad():
            pred, logits, value, hidden = model(x.unsqueeze(0), hidden)

        if prev_encoding is not None:
            error = float(torch.nn.functional.mse_loss(pred.squeeze(0), x).item())
            pred_errors.append(error)

        prev_encoding = x

        try:
            result = env.step(GameAction[f'ACTION{action}'])
            grid = np.array(result.frame[0])
            avail = result.available_actions
        except Exception:
            break

    return np.mean(pred_errors) if pred_errors else float('inf')


def main():
    print("=" * 70)
    print("EXPERIMENT: Does Maze Pre-Training Transfer to Real ARC-AGI-3?")
    print("=" * 70)

    # Phase 1: Train
    print(f"\nPhase 1: Training GRU on maze ({MAZE_EPISODES} eps)...")
    t0 = time.time()

    reality = MazeReality(level=MAZE_LEVEL, step_penalty=-0.05)
    pretrained = WorldModelNet(input_size=D_INPUT, hidden_size=HIDDEN_SIZE, n_actions=4)

    stats, _ = train_on_reality(
        model=pretrained, reality=reality, n_episodes=MAZE_EPISODES,
        lr=3e-4, spatial_loss_weight=1.0, spatial_lookahead_k=5,
        max_steps_per_episode=300, collect_probing_data=False, verbose=True,
    )

    train_time = time.time() - t0
    print(f"\nDone: {train_time:.0f}s, solve={stats.solve_rate:.1%}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(pretrained.state_dict(), "checkpoints/maze_gru_level1.pt")

    # Phase 2: Baseline
    random_model = WorldModelNet(input_size=D_INPUT, hidden_size=HIDDEN_SIZE, n_actions=4)

    # Phase 3: Compare
    print(f"\n{'='*70}")
    print(f"Phase 2: Evaluating on {len(REAL_GAMES)} real games")
    print(f"{'='*70}\n")

    results = {}
    for gid in REAL_GAMES:
        print(f"  {gid}...", end=" ", flush=True)
        try:
            pe = evaluate_on_real_game(pretrained, gid, EVAL_STEPS)
            re = evaluate_on_real_game(random_model, gid, EVAL_STEPS)
            imp = (re - pe) / max(re, 1e-8) * 100
            results[gid] = {'pretrained': pe, 'random': re, 'improvement': imp}
            print(f"pre={pe:.4f} rand={re:.4f} imp={imp:+.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")
            results[gid] = {'error': str(e)}

    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    imps = []
    for gid, r in results.items():
        if 'error' not in r:
            print(f"  {gid:>6}: pre={r['pretrained']:.4f} rand={r['random']:.4f} {r['improvement']:+.1f}%")
            imps.append(r['improvement'])

    if imps:
        avg = np.mean(imps)
        print(f"\n  Avg improvement: {avg:+.1f}%")
        print(f"  Games helped: {sum(1 for x in imps if x > 0)}/{len(imps)}")
        verdict = "TRANSFERS" if avg > 0 else "DOES NOT TRANSFER"
        print(f"\n  VERDICT: Pre-training {verdict}")

    Path("results").mkdir(exist_ok=True)
    with open("results/transfer_experiment.json", "w") as f:
        json.dump({'config': {'episodes': MAZE_EPISODES, 'train_time': train_time,
                              'solve_rate': stats.solve_rate}, 'results': results}, f, indent=2, default=str)
    print("Saved: results/transfer_experiment.json")


if __name__ == "__main__":
    main()
