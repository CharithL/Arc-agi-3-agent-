"""
EXPERIMENT: Path 3.5 -- Does Mirror Model Improve Over Player Alone?

Protocol:
1. Load pre-trained Player checkpoint (do NOT retrain)
2. Train Mirror (phases A, B, C) on maze reality (~10 min)
3. Save Mirror checkpoint
4. Compare on 5 real ARC-AGI-3 games:
   - Condition A: Player alone
   - Condition B: Player + Mirror (dual agent)
   - Metric: prediction error (lower = better)

Usage:
    uv run python scripts/experiment_dual.py
"""
import sys
sys.path.insert(0, "src")

import torch
import numpy as np
import time
import json
from pathlib import Path

from charith.neural.world_model_net import WorldModelNet
from charith.neural.mirror_model import MirrorModel
from charith.neural.mirror_training import MirrorTrainer
from charith.neural.dual_agent import DualAgent
from charith.neural.state_encoder import D_INPUT
from charith.synthetic.maze_reality import MazeReality


# --- Config ---
PLAYER_CHECKPOINT = "checkpoints/maze_gru_level1.pt"
MIRROR_CHECKPOINT = "checkpoints/mirror_model.pt"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
PLAYER_HIDDEN_TOTAL = NUM_LAYERS * HIDDEN_SIZE  # 512
N_ACTIONS = 4

REAL_GAMES = ["ls20", "tr87", "g50t", "wa30", "re86"]
STEPS_PER_GAME = 50

# Mirror training config (smaller for speed -- ~10 min total)
PHASE_A_EPISODES = 200
PHASE_A_EPOCHS = 50
PHASE_B_EPISODES = 200
PHASE_B_EPOCHS = 30
PHASE_C_EPISODES = 500


def load_player() -> WorldModelNet:
    """Load pre-trained Player from checkpoint."""
    player = WorldModelNet(
        input_size=D_INPUT,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        n_actions=N_ACTIONS,
    )

    ckpt_path = Path(PLAYER_CHECKPOINT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Player checkpoint not found at {ckpt_path.absolute()}.\n"
            f"To create it, run:\n"
            f"  uv run python scripts/pretrain_maze.py\n"
            f"Or copy from Vast.ai:\n"
            f"  scp vast:~/charith-arc-agent/checkpoints/maze_gru_level1.pt checkpoints/"
        )

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    player.load_state_dict(state_dict)
    player.eval()
    print(f"Loaded Player from {ckpt_path}")
    return player


def train_mirror(player: WorldModelNet) -> MirrorModel:
    """Train Mirror model on maze reality using the frozen Player."""
    mirror = MirrorModel(
        player_hidden_total=PLAYER_HIDDEN_TOTAL,
        n_decoded_features=10,
        n_actions=N_ACTIONS,
        mirror_hidden=128,
    )

    reality = MazeReality(level=1, step_penalty=-0.05)

    trainer = MirrorTrainer(
        player=player,
        mirror=mirror,
        reality=reality,
        lr=1e-3,
        verbose=True,
    )

    stats = trainer.train_all(
        phase_a_episodes=PHASE_A_EPISODES,
        phase_a_epochs=PHASE_A_EPOCHS,
        phase_b_episodes=PHASE_B_EPISODES,
        phase_b_epochs=PHASE_B_EPOCHS,
        phase_c_episodes=PHASE_C_EPISODES,
    )

    print(f"\nMirror training complete:")
    print(f"  Phase A loss: {stats.phase_a_loss:.6f}")
    print(f"  Phase B loss: {stats.phase_b_loss:.6f}")
    print(f"  Phase C reward: {stats.phase_c_reward:.3f}")
    print(f"  Total episodes: {stats.total_episodes}")

    return mirror


def main():
    print("=" * 70)
    print("EXPERIMENT: Path 3.5 -- Player + Mirror vs Player Alone")
    print("=" * 70)

    # Step 1: Load Player
    print("\n[Step 1] Loading pre-trained Player...")
    try:
        player = load_player()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    # Step 2: Train Mirror
    print(f"\n[Step 2] Training Mirror (phases A/B/C)...")
    t0 = time.time()
    mirror = train_mirror(player)
    train_time = time.time() - t0
    print(f"\nMirror training took {train_time:.0f}s ({train_time / 60:.1f} min)")

    # Step 3: Save Mirror checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(mirror.state_dict(), MIRROR_CHECKPOINT)
    print(f"Saved Mirror to {MIRROR_CHECKPOINT}")

    # Step 4: Compare on real games
    print(f"\n{'=' * 70}")
    print(f"[Step 4] Comparing on {len(REAL_GAMES)} real ARC-AGI-3 games")
    print(f"{'=' * 70}\n")

    dual = DualAgent(player=player, mirror=mirror)

    results = {}
    for gid in REAL_GAMES:
        print(f"  {gid}...", end=" ", flush=True)
        try:
            # Condition A: Player alone
            pe_player = dual.evaluate_player_only(gid, n_steps=STEPS_PER_GAME)

            # Condition B: Player + Mirror
            pe_dual, diag = dual.evaluate_on_game(gid, n_steps=STEPS_PER_GAME)

            improvement = (pe_player - pe_dual) / max(pe_player, 1e-8) * 100

            results[gid] = {
                "player_only": pe_player,
                "player_mirror": pe_dual,
                "improvement_pct": improvement,
                "diagnostics": diag.summary(),
            }
            print(
                f"player={pe_player:.4f} dual={pe_dual:.4f} "
                f"imp={improvement:+.1f}% "
                f"(trust={diag.trust_count} exp={diag.explore_count} "
                f"ovr={diag.override_count} conf={diag.mean_confidence:.2f})"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            results[gid] = {"error": str(e)}

    # Step 5: Print comparison table and verdict
    print(f"\n{'=' * 70}")
    print("RESULTS: Player Alone vs Player + Mirror")
    print(f"{'=' * 70}\n")

    print(f"  {'Game':>6}  {'Player':>10}  {'Dual':>10}  {'Improve':>10}  {'Trust%':>8}")
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    improvements = []
    for gid, r in results.items():
        if "error" not in r:
            trust_pct = r["diagnostics"]["trust_rate"] * 100
            print(
                f"  {gid:>6}  {r['player_only']:10.4f}  "
                f"{r['player_mirror']:10.4f}  "
                f"{r['improvement_pct']:+9.1f}%  "
                f"{trust_pct:7.0f}%"
            )
            improvements.append(r["improvement_pct"])

    if improvements:
        avg_imp = np.mean(improvements)
        games_helped = sum(1 for x in improvements if x > 0)
        print(f"\n  Average improvement: {avg_imp:+.1f}%")
        print(f"  Games helped: {games_helped}/{len(improvements)}")

        if avg_imp > 0:
            verdict = "MIRROR HELPS -- dual agent outperforms Player alone"
        elif avg_imp > -5:
            verdict = "NEUTRAL -- Mirror neither helps nor hurts significantly"
        else:
            verdict = "MIRROR HURTS -- Player alone is better"

        print(f"\n  VERDICT: {verdict}")
    else:
        print("\n  No games completed successfully.")

    # Save results
    Path("results").mkdir(exist_ok=True)
    output = {
        "config": {
            "player_checkpoint": PLAYER_CHECKPOINT,
            "mirror_training_time_s": train_time,
            "phase_a_episodes": PHASE_A_EPISODES,
            "phase_b_episodes": PHASE_B_EPISODES,
            "phase_c_episodes": PHASE_C_EPISODES,
            "games": REAL_GAMES,
            "steps_per_game": STEPS_PER_GAME,
        },
        "results": {},
    }
    # Serialize results (convert non-serializable types)
    for gid, r in results.items():
        if "error" in r:
            output["results"][gid] = {"error": r["error"]}
        else:
            output["results"][gid] = {
                "player_only": r["player_only"],
                "player_mirror": r["player_mirror"],
                "improvement_pct": r["improvement_pct"],
                "diagnostics": r["diagnostics"],
            }

    with open("results/dual_experiment.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: results/dual_experiment.json")


if __name__ == "__main__":
    main()
