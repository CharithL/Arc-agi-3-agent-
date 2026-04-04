"""Run meta-training on procedural games. Usage: uv run python scripts/meta_train.py"""
import sys
sys.path.insert(0, 'src')
from charith.transformer.meta_training import meta_train, MetaTrainingConfig

config = MetaTrainingConfig(
    n_episodes=1000,  # Start small, increase for full run
    max_steps_per_episode=50,
    lr=3e-4,
    n_layers=6,
    d_model=256,
    n_actions=8,
    log_interval=50,
)

print("CHARITH Path 4: Meta-Training Transformer on Procedural Games")
print(f"Episodes: {config.n_episodes}, Steps/ep: {config.max_steps_per_episode}")
print(f"Model: d={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")

model = meta_train(config)
print("\nDone.")
