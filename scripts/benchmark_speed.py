"""Benchmark agent speed (ticks per second)."""
import sys
import time
sys.path.insert(0, 'src')
from charith.agent import CHARITHAgent

game = sys.argv[1] if len(sys.argv) > 1 else "deterministic_movement"
max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

agent = CHARITHAgent()
start = time.perf_counter()
agent.play_game(game, max_actions=max_actions)
elapsed = time.perf_counter() - start

fps = agent._total_actions / max(elapsed, 1e-6)
print(f"Game: {game}")
print(f"Actions: {agent._total_actions}")
print(f"Time: {elapsed:.3f}s")
print(f"Speed: {fps:.0f} ticks/second")
print(f"Target: 500+ ticks/second")
