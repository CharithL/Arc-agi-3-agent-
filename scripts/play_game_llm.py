"""Play one ARC-AGI-3 game with the LLM agent."""

import sys
sys.path.insert(0, "src")

from charith.llm_agent.agent import LLMAgent

game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 50
model = sys.argv[3] if len(sys.argv) > 3 else "gemma4:4b"

print(f"Game: {game_id}, Max actions: {max_actions}, Model: {model}")
agent = LLMAgent(model=model)
result = agent.play_game(game_id, max_actions=max_actions)
print(f"\nResult: {result}")
