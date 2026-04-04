"""Play one ARC-AGI-3 game with the LLM agent."""

import sys
sys.path.insert(0, "src")

from charith.llm_agent.agent import LLMAgent

game_id = sys.argv[1] if len(sys.argv) > 1 else "ls20"
max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 50

agent = LLMAgent()
result = agent.play_game(game_id, max_actions=max_actions)
print(f"\nResult: {result}")
