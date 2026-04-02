"""Play a single mock game with verbose output."""
import sys
sys.path.insert(0, 'src')
from charith.agent import CHARITHAgent

game = sys.argv[1] if len(sys.argv) > 1 else "deterministic_movement"
max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 500

agent = CHARITHAgent()
scorecard = agent.play_game(game, max_actions=max_actions)

print(f"\n{'='*60}")
print(f"Game: {game}")
print(f"Levels completed: {agent._levels_completed}")
print(f"Total actions: {agent._total_actions}")
print(f"Rules learned: {agent.world_model.get_rule_count()}")
print(f"Prediction accuracy: {agent.world_model.get_accuracy():.2%}")
print(f"Ontology expansions: {agent.ontology._expansion_count}")
print(f"Exploring: {agent.explorer.is_exploring()}")
best_goal = agent.goal_discovery.get_best_hypothesis()
if best_goal:
    print(f"Best goal hypothesis: {best_goal.description} ({best_goal.confidence:.2f})")
print(f"{'='*60}")
print(scorecard)
