"""Play all mock games and report aggregate results."""
import sys
sys.path.insert(0, 'src')
from charith.agent import CHARITHAgent

GAMES = ["deterministic_movement", "hidden_goal",
         "context_dependent", "multi_level"]

results = {}
for game_id in GAMES:
    print(f"\nPlaying {game_id}...")
    agent = CHARITHAgent()
    scorecard = agent.play_game(game_id, max_actions=500)
    results[game_id] = {
        'levels': agent._levels_completed,
        'actions': agent._total_actions,
        'rules': agent.world_model.get_rule_count(),
        'accuracy': agent.world_model.get_accuracy(),
        'expansions': agent.ontology._expansion_count,
    }

print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print("="*60)
for gid, r in results.items():
    print(f"{gid:25s}: {r['levels']} levels, {r['actions']:4d} actions, "
          f"{r['accuracy']:.1%} accuracy, {r['expansions']} expansions")
