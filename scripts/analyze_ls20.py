"""
Analyze ls20 to understand its goal structure.

Strategy:
1. Play many random actions and record all state changes
2. Detect the controllable object and track its position
3. Look for special cells (unique colors, patterns) that might be goals
4. Check if the score/level changes correlate with controllable proximity
"""
import sys
sys.path.insert(0, 'src')

import arc_agi
from arcengine import GameAction
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.perception.object_tracker import ObjectTracker
from collections import Counter

def analyze_game(game_id: str = "ls20", max_actions: int = 500):
    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)

    perception = CoreKnowledgePerception()
    tracker = ObjectTracker()

    # Initial observation
    initial = env.reset()
    grid = np.array(initial.frame[0])
    avail = initial.available_actions
    win_levels = initial.win_levels

    print(f"{'='*70}")
    print(f"ANALYSIS: {game_id}")
    print(f"{'='*70}")
    print(f"Grid shape: {grid.shape}")
    print(f"Win levels: {win_levels}")
    print(f"Available actions: {avail}")
    print(f"Unique colors: {sorted(np.unique(grid))}")
    color_counts = dict(zip(*np.unique(grid, return_counts=True)))
    print(f"Color cell counts:")
    for c in sorted(color_counts.keys()):
        pct = color_counts[c] / grid.size * 100
        print(f"  Color {c:2d}: {color_counts[c]:5d} cells ({pct:5.1f}%)")

    # Map the initial grid — what's where?
    percept = perception.perceive(grid)
    print(f"\n--- OBJECTS (sorted by size) ---")
    print(f"Total objects: {percept.object_count}")
    print(f"Background color: {percept.background_color}")

    for obj in sorted(percept.objects, key=lambda o: o.size, reverse=True)[:20]:
        print(f"  Obj {obj.object_id:3d}: color={obj.color:2d}, size={obj.size:4d}, "
              f"centroid=({obj.centroid[0]:5.1f},{obj.centroid[1]:5.1f}), "
              f"bbox=({obj.bbox[0]:2d},{obj.bbox[1]:2d})-({obj.bbox[2]:2d},{obj.bbox[3]:2d}), "
              f"shape={obj.width}x{obj.height}")

    # Look for singleton-color objects (potential goals)
    color_obj_counts = Counter(o.color for o in percept.objects)
    print(f"\n--- SINGLETON-COLOR OBJECTS (potential goals) ---")
    for color, count in color_obj_counts.items():
        if count == 1:
            obj = next(o for o in percept.objects if o.color == color)
            print(f"  SINGLETON: color={color}, size={obj.size}, "
                  f"centroid=({obj.centroid[0]:.0f},{obj.centroid[1]:.0f})")

    # Play actions and track movements
    print(f"\n--- ACTION EFFECTS (first 50 ticks) ---")
    prev_grid = grid.copy()
    prev_percept = percept
    movements = {}  # action -> list of (color, displacement)
    controllable_history = []
    level_completions = []

    for i in range(max_actions):
        # Cycle through available actions
        action_val = avail[i % len(avail)]
        action_enum = GameAction[f'ACTION{action_val}']
        result = env.step(action_enum)

        new_grid = np.array(result.frame[0])
        perception.reset() if i == 0 else None  # Don't reset perception mid-game
        new_percept = perception.perceive(new_grid)

        # Track what moved
        matched = tracker.match(prev_percept.objects, new_percept.objects)
        for prev_id, curr_id in matched:
            prev_obj = next((o for o in prev_percept.objects if o.object_id == prev_id), None)
            curr_obj = next((o for o in new_percept.objects if o.object_id == curr_id), None)
            if prev_obj and curr_obj:
                dr = round(curr_obj.centroid[0] - prev_obj.centroid[0])
                dc = round(curr_obj.centroid[1] - prev_obj.centroid[1])
                if abs(dr) > 0 or abs(dc) > 0:
                    movements.setdefault(action_val, []).append(
                        (prev_obj.color, (dr, dc))
                    )
                    if i < 50:
                        print(f"  Tick {i:3d}: Action {action_val} -> "
                              f"color={prev_obj.color} moved ({dr:+d},{dc:+d}) "
                              f"from ({prev_obj.centroid[0]:.0f},{prev_obj.centroid[1]:.0f}) "
                              f"to ({curr_obj.centroid[0]:.0f},{curr_obj.centroid[1]:.0f})")
                    controllable_history.append({
                        'tick': i,
                        'action': action_val,
                        'color': prev_obj.color,
                        'from': prev_obj.centroid,
                        'to': curr_obj.centroid,
                    })

        # Check level completion
        new_levels = result.levels_completed
        state = str(result.state)
        if new_levels > len(level_completions):
            pos = controllable_history[-1]['to'] if controllable_history else None
            level_completions.append({
                'tick': i,
                'levels': new_levels,
                'controllable_position': pos,
            })
            print(f"\n  *** LEVEL {new_levels} COMPLETE at tick {i}! ***")
            if pos:
                print(f"      Controllable was at: ({pos[0]:.0f}, {pos[1]:.0f})")

            # Show what the grid looks like at level completion
            new_percept_at_complete = perception.perceive(new_grid)
            print(f"      Objects: {new_percept_at_complete.object_count}")

        if 'NOT_FINISHED' not in state:
            print(f"\n  Game ended: {state} at tick {i}")
            break

        prev_grid = new_grid.copy()
        prev_percept = new_percept

    # Summary
    print(f"\n{'='*70}")
    print(f"ACTION → MOVEMENT SUMMARY")
    print(f"{'='*70}")
    for action_val in sorted(movements.keys()):
        moves = movements[action_val]
        colors_moved = Counter(m[0] for m in moves)
        displacements = Counter(m[1] for m in moves)
        print(f"\n  Action {action_val}:")
        print(f"    Colors that moved: {dict(colors_moved)}")
        print(f"    Displacements: {dict(displacements)}")

    if controllable_history:
        all_colors = Counter(m['color'] for m in controllable_history)
        print(f"\n--- CONTROLLABLE OBJECT ---")
        print(f"  Colors that respond to actions: {dict(all_colors)}")
        positions = np.array([m['to'] for m in controllable_history])
        print(f"  Position range: rows [{positions[:,0].min():.0f}-{positions[:,0].max():.0f}], "
              f"cols [{positions[:,1].min():.0f}-{positions[:,1].max():.0f}]")

    if level_completions:
        print(f"\n--- LEVEL COMPLETIONS ---")
        for lc in level_completions:
            print(f"  Level {lc['levels']} at tick {lc['tick']}: "
                  f"position={lc['controllable_position']}")
    else:
        print(f"\n--- NO LEVELS COMPLETED in {max_actions} actions ---")

if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
    actions = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    analyze_game(game, actions)
