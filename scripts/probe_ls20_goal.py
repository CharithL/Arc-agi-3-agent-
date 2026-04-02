"""
Probe ls20's goal by moving the controllable object to suspected targets.

From analyze_ls20.py we know:
- Controllable: color-9 (maroon) + color-12 pair, starting at ~(48,36)
- Suspected goal: color-0 cells at (32,21) and color-1 cells at (32,20)/(33,21)
- Movement: ACTION1=up(-5,0), ACTION2=down(+5,0), ACTION3=left(0,-5), ACTION4=right(0,+5)

Strategy:
1. Move up toward row 32: need ~3x ACTION1 (from 48 to 33)
2. Move left toward col 21: need ~3x ACTION3 (from 36 to 21)
3. Log everything — watch for level_complete, score changes, grid changes
4. If (32,21) doesn't trigger, try nearby cells and other actions
"""
import sys
sys.path.insert(0, 'src')

import arc_agi
from arcengine import GameAction
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception, ObjectnessPrior
from charith.perception.object_tracker import ObjectTracker

def probe_goal():
    arcade = arc_agi.Arcade()
    env = arcade.make('ls20')
    perception = CoreKnowledgePerception()
    tracker = ObjectTracker()

    initial = env.reset()
    grid = np.array(initial.frame[0])
    avail = initial.available_actions

    print(f"Grid: {grid.shape}, Available: {avail}")
    print(f"Win levels: {initial.win_levels}")

    # Find initial controllable position
    percept = perception.perceive(grid)
    ctrl_objs = [o for o in percept.objects if o.color == 9]
    if ctrl_objs:
        ctrl = ctrl_objs[0]
        print(f"\nControllable (color-9): centroid=({ctrl.centroid[0]:.0f},{ctrl.centroid[1]:.0f}), "
              f"size={ctrl.size}")
    else:
        print("WARNING: No color-9 object found!")
        ctrl = None

    # Find target candidates
    target_objs_0 = [o for o in percept.objects if o.color == 0]
    target_objs_1 = [o for o in percept.objects if o.color == 1]
    print(f"\nTarget candidates:")
    for o in target_objs_0:
        print(f"  color-0: centroid=({o.centroid[0]:.1f},{o.centroid[1]:.1f}), size={o.size}, size={o.size}")
    for o in target_objs_1:
        print(f"  color-1: centroid=({o.centroid[0]:.1f},{o.centroid[1]:.1f}), size={o.size}, size={o.size}")

    # ================================================================
    # PHASE 1: Move up (toward row ~32)
    # From row 48, need about 3 ups (48 -> 43 -> 38 -> 33)
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Moving UP toward row 32")
    print(f"{'='*60}")

    prev_grid = grid.copy()
    prev_percept = percept

    # Adaptive: compute how many steps needed to reach target (32, 21)
    target_row, target_col = 32.0, 21.0
    if ctrl:
        start_row, start_col = ctrl.centroid
    else:
        start_row, start_col = 48.0, 36.0

    # Each step = 5 cells
    rows_to_go = target_row - start_row   # negative = need to go up
    cols_to_go = target_col - start_col   # negative = need to go left

    ups = max(0, int(-rows_to_go / 5))     # ACTION1
    downs = max(0, int(rows_to_go / 5))    # ACTION2
    lefts = max(0, int(-cols_to_go / 5))   # ACTION3
    rights = max(0, int(cols_to_go / 5))   # ACTION4

    print(f"\n  Start: ({start_row:.0f},{start_col:.0f}) -> Target: ({target_row:.0f},{target_col:.0f})")
    print(f"  Need: {ups} ups, {downs} downs, {lefts} lefts, {rights} rights")

    action_sequence = (
        [GameAction.ACTION1] * ups +
        [GameAction.ACTION3] * lefts +
        [GameAction.ACTION2] * downs +
        [GameAction.ACTION4] * rights +
        # Fine-tune around target: try all 4 directions
        [GameAction.ACTION1] * 2 +
        [GameAction.ACTION2] * 2 +
        [GameAction.ACTION3] * 2 +
        [GameAction.ACTION4] * 2 +
        # More fine-tuning
        [GameAction.ACTION1] * 1 +
        [GameAction.ACTION3] * 1 +
        [GameAction.ACTION2] * 1 +
        [GameAction.ACTION4] * 1
    )

    for i, action in enumerate(action_sequence):
        result = env.step(action)
        new_grid = np.array(result.frame[0])
        new_percept = perception.perceive(new_grid)

        # Find controllable position
        new_ctrl = None
        matched = tracker.match(prev_percept.objects, new_percept.objects)
        for pid, cid in matched:
            prev_obj = next((o for o in prev_percept.objects if o.object_id == pid), None)
            curr_obj = next((o for o in new_percept.objects if o.object_id == cid), None)
            if prev_obj and curr_obj and prev_obj.color == 9:
                dr = round(curr_obj.centroid[0] - prev_obj.centroid[0])
                dc = round(curr_obj.centroid[1] - prev_obj.centroid[1])
                new_ctrl = curr_obj
                move_str = f"d=({dr:+d},{dc:+d})" if (abs(dr) > 0 or abs(dc) > 0) else "BLOCKED"
                break

        # Check state
        levels = result.levels_completed
        state = str(result.state)
        changed = int(np.sum(prev_grid != new_grid))

        pos_str = f"({new_ctrl.centroid[0]:.0f},{new_ctrl.centroid[1]:.0f})" if new_ctrl else "?"
        action_name = action.name

        # Distance to targets
        if new_ctrl:
            dist_0 = ((new_ctrl.centroid[0] - 31.7)**2 + (new_ctrl.centroid[1] - 21.3)**2)**0.5
            dist_1a = ((new_ctrl.centroid[0] - 32.0)**2 + (new_ctrl.centroid[1] - 20.0)**2)**0.5
        else:
            dist_0 = dist_1a = 999

        flag = ""
        if levels > 0:
            flag = " *** LEVEL COMPLETE! ***"
        if 'NOT_FINISHED' not in state:
            flag = f" *** GAME STATE: {state} ***"
        if changed > 100:
            flag += f" [BIG CHANGE: {changed} cells]"

        print(f"  [{i:2d}] {action_name:8s} -> pos={pos_str:>10s} "
              f"{move_str if new_ctrl else '':>10s} | "
              f"dist_to_0={dist_0:5.1f} dist_to_1={dist_1a:5.1f} | "
              f"changed={changed:4d} | lvls={levels}{flag}")

        if levels > 0:
            print(f"\n  !!! LEVEL COMPLETED at tick {i}!")
            print(f"      Controllable was at: {pos_str}")
            print(f"      Distance to color-0: {dist_0:.1f}")
            print(f"      Distance to color-1: {dist_1a:.1f}")
            # Don't break — keep going to see next level setup

        prev_grid = new_grid.copy()
        prev_percept = new_percept

    # ================================================================
    # PHASE 2: If no level complete yet, try sitting on each target cell
    # ================================================================
    if result.levels_completed == 0:
        print(f"\n{'='*60}")
        print("PHASE 2: No level complete yet. Grid status around targets:")
        print(f"{'='*60}")

        # Show what's at the target locations in current grid
        new_grid = np.array(result.frame[0])
        print(f"\n  Grid values near (32,21):")
        for r in range(29, 36):
            row_vals = [f"{new_grid[r,c]:2d}" for c in range(18, 25)]
            marker = " <-- target row" if r in [31, 32, 33] else ""
            print(f"    row {r}: [{' '.join(row_vals)}]{marker}")

        # Find where controllable is now
        curr_percept = perception.perceive(new_grid)
        ctrl_now = [o for o in curr_percept.objects if o.color == 9]
        if ctrl_now:
            print(f"\n  Controllable now at: ({ctrl_now[0].centroid[0]:.0f},{ctrl_now[0].centroid[1]:.0f})")

        # Show ALL unique singleton objects (rarer = more likely goal)
        print(f"\n  All small objects (<= 10 cells):")
        for obj in sorted(curr_percept.objects, key=lambda o: o.size):
            if obj.size <= 10:
                print(f"    color={obj.color:2d}, size={obj.size:3d}, "
                      f"centroid=({obj.centroid[0]:.0f},{obj.centroid[1]:.0f}), "
                      f"cells={sorted(obj.cells)[:5]}{'...' if len(obj.cells) > 5 else ''}")

    # Final scorecard
    sc = arcade.get_scorecard()
    print(f"\n{'='*60}")
    print(f"SCORECARD:")
    if hasattr(sc, 'score'):
        print(f"  Score: {sc.score}")
    print(f"  Levels completed: {result.levels_completed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    probe_goal()
