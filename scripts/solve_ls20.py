"""
Solve ls20 level 1 by navigating the sprite to the target.

From the ASCII render we know:
- Sprite (color-12) starts at ~(45, 36) on the green playing field
- Target (color-0/1) is at ~(31, 21)
- The green field has an L-shape with a gap at rows 30-39, cols 25-33
- We need to navigate: UP along the right corridor, then LEFT across the top,
  then DOWN to reach the target

Strategy: move UP to row ~26 (top of green area), then LEFT to col ~21,
then DOWN to row ~31 to reach the target. Log every step.
"""
import sys
sys.path.insert(0, 'src')

import arc_agi
from arcengine import GameAction
import numpy as np


def find_c12(grid):
    """Find color-12 sprite centroid."""
    pos = list(zip(*np.where(grid == 12)))
    if pos:
        return (np.mean([r for r, c in pos]), np.mean([c for r, c in pos]))
    return None


def main():
    arcade = arc_agi.Arcade()
    env = arcade.make('ls20')
    frame = env.reset()
    grid = np.array(frame.frame[0])

    pos = find_c12(grid)
    print(f"Start: c12 at ({pos[0]:.0f}, {pos[1]:.0f})")
    print(f"Target: ~(31, 21)")
    print(f"Win levels: {frame.win_levels}\n")

    # Build navigation sequence
    # Phase 1: Go UP to top of right corridor (~row 26)
    # From row ~46, need to go up to row ~26 = 4 ups (each moves 5 cells)
    # Phase 2: Go LEFT across the top to col ~21
    # From col ~36, need to go left to col ~21 = 3 lefts
    # Phase 3: Go DOWN to target row ~31
    # From row ~26, need to go down to row ~31 = 1 down

    actions = []

    # Phase 1: UP to top
    for _ in range(6):
        actions.append(("UP", GameAction.ACTION1))

    # Phase 2: LEFT across top
    for _ in range(5):
        actions.append(("LEFT", GameAction.ACTION3))

    # Phase 3: DOWN to target
    for _ in range(3):
        actions.append(("DOWN", GameAction.ACTION2))

    # Phase 4: Fine-tune - try various positions around target
    # If we overshoot, adjust
    fine_tune = [
        ("UP", GameAction.ACTION1),
        ("LEFT", GameAction.ACTION3),
        ("DOWN", GameAction.ACTION2),
        ("RIGHT", GameAction.ACTION4),
        ("UP", GameAction.ACTION1),
        ("UP", GameAction.ACTION1),
        ("LEFT", GameAction.ACTION3),
        ("DOWN", GameAction.ACTION2),
        ("DOWN", GameAction.ACTION2),
        ("RIGHT", GameAction.ACTION4),
        ("RIGHT", GameAction.ACTION4),
        ("UP", GameAction.ACTION1),
        ("LEFT", GameAction.ACTION3),
        ("LEFT", GameAction.ACTION3),
    ]
    actions.extend(fine_tune)

    # Execute
    prev_levels = 0
    for i, (name, action_enum) in enumerate(actions):
        result = env.step(action_enum)
        grid = np.array(result.frame[0])
        pos = find_c12(grid)
        levels = result.levels_completed
        state = str(result.state)

        pos_str = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "???"

        # Check distance to target area
        if pos:
            dist = ((pos[0] - 31)**2 + (pos[1] - 21)**2)**0.5
        else:
            dist = 999

        flag = ""
        if levels > prev_levels:
            flag = " *** LEVEL COMPLETE! ***"
            prev_levels = levels
        if 'NOT_FINISHED' not in state:
            flag += f" [{state}]"

        print(f"  [{i:2d}] {name:5s} -> {pos_str:>10s}  dist={dist:5.1f}  lvls={levels}{flag}")

        if levels > prev_levels or ('NOT_FINISHED' not in state and 'GAME_OVER' in state):
            break

    # Show final scorecard
    sc = arcade.get_scorecard()
    print(f"\n{'='*60}")
    print(f"Final: levels_completed={result.levels_completed}")
    if hasattr(sc, 'score'):
        print(f"Score: {sc.score}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
