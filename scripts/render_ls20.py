"""
Render ls20 grid as ASCII art to visualize the maze structure.
Then systematically probe movement in all directions to map traversable paths.
"""
import sys
sys.path.insert(0, 'src')

import arc_agi
from arcengine import GameAction
import numpy as np

def render_grid(grid, title=""):
    """Render grid as compact ASCII. Each color gets a symbol."""
    symbols = {
        0: '.', 1: 'B', 2: 'R', 3: '#', 4: ' ', 5: 'W',
        6: 'M', 7: 'O', 8: 'C', 9: '@', 10: 'X', 11: '=', 12: '$'
    }
    if title:
        print(f"\n{title}")
    rows, cols = grid.shape
    # Print column headers every 10
    header = "    " + "".join(f"{c%10}" if c % 5 == 0 else " " for c in range(cols))
    print(header)
    for r in range(rows):
        prefix = f"{r:3d} "
        row_str = "".join(symbols.get(int(grid[r, c]), '?') for c in range(cols))
        print(f"{prefix}{row_str}")


def find_color_blocks(grid):
    """Find contiguous regions of each color and their positions."""
    unique_colors = np.unique(grid)
    print(f"\nColor legend:")
    symbols = {
        0: '.', 1: 'B', 2: 'R', 3: '#', 4: ' ', 5: 'W',
        6: 'M', 7: 'O', 8: 'C', 9: '@', 10: 'X', 11: '=', 12: '$'
    }
    for c in sorted(unique_colors):
        count = int(np.sum(grid == c))
        positions = list(zip(*np.where(grid == c)))
        rows = [r for r, _ in positions]
        cols = [c_ for _, c_ in positions]
        sym = symbols.get(int(c), '?')
        print(f"  Color {c:2d} ({sym}): {count:5d} cells, "
              f"rows [{min(rows)}-{max(rows)}], cols [{min(cols)}-{max(cols)}]")


def probe_movement(env, initial_grid):
    """Systematically test movement from current position."""
    ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}

    print(f"\n--- Movement Probe ---")
    for action_val in [1, 2, 3, 4]:
        # Reset to initial state by creating new env
        arcade2 = arc_agi.Arcade()
        env2 = arcade2.make('ls20')
        frame = env2.reset()
        grid_before = np.array(frame.frame[0])

        # Take one action
        result = env2.step(GameAction[f'ACTION{action_val}'])
        grid_after = np.array(result.frame[0])

        diff = (grid_before != grid_after)
        n_changed = int(np.sum(diff))
        changed_cells = list(zip(*np.where(diff)))

        if n_changed > 0:
            # Find what colors changed where
            from_colors = set()
            to_colors = set()
            for r, c in changed_cells:
                from_colors.add(int(grid_before[r, c]))
                to_colors.add(int(grid_after[r, c]))

            # Find the sprite's movement
            c9_before = set(zip(*np.where(grid_before == 9)))
            c9_after = set(zip(*np.where(grid_after == 9)))
            c12_before = set(zip(*np.where(grid_before == 12)))
            c12_after = set(zip(*np.where(grid_after == 12)))

            new_c9 = c9_after - c9_before
            lost_c9 = c9_before - c9_after
            new_c12 = c12_after - c12_before
            lost_c12 = c12_before - c12_after

            print(f"\n  {ACTION_NAMES[action_val]:5s}: {n_changed} cells changed")
            if new_c12:
                new_rows = [r for r, c in new_c12]
                new_cols = [c for r, c in new_c12]
                lost_rows = [r for r, c in lost_c12]
                lost_cols = [c for r, c in lost_c12]
                if new_rows and lost_rows:
                    dr = int(np.mean(new_rows) - np.mean(lost_rows))
                    dc = int(np.mean(new_cols) - np.mean(lost_cols))
                    print(f"    Color-12 moved: ({dr:+d}, {dc:+d})")
            if new_c9:
                new_rows = [r for r, c in new_c9]
                lost_rows = [r for r, c in lost_c9] if lost_c9 else new_rows
                new_cols = [c for r, c in new_c9]
                lost_cols = [c for r, c in lost_c9] if lost_c9 else new_cols
                if new_rows and lost_rows:
                    dr = int(np.mean(new_rows) - np.mean(lost_rows))
                    dc = int(np.mean(new_cols) - np.mean(lost_cols))
                    print(f"    Color-9 moved: ({dr:+d}, {dc:+d})")
        else:
            print(f"\n  {ACTION_NAMES[action_val]:5s}: BLOCKED (no cells changed)")


def navigate_and_log(env, actions, max_actions=50):
    """Execute a sequence of actions, logging position after each."""
    print(f"\n--- Navigation Sequence ({len(actions)} actions) ---")

    for i, (name, action_enum) in enumerate(actions[:max_actions]):
        result = env.step(action_enum)
        grid = np.array(result.frame[0])
        levels = result.levels_completed
        state = str(result.state)

        # Find color-12 sprite position (the clean controllable)
        c12_pos = list(zip(*np.where(grid == 12)))
        if c12_pos:
            rows = [r for r, c in c12_pos]
            cols = [c for r, c in c12_pos]
            center = (np.mean(rows), np.mean(cols))
        else:
            center = (0, 0)

        flag = ""
        if levels > 0:
            flag = " *** LEVEL COMPLETE ***"
        if 'NOT_FINISHED' not in state:
            flag += f" [{state}]"

        print(f"  [{i:2d}] {name:5s} -> c12=({center[0]:.0f},{center[1]:.0f}) lvls={levels}{flag}")

        if 'NOT_FINISHED' not in state and 'GAME_OVER' in state:
            break

    return grid


def main():
    arcade = arc_agi.Arcade()
    env = arcade.make('ls20')
    frame = env.reset()
    grid = np.array(frame.frame[0])

    print("=" * 70)
    print("LS20 MAZE VISUALIZATION")
    print("=" * 70)
    print(f"Grid: {grid.shape}")
    print(f"Available actions: {frame.available_actions}")
    print(f"Win levels: {frame.win_levels}")

    find_color_blocks(grid)
    render_grid(grid, "Initial Grid (4=space, 3=#maze, 5=W walls, 9=@player, 12=$paired)")

    # Probe movement in each direction
    probe_movement(env, grid)

    # Now try a directed navigation: move in each direction multiple times
    # to map the traversable space
    print(f"\n{'='*70}")
    print("EXPLORATION: Move in each direction as far as possible")
    print(f"{'='*70}")

    # Go UP as far as possible, then DOWN back, then LEFT, then RIGHT
    exploration = (
        [("UP", GameAction.ACTION1)] * 10 +
        [("DOWN", GameAction.ACTION2)] * 20 +
        [("LEFT", GameAction.ACTION3)] * 10 +
        [("RIGHT", GameAction.ACTION4)] * 20 +
        [("UP", GameAction.ACTION1)] * 10
    )
    final_grid = navigate_and_log(env, exploration, max_actions=70)

    # Render final state
    render_grid(final_grid, "After exploration:")


if __name__ == "__main__":
    main()
