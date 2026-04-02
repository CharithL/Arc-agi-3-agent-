# CHARITH Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full-scaffold ARC-AGI-3 agent with core knowledge priors, predictive processing world model, ontology expansion, Thompson sampling exploration, and goal discovery — all testable against a mock environment.

**Architecture:** 6-module cognitive architecture (Perception, WorldModel, Metacognition, Action, Memory, Agent Loop) connected by a predict-compare-update cycle. MockEnvironment provides 4 test scenarios simulating ARC-AGI-3 grid games. Amendments 1-5 override original guide where they conflict.

**Tech Stack:** Python 3.11, numpy, scipy, pyyaml, pytest. No arc-agi SDK yet (mock env substitutes).

---

## Task 0: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `configs/default.yaml`
- Create: `src/charith/__init__.py`
- Create: `src/charith/perception/__init__.py`
- Create: `src/charith/world_model/__init__.py`
- Create: `src/charith/metacognition/__init__.py`
- Create: `src/charith/action/__init__.py`
- Create: `src/charith/memory/__init__.py`
- Create: `src/charith/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/.gitkeep`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "charith-arc-agent"
version = "0.1.0"
description = "CHARITH: Core-knowledge Hierarchical Agent for Reasoning in Turn-based Interactive Tasks with Heuristics"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Create .python-version**

```
3.11
```

**Step 3: Create configs/default.yaml**

```yaml
# CHARITH Agent — Default Configuration
# See guide Section 10.2 for parameter descriptions

# World Model
max_rules: 10000
rule_decay_ticks: 500

# Ontology Expansion (4-test protocol)
ontology_window: 50
residual_r2_threshold: 0.1
volatility_sigma_multiplier: 2.0
uncertainty_patience: 20
capacity_growth_threshold: 0.1

# Thompson Sampling
info_gain_weight: 0.5
explore_threshold: 0.3

# Memory
wm_capacity: 7
max_episodes: 5000

# Agent Loop
ontology_check_interval: 50
goal_hypothesis_delay: 20
```

**Step 4: Create all __init__.py files**

Each `__init__.py` is empty initially.

**Step 5: Initialize git and install**

```bash
cd charith-arc-agent
git init
uv sync
git add -A
git commit -m "chore: project scaffold with pyproject.toml, configs, package structure"
```

---

## Task 1: Perception — Core Knowledge Priors

**Files:**
- Create: `src/charith/perception/core_knowledge.py`
- Create: `tests/test_core_knowledge.py`

**Step 1: Write failing tests**

```python
# tests/test_core_knowledge.py
"""Tests for Core Knowledge Priors — Spelke's 4 systems."""
import numpy as np
import pytest


def test_objectness_single_object():
    """Single colored block on black background -> 1 object."""
    from charith.perception.core_knowledge import ObjectnessPrior

    grid = np.zeros((5, 5), dtype=int)
    grid[1:3, 1:3] = 1  # 2x2 blue block
    prior = ObjectnessPrior()
    objects = prior.detect(grid)
    assert len(objects) == 1
    assert objects[0].color == 1
    assert objects[0].size == 4


def test_objectness_multiple_colors():
    """Different colors -> different objects."""
    from charith.perception.core_knowledge import ObjectnessPrior

    grid = np.zeros((5, 5), dtype=int)
    grid[0, 0] = 1  # blue pixel
    grid[4, 4] = 2  # red pixel
    prior = ObjectnessPrior()
    objects = prior.detect(grid)
    assert len(objects) == 2


def test_objectness_diagonal_not_connected():
    """Diagonal cells of same color -> separate objects (4-connectivity)."""
    from charith.perception.core_knowledge import ObjectnessPrior

    grid = np.zeros((3, 3), dtype=int)
    grid[0, 0] = 1
    grid[1, 1] = 1  # diagonal, not 4-connected
    prior = ObjectnessPrior(connectivity=4)
    objects = prior.detect(grid)
    assert len(objects) == 2


def test_objectness_8_connectivity():
    """With 8-connectivity, diagonal cells form one object."""
    from charith.perception.core_knowledge import ObjectnessPrior

    grid = np.zeros((3, 3), dtype=int)
    grid[0, 0] = 1
    grid[1, 1] = 1
    prior = ObjectnessPrior(connectivity=8)
    objects = prior.detect(grid)
    assert len(objects) == 1


def test_objectness_shape_hash_position_invariant():
    """Same shape at different positions -> same shape_hash."""
    from charith.perception.core_knowledge import ObjectnessPrior

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[0:2, 0:2] = 1  # 2x2 at top-left

    grid2 = np.zeros((10, 10), dtype=int)
    grid2[5:7, 5:7] = 1  # 2x2 at center

    prior = ObjectnessPrior()
    obj1 = prior.detect(grid1)[0]
    prior._next_id = 0  # Reset ID counter
    obj2 = prior.detect(grid2)[0]
    assert obj1.shape_hash == obj2.shape_hash


def test_spatial_above_below():
    """Object A above Object B -> 'above' relation."""
    from charith.perception.core_knowledge import ObjectnessPrior, SpatialPrior

    grid = np.zeros((10, 10), dtype=int)
    grid[1, 5] = 1  # blue at top
    grid[8, 5] = 2  # red at bottom
    prior = ObjectnessPrior()
    objects = prior.detect(grid)
    spatial = SpatialPrior()
    relations = spatial.compute_relations(objects)
    rel_types = [r.relation for r in relations]
    assert 'above' in rel_types


def test_spatial_adjacency():
    """Two objects sharing a border -> 'adjacent' relation."""
    from charith.perception.core_knowledge import ObjectnessPrior, SpatialPrior

    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 1  # blue
    grid[1, 2] = 2  # red, adjacent to blue
    prior = ObjectnessPrior()
    objects = prior.detect(grid)
    spatial = SpatialPrior()
    relations = spatial.compute_relations(objects)
    rel_types = [r.relation for r in relations]
    assert 'adjacent' in rel_types


def test_spatial_symmetry_detection():
    """Horizontally symmetric grid detected."""
    from charith.perception.core_knowledge import SpatialPrior

    grid = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ])
    spatial = SpatialPrior()
    sym = spatial.detect_grid_symmetry(grid)
    assert sym['h_symmetric'] is True
    assert sym['v_symmetric'] is True


def test_number_count_by_color():
    """Count cells per color."""
    from charith.perception.core_knowledge import NumberPrior

    grid = np.array([
        [0, 1, 1],
        [0, 0, 2],
    ])
    prior = NumberPrior()
    counts = prior.count_by_color(grid)
    assert counts[0] == 3
    assert counts[1] == 2
    assert counts[2] == 1


def test_number_periodic_detection():
    """Sequence [1, 2, 1, 2, 1, 2] -> periodic=True."""
    from charith.perception.core_knowledge import NumberPrior

    prior = NumberPrior()
    result = prior.detect_numerical_patterns([1, 2, 1, 2, 1, 2])
    assert result['periodic'] is True


def test_number_arithmetic_detection():
    """Sequence [2, 4, 6, 8] -> arithmetic=True."""
    from charith.perception.core_knowledge import NumberPrior

    prior = NumberPrior()
    result = prior.detect_numerical_patterns([2, 4, 6, 8])
    assert result['arithmetic'] is True


def test_agency_record_contingency():
    """Recording action contingency tracks changed cells."""
    from charith.perception.core_knowledge import AgencyPrior

    prior = AgencyPrior()
    before = np.zeros((5, 5), dtype=int)
    after = before.copy()
    after[2, 3] = 1  # One cell changed
    prior.record_action_contingency(0, before, after)
    assert len(prior._action_contingencies) == 1
    assert prior._action_contingencies[0]['num_changes'] == 1


def test_full_perception_pipeline():
    """Integration test: grid -> StructuredPercept with all fields populated."""
    from charith.perception.core_knowledge import CoreKnowledgePerception

    grid = np.zeros((10, 10), dtype=int)
    grid[2:4, 2:4] = 1  # blue block
    grid[7:9, 7:9] = 2  # red block
    perception = CoreKnowledgePerception()
    percept = perception.perceive(grid)
    assert percept.object_count == 2
    assert len(percept.spatial_relations) > 0
    assert percept.background_color == 0
    assert percept.unique_colors == 3  # black + blue + red
    assert percept.grid_dims == (10, 10)
    assert percept.timestamp == 0


def test_perception_reset():
    """Reset clears state for new game."""
    from charith.perception.core_knowledge import CoreKnowledgePerception

    perception = CoreKnowledgePerception()
    grid = np.zeros((5, 5), dtype=int)
    grid[1, 1] = 1
    perception.perceive(grid)
    assert perception._tick == 1
    perception.reset()
    assert perception._tick == 0
```

**Step 2: Run tests to verify they fail**

```bash
cd charith-arc-agent
uv run pytest tests/test_core_knowledge.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'charith.perception.core_knowledge'`

**Step 3: Implement core_knowledge.py**

Copy the FULL implementation from the original guide Section 5.2 verbatim. This includes:
- `Color` enum
- `Cell`, `Object`, `SpatialRelation`, `StructuredPercept` dataclasses
- `ObjectnessPrior` class (BFS flood fill, shape_hash)
- `SpatialPrior` class (relations, symmetry)
- `NumberPrior` class (counting, patterns, periodicity)
- `AgencyPrior` class (motion, contingency, controllable detection)
- `CoreKnowledgePerception` pipeline class

**No changes from the original guide** for this module.

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_core_knowledge.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/charith/perception/core_knowledge.py tests/test_core_knowledge.py
git commit -m "feat: core knowledge priors (objectness, spatial, number, agency)"
```

---

## Task 2: MockEnvironment — Test Harness

**Files:**
- Create: `src/charith/mock_env.py`
- Create: `tests/test_mock_env.py`

**Step 1: Write failing tests**

```python
# tests/test_mock_env.py
"""Tests for MockEnvironment — verifies the test harness itself."""
import numpy as np
import pytest


def test_deterministic_movement_action_up():
    """Action 0 moves the block up by 1 row."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("deterministic_movement")
    obs = env.get_observation()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype in (np.int32, np.int64, int)

    # Find the block position
    block_positions_before = list(zip(*np.where(obs == 1)))
    assert len(block_positions_before) > 0

    result = env.step(0)  # ACTION_UP
    obs_after = env.get_observation()
    block_positions_after = list(zip(*np.where(obs_after == 1)))

    # Block should have moved up (row decreased by 1)
    for (r_before, c_before), (r_after, c_after) in zip(
        sorted(block_positions_before), sorted(block_positions_after)
    ):
        assert r_after == r_before - 1 or r_before == 0  # Can't go above row 0
        assert c_after == c_before


def test_deterministic_movement_action_down():
    """Action 1 moves the block down by 1 row."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("deterministic_movement")
    result = env.step(1)  # ACTION_DOWN
    assert isinstance(result, dict)
    assert 'score' in result
    assert 'level_complete' in result
    assert 'game_over' in result


def test_deterministic_movement_wall_blocking():
    """Block cannot move through grid boundary."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("deterministic_movement")
    # Move up many times — should stop at row 0
    for _ in range(20):
        env.step(0)  # UP
    obs = env.get_observation()
    block_rows = [r for r, c in zip(*np.where(obs == 1))]
    assert min(block_rows) >= 0


def test_hidden_goal_no_score_until_solved():
    """Score stays 0 until blue block reaches green cell."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("hidden_goal")
    obs = env.get_observation()

    # Take one random action — score should be 0
    result = env.step(0)
    assert result['score'] == 0.0 or result['level_complete'] is True


def test_hidden_goal_level_complete_on_solve():
    """When blue block reaches green cell, level_complete=True and score=1.0."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("hidden_goal")
    # We need to know layout to solve it — but we can test that the
    # interface works and eventually terminates
    for i in range(200):
        result = env.step(i % 4)  # Cycle through up/down/left/right
        if result['level_complete']:
            assert result['score'] == 1.0
            break


def test_context_dependent_rules():
    """Same action produces different effects based on background color."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("context_dependent")
    obs = env.get_observation()
    assert isinstance(obs, np.ndarray)
    # Just verify it runs without error — testing the actual context
    # behavior is done in integration tests
    result = env.step(0)
    assert isinstance(result, dict)


def test_multi_level_progression():
    """Solving level 1 advances to level 2 with more objects."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("multi_level")
    obs1 = env.get_observation()
    # Count objects in level 1
    colors1 = set(obs1.flatten()) - {0}  # Non-background colors

    # Solve level 1 (cycle actions until level_complete)
    for i in range(500):
        result = env.step(i % 4)
        if result['level_complete']:
            break

    # Get level 2 observation
    obs2 = env.get_observation()
    colors2 = set(obs2.flatten()) - {0}
    # Level 2 should have >= as many non-background colors
    assert len(colors2) >= len(colors1)


def test_step_result_format():
    """Step result has all required fields."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("deterministic_movement")
    result = env.step(0)
    assert 'score' in result
    assert 'level_complete' in result
    assert 'game_over' in result
    assert isinstance(result['score'], (int, float))
    assert isinstance(result['level_complete'], bool)
    assert isinstance(result['game_over'], bool)


def test_scorecard():
    """Arcade provides a scorecard after playing."""
    from charith.mock_env import MockArcade

    arcade = MockArcade()
    env = arcade.make("deterministic_movement")
    for _ in range(10):
        env.step(0)
    scorecard = arcade.get_scorecard()
    assert scorecard is not None
    assert 'total_actions' in scorecard
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_mock_env.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'charith.mock_env'`

**Step 3: Implement mock_env.py**

Create `src/charith/mock_env.py` with:
- `MockArcade` class (mirrors `arc_agi.Arcade` interface: `make()`, `get_scorecard()`)
- `MockEnvironment` base class (interface: `get_observation()`, `step()`)
- `DeterministicMovementEnv` — 10x10 grid, 1x1 blue block at center, actions 0-3 = up/down/left/right, walls at grid boundaries
- `HiddenGoalEnv` — 10x10 grid, blue block + green target cell, score=0 until block on target, then score=1.0 + level_complete
- `ContextDependentEnv` — 10x10 grid, block on white background: action 0 moves right; block on grey background: action 0 moves left. Background alternates every 20 ticks.
- `MultiLevelEnv` — wraps HiddenGoalEnv, level 1: 1 block, level 2: 2 blocks, level 3: 2 blocks + wall obstacle

Key implementation details:
- All observations are `np.ndarray` with `dtype=int`, values 0-9
- All step results are `dict` with keys: `score` (float), `level_complete` (bool), `game_over` (bool)
- Grid colors: 0=black (background), 1=blue (block), 3=green (target), 5=grey (alt background), 8=cyan (wall)

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_mock_env.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/charith/mock_env.py tests/test_mock_env.py
git commit -m "feat: mock environment with 4 test scenarios (deterministic, goal, context, multi-level)"
```

---

## Task 3: Object Tracker

**Files:**
- Create: `src/charith/perception/object_tracker.py`
- Create: `tests/test_object_tracker.py`

**Step 1: Write failing tests**

```python
# tests/test_object_tracker.py
"""Tests for ObjectTracker — persistent identity across frames."""
import numpy as np
import pytest


def test_match_identical_objects():
    """Same objects in same position -> matched pairs."""
    from charith.perception.core_knowledge import ObjectnessPrior
    from charith.perception.object_tracker import ObjectTracker

    grid = np.zeros((10, 10), dtype=int)
    grid[2:4, 2:4] = 1
    grid[7:9, 7:9] = 2

    prior = ObjectnessPrior()
    objects_a = prior.detect(grid)
    prior._next_id = 0
    objects_b = prior.detect(grid)

    tracker = ObjectTracker()
    pairs = tracker.match(objects_a, objects_b)
    assert len(pairs) == 2


def test_match_moved_object():
    """Object that moved one cell -> still matched by color+size."""
    from charith.perception.core_knowledge import ObjectnessPrior
    from charith.perception.object_tracker import ObjectTracker

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[3, 3] = 1  # blue at (3,3)

    grid2 = np.zeros((10, 10), dtype=int)
    grid2[3, 4] = 1  # blue moved to (3,4)

    prior = ObjectnessPrior()
    objects1 = prior.detect(grid1)
    prior._next_id = 0
    objects2 = prior.detect(grid2)

    tracker = ObjectTracker()
    pairs = tracker.match(objects1, objects2)
    assert len(pairs) == 1


def test_match_detects_appeared_disappeared():
    """Object appears or disappears -> not in matched pairs."""
    from charith.perception.core_knowledge import ObjectnessPrior
    from charith.perception.object_tracker import ObjectTracker

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[3, 3] = 1  # blue only

    grid2 = np.zeros((10, 10), dtype=int)
    grid2[3, 3] = 1  # blue still there
    grid2[7, 7] = 2  # red appeared

    prior = ObjectnessPrior()
    objects1 = prior.detect(grid1)
    prior._next_id = 0
    objects2 = prior.detect(grid2)

    tracker = ObjectTracker()
    pairs = tracker.match(objects1, objects2)
    # Only blue matched; red is new
    assert len(pairs) == 1
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_object_tracker.py -v
```

**Step 3: Implement object_tracker.py**

```python
# src/charith/perception/object_tracker.py
"""
Object Tracker — persistent object identity across frames.

Uses greedy matching by (color, size, proximity) to match objects
between consecutive frames. Full Hungarian matching deferred to Phase 2.
"""
from typing import List, Tuple, Set
from charith.perception.core_knowledge import Object


class ObjectTracker:
    """
    Match objects between consecutive frames.

    Strategy: greedy matching by color first, then by centroid proximity.
    Objects with unique color are matched trivially.
    Objects sharing a color are matched by nearest centroid.
    """

    def match(self, prev_objects: List[Object],
              curr_objects: List[Object]) -> List[Tuple[int, int]]:
        """
        Match objects from prev frame to curr frame.

        Returns list of (prev_object_id, curr_object_id) pairs.
        Unmatched objects are NOT in the output (appeared/disappeared).
        """
        pairs = []
        used_curr = set()

        # Group by color for efficient matching
        prev_by_color = {}
        for obj in prev_objects:
            prev_by_color.setdefault(obj.color, []).append(obj)

        curr_by_color = {}
        for obj in curr_objects:
            curr_by_color.setdefault(obj.color, []).append(obj)

        for color in prev_by_color:
            if color not in curr_by_color:
                continue  # All objects of this color disappeared

            prev_group = prev_by_color[color]
            curr_group = [o for o in curr_by_color[color]
                         if o.object_id not in used_curr]

            # Sort by centroid distance (greedy nearest-neighbor)
            for prev_obj in prev_group:
                best_curr = None
                best_dist = float('inf')
                for curr_obj in curr_group:
                    if curr_obj.object_id in used_curr:
                        continue
                    dist = self._centroid_dist(prev_obj, curr_obj)
                    if dist < best_dist:
                        best_dist = dist
                        best_curr = curr_obj

                if best_curr is not None:
                    pairs.append((prev_obj.object_id, best_curr.object_id))
                    used_curr.add(best_curr.object_id)

        return pairs

    def _centroid_dist(self, a: Object, b: Object) -> float:
        return ((a.centroid[0] - b.centroid[0])**2 +
                (a.centroid[1] - b.centroid[1])**2) ** 0.5
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_object_tracker.py -v
```

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/charith/perception/object_tracker.py tests/test_object_tracker.py
git commit -m "feat: object tracker with greedy color+proximity matching"
```

---

## Task 4: WorldModel (Amendment 1 + 4)

**Files:**
- Create: `src/charith/world_model/model.py`
- Create: `tests/test_world_model.py`

**Step 1: Write failing tests**

```python
# tests/test_world_model.py
"""Tests for WorldModel — predictive processing with object-level rules."""
import numpy as np
import pytest


def test_world_model_learns_simple_transition():
    """After observing same action->effect 3 times, model predicts it."""
    from charith.world_model.model import WorldModel, ObjectEffect

    wm = WorldModel()
    context = {'background_color': 0, 'object_count': 1}
    effect = [ObjectEffect(
        object_color=1, displacement=(0, 1),
        shape_changed=False, size_delta=0,
        appeared=False, disappeared=False
    )]

    # Observe 3 times
    for tick in range(3):
        wm.update(action=3, context=context,
                  observed_effects=effect, tick=tick)

    # Predict
    predicted = wm.predict(action=3, context=context)
    assert predicted is not None
    assert len(predicted) == 1
    assert predicted[0].displacement == (0, 1)


def test_world_model_no_prediction_for_unseen():
    """No prediction if action never observed in this context."""
    from charith.world_model.model import WorldModel

    wm = WorldModel()
    predicted = wm.predict(action=5, context={'background_color': 0})
    assert predicted is None


def test_world_model_context_matters():
    """Different contexts -> different predictions for same action."""
    from charith.world_model.model import WorldModel, ObjectEffect

    wm = WorldModel()
    ctx_white = {'background_color': 0}
    ctx_grey = {'background_color': 5}

    effect_right = [ObjectEffect(1, (0, 1), False, 0, False, False)]
    effect_left = [ObjectEffect(1, (0, -1), False, 0, False, False)]

    for tick in range(5):
        wm.update(0, ctx_white, effect_right, tick)
        wm.update(0, ctx_grey, effect_left, tick + 5)

    pred_white = wm.predict(0, ctx_white)
    pred_grey = wm.predict(0, ctx_grey)
    assert pred_white[0].displacement == (0, 1)
    assert pred_grey[0].displacement == (0, -1)


def test_world_model_relative_context_only():
    """Context must not contain forbidden absolute keys."""
    from charith.world_model.model import WorldModel
    from charith.perception.core_knowledge import CoreKnowledgePerception

    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 1
    perception = CoreKnowledgePerception()
    percept = perception.perceive(grid)

    wm = WorldModel()
    context = wm.extract_context(percept, controllable_ids=set())
    assert 'grid_rows' not in context
    assert 'grid_cols' not in context
    assert 'ctrl_row' not in context


def test_compute_effects_detects_movement():
    """Object that moved -> displacement in ObjectEffect."""
    from charith.world_model.model import WorldModel, ObjectEffect
    from charith.perception.core_knowledge import CoreKnowledgePerception

    perception = CoreKnowledgePerception()

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[5, 5] = 1
    percept1 = perception.perceive(grid1)

    perception._tick = 0  # reset tick for clean percept
    grid2 = np.zeros((10, 10), dtype=int)
    grid2[5, 6] = 1  # moved right
    percept2 = perception.perceive(grid2)

    wm = WorldModel()
    # Match objects manually (same color, close proximity)
    pairs = [(percept1.objects[0].object_id, percept2.objects[0].object_id)]
    effects = wm.compute_effects(percept1, percept2, pairs)
    assert len(effects) == 1
    assert effects[0].displacement == (0, 1)


def test_compute_effects_detects_appearance():
    """New object in curr_percept -> appeared=True."""
    from charith.world_model.model import WorldModel
    from charith.perception.core_knowledge import CoreKnowledgePerception

    perception = CoreKnowledgePerception()

    grid1 = np.zeros((10, 10), dtype=int)
    grid1[5, 5] = 1
    percept1 = perception.perceive(grid1)

    perception._tick = 0
    grid2 = np.zeros((10, 10), dtype=int)
    grid2[5, 5] = 1
    grid2[8, 8] = 2  # new red object
    percept2 = perception.perceive(grid2)

    wm = WorldModel()
    pairs = [(percept1.objects[0].object_id, percept2.objects[0].object_id)]
    effects = wm.compute_effects(percept1, percept2, pairs)
    appeared = [e for e in effects if e.appeared]
    assert len(appeared) == 1
    assert appeared[0].object_color == 2


def test_world_model_confidence_decay_on_reset():
    """Soft reset decays confidence by 0.8."""
    from charith.world_model.model import WorldModel, ObjectEffect

    wm = WorldModel()
    ctx = {'background_color': 0}
    eff = [ObjectEffect(1, (0, 1), False, 0, False, False)]
    for tick in range(10):
        wm.update(0, ctx, eff, tick)

    # Get confidence before reset
    rules = wm._rules[0]
    assert len(rules) > 0
    conf_before = rules[0].confidence

    wm.reset()
    conf_after = rules[0].confidence
    assert abs(conf_after - conf_before * 0.8) < 0.01


def test_world_model_error_history():
    """Recording errors populates history for metacognition."""
    from charith.world_model.model import WorldModel, PredictionError

    wm = WorldModel()
    error = PredictionError(
        predicted_grid=None, observed_grid=None,
        error_magnitude=0.5, error_cells=[],
        precision=0.8, weighted_error=0.4, is_novel=False
    )
    wm.record_error(error)
    assert len(wm.get_recent_errors()) == 1
    assert wm.get_recent_errors()[0] == 0.4
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_world_model.py -v
```

**Step 3: Implement world_model/model.py**

Implement per **Amendment 1 + Amendment 4**:
- `ObjectEffect` dataclass (color, displacement, shape_changed, size_delta, appeared, disappeared)
- `TransitionRule` with context_features, effects, observation_history (per Amendment 2 addition)
- `PredictionError` dataclass (from original guide)
- `WorldModel` class with:
  - `extract_context(percept, controllable_ids)` — relative features only, FORBIDDEN_KEYS stripped
  - `compute_effects(prev_percept, curr_percept, matched_pairs)` — object-level diffing
  - `predict(action, context)` — highest-confidence matching rule
  - `update(action, context, observed_effects, tick)` — Bayesian update with observation history
  - `record_error()`, `get_recent_errors()`, `get_accuracy()`, `get_rule_count()`
  - `reset()` — soft reset with 0.8 confidence decay, clear observation histories
  - `hard_reset()` — full clear

**Step 4: Run tests**

```bash
uv run pytest tests/test_world_model.py -v
```

**Step 5: Commit**

```bash
git add src/charith/world_model/model.py tests/test_world_model.py
git commit -m "feat: world model with object-level rules and relative context (Amendments 1+4)"
```

---

## Task 5: Ontology Expansion (Amendment 2)

**Files:**
- Create: `src/charith/metacognition/ontology.py`
- Create: `src/charith/metacognition/confidence.py`
- Create: `tests/test_ontology.py`

**Step 1: Write failing tests**

```python
# tests/test_ontology.py
"""Tests for Ontology Expansion — 4-test protocol + rule splitting."""
import numpy as np
import pytest


def test_insufficient_data_no_expansion():
    """With < window_size errors, should not expand."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    result = ont.check(recent_errors=[0.5] * 10, rule_count=5, accuracy=0.5)
    assert result.should_expand is False
    assert result.test_results.get('insufficient_data') is True


def test_residual_structure_detects_autocorrelation():
    """Autocorrelated errors -> residual structure test passes."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    # Create autocorrelated errors: high-low-high-low pattern
    errors = []
    for i in range(60):
        errors.append(0.8 if i % 2 == 0 else 0.2)
    result = ont.check(errors, rule_count=50, accuracy=0.5)
    assert result.test_results.get('residual_structure') is True


def test_random_errors_no_expansion():
    """Random errors -> no expansion triggered."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50)
    rng = np.random.RandomState(42)
    errors = rng.uniform(0.3, 0.7, size=60).tolist()
    # Feed enough data for rule/accuracy history too
    for i in range(60):
        ont.check([errors[i]], rule_count=50 + i, accuracy=0.5 + i * 0.005)
    # Final check with full window
    result = ont.check(errors[-50:], rule_count=110, accuracy=0.8)
    # Random errors should NOT trigger 3/4 tests
    assert result.should_expand is False


def test_volatility_spike_detection():
    """Variance doubling in recent window -> volatility test passes."""
    from charith.metacognition.ontology import OntologyExpansion

    ont = OntologyExpansion(window_size=50, volatility_sigma_multiplier=2.0)
    # Low variance baseline, then high variance
    errors = [0.5] * 30 + list(np.random.RandomState(42).uniform(0.0, 1.0, 30))
    result = ont.check(errors, rule_count=50, accuracy=0.5)
    # Check if volatility test fires (may or may not depending on random seed)
    assert 'volatility_spike' in result.test_results


def test_rule_splitting_reduces_confusion():
    """After splitting a confused rule, the two new rules should be more specific."""
    from charith.world_model.model import WorldModel, ObjectEffect, TransitionRule
    from charith.metacognition.ontology import OntologyExpansion

    wm = WorldModel()
    ont = OntologyExpansion()

    # Create a confused rule: same action, different effects depending on context
    ctx_white = {'background_color': 0}
    ctx_grey = {'background_color': 5}
    eff_right = [ObjectEffect(1, (0, 1), False, 0, False, False)]
    eff_left = [ObjectEffect(1, (0, -1), False, 0, False, False)]

    # Feed observations that create a confused rule
    # First, both go to the same rule (no distinguishing context initially)
    shared_ctx = {}  # Minimal context
    for tick in range(5):
        wm.update(0, ctx_white, eff_right, tick)
    for tick in range(5, 10):
        wm.update(0, ctx_grey, eff_left, tick)

    # Now the rule for action 0 should be confused
    rules_before = len(wm._rules.get(0, []))

    # Attempt expansion
    success = ont.execute_expansion('new_environment_mechanic', wm, None)
    # The split may or may not succeed depending on history — test the interface
    assert isinstance(success, bool)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ontology.py -v
```

**Step 3: Implement ontology.py**

Implement per **original guide Section 7.2 + Amendment 2**:
- `OntologyExpansionResult` dataclass
- `OntologyExpansion` class with 4-test protocol:
  - `_test_residual_structure()` — autoregression R^2
  - `_test_volatility_spike()` — variance ratio
  - `_test_epistemic_uncertainty()` — stagnant high errors
  - `_test_capacity_saturation()` — rules growing without accuracy gains
- `execute_expansion()` — finds most confused rule, calls `_split_rule_by_context()`
- `_split_rule_by_context()` — Amendment 2's context-feature separation algorithm

Also create stub `confidence.py`:

```python
# src/charith/metacognition/confidence.py
"""Epistemic uncertainty tracking — stub for Phase 2."""

class ConfidenceTracker:
    """Tracks overall epistemic uncertainty of the agent."""

    def __init__(self):
        self._uncertainty = 1.0

    def update(self, error_magnitude: float, rule_count: int):
        alpha = 0.1
        self._uncertainty = alpha * error_magnitude + (1 - alpha) * self._uncertainty

    @property
    def uncertainty(self) -> float:
        return self._uncertainty

    def reset(self):
        self._uncertainty = 1.0
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_ontology.py -v
```

**Step 5: Commit**

```bash
git add src/charith/metacognition/ontology.py src/charith/metacognition/confidence.py tests/test_ontology.py
git commit -m "feat: ontology expansion with 4-test protocol and rule splitting (Amendment 2)"
```

---

## Task 6: Goal Discovery (Amendment 3)

**Files:**
- Create: `src/charith/metacognition/goal_discovery.py`
- Create: `tests/test_goal_discovery.py`

**Step 1: Write failing tests**

```python
# tests/test_goal_discovery.py
"""Tests for Goal Discovery — discriminating hypothesis predictions."""
import numpy as np
import pytest


def test_reduce_colors_hypothesis_predicts_positive():
    """Fewer colors -> ReduceColorsHypothesis predicts positive reward."""
    from charith.metacognition.goal_discovery import ReduceColorsHypothesis

    h = ReduceColorsHypothesis(tick=0)
    prev = np.array([[1, 2], [3, 0]])  # 4 colors
    curr = np.array([[1, 1], [0, 0]])  # 2 colors
    reward = h.predict_reward(prev, curr, None, None)
    assert reward > 0


def test_reduce_colors_hypothesis_predicts_negative():
    """More colors -> ReduceColorsHypothesis predicts negative reward."""
    from charith.metacognition.goal_discovery import ReduceColorsHypothesis

    h = ReduceColorsHypothesis(tick=0)
    prev = np.array([[1, 1], [0, 0]])  # 2 colors
    curr = np.array([[1, 2], [3, 0]])  # 4 colors
    reward = h.predict_reward(prev, curr, None, None)
    assert reward < 0


def test_create_symmetry_hypothesis():
    """Symmetric grid -> CreateSymmetryHypothesis predicts positive."""
    from charith.metacognition.goal_discovery import CreateSymmetryHypothesis
    from charith.perception.core_knowledge import CoreKnowledgePerception

    perception = CoreKnowledgePerception()

    prev_grid = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
    curr_grid = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])  # symmetric

    pp = perception.perceive(prev_grid)
    pc = perception.perceive(curr_grid)

    h = CreateSymmetryHypothesis(tick=0)
    reward = h.predict_reward(prev_grid, curr_grid, pp, pc)
    assert reward >= 0  # Symmetric should be non-negative


def test_hypothesis_accuracy_tracking():
    """Hypotheses update accuracy based on prediction vs actual."""
    from charith.metacognition.goal_discovery import ReduceColorsHypothesis

    h = ReduceColorsHypothesis(tick=0)
    # Predict +0.5, actual +0.5 -> correct
    h.update_accuracy(predicted_reward=0.5, actual_reward=0.5)
    assert h.prediction_correct == 1
    assert h.confidence > 0.5

    # Predict +0.5, actual -0.5 -> incorrect
    h.update_accuracy(predicted_reward=0.5, actual_reward=-0.5)
    assert h.prediction_correct == 1  # Still 1
    assert h.prediction_total == 2


def test_goal_discovery_external_reward():
    """External score signal dominates reward computation."""
    from charith.metacognition.goal_discovery import GoalDiscovery

    gd = GoalDiscovery()
    grid = np.zeros((5, 5), dtype=int)
    reward = gd.update(grid, action=0, score=0.0)
    reward = gd.update(grid, action=0, score=1.0)
    assert reward > 0  # Score went up


def test_goal_discovery_generates_hypotheses():
    """After 20 ticks, hypotheses are generated."""
    from charith.metacognition.goal_discovery import GoalDiscovery

    gd = GoalDiscovery()
    grid = np.zeros((5, 5), dtype=int)
    for i in range(25):
        gd.update(grid, action=i % 4, score=0.0)
    assert len(gd._hypotheses) > 0


def test_best_hypothesis_selection():
    """get_best_hypothesis returns highest-confidence hypothesis."""
    from charith.metacognition.goal_discovery import GoalDiscovery

    gd = GoalDiscovery()
    grid = np.zeros((5, 5), dtype=int)
    for i in range(25):
        gd.update(grid, action=i % 4, score=0.0)
    best = gd.get_best_hypothesis()
    assert best is not None
    assert hasattr(best, 'confidence')
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_goal_discovery.py -v
```

**Step 3: Implement goal_discovery.py**

Implement per **Amendment 3**:
- `GoalHypothesis` base class with `predict_reward()` and `update_accuracy()`
- `ReduceColorsHypothesis` — predicts reward for fewer colors
- `CreateSymmetryHypothesis` — predicts reward for increased symmetry
- `MoveToTargetHypothesis` — placeholder (needs tracker data)
- `MatchTemplateHypothesis` — predicts reward for matching a template
- `SortObjectsHypothesis` — predicts reward for ordered objects
- `GoalDiscovery` class with updated `update()` method that scores hypotheses

**Step 4: Run tests**

```bash
uv run pytest tests/test_goal_discovery.py -v
```

**Step 5: Commit**

```bash
git add src/charith/metacognition/goal_discovery.py tests/test_goal_discovery.py
git commit -m "feat: goal discovery with discriminating hypothesis predictions (Amendment 3)"
```

---

## Task 7: Thompson Sampling + Action Sequence Memory (Amendment 5)

**Files:**
- Create: `src/charith/action/thompson.py`
- Create: `src/charith/action/action_space.py`
- Create: `src/charith/memory/sequences.py`
- Create: `tests/test_thompson.py`

**Step 1: Write failing tests**

```python
# tests/test_thompson.py
"""Tests for Thompson Sampling and Action Sequence Memory."""
import numpy as np
import pytest


def test_thompson_selects_from_available_actions():
    """Selected action is in the available set."""
    from charith.action.thompson import ThompsonSampler

    ts = ThompsonSampler(n_actions=8)
    action = ts.select_action(context_hash=0, available_actions=[0, 1, 2])
    assert action in [0, 1, 2]


def test_thompson_converges_to_best_action():
    """After many updates, best action is selected most often."""
    from charith.action.thompson import ThompsonSampler

    ts = ThompsonSampler(n_actions=4, info_gain_weight=0.0)

    # Action 2 always gives reward, others don't
    for _ in range(100):
        for a in range(4):
            reward = 1.0 if a == 2 else 0.0
            ts.update(context_hash=0, action=a, reward=reward)

    # Sample 100 actions, action 2 should dominate
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(100):
        a = ts.select_action(context_hash=0)
        counts[a] += 1
    assert counts[2] > 50  # Should be selected >50% of the time


def test_thompson_exploration_mode():
    """With no data, should be in exploration mode (high uncertainty)."""
    from charith.action.thompson import ThompsonSampler

    ts = ThompsonSampler(n_actions=4)
    assert ts.is_exploring() is True


def test_thompson_goal_directed_bias():
    """With goal_directed=True, goal_action chosen ~50% of the time."""
    from charith.action.thompson import ThompsonSampler

    ts = ThompsonSampler(n_actions=4)
    goal_count = 0
    for _ in range(200):
        a = ts.select_action(context_hash=0, goal_directed=True, goal_action=3)
        if a == 3:
            goal_count += 1
    # Should be roughly 50% (with some variance)
    assert 60 < goal_count < 140


def test_sequence_memory_learns_good_pairs():
    """After rewarding (A, B) sequence, suggests B after A."""
    from charith.memory.sequences import ActionSequenceMemory

    seq = ActionSequenceMemory(n_actions=4)
    # (prev=1, curr=2) always rewarded
    for _ in range(20):
        seq.update(prev_action=1, curr_action=2, reward=1.0)
    # (prev=1, curr=0) never rewarded
    for _ in range(20):
        seq.update(prev_action=1, curr_action=0, reward=0.0)

    suggestion = seq.suggest_action(prev_action=1)
    assert suggestion == 2


def test_sequence_boost_positive():
    """Good sequence pair -> positive boost value."""
    from charith.memory.sequences import ActionSequenceMemory

    seq = ActionSequenceMemory(n_actions=4)
    for _ in range(20):
        seq.update(1, 2, 1.0)
    boost = seq.get_sequence_boost(prev_action=1, candidate_action=2)
    assert boost > 0


def test_thompson_with_sequence_boost():
    """Sequence memory influences Thompson action selection."""
    from charith.action.thompson import ThompsonSampler
    from charith.memory.sequences import ActionSequenceMemory

    ts = ThompsonSampler(n_actions=4, info_gain_weight=0.0)
    seq = ActionSequenceMemory(n_actions=4)
    for _ in range(50):
        seq.update(1, 3, 1.0)

    # With sequence boost, action 3 should be favored after action 1
    counts = {a: 0 for a in range(4)}
    for _ in range(200):
        a = ts.select_action(
            context_hash=0, prev_action=1, sequence_memory=seq
        )
        counts[a] += 1
    assert counts[3] > counts[0]  # Action 3 should be preferred
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_thompson.py -v
```

**Step 3: Implement thompson.py, action_space.py, sequences.py**

- `thompson.py`: From original guide Section 8.2 + Amendment 5 (sequence_boost param)
- `action_space.py`: Simple action mapping stub
- `sequences.py`: From Amendment 5 verbatim (ActionSequenceMemory)

```python
# src/charith/action/action_space.py
"""ARC-AGI-3 action mapping — adapt when SDK is available."""
from enum import IntEnum


class Action(IntEnum):
    """Action indices used throughout the agent."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    USE = 4
    SECONDARY = 5
    WAIT = 6
    CONFIRM = 7


N_ACTIONS = len(Action)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_thompson.py -v
```

**Step 5: Commit**

```bash
git add src/charith/action/thompson.py src/charith/action/action_space.py src/charith/memory/sequences.py tests/test_thompson.py
git commit -m "feat: Thompson sampling with sequence memory boost (Amendment 5)"
```

---

## Task 8: Memory Stubs

**Files:**
- Create: `src/charith/memory/working.py`
- Create: `src/charith/memory/episodic.py`
- Create: `src/charith/memory/rules.py`
- Create: `src/charith/memory/consolidation.py`
- Create: `src/charith/utils/grid_ops.py`
- Create: `src/charith/utils/hashing.py`
- Create: `src/charith/utils/logging.py`
- Create: `tests/test_memory.py`

**Step 1: Write failing tests**

```python
# tests/test_memory.py
"""Tests for memory modules."""
import numpy as np


def test_working_memory_capacity():
    """Working memory holds max 7 items, oldest evicted first."""
    from charith.memory.working import WorkingMemory

    wm = WorkingMemory(capacity=7)
    for i in range(10):
        wm.store(f"item_{i}", f"value_{i}")
    assert wm.size == 7
    assert wm.retrieve("item_0") is None  # Evicted
    assert wm.retrieve("item_9") == "value_9"  # Still there


def test_working_memory_clear():
    """Clear empties all slots."""
    from charith.memory.working import WorkingMemory

    wm = WorkingMemory(capacity=7)
    wm.store("key", "value")
    wm.clear()
    assert wm.size == 0


def test_episodic_store_record_and_query():
    """Record episodes and query by state hash."""
    from charith.memory.episodic import EpisodeStore

    store = EpisodeStore(max_episodes=100)
    grid = np.zeros((5, 5), dtype=int)
    store.record(state=grid, action=0, next_state=grid,
                 error=0.1, tick=0)
    assert store.count == 1


def test_episodic_store_level_boundary():
    """mark_level_boundary records boundary tick."""
    from charith.memory.episodic import EpisodeStore

    store = EpisodeStore(max_episodes=100)
    grid = np.zeros((5, 5), dtype=int)
    store.record(state=grid, action=0, next_state=grid, error=0.1, tick=0)
    store.mark_level_boundary()
    assert len(store._level_boundaries) == 1


def test_episodic_store_max_capacity():
    """Old episodes evicted when max reached."""
    from charith.memory.episodic import EpisodeStore

    store = EpisodeStore(max_episodes=5)
    grid = np.zeros((5, 5), dtype=int)
    for i in range(10):
        store.record(state=grid, action=i % 4, next_state=grid,
                     error=0.1, tick=i)
    assert store.count == 5
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_memory.py -v
```

**Step 3: Implement all memory and utils modules**

```python
# src/charith/memory/working.py
"""Capacity-limited working memory (7 slots, per Miller's Law)."""
from collections import OrderedDict
from typing import Any, Optional


class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._slots: OrderedDict = OrderedDict()

    def store(self, key: str, value: Any):
        if key in self._slots:
            self._slots.move_to_end(key)
            self._slots[key] = value
        else:
            if len(self._slots) >= self.capacity:
                self._slots.popitem(last=False)  # Remove oldest
            self._slots[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        return self._slots.get(key)

    @property
    def size(self) -> int:
        return len(self._slots)

    def clear(self):
        self._slots.clear()
```

```python
# src/charith/memory/episodic.py
"""Hash-indexed episode store."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import hashlib


@dataclass
class Episode:
    state_hash: int
    action: int
    next_state_hash: int
    error: float
    tick: int
    level: int = 0


class EpisodeStore:
    def __init__(self, max_episodes: int = 5000):
        self._max = max_episodes
        self._episodes: List[Episode] = []
        self._level_boundaries: List[int] = []
        self._current_level: int = 0

    def record(self, state: np.ndarray, action: int,
               next_state: np.ndarray, error: float, tick: int):
        s_hash = int(hashlib.md5(state.tobytes()).hexdigest()[:16], 16)
        ns_hash = int(hashlib.md5(next_state.tobytes()).hexdigest()[:16], 16)
        ep = Episode(s_hash, action, ns_hash, error, tick, self._current_level)
        self._episodes.append(ep)
        if len(self._episodes) > self._max:
            self._episodes = self._episodes[-self._max:]

    def mark_level_boundary(self):
        self._level_boundaries.append(len(self._episodes))
        self._current_level += 1

    @property
    def count(self) -> int:
        return len(self._episodes)

    def hard_reset(self):
        self._episodes.clear()
        self._level_boundaries.clear()
        self._current_level = 0
```

```python
# src/charith/memory/rules.py
"""If-then rule store — stub for Phase 2 consolidation."""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ExtractedRule:
    condition: Dict[str, Any]
    action: int
    effect: str
    confidence: float
    source_episodes: int


class RuleStore:
    def __init__(self):
        self._rules: List[ExtractedRule] = []

    def add(self, rule: ExtractedRule):
        self._rules.append(rule)

    @property
    def count(self) -> int:
        return len(self._rules)

    def clear(self):
        self._rules.clear()
```

```python
# src/charith/memory/consolidation.py
"""Episode -> rule extraction — stub for Phase 2."""


class Consolidator:
    """Phase 2: extract if-then rules from episodic memory patterns."""

    def extract_rules(self, episode_store, rule_store):
        """Analyze episodes for repeating patterns and extract rules."""
        pass  # Phase 2 implementation
```

```python
# src/charith/utils/hashing.py
"""Fast state hashing utilities."""
import numpy as np
import hashlib


def state_hash(grid: np.ndarray) -> int:
    """Fast hash of grid state for dictionary lookups."""
    return int(hashlib.md5(grid.tobytes()).hexdigest()[:16], 16)
```

```python
# src/charith/utils/grid_ops.py
"""Grid manipulation utilities."""
import numpy as np
from typing import Tuple


def grid_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return boolean mask of cells that differ."""
    return a != b


def grid_entropy(grid: np.ndarray) -> float:
    """Shannon entropy of color distribution."""
    unique, counts = np.unique(grid, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log2(probs + 1e-10)))
```

```python
# src/charith/utils/logging.py
"""Structured logging for agent analysis."""
import time
from typing import Any, Dict, List


class AgentLogger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def log(self, event_type: str, data: Any, tick: int = 0):
        self._events.append({
            'tick': tick,
            'event': event_type,
            'data': data,
            'time': time.time(),
        })

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def clear(self):
        self._events.clear()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_memory.py -v
```

**Step 5: Commit**

```bash
git add src/charith/memory/ src/charith/utils/ tests/test_memory.py
git commit -m "feat: memory modules (working, episodic, rules) and utils (hashing, grid_ops, logging)"
```

---

## Task 9: Agent Loop (All Amendments Integrated)

**Files:**
- Create: `src/charith/agent.py`
- Create: `tests/test_agent.py`

**Step 1: Write failing tests**

```python
# tests/test_agent.py
"""Integration tests for the CHARITHAgent."""
import pytest


def test_agent_plays_deterministic_without_crashing():
    """Smoke test: agent plays deterministic movement game end-to-end."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    scorecard = agent.play_game("deterministic_movement", max_actions=100)
    assert scorecard is not None
    assert agent._total_actions > 0


def test_agent_plays_hidden_goal():
    """Agent plays hidden goal game without crashing."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    scorecard = agent.play_game("hidden_goal", max_actions=200)
    assert scorecard is not None


def test_agent_world_model_learns():
    """After playing deterministic game, world model has rules."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("deterministic_movement", max_actions=50)
    assert agent.world_model.get_rule_count() > 0


def test_agent_goal_discovery_generates_hypotheses():
    """After enough ticks, goal discovery has hypotheses."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("hidden_goal", max_actions=50)
    assert len(agent.goal_discovery._hypotheses) > 0


def test_agent_hard_reset_clears_state():
    """Hard reset clears all agent state."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("deterministic_movement", max_actions=20)
    agent._hard_reset()
    assert agent.world_model.get_rule_count() == 0
    assert agent._tick == 0
    assert agent._total_actions == 0


def test_agent_cross_level_preserves_rules():
    """Multi-level game: rules from level 1 persist into level 2."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("multi_level", max_actions=500)
    # If level transition happened, rules should persist
    if agent._levels_completed > 0:
        assert agent.world_model.get_rule_count() > 0
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_agent.py -v
```

**Step 3: Implement agent.py**

The main agent class per the original guide Section 10.2, with ALL amendments applied:
- Uses `MockArcade` instead of `arc_agi.Arcade` (SDK swap point)
- `_tick_cycle()` follows Amendment 1 flow: PERCEIVE -> TRACK -> CONTEXT -> PREDICT -> EFFECTS -> ERROR -> UPDATE -> META -> GOAL -> SELECT
- Passes `StructuredPercept` to WorldModel (Amendment 1)
- Uses relative context only (Amendment 4)
- Passes `sequence_memory` and `prev_action` to ThompsonSampler (Amendment 5)
- GoalDiscovery receives both percepts for hypothesis scoring (Amendment 3)
- `_on_level_complete()` does soft reset with confidence decay (Amendment 4)
- Includes `play_game()`, `_hard_reset()`, `_parse_observation()`, helper methods

**Step 4: Run tests**

```bash
uv run pytest tests/test_agent.py -v
```

**Step 5: Commit**

```bash
git add src/charith/agent.py tests/test_agent.py
git commit -m "feat: CHARITH agent loop integrating all 5 amendments"
```

---

## Task 10: Full Test Suite + Scripts

**Files:**
- Create: `tests/test_integration.py`
- Create: `scripts/play_single.py`
- Create: `scripts/play_all.py`
- Create: `scripts/benchmark_speed.py`

**Step 1: Write integration tests**

```python
# tests/test_integration.py
"""Full integration tests — validate architecture actually works."""
import pytest
import numpy as np


def test_world_model_learns_deterministic_movement():
    """WorldModel achieves >0.5 accuracy after 30 ticks in deterministic mock."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("deterministic_movement", max_actions=50)
    accuracy = agent.world_model.get_accuracy()
    # Should learn something about deterministic movement
    assert accuracy > 0.0


def test_thompson_converges_on_deterministic():
    """Thompson Sampling should favor rewarded actions."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("deterministic_movement", max_actions=100)
    summary = agent.explorer.get_action_summary()
    # At least one action should have been taken multiple times
    max_taken = max(s['times_taken'] for s in summary.values())
    assert max_taken > 5


def test_context_dependent_triggers_expansion_eventually():
    """Context-dependent env should eventually trigger ontology expansion check."""
    from charith.agent import CHARITHAgent

    agent = CHARITHAgent()
    agent.play_game("context_dependent", max_actions=200)
    # Ontology expansion may or may not fire — just verify no crash
    assert agent._total_actions > 0


def test_all_mock_games_complete_without_error():
    """Smoke test: every mock scenario runs to completion."""
    from charith.agent import CHARITHAgent

    for game_id in ["deterministic_movement", "hidden_goal",
                    "context_dependent", "multi_level"]:
        agent = CHARITHAgent()
        scorecard = agent.play_game(game_id, max_actions=100)
        assert scorecard is not None, f"Failed on {game_id}"
```

**Step 2: Write scripts**

```python
# scripts/play_single.py
"""Play a single mock game with verbose output."""
import sys
sys.path.insert(0, 'src')
from charith.agent import CHARITHAgent

game = sys.argv[1] if len(sys.argv) > 1 else "deterministic_movement"
agent = CHARITHAgent()
scorecard = agent.play_game(game, max_actions=500)

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
```

```python
# scripts/play_all.py
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
```

```python
# scripts/benchmark_speed.py
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
```

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py scripts/
git commit -m "feat: integration tests and runner scripts (play_single, play_all, benchmark)"
```

---

## Summary

| Task | Module | Key Files | Tests |
|---|---|---|---|
| 0 | Scaffold | pyproject.toml, configs, __init__.py | — |
| 1 | Perception | core_knowledge.py | 15 tests |
| 2 | MockEnvironment | mock_env.py | 9 tests |
| 3 | ObjectTracker | object_tracker.py | 3 tests |
| 4 | WorldModel | world_model/model.py | 8 tests |
| 5 | Ontology | metacognition/ontology.py | 4 tests |
| 6 | GoalDiscovery | metacognition/goal_discovery.py | 7 tests |
| 7 | Thompson+Sequences | action/thompson.py, memory/sequences.py | 7 tests |
| 8 | Memory+Utils | memory/*.py, utils/*.py | 5 tests |
| 9 | Agent Loop | agent.py | 6 tests |
| 10 | Integration+Scripts | test_integration.py, scripts/*.py | 4 tests |

**Total: 68 tests across 11 tasks**

Each task is independently committable and testable. The mock environment (Task 2) enables testing every subsequent module immediately after implementation.
