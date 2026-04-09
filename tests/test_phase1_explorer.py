"""Tests for Phase 1 — Explorer."""
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.alfa_loop.explorer import Explorer
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def _ls20_like_env():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12   # sprite
    return MockArcEnv(
        initial_grid=grid,
        rules={
            1: move_obj_by(12, -1, 0),
            2: move_obj_by(12, 1, 0),
            3: move_obj_by(12, 0, -1),
            4: move_obj_by(12, 0, 1),
        },
    )


def test_explore_returns_8_evidence_entries():
    env = _ls20_like_env()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    explorer = Explorer(env, perception, table)

    evidence = explorer.explore(num_actions=8)
    assert len(evidence) == 8
    for e in evidence:
        assert e.action in range(1, 9)
        assert e.percept_before is not None
        assert e.percept_after is not None


def test_explore_records_in_table():
    env = _ls20_like_env()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    explorer = Explorer(env, perception, table)

    explorer.explore(num_actions=8)
    assert len(table.single_table) == 8


def test_explore_halts_on_done():
    env = _ls20_like_env()
    env.done_on_action = 3
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    explorer = Explorer(env, perception, table)

    evidence = explorer.explore(num_actions=8)
    assert len(evidence) == 3
