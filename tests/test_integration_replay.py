"""Offline replay of recorded episodes (guide ground rule 6 / §2.6).

Feeds recorded grids back through the PATH6 agent with the mock backend — zero
API calls. Exercises the full live loop (tracker -> ledger -> decision) and is
the foundation for the S10 ablation dry-run.
"""
import os

import pytest

from charith.config import PROFILE_PATH6, PROFILE_LEGACY
from charith.llm_agent.agent import LLMAgent
from charith.eval.episode import load_episode

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "episodes")


def _episode(name):
    path = os.path.join(_FIXTURES, f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"fixture {name} not recorded")
    return load_episode(path)


def _replay(agent, episode, limit=None):
    ticks = episode["ticks"][:limit] if limit else episode["ticks"]
    actions = []
    for t in ticks:
        actions.append(agent.act(t["grid"], available_actions=t["available_actions"],
                                 score=t["score"]))
    return actions


@pytest.mark.parametrize("game", ["ls20", "tr87"])
def test_path6_replay_offline(game):
    ep = _episode(game)
    agent = LLMAgent(config=PROFILE_PATH6)
    agent.llm._backend = "mock"
    actions = _replay(agent, ep, limit=15)
    # Every tick recorded a prediction and returned a legal action.
    assert len(agent.ledger.records) == len(actions)
    for a, t in zip(actions, ep["ticks"][:15]):
        assert a in t["available_actions"]
    # Record-before-execute invariant holds for every tick.
    for rec, a in zip(agent.ledger.records, actions):
        assert int(rec.action_id) == a


def test_legacy_replay_offline():
    ep = _episode("ls20")
    agent = LLMAgent(config=PROFILE_LEGACY)
    agent.llm._backend = "mock"
    actions = _replay(agent, ep, limit=10)
    assert len(actions) == 10
    assert len(agent.ledger.records) == 0  # legacy loop never touches the ledger


def test_replay_object_tracking_stable_on_static_scene():
    """On a static grid, oids do not churn tick-to-tick (S2 done-when)."""
    ep = _episode("ls20")
    agent = LLMAgent(config=PROFILE_PATH6)
    agent.llm._backend = "mock"
    # Replay the first few ticks and confirm the tracker never explodes oid count.
    _replay(agent, ep, limit=8)
    total_oids = agent.object_tracker._next_oid
    live = len(agent._last_tracked_oids)
    # Total oids created stays close to the live object count (no per-tick renumber).
    assert total_oids <= live * 3
