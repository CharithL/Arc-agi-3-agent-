"""Tests for S0 — config profiles & CLI flag plumbing."""
import pytest

from charith.config import (
    AgentConfig,
    PROFILE_LEGACY,
    PROFILE_PATH6,
    config_from_cli,
)


def test_path6_defaults_all_on():
    cfg = PROFILE_PATH6
    assert cfg.structured_decisions is True
    assert cfg.persistent_tracking is True
    assert cfg.enforce_predictions is True
    assert cfg.gated_expansion is True
    assert cfg.hypothesis_bank is True
    assert cfg.alfa_controller is True


def test_legacy_disables_the_fixes():
    cfg = PROFILE_LEGACY
    for flag in (
        "structured_decisions", "persistent_tracking", "enforce_predictions",
        "gated_expansion", "hypothesis_bank", "thompson_exploration",
        "goal_population", "affordance_clicks", "probe_registry",
        "concept_lifecycle", "alfa_controller",
    ):
        assert getattr(cfg, flag) is False, f"{flag} should be off in legacy"


def test_popper_bias_default():
    assert PROFILE_PATH6.refute_weight == 2.0
    assert PROFILE_PATH6.refute_weight > 1.0  # must stay a Popper bias


def test_profiles_are_independent_objects():
    # replace() based overrides must not mutate the canonical profiles.
    c = PROFILE_PATH6.with_overrides(gated_expansion=False)
    assert c.gated_expansion is False
    assert PROFILE_PATH6.gated_expansion is True


def test_config_from_cli_profile_selection():
    assert config_from_cli("legacy").structured_decisions is False
    assert config_from_cli("path6").structured_decisions is True


def test_config_from_cli_bool_override():
    cfg = config_from_cli("path6", ["gated_expansion=false"])
    assert cfg.gated_expansion is False
    assert cfg.structured_decisions is True  # untouched


def test_config_from_cli_numeric_overrides():
    cfg = config_from_cli("path6", ["refute_weight=3.5", "max_active_hypotheses=20"])
    assert cfg.refute_weight == 3.5
    assert cfg.max_active_hypotheses == 20


def test_config_from_cli_literal_override():
    cfg = config_from_cli("path6", ["translator_mode=raw"])
    assert cfg.translator_mode == "raw"


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        config_from_cli("nonsense")


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        config_from_cli("path6", ["not_a_real_flag=1"])


def test_bad_override_format_raises():
    with pytest.raises(ValueError):
        config_from_cli("path6", ["missing_equals"])


def test_agent_accepts_config():
    from charith.llm_agent.agent import LLMAgent
    agent = LLMAgent(config=PROFILE_LEGACY)
    assert agent.config is PROFILE_LEGACY
    # default is PATH6
    assert LLMAgent().config is PROFILE_PATH6
