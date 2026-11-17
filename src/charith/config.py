"""Path 6 configuration & feature flags.

Single source of truth for every Path 6 mechanism switch. Two canonical
profiles must always work:

* ``PROFILE_LEGACY`` — reproduces Path-5 behavior (all fixes OFF).
* ``PROFILE_PATH6``  — all fixes ON.

The S10 ablation harness flips *these same flags* to define its conditions.
There is deliberately no second switching mechanism (see guide S0 / ground
rule 3): the rollout safety switch and the milestone evidence switch are the
same object, so they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


@dataclass
class AgentConfig:
    """All Path 6 feature flags. Defaults = PROFILE_PATH6 (everything on)."""

    # --- S1 decision interface ---------------------------------------------
    structured_decisions: bool = True
    max_decision_retries: int = 2
    decision_temperature: float = 0.2
    num_ctx: int = 16384                # Ollama context window; small default truncates SILENTLY

    # --- S2 perception identity --------------------------------------------
    persistent_tracking: bool = True
    occlusion_grace_ticks: int = 3

    # --- S3 hum ------------------------------------------------------------
    enforce_predictions: bool = True
    gated_expansion: bool = True
    expansion_cooldown_ticks: int = 8
    orphan_window: int = 6

    # --- S4 hypotheses -----------------------------------------------------
    hypothesis_bank: bool = True
    thompson_exploration: bool = True
    max_active_hypotheses: int = 12
    refute_weight: float = 2.0          # Popper bias: beta += 2 on refutation

    # --- S5 goals ----------------------------------------------------------
    goal_population: bool = True
    max_goal_hypotheses: int = 6

    # --- S6 actions --------------------------------------------------------
    affordance_clicks: bool = True
    max_affordances: int = 15
    dead_action_min_trials: int = 3
    dead_action_min_state_classes: int = 2

    # --- S7 perception expansion -------------------------------------------
    probe_registry: bool = True
    authored_probes: bool = False       # stretch; keep OFF by default

    # --- S8 ontology lifecycle ---------------------------------------------
    concept_lifecycle: bool = True
    c1_soft_cap: int = 25               # non-core concepts
    concept_stale_ticks: int = 200

    # --- S9 ALFA -----------------------------------------------------------
    alfa_controller: bool = True
    modeler_period: int = 6             # ticks per Explorer batch
    world_model_gate: float = 0.70      # rolling prediction accuracy to unlock Planner
    plan_search_budget: int = 5000      # node expansions

    # --- S10 ablation switches ---------------------------------------------
    translator_mode: Literal["structured", "raw"] = "structured"
    c1c2_enabled: bool = True

    # ---- helpers ----------------------------------------------------------

    def with_overrides(self, **kwargs) -> "AgentConfig":
        """Return a copy with the given fields overridden (for --set key=value)."""
        unknown = set(kwargs) - {f for f in self.__dataclass_fields__}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Canonical profiles
# ---------------------------------------------------------------------------

PROFILE_PATH6 = AgentConfig()

PROFILE_LEGACY = AgentConfig(
    structured_decisions=False,
    persistent_tracking=False,
    enforce_predictions=False,
    gated_expansion=False,
    hypothesis_bank=False,
    thompson_exploration=False,
    goal_population=False,
    affordance_clicks=False,
    probe_registry=False,
    concept_lifecycle=False,
    alfa_controller=False,
)

PROFILES: dict[str, AgentConfig] = {
    "legacy": PROFILE_LEGACY,
    "path6": PROFILE_PATH6,
}


# ---------------------------------------------------------------------------
# CLI plumbing shared by play scripts and the ablation harness
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce(field_name: str, raw: str):
    """Coerce a string CLI value to the type declared on AgentConfig."""
    field_type = AgentConfig.__dataclass_fields__[field_name].type
    # dataclass stores annotations as strings under `from __future__ annotations`.
    type_str = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))

    if "bool" in type_str:
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(f"{field_name}: expected bool, got {raw!r}")
    if "int" in type_str:
        return int(raw)
    if "float" in type_str:
        return float(raw)
    # Literal / str
    return raw


def config_from_cli(profile: str = "path6", overrides: list[str] | None = None) -> AgentConfig:
    """Build an AgentConfig from a --profile name and --set key=value overrides.

    Parameters
    ----------
    profile : str
        "legacy" or "path6".
    overrides : list[str], optional
        Each item is "key=value", e.g. ``["gated_expansion=false", "refute_weight=3"]``.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    cfg = PROFILES[profile]

    parsed: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in AgentConfig.__dataclass_fields__:
            raise ValueError(f"Unknown config key {key!r}")
        parsed[key] = _coerce(key, raw)

    return cfg.with_overrides(**parsed) if parsed else cfg
