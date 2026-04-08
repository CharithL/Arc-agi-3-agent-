# CHARITH Full Stack Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a new ARC-AGI-3 agent (CharithFullStackAgent) that runs a 6-phase ALFA loop on ls20 without crashes, using 2-3 LLM calls per attempt, by composing existing CoreKnowledge perception with a ported C1/C2 causal engine.

**Architecture:** 6 phases — Explore → Hypothesize (LLM) → Verify (interventional) → ErrorCheck (+LLM if expansion needed) → Plan (LLM) → Execute+monitor δ. The LLM generates hypotheses; the C1/C2 engine tests them via intervention. No circular validation.

**Tech Stack:** Python 3.12+, numpy, scipy, statsmodels, existing `charith.perception` + `charith.llm_agent.ollama_client`, existing `arc_agi` package for the environment.

**Source materials:**
- Design doc: `docs/plans/2026-04-08-charith-full-stack-design.md`
- Ported C1/C2 logic lifted from: `C:\Users\chari\OneDrive\Documents\C1C2 model\c1c2-hybrid\src\engine\`
- Existing perception: `src/charith/perception/core_knowledge.py` (class `CoreKnowledgePerception`, method `perceive(grid)`, dataclass `StructuredPercept`)
- Existing object matcher: `src/charith/perception/object_tracker.py` (class `ObjectTracker`, method `match(prev, curr)`)
- Existing LLM client: `src/charith/llm_agent/ollama_client.py`

**Critical rule:** Do not modify any existing files under `src/charith/`. All new work lives under new subpackages.

---

## Phase A — Package Scaffolding (Tasks 1-3)

### Task 1: Create empty package directories and `__init__.py` files

**Files:**
- Create: `src/charith/causal_engine/__init__.py`
- Create: `src/charith/alfa_loop/__init__.py`
- Create: `src/charith/full_stack/__init__.py`
- Create: `tests/fixtures/__init__.py`

**Step 1: Create the directories and empty init files**

```bash
mkdir -p src/charith/causal_engine src/charith/alfa_loop src/charith/full_stack tests/fixtures
touch src/charith/causal_engine/__init__.py
touch src/charith/alfa_loop/__init__.py
touch src/charith/full_stack/__init__.py
touch tests/fixtures/__init__.py
```

**Step 2: Verify structure**

```bash
ls src/charith/causal_engine src/charith/alfa_loop src/charith/full_stack tests/fixtures
```

Expected: each directory exists with `__init__.py`.

**Step 3: Commit**

```bash
git add src/charith/causal_engine src/charith/alfa_loop src/charith/full_stack tests/fixtures
git commit -m "scaffold: create new packages for full-stack agent"
```

---

### Task 2: Port ArcTableModel (table_model.py)

**Files:**
- Create: `src/charith/causal_engine/table_model.py`
- Create: `tests/test_causal_engine_table.py`

**Context:** This is a direct port of `ExpandableTableModel` from c1c2-hybrid. Keys change from `cell` (int 0-63) to `action_id` (int 1-8). Values become frozensets of stringified `Change` descriptions (since ARC change objects aren't directly hashable). All other logic — expansion slots, predict fallback order, one-way `enable_expansion` — is preserved.

**Step 1: Write the failing tests**

Create `tests/test_causal_engine_table.py`:

```python
"""Tests for ArcTableModel — ported from c1c2-hybrid."""
import pytest
from charith.causal_engine.table_model import ArcTableModel


def test_record_and_predict_single_action():
    t = ArcTableModel(num_actions=8)
    t.record(action=1, changes={"red_moved_up"})
    t.record(action=1, changes={"red_moved_up"})
    t.record(action=1, changes={"red_moved_up"})

    pred = t.predict(action=1, target="red_moved_up")
    assert pred[0] is True
    assert pred[1] == 1.0
    assert pred[2] == "single"


def test_predict_unseen_returns_unseen():
    t = ArcTableModel(num_actions=8)
    pred = t.predict(action=5, target="anything")
    assert pred == (False, 0.0, "unseen")


def test_enable_expansion_is_one_way():
    t = ArcTableModel(num_actions=8)
    assert t.enable_expansion("sequential", "test") is True
    assert t.sequence_enabled is True
    # Re-enabling returns False (already enabled)
    assert t.enable_expansion("sequential", "test2") is False


def test_get_active_expansions_reports_enabled():
    t = ArcTableModel(num_actions=8)
    assert t.get_active_expansions() == ["single"]
    t.enable_expansion("sequential", "test")
    assert "sequential" in t.get_active_expansions()
    t.enable_expansion("context", "test")
    assert "context" in t.get_active_expansions()


def test_sequence_prediction_when_enabled():
    t = ArcTableModel(num_actions=8)
    t.enable_expansion("sequential", "test")

    # Record sequence: action 1 -> action 2 -> target changes
    t.record(action=1, changes=set())
    t.record(action=2, changes={"effect_fired"})
    t.record(action=1, changes=set())
    t.record(action=2, changes={"effect_fired"})

    pred = t.predict(action=2, target="effect_fired", prev_action=1)
    assert pred[0] is True
    assert pred[2] == "sequence"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_causal_engine_table.py -v
```

Expected: `ModuleNotFoundError: No module named 'charith.causal_engine.table_model'`

**Step 3: Implement `src/charith/causal_engine/table_model.py`**

```python
"""
Expandable table world model for ARC-AGI-3.

Port of ExpandableTableModel from c1c2-hybrid, adapted for ARC:
  - Keys: action_id (1..num_actions) instead of cell indices
  - Values: frozenset of stringified change descriptions

Starts minimal (single-action table only) and expands when triggered by
error structure analysis. Expansion is one-way.

Levels of representation:
  Level 0 (always active): action -> effects
  Level 1 (sequential):    (prev_action, action) -> effects
  Level 2 (context):       (action, context_hash) -> effects
  Level 3 (spatial):       (action, neighbor_hash) -> effects
"""

from collections import defaultdict
from typing import Iterable, Optional, Tuple


class ArcTableModel:
    def __init__(self, num_actions: int = 8):
        self.num_actions = num_actions

        # Level 0: always active
        self.single_table = defaultdict(list)   # action -> [frozenset(change_str)]

        # Level 1: sequential
        self.sequence_table = defaultdict(list)  # (prev_action, action) -> [frozenset]
        self.sequence_enabled = False

        # Level 2: context
        self.context_table = defaultdict(list)   # (action, ctx_hash) -> [frozenset]
        self.context_enabled = False

        # Level 3: spatial (reserved slot, not wired to predict yet)
        self.spatial_table = defaultdict(list)
        self.spatial_enabled = False

        self.prev_action: Optional[int] = None
        self.expansion_history = []
        self.total_observations = 0

    def record(
        self,
        action: int,
        changes: Iterable,
        context: Optional[dict] = None,
    ) -> None:
        """Record an observation. Writes to all currently active tables."""
        # Changes can be a set of strings, a set of arbitrary objects,
        # or a frozenset. Stringify everything for hashability.
        frozen = frozenset(str(c) for c in changes)
        self.total_observations += 1

        self.single_table[action].append(frozen)

        if self.sequence_enabled and self.prev_action is not None:
            self.sequence_table[(self.prev_action, action)].append(frozen)

        if self.context_enabled and context:
            ctx_key = (action, self._hash_context(context))
            self.context_table[ctx_key].append(frozen)

        self.prev_action = action

    def predict(
        self,
        action: int,
        target: Optional[str] = None,
        prev_action: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> Tuple[bool, float, str]:
        """
        Predict whether activating `action` causes `target` (a change
        description string) to appear in the observed changes.

        Checks most-specific table first and falls back:
            sequence -> context -> single -> unseen

        Returns: (predicts_effect, confidence, source).
        If target is None, returns a no-op default.
        """
        if target is None:
            return False, 0.0, "no_target"

        # Level 1: sequence
        if self.sequence_enabled and prev_action is not None:
            key = (prev_action, action)
            obs = self.sequence_table.get(key)
            if obs and len(obs) >= 2:
                times = sum(1 for o in obs if target in o)
                freq = times / len(obs)
                return freq > 0.5, freq, "sequence"

        # Level 2: context
        if self.context_enabled and context:
            ctx_key = (action, self._hash_context(context))
            obs = self.context_table.get(ctx_key)
            if obs and len(obs) >= 2:
                times = sum(1 for o in obs if target in o)
                freq = times / len(obs)
                return freq > 0.5, freq, "context"

        # Level 0: single
        obs = self.single_table.get(action)
        if obs:
            times = sum(1 for o in obs if target in o)
            freq = times / len(obs)
            return freq > 0.5, freq, "single"

        return False, 0.0, "unseen"

    def enable_expansion(self, expansion_type: str, reason: str) -> bool:
        """One-way enable of an expansion slot. Returns True if newly enabled."""
        if expansion_type == "sequential" and not self.sequence_enabled:
            self.sequence_enabled = True
            self.expansion_history.append(
                {"type": "sequential", "reason": reason, "step": self.total_observations}
            )
            return True
        if expansion_type == "context" and not self.context_enabled:
            self.context_enabled = True
            self.expansion_history.append(
                {"type": "context", "reason": reason, "step": self.total_observations}
            )
            return True
        if expansion_type == "spatial" and not self.spatial_enabled:
            self.spatial_enabled = True
            self.expansion_history.append(
                {"type": "spatial", "reason": reason, "step": self.total_observations}
            )
            return True
        return False

    def get_active_expansions(self) -> list:
        active = ["single"]
        if self.sequence_enabled:
            active.append("sequential")
        if self.context_enabled:
            active.append("context")
        if self.spatial_enabled:
            active.append("spatial")
        return active

    def _hash_context(self, context: dict) -> int:
        return hash(frozenset(context.items()))
```

**Step 4: Run tests**

```bash
pytest tests/test_causal_engine_table.py -v
```

Expected: `5 passed`.

**Step 5: Commit**

```bash
git add src/charith/causal_engine/table_model.py tests/test_causal_engine_table.py
git commit -m "feat(causal_engine): port ArcTableModel from c1c2-hybrid"
```

---

### Task 3: Port ArcErrorAnalyzer (error_analyzer.py)

**Files:**
- Create: `src/charith/causal_engine/error_analyzer.py`
- Create: `tests/test_causal_engine_analyzer.py`

**Context:** Direct port of `ErrorAnalyzer` from c1c2-hybrid with the critical `all-errors` mode (not windowed). Keys renamed `cell` → `action`, `prev_cell` → `prev_action`. All four tests (Ljung-Box, Kruskal-by-prev, Kruskal-by-current, variance ratio) preserved. The neutral `_make_summary` format is preserved verbatim because the LLM prompts in Phase 4 depend on it.

**Step 1: Write the failing tests**

Create `tests/test_causal_engine_analyzer.py`:

```python
"""Tests for ArcErrorAnalyzer — ported from c1c2-hybrid."""
import pytest
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer


def test_insufficient_data_under_20():
    a = ArcErrorAnalyzer()
    for i in range(10):
        a.record(step=i, action=1, predicted_right=True, prev_action=None)
    result = a.analyze()
    assert result["sufficient_data"] is False


def test_no_structure_when_all_correct():
    a = ArcErrorAnalyzer()
    for i in range(30):
        a.record(step=i, action=(i % 4) + 1, predicted_right=True,
                 prev_action=(i - 1) % 4 + 1 if i > 0 else None)
    result = a.analyze()
    assert result["sufficient_data"] is True
    assert result["error_rate"] == 0.0
    assert result["any_structure"] is False


def test_kruskal_fires_on_prev_clustered_errors():
    """When errors cluster by previous action, Kruskal-by-prev should fire."""
    a = ArcErrorAnalyzer()
    # 40 observations: errors only when prev_action == 1
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        # Errors concentrated where prev=1, never where prev=2
        correct = (prev == 2)
        a.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    result = a.analyze()
    assert result["sufficient_data"] is True
    # Kruskal should detect the cluster
    assert result["kruskal_fires"] is True
    assert result["any_structure"] is True


def test_summary_always_shows_all_four_tests():
    """The LLM-facing summary must contain all 4 test names even when none fire."""
    a = ArcErrorAnalyzer()
    for i in range(25):
        a.record(step=i, action=1, predicted_right=True, prev_action=None)
    result = a.analyze()
    summary = result["summary"]
    assert "Ljung-Box" in summary
    assert "Kruskal by PREVIOUS action" in summary
    assert "Kruskal by CURRENT cell" in summary  # ported verbatim
    assert "Variance ratio" in summary
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_causal_engine_analyzer.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/causal_engine/error_analyzer.py`**

Copy the full content from `C:\Users\chari\OneDrive\Documents\C1C2 model\c1c2-hybrid\src\engine\error_analyzer.py` with these minimal renames:

- Class name: `ErrorAnalyzer` → `ArcErrorAnalyzer`
- Parameter name: `cell` → `action` throughout `record()` signature
- Parameter name: `prev_cell` → `prev_action` throughout
- Dictionary key: `'cell'` → `'action'` in the error records
- Dictionary key: `'prev_cell'` → `'prev_action'` in the error records
- Inside `_make_summary`: keep the string `"Kruskal by CURRENT cell"` as-is (the LLM prompt in Phase 4 is designed around this wording — changing it breaks the prompt)

Full ported code to write:

```python
"""
Error structure analyzer for ARC-AGI-3.

Port of c1c2-hybrid ErrorAnalyzer with all-errors mode (not windowed).
Runs Ljung-Box, Kruskal (by prev_action and by action), and variance-ratio
tests on accumulated prediction errors. Produces a structured result AND
a neutral human-readable summary for LLM consumption in Phase 4.
"""

import numpy as np
from scipy import stats
from collections import defaultdict
from typing import Optional


class ArcErrorAnalyzer:
    def __init__(self, window_size: int = 50):
        # window_size retained for API compatibility but not used (all-errors mode).
        self.window_size = window_size
        self.errors = []
        self.error_binary = []

    def record(
        self,
        step: int,
        action: int,
        predicted_right: bool,
        prev_action: Optional[int] = None,
        changed: Optional[set] = None,
    ) -> None:
        self.errors.append({
            "step": step,
            "action": action,
            "correct": predicted_right,
            "prev_action": prev_action,
            "changed": changed or set(),
        })
        self.error_binary.append(0 if predicted_right else 1)

    def analyze(self) -> dict:
        """
        Run all statistical tests on ALL accumulated errors.

        Design note: we use all errors (not a sliding window). The Kruskal
        test needs ALL errors visible at once to detect sparse clustering
        by previous action when rules fire rarely.
        """
        recent = self.errors
        recent_binary = self.error_binary

        if len(recent) < 20:
            return {"sufficient_data": False}

        result = {"sufficient_data": True, "n": len(recent)}
        result["error_rate"] = sum(recent_binary) / len(recent_binary)

        # Test 1: Ljung-Box autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            if np.std(recent_binary) > 0:
                lb = acorr_ljungbox(recent_binary, lags=[1, 2, 3], return_df=True)
                lb_pmin = float(lb["lb_pvalue"].min())
                result["ljung_box_p"] = lb_pmin
                result["ljung_box_fires"] = lb_pmin < 0.05
            else:
                result["ljung_box_p"] = 1.0
                result["ljung_box_fires"] = False
        except Exception:
            result["ljung_box_p"] = 1.0
            result["ljung_box_fires"] = False

        # Test 2: Kruskal — errors grouped by previous action
        prev_groups = defaultdict(list)
        for e in recent:
            if e["prev_action"] is not None:
                prev_groups[e["prev_action"]].append(0 if e["correct"] else 1)

        valid_prev = [g for g in prev_groups.values() if len(g) >= 3]
        if len(valid_prev) >= 2 and any(np.std(g) > 0 for g in valid_prev):
            try:
                kr = stats.kruskal(*valid_prev)
                result["kruskal_p"] = float(kr.pvalue)
                result["kruskal_fires"] = kr.pvalue < 0.05
            except Exception:
                result["kruskal_p"] = 1.0
                result["kruskal_fires"] = False
        else:
            result["kruskal_p"] = 1.0
            result["kruskal_fires"] = False

        # Test 3: Kruskal — errors grouped by current action
        action_groups = defaultdict(list)
        for e in recent:
            action_groups[e["action"]].append(0 if e["correct"] else 1)

        valid_act = [g for g in action_groups.values() if len(g) >= 3]
        if len(valid_act) >= 2 and any(np.std(g) > 0 for g in valid_act):
            try:
                kr_a = stats.kruskal(*valid_act)
                result["kruskal_cell_p"] = float(kr_a.pvalue)
                result["kruskal_cell_fires"] = kr_a.pvalue < 0.05
            except Exception:
                result["kruskal_cell_p"] = 1.0
                result["kruskal_cell_fires"] = False
        else:
            result["kruskal_cell_p"] = 1.0
            result["kruskal_cell_fires"] = False

        # Test 4: Variance ratio (early vs late errors)
        mid = len(recent_binary) // 2
        var_early = np.var(recent_binary[:mid]) if mid > 0 else 0
        var_late = np.var(recent_binary[mid:]) if mid > 0 else 0
        result["variance_ratio"] = float(var_late / (var_early + 1e-10))
        result["variance_fires"] = result["variance_ratio"] > 2.0

        result["any_structure"] = any([
            result.get("ljung_box_fires", False),
            result.get("kruskal_fires", False),
            result.get("kruskal_cell_fires", False),
            result.get("variance_fires", False),
        ])
        result["summary"] = self._make_summary(result)
        return result

    def _make_summary(self, result: dict) -> str:
        """
        Neutral human-readable error report for LLM consumption.

        ALWAYS shows all 4 p-values (firing or not) so the LLM can see
        near-misses. Interpretations are kept neutral.
        """
        lines = [
            f"Error Analysis Report ({result['n']} recent predictions)",
            f"Overall error rate: {result['error_rate']:.1%}",
            "",
            "Statistical tests (p < 0.05 = FIRES):",
        ]

        lb_p = result.get("ljung_box_p", 1.0)
        lb_mark = "FIRES" if result.get("ljung_box_fires") else "random"
        lines.append(f"  [1] Ljung-Box autocorrelation:       p={lb_p:.4f}  [{lb_mark}]")
        lines.append("      -> Tests if consecutive errors correlate in time")

        kr_p = result.get("kruskal_p", 1.0)
        kr_mark = "FIRES" if result.get("kruskal_fires") else "random"
        lines.append(f"  [2] Kruskal by PREVIOUS action:      p={kr_p:.4f}  [{kr_mark}]")
        lines.append("      -> Tests if errors depend on what was activated LAST step")
        lines.append("      -> If fires: suggests SEQUENTIAL structure (needs prev-action memory)")

        krc_p = result.get("kruskal_cell_p", 1.0)
        krc_mark = "FIRES" if result.get("kruskal_cell_fires") else "random"
        lines.append(f"  [3] Kruskal by CURRENT cell:         p={krc_p:.4f}  [{krc_mark}]")
        lines.append("      -> Tests if errors depend on WHICH action is being taken")
        lines.append("      -> Ambiguous: can fire for sequential OR context-dependent rules")

        vr = result.get("variance_ratio", 1.0)
        vr_mark = "FIRES" if result.get("variance_fires") else "stable"
        lines.append(f"  [4] Variance ratio (late/early):     {vr:.2f}     [{vr_mark}]")
        lines.append("      -> Tests if recent errors are more variable than earlier ones")
        lines.append("      -> If fires: could mean regime change OR new rule firing")

        lines.append("")
        if result.get("any_structure"):
            firing = []
            if result.get("ljung_box_fires"):
                firing.append("Ljung-Box (autocorrelation)")
            if result.get("kruskal_fires"):
                firing.append("Kruskal-by-prev (sequential)")
            if result.get("kruskal_cell_fires"):
                firing.append("Kruskal-by-action (ambiguous)")
            if result.get("variance_fires"):
                firing.append("variance-ratio (regime change)")
            lines.append(f"VERDICT: STRUCTURED errors. Firing tests: {', '.join(firing)}")
            lines.append("  -> Vocabulary may be insufficient. Consider expansion.")
        else:
            lines.append("VERDICT: RANDOM errors, no structure detected.")
            lines.append("  -> Current vocabulary appears sufficient.")

        return "\n".join(lines)
```

**Step 4: Run tests**

```bash
pytest tests/test_causal_engine_analyzer.py -v
```

Expected: `4 passed`.

**Step 5: Commit**

```bash
git add src/charith/causal_engine/error_analyzer.py tests/test_causal_engine_analyzer.py
git commit -m "feat(causal_engine): port ArcErrorAnalyzer with all-errors mode"
```

---

## Phase B — Dataclasses and Match Scoring (Tasks 4-7)

### Task 4: Create hypothesis schema dataclasses

**Files:**
- Create: `src/charith/full_stack/hypothesis_schema.py`
- Create: `tests/test_hypothesis_schema.py`

**Step 1: Write the failing test**

Create `tests/test_hypothesis_schema.py`:

```python
"""Tests for hypothesis_schema dataclasses."""
import pytest
from charith.full_stack.hypothesis_schema import (
    ExpectedOutcome, Hypothesis, ActualObservation
)


def test_expected_outcome_defaults_all_none():
    e = ExpectedOutcome()
    assert e.direction is None
    assert e.magnitude_cells is None
    assert e.no_effect is False


def test_hypothesis_initial_status_untested():
    h = Hypothesis(
        rule="test rule", confidence=0.5, test_action=1,
        expected=ExpectedOutcome(direction="up")
    )
    assert h.status == "untested"
    assert h.match_score is None


def test_actual_observation_construction():
    a = ActualObservation(
        controllable_displacement=None,
        controllable_direction=None,
        controllable_magnitude=0,
        any_color_changes=[],
        new_objects=[],
        removed_objects=[],
        score_changed=False,
    )
    assert a.controllable_magnitude == 0
    assert a.score_changed is False
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_hypothesis_schema.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/full_stack/hypothesis_schema.py`**

```python
"""
Hypothesis data contracts for the CHARITH full stack agent.

The LLM (Phase 2) emits Hypothesis objects with structured ExpectedOutcome.
The Verifier (Phase 3) converts StructuredPercept diffs into ActualObservation
and compares using compute_match_score (see match_score.py).
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Any

Direction = Literal["up", "down", "left", "right", "none"]
HypothesisStatus = Literal[
    "untested", "confirmed", "refuted", "untestable", "ambiguous"
]


@dataclass
class ExpectedOutcome:
    """
    Structured prediction. Every field is optional — None means
    "no claim made" (not penalized, not rewarded).
    """
    direction: Optional[Direction] = None
    magnitude_cells: Optional[int] = None
    object_ref: Optional[str] = None
    color_change_to: Optional[str] = None
    object_appears: Optional[bool] = None
    object_disappears: Optional[bool] = None
    score_change: Optional[bool] = None
    no_effect: bool = False


@dataclass
class Hypothesis:
    rule: str
    confidence: float
    test_action: int
    expected: ExpectedOutcome
    status: HypothesisStatus = "untested"
    actual_summary: Optional[str] = None
    match_score: Optional[float] = None


@dataclass
class ActualObservation:
    """
    Distilled from a StructuredPercept diff, ready for structured matching.

    Any 'Object' here refers to charith.perception.core_knowledge.Object —
    typed as Any to avoid import cycles during scaffolding.
    """
    controllable_displacement: Optional[Tuple[int, int]]
    controllable_direction: Optional[Direction]
    controllable_magnitude: int
    any_color_changes: List[Tuple[Any, int, int]]  # (obj, old_color, new_color)
    new_objects: List[Any]
    removed_objects: List[Any]
    score_changed: bool
```

**Step 4: Run tests**

```bash
pytest tests/test_hypothesis_schema.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add src/charith/full_stack/hypothesis_schema.py tests/test_hypothesis_schema.py
git commit -m "feat(full_stack): add Hypothesis / ExpectedOutcome / ActualObservation dataclasses"
```

---

### Task 5: Write match_score golden tests (RED)

**Files:**
- Create: `tests/test_match_score.py`

**Context:** These 7 golden tests lock in the semantics the user agreed to in the design doc §5.3–5.4. Any future refactor must keep them green.

**Step 1: Write the 7 golden tests**

Create `tests/test_match_score.py`:

```python
"""
Golden tests for compute_match_score.
Locks in the semantics from the design doc §5.3–5.4.
"""
import pytest
from charith.full_stack.hypothesis_schema import ExpectedOutcome, ActualObservation
from charith.full_stack.match_score import compute_match_score


def _empty_actual() -> ActualObservation:
    return ActualObservation(
        controllable_displacement=None,
        controllable_direction=None,
        controllable_magnitude=0,
        any_color_changes=[],
        new_objects=[],
        removed_objects=[],
        score_changed=False,
    )


def test_no_effect_perfect_match():
    expected = ExpectedOutcome(no_effect=True)
    actual = _empty_actual()
    assert compute_match_score(expected, actual) == 1.0


def test_no_effect_violated():
    expected = ExpectedOutcome(no_effect=True)
    actual = _empty_actual()
    actual.controllable_displacement = (-5, 0)
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 5
    assert compute_match_score(expected, actual) == 0.0


def test_direction_match_magnitude_off_by_one_tolerated():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5)
    actual = _empty_actual()
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 4  # off by 1
    assert compute_match_score(expected, actual) == 1.0


def test_direction_match_magnitude_off_by_two_fails_magnitude():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5)
    actual = _empty_actual()
    actual.controllable_direction = "up"
    actual.controllable_magnitude = 3  # off by 2
    # direction: 1.0, magnitude: 0.0, mean: 0.5
    assert compute_match_score(expected, actual) == 0.5


def test_mean_aggregation_three_fields():
    expected = ExpectedOutcome(direction="up", magnitude_cells=5, score_change=True)
    actual = _empty_actual()
    actual.controllable_direction = "up"        # match
    actual.controllable_magnitude = 5           # match
    actual.score_changed = False                # miss
    # 1 + 1 + 0 = 2/3 ≈ 0.667
    assert abs(compute_match_score(expected, actual) - 2/3) < 0.01


def test_empty_expected_returns_neutral():
    expected = ExpectedOutcome()  # all None, no_effect=False
    actual = _empty_actual()
    assert compute_match_score(expected, actual) == 0.5


def test_color_match_case_insensitive():
    expected = ExpectedOutcome(color_change_to="Green")
    actual = _empty_actual()
    actual.any_color_changes = [(None, 0, "green")]
    assert compute_match_score(expected, actual) == 1.0
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_match_score.py -v
```

Expected: `ModuleNotFoundError: No module named 'charith.full_stack.match_score'`.

**Step 3: Commit the RED tests**

```bash
git add tests/test_match_score.py
git commit -m "test(match_score): add 7 golden tests for compute_match_score"
```

---

### Task 6: Implement compute_match_score (GREEN)

**Files:**
- Create: `src/charith/full_stack/match_score.py`

**Context:** This is the user's exact implementation from the design doc §5.3, copied verbatim. Mean aggregation, ±1 magnitude tolerance, liberal object_ref. Do not modify the logic — the golden tests lock in these semantics.

**Step 1: Implement the module**

```python
"""
compute_match_score — locks in Section 5.3–5.4 of the design doc.

Compares an LLM-emitted ExpectedOutcome to a mechanistically-derived
ActualObservation and returns a score in [0, 1].

Design calls (user-locked):
  - Mean aggregation (partial credit)
  - ±1 cell magnitude tolerance
  - object_ref deliberately NOT checked (liberal for ls20)
"""

from charith.full_stack.hypothesis_schema import ExpectedOutcome, ActualObservation


def compute_match_score(expected: ExpectedOutcome, actual: ActualObservation) -> float:
    """
    Return a match score in [0, 1].

    See design doc §5.3 for exact rules.
    """
    # Special case: no_effect claim
    if expected.no_effect:
        is_empty = (
            actual.controllable_magnitude == 0
            and not actual.any_color_changes
            and not actual.new_objects
            and not actual.removed_objects
            and not actual.score_changed
        )
        return 1.0 if is_empty else 0.0

    scores = []

    # Direction: strict match
    if expected.direction is not None:
        if expected.direction == "none":
            scores.append(1.0 if actual.controllable_magnitude == 0 else 0.0)
        else:
            scores.append(
                1.0 if actual.controllable_direction == expected.direction else 0.0
            )

    # Magnitude: ±1 tolerance
    if expected.magnitude_cells is not None and actual.controllable_magnitude > 0:
        diff = abs(actual.controllable_magnitude - expected.magnitude_cells)
        scores.append(1.0 if diff <= 1 else 0.0)

    # Color change: substring, case-insensitive
    if expected.color_change_to is not None:
        matched = any(
            expected.color_change_to.lower() in str(new_c).lower()
            for _, _, new_c in actual.any_color_changes
        )
        scores.append(1.0 if matched else 0.0)

    # Object appears / disappears
    if expected.object_appears is not None:
        scores.append(1.0 if expected.object_appears == bool(actual.new_objects) else 0.0)
    if expected.object_disappears is not None:
        scores.append(
            1.0 if expected.object_disappears == bool(actual.removed_objects) else 0.0
        )

    # Score change
    if expected.score_change is not None:
        scores.append(1.0 if expected.score_change == actual.score_changed else 0.0)

    # Aggregation: MEAN
    if not scores:
        return 0.5  # LLM claimed nothing specific — ambiguous
    return sum(scores) / len(scores)
```

**Step 2: Run tests**

```bash
pytest tests/test_match_score.py -v
```

Expected: `7 passed`.

**Step 3: Commit**

```bash
git add src/charith/full_stack/match_score.py
git commit -m "feat(full_stack): implement compute_match_score (mean, ±1, liberal)"
```

---

### Task 7: Add budgets and results dataclasses

**Files:**
- Create: `src/charith/full_stack/budgets.py`
- Create: `src/charith/full_stack/results.py`
- Create: `tests/test_full_stack_results.py`

**Step 1: Write the failing test**

Create `tests/test_full_stack_results.py`:

```python
"""Tests for budgets and result dataclasses."""
from charith.full_stack.budgets import AgentBudgets
from charith.full_stack.results import AttemptResult, LevelResult


def test_budgets_default_values():
    b = AgentBudgets()
    assert b.max_actions_per_level == 150
    assert b.max_llm_calls_per_level == 8
    assert b.max_plan_length == 20
    assert b.consecutive_surprises_to_halt == 3


def test_attempt_result_construction():
    r = AttemptResult(
        completed=False, actions_taken=25, llm_calls=3,
        reason="delta_spike", phase_reached=6,
        hypotheses_generated=4, hypotheses_confirmed=2,
        hypotheses_refuted=1, expansions_triggered=[],
        final_error_summary="ok",
    )
    assert r.completed is False
    assert r.reason == "delta_spike"


def test_level_result_totals():
    a1 = AttemptResult(
        completed=True, actions_taken=20, llm_calls=3,
        reason="success", phase_reached=6,
        hypotheses_generated=3, hypotheses_confirmed=3,
        hypotheses_refuted=0, expansions_triggered=[],
        final_error_summary="ok",
    )
    level = LevelResult(
        completed=True, attempts=[a1],
        total_actions=20, total_llm_calls=3, final_table_stats={}
    )
    assert level.completed is True
    assert level.total_actions == 20
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_full_stack_results.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/full_stack/budgets.py`**

```python
"""Agent budget caps. Frozen dataclass to prevent accidental mutation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBudgets:
    max_actions_per_level: int = 150
    max_llm_calls_per_level: int = 8
    max_attempts_per_level: int = 3
    max_expansion_cycles_per_attempt: int = 2
    explore_num_actions: int = 8
    max_hypotheses_to_verify: int = 5
    max_plan_length: int = 20
    consecutive_surprises_to_halt: int = 3
```

**Step 4: Implement `src/charith/full_stack/results.py`**

```python
"""Result dataclasses for the full-stack agent."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AttemptResult:
    completed: bool
    actions_taken: int
    llm_calls: int
    reason: str
    phase_reached: int
    hypotheses_generated: int
    hypotheses_confirmed: int
    hypotheses_refuted: int
    expansions_triggered: List[str]
    final_error_summary: str


@dataclass
class LevelResult:
    completed: bool
    attempts: List[AttemptResult]
    total_actions: int
    total_llm_calls: int
    final_table_stats: Dict
```

**Step 5: Run tests**

```bash
pytest tests/test_full_stack_results.py -v
```

Expected: `3 passed`.

**Step 6: Commit**

```bash
git add src/charith/full_stack/budgets.py src/charith/full_stack/results.py tests/test_full_stack_results.py
git commit -m "feat(full_stack): add budgets and result dataclasses"
```

---

## Phase C — StructuredPercept Adapter (Task 8)

### Task 8: Implement StructuredPercept diff → ActualObservation adapter

**Files:**
- Create: `src/charith/full_stack/percept_diff.py`
- Create: `tests/test_percept_diff.py`

**Context (read before implementing):** This is the critical adapter that converts the existing perception output into the format the verifier needs. From `src/charith/perception/core_knowledge.py`:
- `StructuredPercept.objects: List[Object]` — no `controllable` field; you must find it.
- `Object.centroid: Tuple[float, float]` — `(row, col)` floats.
- `Object.color: int` — integer color ID.
- `ObjectTracker.match(prev_objects, curr_objects) -> List[Tuple[int, int]]` — returns `(prev_id, curr_id)` pairs matched by color then proximity.

**Controllable detection strategy:** For Milestone 1 we use a simple heuristic — the controllable is the object that MOVED the furthest between before and after percepts. In ls20 this reliably finds the sprite. For multi-object games we'll need the `AgencyPrior.detect_controllable_objects` path; that's Milestone 2.

**Score detection:** Compare sums of `raw_grid` pixels in a fixed score region. ls20 doesn't expose a score channel directly; the simplest heuristic is "did ANY cell outside the main playing area change". We'll use a conservative stub: `score_changed = False` for Milestone 1 and revisit in smoke test.

**Step 1: Write the failing tests**

Create `tests/test_percept_diff.py`:

```python
"""Tests for StructuredPercept diff adapter."""
import numpy as np
from charith.perception.core_knowledge import Object, Cell, StructuredPercept
from charith.full_stack.percept_diff import diff_to_actual_observation


def _make_percept(objects):
    return StructuredPercept(
        raw_grid=np.zeros((10, 10), dtype=int),
        objects=objects,
        spatial_relations=[],
        color_counts={},
        grid_dims=(10, 10),
        background_color=0,
        symmetry={},
        unique_colors=set(),
        object_count=len(objects),
        timestamp=0.0,
    )


def _make_obj(oid, color, centroid):
    return Object(
        object_id=oid,
        cells=frozenset(),
        color=color,
        bbox=(0, 0, 0, 0),
        size=1,
        centroid=centroid,
        shape_hash=0,
    )


def test_diff_no_change_gives_empty_observation():
    obj = _make_obj(1, 5, (3.0, 3.0))
    before = _make_percept([obj])
    after = _make_percept([obj])
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_magnitude == 0
    assert actual.any_color_changes == []
    assert actual.new_objects == []
    assert actual.removed_objects == []


def test_diff_detects_upward_movement():
    before = _make_percept([_make_obj(1, 5, (8.0, 4.0))])
    after = _make_percept([_make_obj(1, 5, (3.0, 4.0))])   # row decreased → up
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_direction == "up"
    assert actual.controllable_magnitude == 5


def test_diff_detects_rightward_movement():
    before = _make_percept([_make_obj(1, 5, (4.0, 2.0))])
    after = _make_percept([_make_obj(1, 5, (4.0, 7.0))])
    actual = diff_to_actual_observation(before, after)
    assert actual.controllable_direction == "right"
    assert actual.controllable_magnitude == 5


def test_diff_detects_new_object():
    before = _make_percept([_make_obj(1, 5, (3.0, 3.0))])
    after = _make_percept([
        _make_obj(1, 5, (3.0, 3.0)),
        _make_obj(2, 7, (7.0, 7.0)),   # new object, different color
    ])
    actual = diff_to_actual_observation(before, after)
    assert len(actual.new_objects) == 1


def test_diff_detects_removed_object():
    before = _make_percept([
        _make_obj(1, 5, (3.0, 3.0)),
        _make_obj(2, 7, (7.0, 7.0)),
    ])
    after = _make_percept([_make_obj(1, 5, (3.0, 3.0))])
    actual = diff_to_actual_observation(before, after)
    assert len(actual.removed_objects) == 1
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_percept_diff.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/full_stack/percept_diff.py`**

```python
"""
Convert a StructuredPercept diff into an ActualObservation.

For Milestone 1 (ls20 focus):
- Controllable = the matched object that moved the most
- Direction = sign of the larger axis displacement
- Magnitude = L-inf distance (max of |row_delta|, |col_delta|)
- Color changes = object pairs matched with different colors
- New / removed objects = unmatched after / unmatched before
- Score = stub False for Milestone 1

Uses ObjectTracker.match() for identity persistence across frames.
"""

from typing import Optional, Tuple

from charith.perception.core_knowledge import StructuredPercept
from charith.perception.object_tracker import ObjectTracker
from charith.full_stack.hypothesis_schema import ActualObservation, Direction


_TRACKER = ObjectTracker()


def diff_to_actual_observation(
    before: StructuredPercept,
    after: StructuredPercept,
) -> ActualObservation:
    pairs = _TRACKER.match(before.objects, after.objects)

    # Build lookup tables
    before_by_id = {o.object_id: o for o in before.objects}
    after_by_id = {o.object_id: o for o in after.objects}

    # Compute per-object displacements
    max_disp: Optional[Tuple[int, int]] = None
    max_magnitude = 0
    for prev_id, curr_id in pairs:
        prev = before_by_id[prev_id]
        curr = after_by_id[curr_id]
        dr = int(round(curr.centroid[0] - prev.centroid[0]))
        dc = int(round(curr.centroid[1] - prev.centroid[1]))
        mag = max(abs(dr), abs(dc))
        if mag > max_magnitude:
            max_magnitude = mag
            max_disp = (dr, dc)

    # Derive direction from the dominant-axis displacement
    direction: Optional[Direction] = None
    if max_disp is not None and max_magnitude > 0:
        dr, dc = max_disp
        if abs(dr) >= abs(dc):
            direction = "up" if dr < 0 else "down"
        else:
            direction = "left" if dc < 0 else "right"

    # Color changes: matched pairs where color differs
    any_color_changes = []
    for prev_id, curr_id in pairs:
        prev = before_by_id[prev_id]
        curr = after_by_id[curr_id]
        if prev.color != curr.color:
            any_color_changes.append((curr, prev.color, curr.color))

    # New objects: in after but not in any matched pair
    matched_curr = {curr_id for _, curr_id in pairs}
    new_objects = [o for o in after.objects if o.object_id not in matched_curr]

    # Removed objects: in before but not in any matched pair
    matched_prev = {prev_id for prev_id, _ in pairs}
    removed_objects = [o for o in before.objects if o.object_id not in matched_prev]

    return ActualObservation(
        controllable_displacement=max_disp,
        controllable_direction=direction,
        controllable_magnitude=max_magnitude,
        any_color_changes=any_color_changes,
        new_objects=new_objects,
        removed_objects=removed_objects,
        score_changed=False,  # Milestone 1 stub
    )
```

**Step 4: Run tests**

```bash
pytest tests/test_percept_diff.py -v
```

Expected: `5 passed`.

**Step 5: Commit**

```bash
git add src/charith/full_stack/percept_diff.py tests/test_percept_diff.py
git commit -m "feat(full_stack): add StructuredPercept diff → ActualObservation adapter"
```

---

## Phase D — Mock Fixtures (Tasks 9-10)

### Task 9: Implement MockArcEnv

**Files:**
- Create: `tests/fixtures/mock_env.py`
- Create: `tests/test_mock_env.py`

**Step 1: Write the failing test**

Create `tests/test_mock_env.py`:

```python
"""Smoke test for MockArcEnv — the mock must behave deterministically."""
import numpy as np
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def test_reset_returns_initial_grid():
    grid = np.zeros((10, 10), dtype=int)
    grid[4, 4] = 12
    env = MockArcEnv(initial_grid=grid, rules={})
    frame = env.reset()
    assert frame.grid[4, 4] == 12


def test_step_applies_rule():
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 12
    env = MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(color=12, dr=-1, dc=0)},
    )
    env.reset()
    frame, _, done, _ = env.step(1)
    # Sprite at (5,5) -> (4,5)
    assert frame.grid[4, 5] == 12
    assert frame.grid[5, 5] == 0
    assert done is False


def test_step_with_no_rule_is_noop():
    grid = np.zeros((10, 10), dtype=int)
    grid[5, 5] = 12
    env = MockArcEnv(initial_grid=grid, rules={})
    env.reset()
    frame, _, done, _ = env.step(99)
    assert frame.grid[5, 5] == 12
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_mock_env.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `tests/fixtures/mock_env.py`**

```python
"""
Deterministic mock for arc_agi.Environment.

Designed to exercise the full 6-phase loop without a real game.
Rules are callables: (grid, state) -> (new_grid, new_state).
The `state` dict lets rules remember things (e.g., prev_action).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MockFrame:
    grid: np.ndarray
    win_levels: List[int] = field(default_factory=lambda: [1])
    score: int = 0
    frame: Optional[list] = None  # populated by __post_init__

    def __post_init__(self):
        # Expose frame[0] as the grid for compatibility with arc_agi
        self.frame = [self.grid]


RuleFn = Callable[[np.ndarray, dict], Tuple[np.ndarray, dict]]


class MockArcEnv:
    """Deterministic fake environment driven by a dict of grid transforms."""

    def __init__(
        self,
        initial_grid: np.ndarray,
        rules: Dict[int, RuleFn],
        score_triggers: Optional[Dict[int, int]] = None,
        done_on_action: Optional[int] = None,
    ):
        self.initial_grid = initial_grid.copy()
        self.rules = rules
        self.score_triggers = score_triggers or {}
        self.done_on_action = done_on_action
        self.grid: np.ndarray = self.initial_grid.copy()
        self.score: int = 0
        self.done: bool = False
        self._internal_state: dict = {}

    def reset(self) -> MockFrame:
        self.grid = self.initial_grid.copy()
        self.score = 0
        self.done = False
        self._internal_state = {}
        return MockFrame(self.grid.copy(), [1], self.score)

    def get_observation(self) -> MockFrame:
        return MockFrame(self.grid.copy(), [1], self.score)

    def step(self, action_id: int) -> Tuple[MockFrame, float, bool, dict]:
        if action_id in self.rules:
            self.grid, self._internal_state = self.rules[action_id](
                self.grid, self._internal_state
            )
        if action_id in self.score_triggers:
            self.score += self.score_triggers[action_id]
        if self.done_on_action is not None and action_id == self.done_on_action:
            self.done = True
        return MockFrame(self.grid.copy(), [1], self.score), 0.0, self.done, {}


def move_obj_by(color: int, dr: int, dc: int) -> RuleFn:
    """Factory: returns a rule that moves all pixels of `color` by (dr, dc)."""
    def rule(grid: np.ndarray, state: dict) -> Tuple[np.ndarray, dict]:
        new_grid = grid.copy()
        rows, cols = np.where(grid == color)
        # Clear old positions
        for r, c in zip(rows, cols):
            new_grid[r, c] = 0
        # Set new positions (clipped to grid)
        h, w = grid.shape
        for r, c in zip(rows, cols):
            nr, nc = max(0, min(h - 1, r + dr)), max(0, min(w - 1, c + dc))
            new_grid[nr, nc] = color
        return new_grid, state
    return rule
```

**Step 4: Run tests**

```bash
pytest tests/test_mock_env.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add tests/fixtures/mock_env.py tests/test_mock_env.py
git commit -m "test(fixtures): add MockArcEnv with move-by rule factory"
```

---

### Task 10: Implement MockOllamaReasoner

**Files:**
- Create: `tests/fixtures/mock_llm.py`
- Create: `tests/test_mock_llm.py`

**Step 1: Write the failing test**

Create `tests/test_mock_llm.py`:

```python
"""Smoke test for MockOllamaReasoner."""
from tests.fixtures.mock_llm import MockOllamaReasoner, DEFAULT_HYPOTHESIZE_RESPONSE


def test_mock_llm_returns_fixed_hypothesize_response():
    llm = MockOllamaReasoner()
    result = llm.reason_json("You are discovering the rules...", "evidence here")
    assert "hypotheses" in result
    assert llm.call_count == 1


def test_mock_llm_routes_by_keyword():
    llm = MockOllamaReasoner(
        expansion_response={"type": "sequential", "reason": "test"},
        plan_response={"plan": [1, 2, 3], "reasoning": "r"},
    )
    exp = llm.reason_json("Available expansions: sequential...", "user")
    assert exp["type"] == "sequential"
    plan = llm.reason_json("You are planning actions...", "user")
    assert plan["plan"] == [1, 2, 3]
    assert llm.call_count == 2


def test_mock_llm_records_call_log():
    llm = MockOllamaReasoner()
    llm.reason_json("sys1", "user1")
    llm.reason_json("sys2", "user2")
    assert len(llm.calls) == 2
    assert llm.calls[0] == ("sys1", "user1")
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_mock_llm.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `tests/fixtures/mock_llm.py`**

```python
"""
Canned-response LLM for testing.

Routes responses by keyword in the system prompt:
  - "discover" or "hypothesis" → hypothesize_response
  - "expansion"                → expansion_response
  - "planning" or "plan"       → plan_response

All responses are dicts matching the real reason_json output shape.
"""

from typing import Any, Dict, List, Optional, Tuple


DEFAULT_HYPOTHESIZE_RESPONSE = {
    "hypotheses": [
        {
            "rule": "Action 1 moves the controllable up by 5 cells",
            "confidence": 0.8,
            "test_action": 1,
            "expected": {
                "direction": "up",
                "magnitude_cells": 5,
                "object_ref": "controllable",
                "no_effect": False,
            },
        }
    ],
    "goal_guess": "Move controllable to target",
}

DEFAULT_EXPANSION_RESPONSE = {"type": "none", "reason": "errors are random"}

DEFAULT_PLAN_RESPONSE = {"plan": [1, 1, 1, 1], "reasoning": "repeat action 1"}


class MockOllamaReasoner:
    """Returns fixed JSON responses routed by keyword in system prompt."""

    def __init__(
        self,
        hypothesize_response: Optional[Dict] = None,
        expansion_response: Optional[Dict] = None,
        plan_response: Optional[Dict] = None,
    ):
        self.hypothesize_response = hypothesize_response or DEFAULT_HYPOTHESIZE_RESPONSE
        self.expansion_response = expansion_response or DEFAULT_EXPANSION_RESPONSE
        self.plan_response = plan_response or DEFAULT_PLAN_RESPONSE
        self.call_count: int = 0
        self.calls: List[Tuple[str, str]] = []

    def reason_json(self, system: str, user: str) -> Dict[str, Any]:
        self.call_count += 1
        self.calls.append((system, user))

        lower = system.lower()
        if "discover" in lower or "hypothesis" in lower or "hypotheses" in lower:
            return self.hypothesize_response
        if "expansion" in lower:
            return self.expansion_response
        if "planning" in lower or "plan " in lower or "action sequence" in lower:
            return self.plan_response
        return {"raw": "unknown", "parse_error": True}

    def reason(self, system: str, user: str) -> str:
        """Fallback plain-text method for counterfactual queries (unused in Milestone 1)."""
        self.call_count += 1
        return "mock response"
```

**Step 4: Run tests**

```bash
pytest tests/test_mock_llm.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add tests/fixtures/mock_llm.py tests/test_mock_llm.py
git commit -m "test(fixtures): add MockOllamaReasoner with keyword routing"
```

---

## Phase E — ALFA Loop Phases (Tasks 11-16)

**Note on Phase tests:** Each of the next 6 tasks follows the same pattern — write a minimal phase module, write a phase test using the mocks, get them green, commit. We build from Phase 1 (simplest) to Phase 6 (most complex). Phase 3 is the most critical — pay extra attention there.

### Task 11: Phase 1 — Explorer

**Files:**
- Create: `src/charith/alfa_loop/explorer.py`
- Create: `tests/test_phase1_explorer.py`

**Step 1: Write the failing test**

Create `tests/test_phase1_explorer.py`:

```python
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
    # After exploring, table should have entries for all 8 actions
    assert len(table.single_table) == 8


def test_explore_halts_on_done():
    env = _ls20_like_env()
    env.done_on_action = 3
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    explorer = Explorer(env, perception, table)

    evidence = explorer.explore(num_actions=8)
    # Should stop at action 3 (which triggers done)
    assert len(evidence) == 3
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase1_explorer.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/explorer.py`**

```python
"""
Phase 1: Explore.

Try each action (1..num_actions) once. Record what happens in the causal
table. No LLM calls. No error recording (no predictions yet).

If the environment signals done=True mid-exploration, return immediately
with whatever evidence was gathered.
"""

from dataclasses import dataclass
from typing import List, Optional

from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept
from charith.causal_engine.table_model import ArcTableModel


@dataclass
class Evidence:
    action: int
    percept_before: StructuredPercept
    percept_after: StructuredPercept
    changes: list        # raw diff objects (for downstream use)
    description: str     # short text for LLM prompt
    reward: float
    done: bool


class Explorer:
    def __init__(self, env, perception: CoreKnowledgePerception, table: ArcTableModel):
        self.env = env
        self.perception = perception
        self.table = table

    def explore(self, num_actions: int = 8) -> List[Evidence]:
        evidence: List[Evidence] = []

        for action_id in range(1, num_actions + 1):
            obs_before = self.env.get_observation()
            percept_before = self.perception.perceive(obs_before.frame[0])

            obs_after, reward, done, _info = self.env.step(action_id)
            percept_after = self.perception.perceive(obs_after.frame[0])

            changes = self._summarise_changes(percept_before, percept_after)
            description = self._describe(action_id, changes)

            # Record in the table as stringified change summary
            self.table.record(action=action_id, changes=[description])

            evidence.append(
                Evidence(
                    action=action_id,
                    percept_before=percept_before,
                    percept_after=percept_after,
                    changes=changes,
                    description=description,
                    reward=float(reward),
                    done=bool(done),
                )
            )

            if done:
                break

        return evidence

    def _summarise_changes(self, before: StructuredPercept, after: StructuredPercept) -> list:
        """Extract a coarse change description for the LLM prompt."""
        diffs = []
        before_ids = {o.object_id for o in before.objects}
        after_ids = {o.object_id for o in after.objects}
        if len(before.objects) != len(after.objects):
            diffs.append(f"object count {len(before.objects)}->{len(after.objects)}")
        return diffs

    def _describe(self, action_id: int, changes: list) -> str:
        if not changes:
            return f"action {action_id}: no change"
        return f"action {action_id}: {'; '.join(changes)}"
```

**Step 4: Run tests**

```bash
pytest tests/test_phase1_explorer.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/explorer.py tests/test_phase1_explorer.py
git commit -m "feat(alfa_loop): implement Phase 1 Explorer"
```

---

### Task 12: Phase 2 — Hypothesizer

**Files:**
- Create: `src/charith/alfa_loop/hypothesizer.py`
- Create: `tests/test_phase2_hypothesizer.py`

**Step 1: Write the failing test**

Create `tests/test_phase2_hypothesizer.py`:

```python
"""Tests for Phase 2 — Hypothesizer."""
from charith.alfa_loop.explorer import Evidence
from charith.alfa_loop.hypothesizer import Hypothesizer
from charith.full_stack.hypothesis_schema import ExpectedOutcome
from tests.fixtures.mock_llm import MockOllamaReasoner


def _fake_evidence() -> list:
    return [
        Evidence(
            action=i,
            percept_before=None,
            percept_after=None,
            changes=[],
            description=f"action {i}: no change",
            reward=0.0,
            done=False,
        )
        for i in range(1, 9)
    ]


def test_hypothesize_calls_llm_once():
    llm = MockOllamaReasoner()
    h = Hypothesizer(llm)
    hypotheses, goal = h.generate(_fake_evidence(), active_expansions=["single"])
    assert llm.call_count == 1
    assert len(hypotheses) >= 1
    assert isinstance(goal, str)


def test_hypothesize_parses_structured_expected_outcome():
    llm = MockOllamaReasoner()
    h = Hypothesizer(llm)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    hyp = hypotheses[0]
    assert isinstance(hyp.expected, ExpectedOutcome)
    assert hyp.expected.direction == "up"
    assert hyp.expected.magnitude_cells == 5


def test_hypothesize_empty_llm_response_returns_empty_list():
    llm = MockOllamaReasoner(hypothesize_response={"hypotheses": []})
    h = Hypothesizer(llm)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    assert hypotheses == []


def test_hypothesize_invalid_test_action_marked_untestable():
    llm = MockOllamaReasoner(
        hypothesize_response={
            "hypotheses": [
                {"rule": "bad", "confidence": 0.5, "test_action": 99,
                 "expected": {"direction": "up"}}
            ],
            "goal_guess": "x",
        }
    )
    h = Hypothesizer(llm, num_actions=8)
    hypotheses, _ = h.generate(_fake_evidence(), active_expansions=["single"])
    assert hypotheses[0].status == "untestable"
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase2_hypothesizer.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/hypothesizer.py`**

```python
"""
Phase 2: Hypothesize.

One LLM call. Build prompt from evidence + current vocabulary, get
structured hypotheses back. Validate and normalize into Hypothesis objects.
"""

from typing import List, Tuple

from charith.alfa_loop.explorer import Evidence
from charith.full_stack.hypothesis_schema import (
    ExpectedOutcome, Hypothesis,
)


SYSTEM_PROMPT = """You are discovering the rules of an unknown grid game by hypothesis.
You just observed 8 actions being tried once each.

Generate 3-5 hypotheses about what the game's rules might be.

CRITICAL: Each hypothesis must be TESTABLE. Your expected_outcome MUST be STRUCTURED
(schema below). Do not describe expected outcomes in prose. Only populate fields you
are actually predicting; leave others null.

ExpectedOutcome schema:
{
  "direction": "up|down|left|right|none",   // optional
  "magnitude_cells": 5,                       // optional
  "object_ref": "controllable|red|blue|...",  // optional
  "color_change_to": "green",                 // optional
  "object_appears": false,                    // optional
  "object_disappears": false,                 // optional
  "score_change": false,                      // optional
  "no_effect": false                          // set true if you predict nothing
}

Respond ONLY with JSON:
{
  "hypotheses": [
    {
      "rule": "Action 1 moves the controllable up by 5 cells",
      "confidence": 0.8,
      "test_action": 1,
      "expected": {"direction": "up", "magnitude_cells": 5, "object_ref": "controllable"}
    }
  ],
  "goal_guess": "Move controllable to the target"
}
"""


class Hypothesizer:
    def __init__(self, llm, num_actions: int = 8):
        self.llm = llm
        self.num_actions = num_actions

    def generate(
        self,
        evidence: List[Evidence],
        active_expansions: List[str],
    ) -> Tuple[List[Hypothesis], str]:
        evidence_text = "\n".join(f"  {e.description}" for e in evidence)
        user_prompt = (
            f"Exploration results (each action tried once):\n{evidence_text}\n\n"
            f"Current vocabulary: {active_expansions}\n"
            f"What are the game's rules? What is the goal?"
        )

        result = self.llm.reason_json(SYSTEM_PROMPT, user_prompt)

        raw_hyps = result.get("hypotheses", []) if isinstance(result, dict) else []
        goal = result.get("goal_guess", "") if isinstance(result, dict) else ""

        parsed: List[Hypothesis] = []
        for rh in raw_hyps:
            try:
                test_action = int(rh.get("test_action", -1))
            except (TypeError, ValueError):
                test_action = -1

            expected_dict = rh.get("expected", {}) or {}
            expected = ExpectedOutcome(
                direction=expected_dict.get("direction"),
                magnitude_cells=expected_dict.get("magnitude_cells"),
                object_ref=expected_dict.get("object_ref"),
                color_change_to=expected_dict.get("color_change_to"),
                object_appears=expected_dict.get("object_appears"),
                object_disappears=expected_dict.get("object_disappears"),
                score_change=expected_dict.get("score_change"),
                no_effect=bool(expected_dict.get("no_effect", False)),
            )

            h = Hypothesis(
                rule=str(rh.get("rule", "")),
                confidence=float(rh.get("confidence", 0.5)),
                test_action=test_action,
                expected=expected,
            )

            if not (1 <= h.test_action <= self.num_actions):
                h.status = "untestable"

            parsed.append(h)

        return parsed, goal
```

**Step 4: Run tests**

```bash
pytest tests/test_phase2_hypothesizer.py -v
```

Expected: `4 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/hypothesizer.py tests/test_phase2_hypothesizer.py
git commit -m "feat(alfa_loop): implement Phase 2 Hypothesizer with structured schema"
```

---

### Task 13: Phase 3 — Verifier (CRITICAL — read carefully)

**Files:**
- Create: `src/charith/alfa_loop/verifier.py`
- Create: `tests/test_phase3_verifier.py`

**Context:** This is the core contribution of the architecture. The verifier executes each LLM hypothesis as an intervention, computes the actual observation via `diff_to_actual_observation`, and scores with `compute_match_score`. Thresholds: ≥0.70 → confirmed, <0.30 → refuted, otherwise ambiguous.

**Step 1: Write the failing tests**

Create `tests/test_phase3_verifier.py`:

```python
"""Tests for Phase 3 — Verifier (the critical novel phase)."""
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.verifier import Verifier
from charith.full_stack.hypothesis_schema import Hypothesis, ExpectedOutcome
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def _env_action1_moves_up():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12
    return MockArcEnv(
        initial_grid=grid,
        rules={1: move_obj_by(12, -1, 0), 2: move_obj_by(12, 1, 0)},
    )


def test_verify_confirms_correct_hypothesis():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up", magnitude_cells=1),
    )
    verified = v.verify([h])
    assert verified[0].status == "confirmed"
    assert verified[0].match_score >= 0.70


def test_verify_refutes_wrong_direction():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves DOWN",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="down", magnitude_cells=1),
    )
    verified = v.verify([h])
    # Direction: 0, magnitude: 1 → mean 0.5 → ambiguous per thresholds
    # But since magnitude check only fires if actual > 0, and the movement IS
    # by 1, magnitude=1 matches expected=1. So score = (0 + 1) / 2 = 0.5.
    assert verified[0].status == "ambiguous"


def test_verify_records_errors_in_analyzer():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up", magnitude_cells=1),
    )
    v.verify([h])
    assert len(analyzer.errors) == 1


def test_verify_skips_untestable_hypotheses():
    env = _env_action1_moves_up()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    v = Verifier(env, perception, table, analyzer)

    h = Hypothesis(
        rule="bad",
        confidence=0.5,
        test_action=99,  # out of range
        expected=ExpectedOutcome(direction="up"),
    )
    h.status = "untestable"
    verified = v.verify([h])
    assert verified[0].status == "untestable"
    # No env.step taken for this hypothesis
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase3_verifier.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/verifier.py`**

```python
"""
Phase 3: Verify. The novel phase.

For each LLM hypothesis, execute its test_action (an intervention do(X)),
compute the actual observation via structured diff, score via
compute_match_score, and assign status:

    match_score >= 0.70 → 'confirmed'
    match_score <  0.30 → 'refuted'
    otherwise           → 'ambiguous'

No LLM calls here. Pure mechanistic testing.
"""

from typing import List, Optional

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.full_stack.hypothesis_schema import Hypothesis
from charith.full_stack.match_score import compute_match_score
from charith.full_stack.percept_diff import diff_to_actual_observation


CONFIRM_THRESHOLD = 0.70
REFUTE_THRESHOLD = 0.30


class Verifier:
    def __init__(
        self,
        env,
        perception: CoreKnowledgePerception,
        table: ArcTableModel,
        error_analyzer: ArcErrorAnalyzer,
    ):
        self.env = env
        self.perception = perception
        self.table = table
        self.error_analyzer = error_analyzer
        self._step_counter = 0

    def verify(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Test each hypothesis in order. State drift between tests is
        accepted per design §2.4 (user-approved).
        """
        prev_action: Optional[int] = self.table.prev_action

        for h in hypotheses:
            if h.status == "untestable":
                continue

            obs_before = self.env.get_observation()
            percept_before = self.perception.perceive(obs_before.frame[0])

            obs_after, _reward, done, _info = self.env.step(h.test_action)
            percept_after = self.perception.perceive(obs_after.frame[0])

            actual = diff_to_actual_observation(percept_before, percept_after)
            score = compute_match_score(h.expected, actual)

            h.match_score = score
            h.actual_summary = self._summarise(actual)
            if score >= CONFIRM_THRESHOLD:
                h.status = "confirmed"
                predicted_right = True
            elif score < REFUTE_THRESHOLD:
                h.status = "refuted"
                predicted_right = False
            else:
                h.status = "ambiguous"
                predicted_right = False

            # Record in table (action → description)
            change_desc = f"direction={actual.controllable_direction},mag={actual.controllable_magnitude}"
            self.table.record(action=h.test_action, changes=[change_desc])

            # Record in error analyzer
            self.error_analyzer.record(
                step=self._step_counter,
                action=h.test_action,
                predicted_right=predicted_right,
                prev_action=prev_action,
            )
            self._step_counter += 1
            prev_action = h.test_action

            if done:
                break

        return hypotheses

    def _summarise(self, actual) -> str:
        parts = []
        if actual.controllable_magnitude > 0:
            parts.append(f"moved {actual.controllable_direction} by {actual.controllable_magnitude}")
        if actual.any_color_changes:
            parts.append(f"{len(actual.any_color_changes)} color change(s)")
        if actual.new_objects:
            parts.append(f"{len(actual.new_objects)} new object(s)")
        if actual.removed_objects:
            parts.append(f"{len(actual.removed_objects)} removed")
        return ", ".join(parts) if parts else "no change"
```

**Step 4: Run tests**

```bash
pytest tests/test_phase3_verifier.py -v
```

Expected: `4 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/verifier.py tests/test_phase3_verifier.py
git commit -m "feat(alfa_loop): implement Phase 3 Verifier (interventional testing)"
```

---

### Task 14: Phase 4 — Error Checker

**Files:**
- Create: `src/charith/alfa_loop/error_checker.py`
- Create: `tests/test_phase4_error_checker.py`

**Step 1: Write the failing test**

Create `tests/test_phase4_error_checker.py`:

```python
"""Tests for Phase 4 — ErrorChecker."""
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.error_checker import ErrorChecker
from tests.fixtures.mock_llm import MockOllamaReasoner


def test_no_structure_returns_no_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    # 30 correct predictions → no structure
    for i in range(30):
        analyzer.record(step=i, action=(i % 4) + 1, predicted_right=True, prev_action=None)
    llm = MockOllamaReasoner()
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is False
    assert llm.call_count == 0


def test_kruskal_fires_triggers_llm_and_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    # Cluster errors by prev_action → Kruskal fires
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        correct = (prev == 2)  # errors only when prev=1
        analyzer.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    llm = MockOllamaReasoner(expansion_response={"type": "sequential", "reason": "Kruskal fired"})
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is True
    assert result["expansion_type"] == "sequential"
    assert table.sequence_enabled is True
    assert llm.call_count == 1


def test_llm_suggests_none_no_expansion():
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    for i in range(40):
        prev = 1 if i % 2 == 0 else 2
        correct = (prev == 2)
        analyzer.record(step=i, action=3, predicted_right=correct, prev_action=prev)
    llm = MockOllamaReasoner(expansion_response={"type": "none", "reason": "noise"})
    checker = ErrorChecker(table, analyzer, llm)
    result = checker.check()
    assert result["expanded"] is False
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase4_error_checker.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/error_checker.py`**

```python
"""
Phase 4: Error Check.

If error_analyzer detects structured errors, ask the LLM what expansion
is needed and apply it to the table. 0 or 1 LLM calls.
"""

from typing import Dict

from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer


SYSTEM_PROMPT = """You analyze error patterns in a game-learning system.
The system currently has a limited vocabulary and its errors are showing structure.

Available expansions:
- "sequential": Remember previous action. Use when errors depend on action order.
- "context": Remember game state. Use when same action has different effects in different states.
- "none": No expansion needed.

Respond ONLY with JSON: {"type": "sequential|context|none", "reason": "one sentence"}
"""


class ErrorChecker:
    def __init__(self, table: ArcTableModel, analyzer: ArcErrorAnalyzer, llm):
        self.table = table
        self.analyzer = analyzer
        self.llm = llm

    def check(self) -> Dict:
        analysis = self.analyzer.analyze()

        if not analysis.get("sufficient_data"):
            return {"expanded": False, "reason": "insufficient_data"}

        if not analysis.get("any_structure"):
            return {"expanded": False, "reason": "random_errors"}

        # Structured errors → ask LLM
        user_prompt = f"Error analysis report:\n\n{analysis['summary']}"
        result = self.llm.reason_json(SYSTEM_PROMPT, user_prompt)

        if not isinstance(result, dict):
            return {"expanded": False, "reason": "llm_parse_error"}

        exp_type = result.get("type", "none")
        reason = result.get("reason", "")

        if exp_type == "none":
            return {"expanded": False, "reason": f"llm_said_none: {reason}"}

        applied = self.table.enable_expansion(exp_type, reason)
        return {
            "expanded": applied,
            "expansion_type": exp_type,
            "reason": reason,
        }
```

**Step 4: Run tests**

```bash
pytest tests/test_phase4_error_checker.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/error_checker.py tests/test_phase4_error_checker.py
git commit -m "feat(alfa_loop): implement Phase 4 ErrorChecker with LLM expansion trigger"
```

---

### Task 15: Phase 5 — Planner (with emergency fallback)

**Files:**
- Create: `src/charith/alfa_loop/planner.py`
- Create: `tests/test_phase5_planner.py`

**Step 1: Write the failing test**

Create `tests/test_phase5_planner.py`:

```python
"""Tests for Phase 5 — Planner."""
from charith.causal_engine.table_model import ArcTableModel
from charith.alfa_loop.planner import Planner
from charith.full_stack.hypothesis_schema import Hypothesis, ExpectedOutcome
from tests.fixtures.mock_llm import MockOllamaReasoner


def _confirmed_hyp():
    return Hypothesis(
        rule="action 1 moves up",
        confidence=0.8,
        test_action=1,
        expected=ExpectedOutcome(direction="up"),
        status="confirmed",
    )


def test_plan_calls_llm_once_with_verified_rules():
    llm = MockOllamaReasoner(plan_response={"plan": [1, 1, 3], "reasoning": "r"})
    table = ArcTableModel(num_actions=8)
    planner = Planner(llm, table)
    plan = planner.plan([_confirmed_hyp()], goal="reach target", state_desc="test", num_actions=8)
    assert plan == [1, 1, 3]
    assert llm.call_count == 1


def test_plan_filters_invalid_actions():
    llm = MockOllamaReasoner(plan_response={"plan": [1, 99, 3], "reasoning": "r"})
    table = ArcTableModel(num_actions=8)
    planner = Planner(llm, table)
    plan = planner.plan([_confirmed_hyp()], goal="reach target", state_desc="test", num_actions=8)
    assert plan == [1, 3]


def test_plan_zero_confirmed_triggers_emergency_fallback():
    llm = MockOllamaReasoner()
    table = ArcTableModel(num_actions=8)
    # Give table some observations so emergency fallback has weights
    for _ in range(5):
        table.record(action=1, changes=["change"])
        table.record(action=2, changes=["change"])
    planner = Planner(llm, table)
    plan = planner.plan([], goal="?", state_desc="test", num_actions=8)
    # Emergency fallback doesn't call LLM
    assert llm.call_count == 0
    # Should return a list of valid actions
    assert len(plan) > 0
    for a in plan:
        assert 1 <= a <= 8
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase5_planner.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/planner.py`**

```python
"""
Phase 5: Plan.

Given verified hypotheses and a goal, ask the LLM for an action sequence.
Zero-confirmed edge case → emergency fallback using table predictions,
per design §6.4 (Option C: when LLM fails, fall back to mechanistic
component).
"""

from typing import List
import math
import random

from charith.causal_engine.table_model import ArcTableModel
from charith.full_stack.hypothesis_schema import Hypothesis


SYSTEM_PROMPT = """You are planning actions in a grid game.
You receive VERIFIED causal rules (tested through intervention — ground truth).
You also have a goal hypothesis and a description of the current state.

Plan an efficient sequence of actions to achieve the goal.
Use only actions that appear in CONFIRMED rules. Avoid REFUTED rules.

Respond ONLY with JSON: {"plan": [action_id, action_id, ...], "reasoning": "..."}
"""


class Planner:
    def __init__(self, llm, table: ArcTableModel, max_plan_length: int = 20):
        self.llm = llm
        self.table = table
        self.max_plan_length = max_plan_length

    def plan(
        self,
        verified: List[Hypothesis],
        goal: str,
        state_desc: str,
        num_actions: int = 8,
    ) -> List[int]:
        confirmed = [h for h in verified if h.status == "confirmed"]

        if not confirmed:
            return self._emergency_fallback(num_actions)

        confirmed_text = "\n".join(
            f"  CONFIRMED: action {h.test_action} → {h.actual_summary or h.rule}"
            for h in confirmed
        )
        refuted = [h for h in verified if h.status == "refuted"]
        refuted_text = "\n".join(f"  REFUTED: {h.rule}" for h in refuted)

        user = (
            f"Current state: {state_desc}\n"
            f"Goal: {goal}\n\n"
            f"Verified rules (ground truth):\n{confirmed_text}\n\n"
            f"Refuted hypotheses (do NOT use):\n{refuted_text or '  (none)'}\n\n"
            f"Plan the shortest action sequence to reach the goal."
        )

        result = self.llm.reason_json(SYSTEM_PROMPT, user)

        raw_plan = result.get("plan", []) if isinstance(result, dict) else []
        valid_plan = []
        for a in raw_plan:
            try:
                ia = int(a)
                if 1 <= ia <= num_actions:
                    valid_plan.append(ia)
            except (TypeError, ValueError):
                continue

        if not valid_plan:
            return self._emergency_fallback(num_actions)

        return valid_plan[: self.max_plan_length]

    def _emergency_fallback(self, num_actions: int) -> List[int]:
        """
        When no confirmed hypothesis exists, fall back to table-weighted
        random walk. Per design §6.4: when the LLM fails, use the
        mechanistic component (the table), not the LLM.
        """
        weights = []
        for action_id in range(1, num_actions + 1):
            pred = self.table.predict(
                action=action_id,
                target=None,
                prev_action=self.table.prev_action,
            )
            confidence = pred[1]
            obs = self.table.single_table.get(action_id, [])
            n = len(obs)
            weight = confidence * math.log1p(n)
            weights.append(max(weight, 0.05))

        total = sum(weights)
        probs = [w / total for w in weights]

        # Sample 10 actions from this distribution
        rng = random.Random(0)  # deterministic for tests
        plan = []
        for _ in range(10):
            r = rng.random()
            cum = 0.0
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    plan.append(idx + 1)
                    break
        return plan
```

**Step 4: Run tests**

```bash
pytest tests/test_phase5_planner.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/planner.py tests/test_phase5_planner.py
git commit -m "feat(alfa_loop): implement Phase 5 Planner with emergency fallback"
```

---

### Task 16: Phase 6 — Executor

**Files:**
- Create: `src/charith/alfa_loop/executor.py`
- Create: `tests/test_phase6_executor.py`

**Step 1: Write the failing test**

Create `tests/test_phase6_executor.py`:

```python
"""Tests for Phase 6 — Executor."""
import numpy as np
from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.executor import Executor
from tests.fixtures.mock_env import MockArcEnv, move_obj_by


def _env():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12
    return MockArcEnv(
        initial_grid=grid,
        rules={
            1: move_obj_by(12, -1, 0),
            2: move_obj_by(12, 1, 0),
            3: move_obj_by(12, 0, -1),
            4: move_obj_by(12, 0, 1),
        },
    )


def test_execute_happy_path_completes_plan():
    env = _env()
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    # Pre-populate table so predictions have some confidence
    for _ in range(3):
        table.record(action=1, changes=["direction=up,mag=1"])
    executor = Executor(env, perception, table, analyzer)
    result = executor.execute([1, 1, 1])
    assert result["actions_taken"] == 3
    assert result["completed"] is False  # not done, but plan finished


def test_execute_returns_on_done():
    env = _env()
    env.done_on_action = 2
    env.reset()
    perception = CoreKnowledgePerception()
    table = ArcTableModel(num_actions=8)
    analyzer = ArcErrorAnalyzer()
    executor = Executor(env, perception, table, analyzer)
    result = executor.execute([1, 2, 3])
    assert result["completed"] is True
    assert result["actions_taken"] == 2  # stopped at action 2
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_phase6_executor.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/alfa_loop/executor.py`**

```python
"""
Phase 6: Execute + monitor δ.

Walk the plan. At each step:
  - predict via table
  - execute
  - compare prediction to actual
  - record
  - halt if consecutive_surprises hits threshold
  - success if env returns done=True
"""

from typing import Dict, List

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.full_stack.percept_diff import diff_to_actual_observation


class Executor:
    def __init__(
        self,
        env,
        perception: CoreKnowledgePerception,
        table: ArcTableModel,
        error_analyzer: ArcErrorAnalyzer,
        halt_threshold: int = 3,
    ):
        self.env = env
        self.perception = perception
        self.table = table
        self.error_analyzer = error_analyzer
        self.halt_threshold = halt_threshold
        self._step_counter = 0

    def execute(self, plan: List[int]) -> Dict:
        consecutive_surprises = 0
        actions_taken = 0
        prev_action = self.table.prev_action

        for action_id in plan:
            obs_before = self.env.get_observation()
            percept_before = self.perception.perceive(obs_before.frame[0])

            # Table prediction (by single_table lookup — coarse)
            prediction = self.table.predict(
                action=action_id,
                target=None,
                prev_action=prev_action,
            )

            obs_after, _reward, done, _info = self.env.step(action_id)
            percept_after = self.perception.perceive(obs_after.frame[0])
            actual = diff_to_actual_observation(percept_before, percept_after)

            change_desc = (
                f"direction={actual.controllable_direction},mag={actual.controllable_magnitude}"
            )
            self.table.record(action=action_id, changes=[change_desc])
            actions_taken += 1

            # Simple match: did anything change at all?
            matched = self._matches(prediction, actual)
            if matched:
                consecutive_surprises = 0
            else:
                consecutive_surprises += 1

            self.error_analyzer.record(
                step=self._step_counter,
                action=action_id,
                predicted_right=matched,
                prev_action=prev_action,
            )
            self._step_counter += 1
            prev_action = action_id

            if consecutive_surprises >= self.halt_threshold:
                return {
                    "completed": False,
                    "actions_taken": actions_taken,
                    "reason": "delta_spike",
                }

            if done:
                return {
                    "completed": True,
                    "actions_taken": actions_taken,
                    "reason": "success",
                }

        return {
            "completed": False,
            "actions_taken": actions_taken,
            "reason": "plan_exhausted",
        }

    def _matches(self, prediction, actual) -> bool:
        """
        Coarse match: if confidence is low, accept anything.
        Otherwise: require some change if the table predicted effects.
        """
        _predicts_effect, confidence, _source = prediction
        if confidence < 0.3:
            return True
        # With confidence, we require actual to have SOME change
        return (
            actual.controllable_magnitude > 0
            or bool(actual.any_color_changes)
            or bool(actual.new_objects)
            or bool(actual.removed_objects)
        )
```

**Step 4: Run tests**

```bash
pytest tests/test_phase6_executor.py -v
```

Expected: `2 passed`.

**Step 5: Commit**

```bash
git add src/charith/alfa_loop/executor.py tests/test_phase6_executor.py
git commit -m "feat(alfa_loop): implement Phase 6 Executor with δ monitoring"
```

---

## Phase F — Orchestrator and Integration (Tasks 17-19)

### Task 17: LLMReasoner adapter (thin wrapper over existing OllamaClient)

**Files:**
- Create: `src/charith/full_stack/llm_reasoner.py`
- Create: `tests/test_llm_reasoner.py`

**Context:** The existing `charith.llm_agent.ollama_client.OllamaClient` doesn't have a `reason_json()` method. We add a thin wrapper that provides it without modifying the old file.

**Step 1: Write the failing test**

Create `tests/test_llm_reasoner.py`:

```python
"""Tests for LLMReasoner adapter."""
import json
from charith.full_stack.llm_reasoner import LLMReasoner


class _FakeOllama:
    def __init__(self, response_text):
        self.response_text = response_text

    def generate(self, system_prompt, user_prompt, **kwargs):
        return self.response_text


def test_reason_json_parses_clean_json():
    ollama = _FakeOllama('{"type": "sequential", "reason": "x"}')
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result["type"] == "sequential"


def test_reason_json_handles_markdown_fence():
    ollama = _FakeOllama('```json\n{"type": "none"}\n```')
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result["type"] == "none"


def test_reason_json_returns_parse_error_dict_on_invalid():
    ollama = _FakeOllama("not json at all")
    llm = LLMReasoner(ollama)
    result = llm.reason_json("sys", "user")
    assert result.get("parse_error") is True
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_llm_reasoner.py -v
```

Expected: `ModuleNotFoundError`.

**Step 3: Implement `src/charith/full_stack/llm_reasoner.py`**

```python
"""
Thin wrapper around existing OllamaClient.

Adds:
  - reason_json(): parses markdown fences and returns dict (or parse_error)
  - Duck-typed: accepts anything with a `generate(system, user, ...)` method,
    which is what the mock and real client both implement.
"""

import json
import re
from typing import Any, Dict


_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


class LLMReasoner:
    def __init__(self, backend):
        """
        backend: any object with `generate(system_prompt, user_prompt, ...) -> str`.
        Real: charith.llm_agent.ollama_client.OllamaClient
        Test: _FakeOllama / MockOllamaReasoner (though the latter provides
              reason_json directly)
        """
        self.backend = backend
        self.call_count = 0

    def reason_json(self, system: str, user: str) -> Dict[str, Any]:
        self.call_count += 1
        raw = self.backend.generate(system, user + "\n\nRespond ONLY with valid JSON.")
        return self._parse(raw)

    def reason(self, system: str, user: str) -> str:
        self.call_count += 1
        return self.backend.generate(system, user)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        if not isinstance(raw, str):
            return {"raw": raw, "parse_error": True}

        stripped = raw.strip()

        # Strip markdown fences if present
        m = _FENCE_RE.search(stripped)
        if m:
            stripped = m.group(1)

        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return {"raw": raw, "parse_error": True}
```

**Step 4: Run tests**

```bash
pytest tests/test_llm_reasoner.py -v
```

Expected: `3 passed`.

**Step 5: Commit**

```bash
git add src/charith/full_stack/llm_reasoner.py tests/test_llm_reasoner.py
git commit -m "feat(full_stack): add LLMReasoner adapter with JSON fence parsing"
```

---

### Task 18: CharithFullStackAgent orchestrator

**Files:**
- Create: `src/charith/full_stack/charith_full_stack_agent.py`

**Context:** Wires all 6 phases together. Does NOT have its own dedicated test — the next task's integration tests exercise it end-to-end. Keep it short and focused: construction, `play_level()` method, `play_game()` method.

**Step 1: Implement `src/charith/full_stack/charith_full_stack_agent.py`**

```python
"""
CharithFullStackAgent — orchestrates the 6-phase ALFA loop.

Composes:
  - CoreKnowledgePerception (existing)
  - ArcTableModel (new, ported)
  - ArcErrorAnalyzer (new, ported)
  - Explorer, Hypothesizer, Verifier, ErrorChecker, Planner, Executor (new)

Persistence: table and error_analyzer NEVER reset within a game, only
between games. See design §6.3.
"""

from typing import Optional

from charith.perception.core_knowledge import CoreKnowledgePerception
from charith.causal_engine.table_model import ArcTableModel
from charith.causal_engine.error_analyzer import ArcErrorAnalyzer
from charith.alfa_loop.explorer import Explorer
from charith.alfa_loop.hypothesizer import Hypothesizer
from charith.alfa_loop.verifier import Verifier
from charith.alfa_loop.error_checker import ErrorChecker
from charith.alfa_loop.planner import Planner
from charith.alfa_loop.executor import Executor
from charith.full_stack.budgets import AgentBudgets
from charith.full_stack.results import AttemptResult, LevelResult


class CharithFullStackAgent:
    def __init__(
        self,
        env,
        llm,
        num_actions: int = 8,
        budgets: Optional[AgentBudgets] = None,
    ):
        self.env = env
        self.llm = llm
        self.num_actions = num_actions
        self.budgets = budgets or AgentBudgets()

        # Perception (existing, read-only)
        self.perception = CoreKnowledgePerception()

        # Causal engine (persists across attempts and levels within a game)
        self.table = ArcTableModel(num_actions=num_actions)
        self.error_analyzer = ArcErrorAnalyzer()

        # ALFA phases
        self.explorer = Explorer(env, self.perception, self.table)
        self.hypothesizer = Hypothesizer(llm, num_actions=num_actions)
        self.verifier = Verifier(env, self.perception, self.table, self.error_analyzer)
        self.error_checker = ErrorChecker(self.table, self.error_analyzer, llm)
        self.planner = Planner(llm, self.table, max_plan_length=self.budgets.max_plan_length)
        self.executor = Executor(
            env, self.perception, self.table, self.error_analyzer,
            halt_threshold=self.budgets.consecutive_surprises_to_halt,
        )

        self._total_actions_this_level = 0
        self._total_llm_calls_this_level = 0

    def play_level(self) -> LevelResult:
        """Play one level — up to max_attempts_per_level attempts."""
        attempts = []
        for attempt_idx in range(self.budgets.max_attempts_per_level):
            result = self._play_attempt()
            attempts.append(result)
            if result.completed:
                break

        return LevelResult(
            completed=any(a.completed for a in attempts),
            attempts=attempts,
            total_actions=sum(a.actions_taken for a in attempts),
            total_llm_calls=sum(a.llm_calls for a in attempts),
            final_table_stats={
                "num_observations": self.table.total_observations,
                "active_expansions": self.table.get_active_expansions(),
            },
        )

    def _play_attempt(self) -> AttemptResult:
        """One attempt: up to max_expansion_cycles_per_attempt through phases 1-4, then 5-6."""
        llm_calls_before = self._current_llm_call_count()
        actions_before = self.table.total_observations

        hypotheses = []
        goal = ""
        verified = []
        expansions_triggered = []

        phase_reached = 1
        for cycle in range(self.budgets.max_expansion_cycles_per_attempt):
            # Phase 1
            evidence = self.explorer.explore(num_actions=self.budgets.explore_num_actions)
            phase_reached = 2

            # Phase 2
            hypotheses, goal = self.hypothesizer.generate(
                evidence, self.table.get_active_expansions()
            )
            phase_reached = 3

            # Phase 3
            verified = self.verifier.verify(hypotheses[: self.budgets.max_hypotheses_to_verify])
            phase_reached = 4

            # Phase 4
            expansion_result = self.error_checker.check()
            if expansion_result.get("expanded"):
                expansions_triggered.append(expansion_result.get("expansion_type", "?"))
                continue  # loop back to Phase 1 with expanded vocabulary
            else:
                break

        # Phase 5
        phase_reached = 5
        state_obs = self.env.get_observation()
        state_percept = self.perception.perceive(state_obs.frame[0])
        state_desc = f"objects={state_percept.object_count}"
        plan = self.planner.plan(
            verified, goal, state_desc, num_actions=self.num_actions
        )

        # Phase 6
        phase_reached = 6
        exec_result = self.executor.execute(plan)

        actions_this_attempt = self.table.total_observations - actions_before
        llm_calls_this_attempt = self._current_llm_call_count() - llm_calls_before

        return AttemptResult(
            completed=bool(exec_result.get("completed")),
            actions_taken=actions_this_attempt,
            llm_calls=llm_calls_this_attempt,
            reason=exec_result.get("reason", "unknown"),
            phase_reached=phase_reached,
            hypotheses_generated=len(hypotheses),
            hypotheses_confirmed=sum(1 for h in verified if h.status == "confirmed"),
            hypotheses_refuted=sum(1 for h in verified if h.status == "refuted"),
            expansions_triggered=expansions_triggered,
            final_error_summary=self.error_analyzer.analyze().get("summary", ""),
        )

    def _current_llm_call_count(self) -> int:
        """Total LLM calls — works for both real and mock LLMs."""
        return getattr(self.llm, "call_count", 0)
```

**Step 2: Commit (no test yet — exercised by integration tests next task)**

```bash
git add src/charith/full_stack/charith_full_stack_agent.py
git commit -m "feat(full_stack): add CharithFullStackAgent orchestrator"
```

---

### Task 19: Integration tests (3 end-to-end mock tests)

**Files:**
- Create: `tests/test_full_stack_integration.py`

**Step 1: Write the integration tests**

Create `tests/test_full_stack_integration.py`:

```python
"""
End-to-end integration tests with MockArcEnv + MockOllamaReasoner.
The agent runs the full 6-phase loop without any real deps.
"""
import numpy as np

from charith.full_stack.charith_full_stack_agent import CharithFullStackAgent
from tests.fixtures.mock_env import MockArcEnv, move_obj_by
from tests.fixtures.mock_llm import MockOllamaReasoner


def _env_basic():
    grid = np.zeros((20, 20), dtype=int)
    grid[10, 10] = 12
    return MockArcEnv(
        initial_grid=grid,
        rules={
            1: move_obj_by(12, -1, 0),
            2: move_obj_by(12, 1, 0),
            3: move_obj_by(12, 0, -1),
            4: move_obj_by(12, 0, 1),
        },
    )


def test_full_loop_runs_without_crashing():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner()
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_level()
    # Level not required to complete; just must not crash
    assert result is not None
    assert len(result.attempts) >= 1


def test_full_loop_respects_llm_budget():
    env = _env_basic()
    env.reset()
    llm = MockOllamaReasoner()
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_level()
    # LLM budget is per-attempt ~3 × max_attempts_per_level=3 = ~9 max
    # Plus any expansion cycles. Upper bound is loose but non-zero.
    assert llm.call_count <= 20
    assert llm.call_count >= 1


def test_full_loop_refuted_hypotheses_fall_back_to_emergency():
    env = _env_basic()
    env.reset()
    # LLM always says "down" but the env moves up when action 1 → refuted
    # Then Phase 5 hits zero confirmed → emergency fallback
    llm = MockOllamaReasoner(
        hypothesize_response={
            "hypotheses": [
                {
                    "rule": "action 1 moves DOWN by 5",
                    "confidence": 0.9,
                    "test_action": 1,
                    "expected": {"direction": "down", "magnitude_cells": 5},
                }
            ],
            "goal_guess": "unknown",
        },
        plan_response={"plan": [], "reasoning": "empty"},  # unused if emergency fires
    )
    agent = CharithFullStackAgent(env, llm, num_actions=8)

    result = agent.play_level()
    # Just assert no crash and that all 3 attempts ran
    assert len(result.attempts) == 3
```

**Step 2: Run tests**

```bash
pytest tests/test_full_stack_integration.py -v
```

Expected: `3 passed`. (Note: these tests may reveal issues in the phase modules that weren't caught by isolated tests. Fix them in-place and re-run.)

**Step 3: Run the FULL test suite to verify nothing regressed**

```bash
pytest tests/ -v --ignore=tests/test_perception.py
```

Expected: All ~30 new tests pass. Existing `test_perception.py` etc. remain green.

**Step 4: Commit**

```bash
git add tests/test_full_stack_integration.py
git commit -m "test(integration): add 3 end-to-end mock integration tests"
```

---

## Phase G — Entry Script and Smoke Test (Tasks 20-21)

### Task 20: CLI entry script

**Files:**
- Create: `scripts/play_full_stack.py`

**Step 1: Implement `scripts/play_full_stack.py`**

```python
"""
CLI entry point for the CHARITH full stack agent.

Usage:
    python scripts/play_full_stack.py --game ls20 --model gemma4:latest
"""

import argparse
import sys
import time

sys.path.insert(0, "src")

import arc_agi  # noqa: E402

from charith.full_stack.charith_full_stack_agent import CharithFullStackAgent  # noqa: E402
from charith.full_stack.llm_reasoner import LLMReasoner  # noqa: E402
from charith.llm_agent.ollama_client import OllamaClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 with CHARITH full stack agent")
    parser.add_argument("--game", default="ls20", help="ARC game id")
    parser.add_argument("--model", default="gemma4:latest", help="Ollama model name")
    parser.add_argument("--num-actions", type=int, default=8, help="Action space size")
    args = parser.parse_args()

    print(f"[charith-full-stack] game={args.game} model={args.model}")

    arcade = arc_agi.Arcade()
    env = arcade.make(args.game)
    env.reset()

    ollama = OllamaClient(model=args.model)
    llm = LLMReasoner(ollama)

    agent = CharithFullStackAgent(env, llm, num_actions=args.num_actions)

    start = time.time()
    result = agent.play_level()
    elapsed = time.time() - start

    print()
    print("=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Level completed: {result.completed}")
    print(f"  Attempts: {len(result.attempts)}")
    print(f"  Total actions: {result.total_actions}")
    print(f"  Total LLM calls: {result.total_llm_calls}")
    print(f"  Active expansions: {result.final_table_stats['active_expansions']}")
    print(f"  Wall time: {elapsed:.1f}s")
    for i, a in enumerate(result.attempts):
        print(
            f"  Attempt {i+1}: phase_reached={a.phase_reached} "
            f"reason={a.reason} "
            f"confirmed={a.hypotheses_confirmed}/{a.hypotheses_generated}"
        )
    return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Dry-run the script (syntax check only, no real env needed)**

```bash
python -c "import ast; ast.parse(open('scripts/play_full_stack.py').read()); print('syntax ok')"
```

Expected: `syntax ok`.

**Step 3: Commit**

```bash
git add scripts/play_full_stack.py
git commit -m "feat(scripts): add play_full_stack.py CLI entry point"
```

---

### Task 21: Manual smoke test on real ls20

**Files:**
- None (manual step)

**Context:** This is the one task that requires real dependencies: Ollama running with the chosen model, `arc_agi` package installed, `ls20` downloaded. Per design §7.6, this is NOT part of the automated test suite — it's a manual milestone gate.

**Step 1: Verify prerequisites**

```bash
# Check Ollama is running and model is available
ollama list | grep gemma

# Check arc_agi is installed
python -c "import arc_agi; print('arc_agi version:', arc_agi.__version__ if hasattr(arc_agi, '__version__') else 'unknown')"

# Check ls20 files exist
ls environment_files/ls20/
```

**Step 2: Run the smoke test**

```bash
python scripts/play_full_stack.py --game ls20 --model gemma4:latest
```

**Success criteria (Milestone 1):**

- [ ] Script runs to completion without unhandled exception
- [ ] All 6 phases are reached in at least one attempt (`phase_reached >= 6`)
- [ ] Total LLM calls ≤ 20 (within budget × 3 attempts × small slack)
- [ ] Total actions ≤ 150
- [ ] `LevelResult` is structured and printable
- [ ] Level completion is a **bonus**, not required

**Step 3: If it passes, commit the smoke run log**

```bash
mkdir -p logs/smoke
python scripts/play_full_stack.py --game ls20 --model gemma4:latest > logs/smoke/ls20_milestone1.log 2>&1 || true
git add logs/smoke/ls20_milestone1.log
git commit -m "test(smoke): ls20 milestone 1 smoke test log"
```

**If it fails**, do not commit. Instead:

1. Run with the `--verbose` flag if available
2. Check the LLM output in the exception trace
3. Most likely failure modes and where to look:
   - **LLM returns malformed JSON** → `LLMReasoner._parse` returned `parse_error`, check `hypothesizer.generate` fallback path
   - **`diff_to_actual_observation` raises** on real ls20 percepts → inspect the raw StructuredPercept, the adapter assumptions don't hold
   - **All hypotheses `untestable`** → `test_action` fields from LLM aren't being parsed as ints
   - **ObjectTracker returns no matches** → real ls20 sprite may have different color or cell count than the mock; update the controllable-detection heuristic in `percept_diff.py`

Each of these is a systematic debugging task, not a plan step — use the `superpowers:systematic-debugging` skill.

---

## Verification Checklist

Before declaring Milestone 1 complete, verify:

- [ ] All unit tests pass: `pytest tests/test_causal_engine_table.py tests/test_causal_engine_analyzer.py tests/test_hypothesis_schema.py tests/test_match_score.py tests/test_percept_diff.py tests/test_full_stack_results.py tests/test_llm_reasoner.py -v`
- [ ] All mock-fixture smoke tests pass: `pytest tests/test_mock_env.py tests/test_mock_llm.py -v`
- [ ] All phase tests pass: `pytest tests/test_phase1_explorer.py tests/test_phase2_hypothesizer.py tests/test_phase3_verifier.py tests/test_phase4_error_checker.py tests/test_phase5_planner.py tests/test_phase6_executor.py -v`
- [ ] All integration tests pass: `pytest tests/test_full_stack_integration.py -v`
- [ ] Existing test suite is still green: `pytest tests/ -v`
- [ ] Manual smoke test on real ls20 runs without exception (level completion optional)
- [ ] Total new files: 14 source files + 13 test files = **27 new files**
- [ ] No existing files modified: `git log --name-only -- src/charith/perception src/charith/llm_agent/agent.py src/charith/agent.py` shows no commits in this plan
- [ ] Design doc committed: `git log docs/plans/2026-04-08-charith-full-stack-design.md` shows commit `57424ac`

---

## What Milestone 2 will tackle (out of scope for this plan)

- **Tighter `object_ref` matching** for multi-object games (tr87 etc.)
- **Real score detection** (currently stubbed `score_changed=False`)
- **Cross-level transfer tests**
- **Prioritized hypothesis verification** (cap via confidence instead of strict ordering)
- **Level completion as a requirement**, not a bonus
- **Performance profiling** and LLM call cost reduction
