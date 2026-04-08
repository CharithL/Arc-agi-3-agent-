# CHARITH Full Stack Agent — Design

**Status:** Approved by user, ready for implementation planning
**Target milestone:** Full 6-phase ALFA loop runs cleanly on ls20 (Level completion is a bonus, not required)
**Authors:** Charith Suranga + Claude Code (brainstorming session 2026-04-08)

---

## 1. Context & Purpose

Every prior approach to ARC-AGI-3 in this repo failed for a specific reason:

- **Path 2 (handcrafted):** Knew how to move, not where to go. 0 levels.
- **Path 3 (GRU):** Needed 1500 episodes to learn what humans learn in 5 actions.
- **Path 5 (LLM every tick):** 100 LLM calls per attempt, 0 levels. Circular validation — LLM reasoned over its own prior outputs.
- **Path 5.5 (ALFA Factory):** Better LLM usage but text-based world model. Still circular.

**The full stack breaks the circular validation problem.** The LLM generates hypotheses about game mechanics. A C1/C2 engine (validated in `c1c2-hybrid`) TESTS those hypotheses through targeted interventions and measures whether they hold. The LLM never validates itself — the environment does, through the C1/C2 causal verification loop.

**Architecture in one sentence:** CoreKnowledge sees objects → LLM guesses rules → C1/C2 tests rules through action → Error analyzer detects when guesses are wrong → LLM revises → repeat until rules are confirmed → LLM plans → execute plan → score.

**LLM budget per attempt:** 2-3 calls (vs Path 5's 100).

---

## 2. Architecture Overview

```
ARC-AGI-3 Game (arc_agi.Arcade().make('ls20'))
      │
      ▼ raw frame
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: PERCEPTION  (EXISTING — Path 2/3 CoreKnowledge) │
│   CoreKnowledgePerception → StructuredPercept           │
│   ObjectTracker → persistence/cohesion                   │
│   PerceptTranslator → natural language summaries        │
└──────────────────────────┬──────────────────────────────┘
                           │ StructuredPercept + text
                           ▼
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: CAUSAL ENGINE  (NEW — port from c1c2-hybrid)   │
│   ArcTableModel        → action → effects table         │
│   ArcErrorAnalyzer     → Kruskal + Ljung-Box (all-errors)│
│   DeltaTracker         → δ computation per step         │
└──────────────────────────┬──────────────────────────────┘
                           │ predictions + error analysis
                           ▼
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: ALFA LOOP  (NEW — 6 phases)                    │
│   Phase 1: Explore (0 LLM, 8 actions)                    │
│   Phase 2: Hypothesize (1 LLM, 0 actions)                │
│   Phase 3: Verify (0 LLM, ~5 actions)  ← novel           │
│   Phase 4: Error check (0-1 LLM, 0 actions) ← novel      │
│   Phase 5: Plan (1 LLM, 0 actions)                       │
│   Phase 6: Execute + monitor δ (0 LLM, N actions)        │
└──────────────────────────┬──────────────────────────────┘
                           │ LevelResult
                           ▼
                      Scorecard
```

**What makes this different from every other ARC-AGI-3 approach:**

1. **LLM never validates itself** — Phase 3 tests LLM hypotheses through intervention.
2. **Automatic C1 expansion** — Phase 4 detects when vocabulary is insufficient via Kruskal/Ljung-Box and expands via 1 LLM call.
3. **2-3 LLM calls per attempt** instead of 100.
4. **Verified causal model** — the planner works from CONFIRMED rules, not guesses.
5. **δ-monitoring during execution** — Phase 6 halts if the plan diverges from reality.

---

## 3. File Layout & Integration Boundaries

**Integration rule: nothing existing is modified. Everything new lives under two new packages.**

```
src/charith/
├── perception/                  [EXISTING — read only]
│   ├── core_knowledge.py       → CoreKnowledgePerception, StructuredPercept
│   └── object_tracker.py        → ObjectTracker
│
├── llm_agent/                   [EXISTING — reuse 3 files, ignore the rest]
│   ├── translator.py           → PerceptTranslator           (reused)
│   ├── ollama_client.py        → OllamaClient                (reused)
│   └── response_parser.py      → ResponseParser              (reused)
│
├── causal_engine/               [NEW — port from c1c2-hybrid]
│   ├── __init__.py
│   ├── table_model.py          → ArcTableModel (action → effects + expansion slots)
│   ├── error_analyzer.py       → ArcErrorAnalyzer (all-errors Kruskal + Ljung-Box)
│   └── delta_tracker.py        → δ between predicted and actual StructuredPercept diff
│
├── alfa_loop/                   [NEW — 6 phases]
│   ├── __init__.py
│   ├── explorer.py             → Phase 1
│   ├── hypothesizer.py         → Phase 2
│   ├── verifier.py             → Phase 3
│   ├── error_checker.py        → Phase 4
│   ├── planner.py              → Phase 5
│   └── executor.py             → Phase 6
│
└── full_stack/                  [NEW — wiring layer]
    ├── __init__.py
    ├── hypothesis_schema.py    → Hypothesis, ExpectedOutcome, ActualObservation
    ├── budgets.py              → AgentBudgets (hard caps)
    ├── results.py              → AttemptResult, LevelResult, FailedAttempt
    └── charith_full_stack_agent.py → CharithFullStackAgent (orchestrator)

scripts/
└── play_full_stack.py           [NEW] → CLI entry point

tests/
├── test_causal_engine_table.py          [NEW]
├── test_causal_engine_analyzer.py       [NEW]
├── test_hypothesis_schema.py            [NEW]
├── test_match_score.py                  [NEW — 7 golden tests]
├── test_actual_observation.py           [NEW]
├── test_phase1_explorer.py              [NEW]
├── test_phase2_hypothesizer.py          [NEW]
├── test_phase3_verifier.py              [NEW]
├── test_phase4_error_checker.py         [NEW]
├── test_phase5_planner.py               [NEW]
├── test_phase6_executor.py              [NEW]
├── test_full_stack_integration.py       [NEW — 3 integration tests]
└── fixtures/
    ├── mock_env.py                      [NEW]
    └── mock_llm.py                      [NEW]
```

**Integration contracts (the only places new code touches old code):**

1. `from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept` — read-only.
2. `from charith.llm_agent.translator import PerceptTranslator` — `.translate(percept, prev)` → text.
3. `from charith.llm_agent.ollama_client import OllamaClient` — wrapped by a thin `LLMReasoner` adapter in `full_stack/` that adds a `reason_json()` helper (same as c1c2-hybrid).
4. `import arc_agi; env = arc_agi.Arcade().make('ls20')` — env is injected into the agent constructor.

**No import cycles.** `causal_engine` depends only on numpy/scipy/statsmodels. `alfa_loop` depends on `causal_engine` + existing perception/translator. `full_stack` depends on everything; nothing depends on it.

**Old agents keep working.** `scripts/play_game_llm.py`, `solve_ls20.py`, `play_single.py` untouched.

---

## 4. Data Flow Through the 6 Phases

### 4.1 Agent-level state

```python
class CharithFullStackAgent:
    env: arc_agi.Environment        # injected
    perception: CoreKnowledgePerception
    translator: PerceptTranslator
    llm: LLMReasoner                 # thin wrapper over OllamaClient
    table: ArcTableModel             # persists across levels (continual learning)
    error_analyzer: ArcErrorAnalyzer # persists across levels
    total_actions: int               # resets per level
    total_llm_calls: int             # resets per level
    prev_action: Optional[int]       # sequential context
```

### 4.2 Phase-by-phase data contracts

**Phase 1 — Explore** (0 LLM, 8 actions)
- **In:** env, 8 actions to try
- **Does:** For each action 1..8: observe → act → observe → detect changes → record in table (no error recorded yet — no prediction to compare against)
- **Out:** `evidence: List[Evidence]` where `Evidence = {action, percept_before, percept_after, changes, description, reward, done}`

**Phase 2 — Hypothesize** (1 LLM call)
- **In:** evidence from Phase 1, active_expansions from table
- **Does:** Build prompt with 8 observations, 1 LLM call → structured JSON
- **Out:** `hypotheses: List[Hypothesis]`, `goal_guess: str`
- **Structured prompt constraint:** LLM must fill `ExpectedOutcome` dataclass fields (see §5), not prose descriptions.

**Phase 3 — Verify** (0 LLM, ~5 actions) — *novel*
- **In:** hypotheses from Phase 2
- **Does:** For each hypothesis (sequentially, state drift accepted):
  - Observe current state
  - Execute `h.test_action` (this is Pearl's do(X))
  - Observe new state
  - Compute `ActualObservation` from StructuredPercept diff
  - Compute `match_score = compute_match_score(h.expected, actual)` (see §5.3)
  - `h.status = 'confirmed' if score ≥ 0.70 else 'refuted' if score < 0.30 else 'ambiguous'`
  - Record action in table, record correctness in error_analyzer
- **Out:** `verified: List[Hypothesis]` with statuses and match_scores filled in

**Phase 4 — Error check** (0-1 LLM) — *novel*
- **In:** error_analyzer's accumulated errors (ALL, not windowed — see §6.3)
- **Does:**
  - `analysis = error_analyzer.analyze()` (Kruskal by prev_action, Ljung-Box autocorrelation, variance ratio)
  - If `analysis.any_structure` → 1 LLM call asking what expansion to add
  - If LLM returns valid expansion → `table.enable_expansion(type, reason)` and signal loop-back to Phase 1
- **Out:** `{expanded: bool, expansion_type: str | None}`
- **Loop-back rule:** max 2 expansion cycles per attempt before forcing Phase 5.

**Phase 5 — Plan** (1 LLM call)
- **In:** verified hypotheses (confirmed + refuted lists), goal_guess, current state description
- **Does:** 1 LLM call with "VERIFIED RULES: ..., REFUTED: ..., GOAL: ..." → `{plan: [action_id, ...], reasoning: str}`
- **Out:** `plan: List[int]`
- **Edge case:** If zero confirmed hypotheses → emergency fallback (see §6.4).

**Phase 6 — Execute + monitor** (0 LLM)
- **In:** plan from Phase 5
- **Does:** For each action in plan:
  - Get `prediction = table.predict(action, prev_action)`
  - Execute action
  - Compare prediction to actual
  - If mismatch → `consecutive_surprises += 1` else `= 0`
  - Record in table and error_analyzer
  - If `consecutive_surprises ≥ 3` → halt (δ-spike)
  - If `done=True` → success, return
- **Out:** `AttemptResult{completed, actions_taken, reason, phase_reached, ...}`

### 4.3 Loop structure

```python
def play_level_attempt(self) -> AttemptResult:
    for cycle in range(self.budgets.max_expansion_cycles_per_attempt):   # default 2
        evidence = self.explorer.explore()                  # Phase 1
        hypotheses, goal = self.hypothesizer(evidence)      # Phase 2
        verified = self.verifier.verify(hypotheses)         # Phase 3
        expansion = self.error_checker.check()              # Phase 4
        if not expansion.expanded:
            break    # vocabulary sufficient, proceed
        # else: loop back to Phase 1 with expanded table

    plan = self.planner.plan(verified, goal, ...)           # Phase 5
    return self.executor.execute(plan)                      # Phase 6

def play_level(self, max_attempts=3) -> LevelResult:
    attempts = []
    for attempt in range(max_attempts):
        result = self.play_level_attempt()
        attempts.append(result)
        if result.completed:
            return LevelResult(completed=True, attempts=attempts, ...)
    return LevelResult(completed=False, attempts=attempts, ...)
```

### 4.4 Data flow invariants

1. **`table.record()` is called in every phase that takes an action** (Phases 1, 3, 6).
2. **`error_analyzer.record()` is called only in Phases 3 and 6** — Phase 1 has no predictions yet.
3. **The LLM never sees the table directly.** It sees (a) evidence text in Phase 2, (b) error analysis summary in Phase 4, (c) verified/refuted rules text in Phase 5.
4. **`prev_action` is tracked at the agent level** so Phases 3 and 6 pass it to `table.predict()`.
5. **`table` and `error_analyzer` persist across attempts AND across levels within a game.** This is what makes cross-level transfer possible. Reset only between games.

### 4.5 Two loop-back signals

- **Phase 4 → Phase 1** (vocabulary expansion): Kruskal detects structured errors AND LLM suggests a valid expansion type. Max 2 cycles per attempt.
- **Phase 6 → Phase 2** (plan failure): δ spikes during execution. This is the outer `max_attempts` loop.

---

## 5. Hypothesis Schema & Match Scoring

### 5.1 Data classes

```python
# src/charith/full_stack/hypothesis_schema.py

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
# Object imported from charith.perception.core_knowledge

Direction = Literal['up', 'down', 'left', 'right', 'none']

@dataclass
class ExpectedOutcome:
    """
    Structured prediction the LLM emits in Phase 2.
    All fields optional — None = "no claim made" (not penalized, not rewarded).
    """
    direction: Optional[Direction] = None
    magnitude_cells: Optional[int] = None
    object_ref: Optional[str] = None              # 'controllable' | 'red' | 'blue' | ...
    color_change_to: Optional[str] = None
    object_appears: Optional[bool] = None
    object_disappears: Optional[bool] = None
    score_change: Optional[bool] = None           # score pixels changed (progress signal)
    no_effect: bool = False                       # null hypothesis — nothing should change

@dataclass
class Hypothesis:
    rule: str
    confidence: float
    test_action: int
    expected: ExpectedOutcome
    status: Literal['untested', 'confirmed', 'refuted', 'untestable', 'ambiguous'] = 'untested'
    actual_summary: Optional[str] = None          # filled by verifier
    match_score: Optional[float] = None           # filled by verifier (0-1)

@dataclass
class ActualObservation:
    """Distilled from StructuredPercept diff, ready for structured matching."""
    controllable_displacement: Optional[Tuple[int, int]]  # (row_delta, col_delta)
    controllable_direction: Optional[Direction]
    controllable_magnitude: int
    any_color_changes: List[Tuple['Object', int, int]]    # (obj, old_color, new_color)
    new_objects: List['Object']
    removed_objects: List['Object']
    score_changed: bool
```

### 5.2 Match thresholds

```python
CONFIRM_THRESHOLD = 0.70   # match_score >= 0.70 → status = 'confirmed'
REFUTE_THRESHOLD = 0.30    # match_score <  0.30 → status = 'refuted'
# 0.30 ≤ score < 0.70      → status = 'ambiguous' (kept, doesn't block planning)
```

### 5.3 `compute_match_score` — the critical algorithm

**User-designed. Locked in.**

```python
def compute_match_score(expected: ExpectedOutcome, actual: ActualObservation) -> float:
    # Special case: no_effect claim
    if expected.no_effect:
        is_empty = (actual.controllable_magnitude == 0
                    and not actual.any_color_changes
                    and not actual.new_objects
                    and not actual.removed_objects
                    and not actual.score_changed)
        return 1.0 if is_empty else 0.0

    scores = []

    # Direction: strict match (up ≠ down)
    if expected.direction is not None:
        if expected.direction == 'none':
            scores.append(1.0 if actual.controllable_magnitude == 0 else 0.0)
        else:
            scores.append(1.0 if actual.controllable_direction == expected.direction else 0.0)

    # Magnitude: ±1 cell tolerance
    if expected.magnitude_cells is not None and actual.controllable_magnitude > 0:
        diff = abs(actual.controllable_magnitude - expected.magnitude_cells)
        scores.append(1.0 if diff <= 1 else 0.0)

    # Color change: name match (case-insensitive, substring)
    if expected.color_change_to is not None:
        matched = any(
            expected.color_change_to.lower() in str(new_c).lower()
            for _, _, new_c in actual.any_color_changes
        )
        scores.append(1.0 if matched else 0.0)

    # Object appears/disappears
    if expected.object_appears is not None:
        scores.append(1.0 if expected.object_appears == bool(actual.new_objects) else 0.0)
    if expected.object_disappears is not None:
        scores.append(1.0 if expected.object_disappears == bool(actual.removed_objects) else 0.0)

    # Score change
    if expected.score_change is not None:
        scores.append(1.0 if expected.score_change == actual.score_changed else 0.0)

    # Aggregation: MEAN (partial credit)
    if not scores:
        return 0.5   # LLM claimed nothing specific — ambiguous by construction
    return sum(scores) / len(scores)
```

### 5.4 Three design calls (user-locked, with rationale)

1. **Mean aggregation (not min or product).** In a sparse-reward game where the agent has ~3 confirmed rules to plan from, losing a partially-correct hypothesis hurts more than trusting a mostly-correct one. If direction is right but magnitude is off by 2 cells, mean gives 0.5 (ambiguous) — it doesn't get confirmed but it doesn't get hard-refuted. Min or product would zero it out and the planner would have nothing.

2. **±1 cell magnitude tolerance.** ls20 moves ~5 cells per action. The LLM might guess 4 or 6 from noisy observation. That's the right mechanic, slightly wrong parameter. Confirming it gives the planner a usable rule. Strict matching would reject correct understanding over a rounding error.

3. **`object_ref` deliberately NOT checked.** In ls20 there's one controllable. "Controllable," "red," "the sprite" all mean the same thing. Strict object_ref matching would reject correct hypotheses over naming conventions. If direction and magnitude match, the LLM understood the rule regardless of label. **For multi-object games later (tr87 etc.), this must be tightened.**

### 5.5 Phase 2 prompt schema

The LLM is instructed to emit `ExpectedOutcome` as structured JSON:

```
ExpectedOutcome schema:
{
  "direction": "up|down|left|right|none",    // optional
  "magnitude_cells": 5,                       // optional
  "object_ref": "controllable|red|blue|...",  // optional
  "color_change_to": "green",                 // optional
  "object_appears": false,                    // optional
  "object_disappears": false,                 // optional
  "score_change": false,                      // optional
  "no_effect": false                          // set true if you predict nothing happens
}
```

The LLM is told: "Only populate fields you are actually predicting; leave others null."

---

## 6. Error Handling & Budgets

### 6.1 Hard budget caps

```python
# src/charith/full_stack/budgets.py

@dataclass(frozen=True)
class AgentBudgets:
    max_actions_per_level: int = 150
    max_llm_calls_per_level: int = 8
    max_attempts_per_level: int = 3
    max_expansion_cycles_per_attempt: int = 2
    explore_num_actions: int = 8
    max_hypotheses_to_verify: int = 5
    max_plan_length: int = 20                 # lowered from 30 per user decision
    consecutive_surprises_to_halt: int = 3
```

### 6.2 Failure taxonomy (summary)

| Phase | Failure | Handling |
|-------|---------|----------|
| 1 | env.step raises | Abort attempt, return `FailedAttempt` |
| 1 | done=True during exploration | **SUCCESS** — return immediately |
| 2 | LLM raises (Ollama down) | Abort level, return `FailedLevel` |
| 2 | LLM parse_error | Retry once with stricter prompt; else empty hypothesis list |
| 2 | LLM returns 0 hypotheses | Skip Phases 3/4, go directly to Phase 5 with empty verified list |
| 2 | Invalid test_action | Mark hypothesis `untestable`, keep others |
| 3 | done=True during verify | **SUCCESS** — return |
| 3 | All hypotheses refuted | Continue to Phase 4 (error analysis may trigger expansion) |
| 3 | env.step raises mid-verify | Mark that hypothesis ambiguous, continue |
| 4 | Not enough errors (<10) | Skip expansion, proceed to Phase 5 |
| 4 | LLM suggests already-active expansion | Proceed (avoid infinite loop) |
| 5 | **Zero confirmed hypotheses** | **Emergency fallback** (see 6.4) |
| 5 | Empty LLM plan | Repeat highest-confidence unrefuted test_action 5x |
| 5 | Invalid actions in plan | Filter; if empty → emergency fallback |
| 6 | 3 consecutive surprises | Halt, trigger next attempt (Phase 2 with new evidence) |
| 6 | done=True | **SUCCESS** |
| 6 | Plan exhausted | `completed=False, reason='plan_exhausted'`, next attempt |
| ALL | Action budget exhausted | `FailedLevel(reason='action_budget')` |
| ALL | LLM budget exhausted | Switch to "LLM-off mode" — table predictions only |

### 6.3 Persistence rule

**The table and error_analyzer NEVER reset within a game.** They persist across attempts and across levels. Only reset between games.

**Rationale (user-approved):** The error analyzer's all-errors mode (ported from c1c2-hybrid, validated fix) looks at the full distribution, not a recent window. Old correct predictions dilute rather than dominate. A soft reset would throw away the very data that makes cross-level transfer work.

### 6.4 The critical edge case — Phase 5 with zero confirmed hypotheses

**Chosen strategy: Option C — Emergency fallback using table predictions (not LLM).**

```python
# src/charith/alfa_loop/planner.py

def emergency_fallback_plan(self, num_actions: int = 10) -> List[int]:
    """
    Called when Phase 5 has zero confirmed hypotheses.
    Uses the table (validated mechanistic component) instead of the LLM
    (weaker reasoning component) when the LLM's reasoning has failed.
    """
    action_weights = []
    for action_id in range(1, N_ACTIONS + 1):
        pred = self.table.predict(action_id, prev_action=self.prev_action)
        # Prefer well-observed, changeful actions
        weight = pred['confidence'] * np.log1p(pred['n_observations'])
        # Floor so untried actions still get some probability
        action_weights.append(max(weight, 0.05))

    probs = np.array(action_weights) / sum(action_weights)
    return list(np.random.choice(
        range(1, N_ACTIONS + 1), size=num_actions, p=probs
    ))
```

**Why Option C:** When the LLM fails, fall back to the mechanistic component. That's the whole thesis of the C1/C2 architecture. (User direct quote from Section 4 review.)

### 6.5 Graceful degradation — structured results

Every failure path returns structured results, never raises:

```python
@dataclass
class AttemptResult:
    completed: bool
    actions_taken: int
    llm_calls: int
    reason: str                                  # 'success' | 'delta_spike' | 'plan_exhausted' | 'env_error' | ...
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

**Explicit non-goals (these are allowed to crash):**
- Ollama not running → exception at construction
- `arc_agi` not installed → ImportError
- Invalid game_id → arc_agi's problem
- OS-level failures (disk full, OOM)

Everything *inside* a running game is caught and reported structurally.

---

## 7. Testing Strategy

### 7.1 Four-tier pyramid

```
Tier 4: Smoke      1 test,  manual,      real Ollama + real ls20
Tier 3: Integration 3 tests, mock deps,  full 6-phase loop
Tier 2: Phase       ~20 tests, mock deps, one per phase
Tier 1: Unit        ~20 tests, pure,     no env, no LLM
```

**Target: ~30 tests. Unit+phase tier runs in <2s. Integration in <10s. Smoke is manual.**

### 7.2 Unit tier

```
tests/
├── test_causal_engine_table.py          # ArcTableModel — port from c1c2-hybrid
├── test_causal_engine_analyzer.py       # ArcErrorAnalyzer — port from c1c2-hybrid
├── test_hypothesis_schema.py            # ExpectedOutcome, Hypothesis serialization
├── test_match_score.py                  # compute_match_score — 7 GOLDEN TESTS
└── test_actual_observation.py           # StructuredPercept diff → ActualObservation
```

**The 7 golden tests for `compute_match_score` lock in the matching semantics** as the spec any future refactor must preserve:

1. `test_no_effect_perfect_match` — expected.no_effect=True, actual empty → 1.0
2. `test_no_effect_violated` — expected.no_effect=True, actual has movement → 0.0
3. `test_direction_match_magnitude_off_by_one_tolerated` → 1.0
4. `test_direction_match_magnitude_off_by_two_fails_magnitude` → 0.5 (ambiguous)
5. `test_mean_aggregation_three_fields` — two match, one miss → ~0.667
6. `test_empty_expected_returns_neutral` — no fields populated → 0.5
7. `test_color_match_case_insensitive` — 'Green' matches 'green'

### 7.3 Mock fixtures

**`MockArcEnv`** — deterministic fake driven by a dict of `action_id → grid_transform`:

```python
class MockArcEnv:
    def __init__(self, initial_grid, rules, score_triggers=None):
        self.initial_grid = initial_grid
        self.rules = rules  # action_id → Callable[(grid, state), (new_grid, new_state)]
        self.score_triggers = score_triggers or {}
        # ...
    def reset(self) -> MockFrame: ...
    def get_observation(self) -> MockFrame: ...
    def step(self, action_id) -> Tuple[MockFrame, float, bool, dict]: ...
```

**`MockOllamaReasoner`** — returns canned JSON responses routed by keyword in the system prompt:

```python
class MockOllamaReasoner:
    def __init__(self, hypothesize_response=None, expansion_response=None, plan_response=None):
        self.call_count = 0
        self.calls = []   # list of (system, user) for assertions
    def reason_json(self, system, user) -> dict:
        # Route by keyword: 'hypothesis' → hypothesize_response, etc.
```

Both fakes have their own smoke tests (`test_mock_env.py`, `test_mock_llm.py`) to verify they behave deterministically.

### 7.4 Phase tier (6 files, ~20 tests total)

Per-phase tests with assertions targeting each phase's contract:

| Phase | Key assertions |
|-------|---------------|
| 1 Explore | returns 8 evidence entries; records in table; halts on done=True |
| 2 Hypothesize | calls LLM exactly once; parses structured ExpectedOutcome; handles empty LLM response; marks invalid test_action as untestable |
| 3 Verify | confirms correct hypothesis; refutes wrong direction; records errors in analyzer; handles done mid-sequence |
| 4 ErrorCheck | no structure → no expansion; Kruskal firing → 1 LLM call; applies returned expansion to table |
| 5 Plan | LLM called with verified rules text; zero confirmed → emergency fallback; filters invalid actions |
| 6 Execute | happy path completes plan; halts on 3 consecutive surprises; returns on done=True |

### 7.5 Integration tier (3 tests)

1. **`test_full_loop_no_crashes_happy_path`** — MockEnv with 4 working actions, MockLLM with correct hypotheses. Assert all 6 phases execute, LevelResult returned, LLM count ≤ budget.

2. **`test_full_loop_triggers_expansion`** — MockEnv where action 2 only works after action 1 (sequential rule). Assert `table.sequence_enabled = True` after the expansion cycle completes.

3. **`test_full_loop_exhausts_attempts_gracefully`** — MockLLM returns always-refuted hypotheses. Assert 3 attempts run, `LevelResult(completed=False)`, actions ≤ 150.

### 7.6 Smoke tier (manual)

`scripts/smoke_test_ls20.py` — **run manually at end of Milestone 1.** Prerequisites: Ollama running, arc_agi installed, ls20 downloaded. Success criteria:

- All 6 phases execute without exception
- LLM calls ≤ 8
- Actions ≤ 150
- LevelResult is structured
- Level completion is a bonus

### 7.7 TDD order

**Phase A — Port existing (Day 1)**
1. Copy c1c2-hybrid `ExpandableTableModel` tests → rename to `ArcTableModel` (key = action_id)
2. Copy c1c2-hybrid `ErrorAnalyzer` tests → rename to `ArcErrorAnalyzer`
3. Both should pass immediately after the port.

**Phase B — New unit tests (Day 1)**
4. Write `test_match_score.py` with 7 golden cases (RED)
5. Implement `compute_match_score` from user's code (GREEN)
6. Write `test_actual_observation.py` for percept diff → ActualObservation adapter (RED → GREEN)

**Phase C — Mock fixtures (Day 2)**
7. Build `MockArcEnv` + ls20-like transition rules
8. Build `MockOllamaReasoner` with keyword routing
9. Smoke tests on the mocks themselves

**Phase D — Per-phase (Day 2-3)**
10-15. Explorer → Hypothesizer → Verifier → ErrorChecker → Planner → Executor (RED → GREEN each)

**Phase E — Integration (Day 3)**
16. Three integration tests (RED → GREEN)
17. Manual smoke test against real ls20

### 7.8 Non-goals

- LLM output quality (tested only at smoke time)
- Real ARC env edge cases (mock env handles the contract)
- Performance / timing (no wall-clock assertions in unit tests)
- Perception module (already tested elsewhere)
- Translator (already working, read-only)
- Multi-level transfer (Milestone 2)

---

## 8. Explicit Non-Goals for Milestone 1

- **Level completion is a bonus, not required.** Success = full 6-phase loop runs cleanly.
- **Cross-level transfer** — the table persists across levels by design, but we don't test cross-level behavior in Milestone 1.
- **Multi-object games (tr87, etc.)** — the liberal `object_ref` matching will need tightening. Milestone 2.
- **Real-time performance** — no frame-rate assertions.
- **Replacing the old Path 2/3/5 agents** — they stay untouched. New agent lives alongside.

---

## 9. Implementation Order (handoff to writing-plans)

The implementation plan will derive from the TDD order in §7.7. High-level sequence:

1. **Port from c1c2-hybrid:** causal_engine/table_model.py + error_analyzer.py + their tests
2. **New dataclasses:** full_stack/hypothesis_schema.py + full_stack/budgets.py + full_stack/results.py
3. **New match logic:** compute_match_score + percept diff adapter + tests
4. **Mocks:** tests/fixtures/mock_env.py + mock_llm.py
5. **Phase 1-6 modules** with per-phase tests (one phase at a time, RED→GREEN)
6. **Orchestrator:** full_stack/charith_full_stack_agent.py
7. **Entry script:** scripts/play_full_stack.py
8. **Integration tests**
9. **Manual smoke test**

**Total implementation files: ~14.** Target: 3 days of focused work.

---

## 10. Open Risks (to watch during implementation)

1. **LLM structured output compliance.** Gemma models sometimes emit prose alongside JSON. The existing `response_parser.py` handles markdown code fences but may need hardening for the new nested `ExpectedOutcome` schema. Mitigation: retry-once-with-stricter-prompt in Phase 2.

2. **StructuredPercept → ActualObservation adapter complexity.** The conversion depends on exact shape of `StructuredPercept` from existing Path 2 code. If the percept schema doesn't expose displacement directly, we'll need to compute it from before/after object positions. Mitigation: write `test_actual_observation.py` first (RED) against real StructuredPercept fixtures captured from ls20 to nail down the adapter.

3. **Rule firing density on real ls20.** In c1c2-hybrid we found sequential rules fire ~1.6% of actions. If ls20's rules fire even more rarely, Phase 4's Kruskal may not trigger expansion within the 3-attempt budget. Mitigation: the all-errors mode we already fixed handles sparse signals correctly; but if real ls20 is sparser than expected, we may need to bias exploration toward suspected sources.

4. **Cross-level continuity.** The table persists but Phase 1's `prev_action` resets between levels. If ls20 Level 2 requires sequential knowledge from Level 1 with specific `prev_action` context, we may see unexpected Phase 4 triggers on Level 2's first 8 actions. Mitigation: document this explicitly, defer to Milestone 2.

---

## 11. Approval & Next Step

**Design approved by user** through incremental section-by-section review (Sections 1-5, with 4 user code contributions: `compute_match_score`, budget lowering, aggregation strategy rationale, and emergency fallback choice).

**Next step:** Invoke `superpowers:writing-plans` skill to turn this design into a step-by-step implementation plan with clear checkpoints.
