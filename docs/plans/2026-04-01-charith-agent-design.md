# CHARITH Agent — Design Document

**Date:** 2026-04-01
**Author:** Charith Suranga + Claude Code
**Status:** Approved

## Overview

Full scaffold implementation of the CHARITH ARC-AGI-3 agent following the
original guide (ARC_AGI_3_AGENT_CLAUDE_CODE_GUIDE.md) with all 5 amendments
from CHARITH_GUIDE_AMENDMENTS_V1.md applied.

## Key Decisions

1. **Approach A: Exact guide replication** with amendments overriding where they conflict
2. **MockEnvironment** built first (before WorldModel) — serves as test harness
3. **No API key yet** — mock env simulates arc-agi SDK interface
4. **Full scaffold** — all modules implemented, Phase 2 stubs clearly marked

## Amendments Applied

| # | Amendment | Impact |
|---|---|---|
| 1 | WorldModel operates on StructuredPercepts + ObjectEffects | Critical — enables generalization |
| 2 | One working ontology expansion (context-conditioned rule splitting) | Critical — novel contribution works |
| 3 | GoalHypotheses make discriminating predictions | High — enables real goal inference |
| 4 | Relative context features for cross-level transfer | High — enables level progression |
| 5 | Action sequence memory (bigram model) | Medium — enables sequence learning |

## MockEnvironment Design

4 scenarios targeting specific modules:

| Scenario | Tests | Mechanic |
|---|---|---|
| DeterministicMovement | WorldModel | Actions 1-4 = up/down/left/right |
| HiddenGoal | GoalDiscovery | Move blue onto green, score jumps |
| ContextDependent | OntologyExpansion | Action effect depends on background color |
| MultiLevel | Cross-level transfer | Increasing difficulty across levels |

Format: observations as np.ndarray (int, 0-9), step returns dict with
grid/score/level_complete/game_over.

## Implementation Order

1. Perception (core_knowledge.py — verbatim from guide)
2. MockEnvironment (DeterministicMovement first, then other scenarios)
3. ObjectTracker (Hungarian matching for identity across frames)
4. WorldModel (Amendment 1+4: ObjectEffect, context-based, relative features)
5. Ontology Expansion (Amendment 2: context-conditioned rule splitting)
6. GoalDiscovery (Amendment 3: discriminating hypothesis subclasses)
7. Thompson Sampling + Sequences (Amendment 5: bigram boost)
8. Memory stubs (working, episodic, rules, consolidation)
9. Agent loop (agent.py — amendments 1-5 integrated)
10. Full test suite (unit + integration + ablation validation)

## Project Structure

```
charith-arc-agent/
  pyproject.toml
  .python-version
  configs/default.yaml
  src/charith/
    __init__.py
    agent.py
    mock_env.py
    perception/ (core_knowledge, object_tracker, spatial)
    world_model/ (model, transition — rewritten per Amend 1+4)
    metacognition/ (ontology — Amend 2, goal_discovery — Amend 3, confidence)
    action/ (thompson — Amend 5, planner, action_space)
    memory/ (working, episodic, rules, consolidation, sequences — Amend 5)
    utils/ (grid_ops, hashing, logging)
  scripts/ (play_single, play_all, evaluate, benchmark)
  tests/ (unit + integration)
  docs/plans/
```
