"""LLMAgent -- Path 5 agent that uses a pre-trained LLM for reasoning.

The innovation is the TRANSLATION LAYER: CoreKnowledge StructuredPercepts are
converted to natural language, wrapped with C1/C2 context, and sent to the LLM.
The LLM's response is parsed back into an action + hypothesis updates.

No training required -- the LLM already knows how to reason.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept
from charith.perception.object_tracker import ObjectTracker
from charith.llm_agent.translator import PerceptTranslator
from charith.llm_agent.c1c2_framework import C1C2Framework
from charith.llm_agent.context_manager import ContextManager
from charith.llm_agent.llm_client import LLMClient
from charith.llm_agent.response_parser import ResponseParser


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are CHARITH -- an intelligent agent playing an unknown game on a grid.

You observe the grid state as natural language descriptions of objects, their colours,
positions, spatial relations, and changes from the previous step.

## Conceptual Framework (C1/C2)

You have a CONCEPTUAL VOCABULARY (C1) -- the set of concepts you can reason about.
Initially these come from core developmental knowledge (objects, movement, walls, goals,
spatial relations, counting, symmetry). As you play, you may DISCOVER new concepts
that expand this vocabulary (e.g., GRAVITY, TELEPORT, COLOR_CHANGE, KEY_LOCK).

You maintain HYPOTHESES (C2) -- your current beliefs about the game mechanics.
Each hypothesis has a confidence level (low/med/high). Update these as you gather evidence.

## Strategy (3 phases)

1. EXPLORE (ticks 1-15): Try each action systematically to learn what they do.
   Map actions to effects. Discover the controllable object.
2. HYPOTHESIZE (ticks 15-40): Form and test hypotheses about the game goal
   and mechanics. Look for patterns in how the environment responds.
3. EXPLOIT (ticks 40+): Execute your best strategy based on confirmed hypotheses.
   If stuck, return to exploration.

## Response Format (STRICT)

You MUST respond in exactly this format:

ACTION: <integer from available actions>
HYPOTHESIS: <your current best hypothesis about what to do and why>
CONFIDENCE: <low|med|high>
C1_EXPANSION: <NEW_CONCEPT_NAME: description> or NONE
REASONING: <1-2 sentences explaining your action choice>

Example:
ACTION: 2
HYPOTHESIS: Moving right towards the blue goal object
CONFIDENCE: med
C1_EXPANSION: NONE
REASONING: The blue object appears to be the goal. Action 2 moves right, which decreases distance.

Example with C1 expansion:
ACTION: 3
HYPOTHESIS: Need to collect keys before reaching the door
CONFIDENCE: low
C1_EXPANSION: KEY_LOCK: Some objects act as keys that unlock barriers when touched
REASONING: After touching the yellow object, a wall disappeared. This suggests a key-lock mechanic.
"""


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Path 5 agent: perception -> translate -> LLM reason -> act."""

    def __init__(
        self,
        model: str = "gemma3:12b",
        temperature: float = 0.3,
    ) -> None:
        # Perception
        self.perception = CoreKnowledgePerception()
        self.object_tracker = ObjectTracker()

        # LLM pipeline
        self.translator = PerceptTranslator()
        self.c1c2 = C1C2Framework()
        self.context = ContextManager()
        self.llm = LLMClient(model=model, temperature=temperature)
        self.parser = ResponseParser()

        # State tracking
        self._tick: int = 0
        self._prev_percept: Optional[StructuredPercept] = None
        self._prev_grid: Optional[np.ndarray] = None
        self._controllable_ids: Set[int] = set()
        self._last_action: int = 0
        self._available_actions: List[int] = [1, 2, 3, 4]

    # ---- public API --------------------------------------------------------

    def act(self, observation, available_actions: Optional[List[int]] = None) -> int:
        """Full pipeline: perceive -> translate -> prompt -> LLM -> parse -> action.

        Parameters
        ----------
        observation : array-like or SDK observation
            The current game frame (grid).
        available_actions : list of int, optional
            Which actions are valid this tick.

        Returns
        -------
        int
            The chosen action.
        """
        if available_actions is not None:
            self._available_actions = available_actions

        self._tick += 1

        # 1. Parse observation to grid
        grid = self._parse_observation(observation)

        # 2. Record action contingency (from previous action)
        if self._prev_grid is not None and self._tick > 1:
            self.perception.agency.record_action_contingency(
                str(self._last_action), self._prev_grid, grid
            )

        # 3. Perceive
        percept = self.perception.perceive(grid)

        # 4. Track objects and detect controllable
        if self._prev_percept is not None:
            self.object_tracker.match(self._prev_percept.objects, percept.objects)

        if self._tick > 2 and self.perception.agency._contingencies:
            ctrl_ids = self.perception.agency.detect_controllable_objects(
                percept.objects, self.perception.agency._contingencies
            )
            if ctrl_ids:
                self._controllable_ids = set(ctrl_ids)

        # 5. Record displacement for controllable objects
        for obj in percept.objects:
            if obj.object_id in self._controllable_ids:
                self.perception.agency.record_object_displacement(
                    obj.object_id, obj.centroid
                )

        # 6. Translate to natural language
        observation_text = self.translator.translate(
            percept,
            prev_percept=self._prev_percept,
            controllable_ids=self._controllable_ids,
            tick=self._tick,
            available_actions=self._available_actions,
        )

        # 7. Describe effect of last action
        if self._prev_percept is not None:
            effect = self._describe_action_effect(self._prev_percept, percept, self._last_action)
            self.context.record_action_effect(self._last_action, effect)

        # 8. Build prompt and query LLM
        user_prompt = self._build_prompt(observation_text)
        response_text = self.llm.query(SYSTEM_PROMPT, user_prompt)

        # 9. Parse LLM response
        parsed = self.parser.parse(response_text, self._available_actions)

        # Force exploration: cycle through actions in first N ticks
        n_explore = len(self._available_actions) * 2  # Try each action twice
        if self._tick < n_explore:
            forced_action = self._available_actions[self._tick % len(self._available_actions)]
            parsed['action'] = forced_action
            parsed['reasoning'] = f"[FORCED EXPLORE tick {self._tick}] " + parsed.get('reasoning', '')

        # 10. Update C1/C2
        if parsed["hypothesis"]:
            self.c1c2.update_hypothesis(parsed["hypothesis"], parsed["confidence"])
        if parsed["c1_expansion"]:
            # Parse "NAME: description" format
            parts = parsed["c1_expansion"].split(":", 1)
            name = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else name
            self.c1c2.expand_c1(name, desc, tick=self._tick)

        # 11. Record in context history
        is_important = (
            parsed["c1_expansion"] is not None
            or parsed["confidence"] == "high"
            or self._tick <= 3
        )
        self.context.add_tick(
            tick=self._tick,
            observation=observation_text[:200],  # truncate for storage
            action=parsed["action"],
            result_description=effect if self._prev_percept else "initial observation",
            hypothesis=parsed["hypothesis"],
            c1_expansion=parsed["c1_expansion"],
            is_important=is_important,
        )

        # 12. Update state for next tick
        self._prev_percept = percept
        self._prev_grid = grid.copy()
        self._last_action = parsed["action"]

        return parsed["action"]

    def play_game(self, game_id: str, max_actions: int = 200):
        """Play a real ARC-AGI-3 game via the SDK.

        Parameters
        ----------
        game_id : str
            Arcade game identifier (e.g. 'ls20').
        max_actions : int
            Maximum number of actions before stopping.

        Returns
        -------
        scorecard or dict with results
        """
        try:
            import arc_agi
            from arcengine import GameAction
        except ImportError:
            print("[LLMAgent] arc_agi / arcengine not installed -- returning mock result")
            return {"game_id": game_id, "status": "sdk_not_available"}

        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        if env is None:
            print(f"[LLMAgent] Game '{game_id}' not found")
            return {"game_id": game_id, "status": "game_not_found"}
        frame = env.reset()

        print(f"[LLMAgent] Playing game: {game_id} (max {max_actions} actions)")
        self._levels_completed = 0
        self._available_actions = getattr(frame, 'available_actions', [1, 2, 3, 4])

        for step in range(max_actions):
            grid = self._parse_observation(frame)
            action = self.act(grid)

            # Log what the LLM decided
            h = self.c1c2.hypotheses[-1] if self.c1c2.hypotheses else None
            last_hyp = (h.text if hasattr(h, 'text') else h.get('text', '?')) if h else 'none'
            last_conf = (h.confidence if hasattr(h, 'confidence') else h.get('confidence', '?')) if h else '?'
            effects = self.context.get_discovered_effects()
            n_c1 = len(self.c1c2.c1_expansions)
            effects = "; ".join(f"A{k}:{v[:30]}" for k, v in self.context.action_effect_map.items())
            print(f"  [Tick {step+1:2d}] ACTION={action} | hyp: {last_hyp[:60]} ({last_conf}) | C1+={n_c1} | effects: {effects[:80]}")

            # Map action int to GameAction
            action_map = {
                1: GameAction.ACTION1,
                2: GameAction.ACTION2,
                3: GameAction.ACTION3,
                4: GameAction.ACTION4,
                5: GameAction.ACTION5,
                6: GameAction.ACTION6,
                7: GameAction.ACTION7,
            }
            game_action = action_map.get(action, GameAction.ACTION1)
            frame = env.step(game_action)

            # Check for game over
            state_str = str(getattr(frame, 'state', ''))
            if 'NOT_FINISHED' not in state_str and state_str:
                print(f"[LLMAgent] Game ended at step {step + 1}: {state_str}")
                break
            if hasattr(frame, 'levels_completed') and frame.levels_completed > self._levels_completed:
                self._levels_completed = frame.levels_completed
                print(f"[LLMAgent] Level {self._levels_completed} complete!")

        scorecard = arcade.get_scorecard()
        print(f"[LLMAgent] Done. Levels: {self._levels_completed}")
        return {
            'game_id': game_id,
            'levels_completed': self._levels_completed,
            'actions_taken': step + 1,
            'scorecard': str(scorecard)[:200],
        }

    # ---- internal ----------------------------------------------------------

    def _parse_observation(self, obs) -> np.ndarray:
        """Parse SDK observation to numpy grid."""
        if hasattr(obs, "frame") and obs.frame is not None:
            frame = obs.frame
            if isinstance(frame, list) and len(frame) > 0:
                return np.array(frame[0], dtype=int)
            return np.array(frame, dtype=int)
        if isinstance(obs, np.ndarray):
            return obs.astype(int)
        return np.array(obs, dtype=int)

    def _build_prompt(self, observation_text: str) -> str:
        """Combine C1 text + hypotheses + history + current observation."""
        parts = [
            self.c1c2.get_c1_text(),
            "",
            self.c1c2.get_hypotheses_text(),
            "",
            self.context.get_discovered_effects(),
            "",
            self.context.get_history_text(),
            "",
            "--- CURRENT OBSERVATION ---",
            observation_text,
            "",
            f"Choose an action from: {self._available_actions}",
        ]
        return "\n".join(parts)

    def _describe_action_effect(
        self,
        prev_percept: StructuredPercept,
        curr_percept: StructuredPercept,
        action: int,
    ) -> str:
        """Describe what changed as a result of the action."""
        # Check controllable movement
        for oid in self._controllable_ids:
            prev_obj = next((o for o in prev_percept.objects if o.object_id == oid), None)
            curr_obj = next((o for o in curr_percept.objects if o.object_id == oid), None)
            if prev_obj and curr_obj:
                dr = curr_obj.centroid[0] - prev_obj.centroid[0]
                dc = curr_obj.centroid[1] - prev_obj.centroid[1]
                dist = (dr ** 2 + dc ** 2) ** 0.5
                if dist < 0.5:
                    return "No movement (wall?)"
                direction = "UP" if dr < 0 else "DOWN" if dr > 0 else ""
                if dc < 0:
                    direction += "LEFT" if not direction else "-LEFT"
                elif dc > 0:
                    direction += "RIGHT" if not direction else "-RIGHT"
                return f"Controllable moved {direction} by {dist:.0f} cells"

        # Fallback: count pixel changes
        if prev_percept.raw_grid.shape == curr_percept.raw_grid.shape:
            changed = int((prev_percept.raw_grid != curr_percept.raw_grid).sum())
            if changed == 0:
                return "No visible change"
            return f"{changed} pixels changed"

        return "Grid dimensions changed"
