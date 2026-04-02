"""
CHARITH Agent — Core-knowledge Hierarchical Agent for Reasoning
in Turn-based Interactive Tasks with Heuristics.

The main agent loop. Derived from COGITO's "The Hum" but stripped to:
- No fixed Hz (runs as fast as the game)
- No Experiential Merkle Tree (simple dict-based stores)
- No Pain System (prediction errors are weighted floats)
- No Z3 verification (no formal logic needed for grid games)
- No consciousness monitoring (not needed for task performance)

What IS preserved:
- Predict -> Compare -> Update cycle (Active Inference core)
- Core Knowledge Priors (Spelke systems)
- Ontology Expansion detection (HIMARI 4-test)
- Thompson Sampling exploration (HIMARI budget allocation)
- Cross-level transfer (soft reset between levels, hard between games)

All 5 amendments applied:
- Amendment 1: WorldModel operates on StructuredPercepts/ObjectEffects
- Amendment 2: Ontology expansion rule splitting
- Amendment 3: Discriminating goal hypotheses
- Amendment 4: Relative context for cross-level transfer
- Amendment 5: Action sequence memory (bigram model)
"""

import numpy as np
import time
from typing import Optional, Dict, List, Set
from pathlib import Path
import yaml

from charith.perception.core_knowledge import CoreKnowledgePerception, StructuredPercept
from charith.perception.object_tracker import ObjectTracker
from charith.world_model.model import WorldModel, PredictionError, ObjectEffect
from charith.metacognition.ontology import OntologyExpansion
from charith.metacognition.goal_discovery import GoalDiscovery
from charith.metacognition.confidence import ConfidenceTracker
from charith.action.thompson import ThompsonSampler
from charith.action.action_space import N_ACTIONS
from charith.memory.working import WorkingMemory
from charith.memory.episodic import EpisodeStore
from charith.memory.sequences import ActionSequenceMemory
from charith.utils.logging import AgentLogger


class CHARITHAgent:
    """
    Main agent class.

    Usage:
        agent = CHARITHAgent()
        scorecard = agent.play_game("deterministic_movement")
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # Initialize modules
        self.perception = CoreKnowledgePerception()
        self.object_tracker = ObjectTracker()
        self.world_model = WorldModel(
            max_rules=self.config.get('max_rules', 10000),
        )
        self.ontology = OntologyExpansion(
            window_size=self.config.get('ontology_window', 50),
        )
        self.goal_discovery = GoalDiscovery()
        self.confidence = ConfidenceTracker()
        self.explorer = ThompsonSampler(
            n_actions=N_ACTIONS,
            info_gain_weight=self.config.get('info_gain_weight', 0.5),
        )
        self.working_memory = WorkingMemory(
            capacity=self.config.get('wm_capacity', 7),
        )
        self.episode_store = EpisodeStore(
            max_episodes=self.config.get('max_episodes', 5000),
        )
        self.sequence_memory = ActionSequenceMemory(n_actions=N_ACTIONS)
        self.logger = AgentLogger()

        # State tracking
        self._tick = 0
        self._prev_grid: Optional[np.ndarray] = None
        self._prev_percept: Optional[StructuredPercept] = None
        self._last_action: int = 0
        self._last_info_gain: float = 0.0
        self._controllable_ids: Set[int] = set()
        self._total_actions = 0
        self._levels_completed = 0

    def play_game(self, game_id: str,
                  max_actions: int = 10000,
                  render: bool = False):
        """
        Play a complete game (all levels).

        Args:
            game_id: game identifier (e.g., "deterministic_movement", "hidden_goal")
            max_actions: maximum total actions before giving up
            render: whether to render in terminal (not implemented for mock)
        """
        # Determine if this is a mock game or real ARC-AGI-3 game
        from charith.mock_env import MockArcade
        MOCK_GAMES = MockArcade.GAME_IDS

        self._use_real_sdk = False
        if game_id not in MOCK_GAMES:
            try:
                import arc_agi
                from arcengine import GameAction
                arcade = arc_agi.Arcade()
                env = arcade.make(game_id, render_mode="terminal" if render else None)
                self._use_real_sdk = True
                self._game_action_map = {
                    0: GameAction.ACTION1, 1: GameAction.ACTION2,
                    2: GameAction.ACTION3, 3: GameAction.ACTION4,
                    4: GameAction.ACTION5, 5: GameAction.ACTION6,
                    6: GameAction.ACTION7,
                }
            except Exception:
                arcade = MockArcade()
                env = arcade.make(game_id)
                self._game_action_map = None
        else:
            arcade = MockArcade()
            env = arcade.make(game_id)
            self._game_action_map = None

        self._hard_reset()

        # Real SDK: call reset() to get initial observation
        # Mock: call get_observation() directly
        if self._use_real_sdk:
            initial_frame = env.reset()
            grid = self._parse_observation(initial_frame)
            self._available_actions = getattr(initial_frame, 'available_actions', list(range(N_ACTIONS)))
        else:
            grid = self._parse_observation(env.get_observation())
            self._available_actions = list(range(N_ACTIONS))

        for action_num in range(max_actions):
            if grid is None:
                break

            # ===== THE HUM (stripped) =====
            action = self._tick_cycle(grid)

            # ===== ACT =====
            # Map action int to GameAction for real SDK
            if self._game_action_map and action in self._game_action_map:
                game_action = self._game_action_map[action]
            else:
                game_action = action
            step_result = env.step(game_action)

            # ===== PARSE NEXT OBSERVATION =====
            grid = self._parse_observation(step_result)

            # Update available actions from SDK frame
            if hasattr(step_result, 'available_actions') and step_result.available_actions:
                self._available_actions = step_result.available_actions

            # ===== PROCESS FEEDBACK =====
            score = self._extract_score(step_result)
            level_complete = self._check_level_complete(step_result)
            game_over = self._check_game_over(step_result)

            # ===== REWARD (Amendment 3: discriminating hypotheses) =====
            reward = self.goal_discovery.update(
                grid if grid is not None else np.zeros((1, 1), dtype=int),
                action, score=score,
                level_complete=level_complete, game_over=game_over,
                percept_prev=self._prev_percept,
                percept_curr=self.perception.perceive(grid) if (self._prev_percept and grid is not None) else None,
            )

            # ===== UPDATE EXPLORATION =====
            self.explorer.update(
                context_hash=self._state_hash(grid) if grid is not None else 0,
                action=action,
                reward=reward,
                info_gain=self._last_info_gain,
            )

            # ===== UPDATE SEQUENCE MEMORY (Amendment 5) =====
            if self._total_actions > 0:
                self.sequence_memory.update(
                    self._last_action, action, reward
                )

            # ===== LEVEL TRANSITION =====
            if level_complete:
                self._on_level_complete()

            if game_over:
                break

            self._total_actions += 1

            # For mock env, get next observation separately
            if not self._use_real_sdk:
                grid = self._parse_observation(env.get_observation())

        return arcade.get_scorecard()

    def _tick_cycle(self, grid: np.ndarray) -> int:
        """
        One cycle of The Hum (stripped).

        PERCEIVE -> TRACK -> CONTEXT -> PREDICT -> EFFECTS -> ERROR ->
        UPDATE -> CONTINGENCY -> META -> GOAL -> SELECT
        """
        # 1. PERCEIVE: Apply core knowledge priors
        percept = self.perception.perceive(grid)

        # 2. TRACK OBJECTS: Match across frames
        matched_pairs = []
        if self._prev_percept is not None:
            matched_pairs = self.object_tracker.match(
                self._prev_percept.objects, percept.objects
            )
            # Record displacements for goal-directed motion detection
            curr_map = {o.object_id: o for o in percept.objects}
            for _, curr_id in matched_pairs:
                if curr_id in curr_map:
                    self.perception.agency.record_object_displacement(
                        curr_id, curr_map[curr_id].centroid
                    )

        # 3. IDENTIFY CONTROLLABLE OBJECTS
        self._controllable_ids = self.perception.agency.detect_controllable_objects(
            percept.objects, self.perception.agency._contingencies
        )

        # 4. EXTRACT CONTEXT (Amendment 1+4: relative features only)
        context = self.world_model.extract_context(percept, self._controllable_ids)

        # 5. PREDICT: What did we expect?
        predicted_effects = None
        if self._prev_percept is not None:
            predicted_effects = self.world_model.predict(
                self._last_action, context
            )

        # 6. COMPUTE ACTUAL EFFECTS (Amendment 1: object-level)
        actual_effects = []
        if self._prev_percept is not None:
            actual_effects = self.world_model.compute_effects(
                self._prev_percept, percept, matched_pairs
            )

        # 7. COMPUTE ERROR (object-level)
        error = self._compute_object_error(predicted_effects, actual_effects)
        self._last_info_gain = error.error_magnitude

        # 8. UPDATE WORLD MODEL
        if self._prev_percept is not None:
            self.world_model.update(
                self._last_action, context, actual_effects, self._tick
            )
        self.world_model.record_error(error)

        # 9. RECORD ACTION CONTINGENCY
        if self._prev_grid is not None:
            self.perception.agency.record_action_contingency(
                self._last_action, self._prev_grid, grid
            )

        # 10. STORE EPISODE
        self.episode_store.record(
            state=grid,
            action=self._last_action,
            next_state=grid,
            error=error.weighted_error,
            tick=self._tick,
        )

        # 11. META: Ontology expansion check (every N ticks)
        check_interval = self.config.get('ontology_check_interval', 50)
        if self._tick > 0 and self._tick % check_interval == 0:
            expansion_result = self.ontology.check(
                self.world_model.get_recent_errors(),
                self.world_model.get_rule_count(),
                self.world_model.get_accuracy(),
            )
            if expansion_result.should_expand:
                self.ontology.execute_expansion(
                    expansion_result.suggested_type,
                    self.world_model,
                    self.perception,
                )
                self.logger.log('ontology_expansion',
                               expansion_result.suggested_type, self._tick)

        # 12. UPDATE CONFIDENCE
        self.confidence.update(error.error_magnitude,
                               self.world_model.get_rule_count())

        # 13. SELECT ACTION (Amendment 5: sequence boost)
        goal_action = None
        goal_directed = False
        best_goal = self.goal_discovery.get_best_hypothesis()
        if best_goal and best_goal.confidence > 0.6:
            goal_directed = True
            goal_action = self._goal_to_action(best_goal, percept)

        s_hash = self._state_hash(grid)
        # Map SDK action indices (1-7) to our 0-indexed space
        avail = self._map_available_actions(self._available_actions)
        action = self.explorer.select_action(
            context_hash=s_hash,
            available_actions=avail if avail else None,
            goal_directed=goal_directed,
            goal_action=goal_action,
            prev_action=self._last_action if self._total_actions > 0 else None,
            sequence_memory=self.sequence_memory,
        )

        # Update state
        self._prev_grid = grid.copy()
        self._prev_percept = percept
        self._last_action = action
        self._tick += 1

        return action

    def _compute_object_error(self, predicted: Optional[List[ObjectEffect]],
                               actual: List[ObjectEffect]) -> PredictionError:
        """Compute error at the object level using significant effects only.

        Compares only effects that carry information (moved, appeared,
        disappeared, shape changed). Static d=(0,0) objects are noise.
        """
        if predicted is None:
            return PredictionError(
                predicted_grid=None, observed_grid=None,
                error_magnitude=1.0, error_cells=[],
                precision=0.0, weighted_error=0.0, is_novel=True
            )

        self.world_model._total_predictions += 1

        # Filter to significant effects only
        sig_pred = self.world_model._significant_effects(predicted)
        sig_actual = self.world_model._significant_effects(actual)

        # Both predict "nothing moves" and nothing moved = correct
        if not sig_pred and not sig_actual:
            self.world_model._correct_predictions += 1
            return PredictionError(
                predicted_grid=None, observed_grid=None,
                error_magnitude=0.0, error_cells=[],
                precision=0.8, weighted_error=0.0, is_novel=False
            )

        # Match significant effects by color + displacement
        matches = 0
        total = max(len(sig_pred), len(sig_actual), 1)
        used = set()
        for p_eff in sig_pred:
            for i, a_eff in enumerate(sig_actual):
                if i in used:
                    continue
                if (p_eff.object_color == a_eff.object_color
                        and p_eff.displacement == a_eff.displacement):
                    matches += 1
                    used.add(i)
                    break

        accuracy = matches / total
        magnitude = 1.0 - accuracy
        precision = min(0.9, 0.3 + 0.1 * len(sig_pred))  # More effects = higher precision

        if magnitude < 0.01:
            self.world_model._correct_predictions += 1

        return PredictionError(
            predicted_grid=None, observed_grid=None,
            error_magnitude=magnitude, error_cells=[],
            precision=precision, weighted_error=magnitude * precision,
            is_novel=False
        )

    @staticmethod
    def _map_available_actions(sdk_actions) -> Optional[List[int]]:
        """Map SDK GameAction values (1-7) to our 0-indexed action space.

        SDK returns [1, 2, 3, 4] meaning GameAction.ACTION1-ACTION4.
        We map to [0, 1, 2, 3] for Thompson Sampler.
        """
        if not sdk_actions:
            return None
        return [a - 1 for a in sdk_actions if isinstance(a, int) and 1 <= a <= 7]

    def _goal_to_action(self, goal, percept: StructuredPercept) -> Optional[int]:
        """Map a goal hypothesis to a concrete action. Phase 2: proper planning."""
        return None

    def _on_level_complete(self):
        """Handle level completion — soft reset."""
        self._levels_completed += 1
        self.logger.log('level_complete', self._levels_completed, self._tick)

        # Soft reset: keep learned rules and hypotheses
        self.perception.reset()
        self.ontology.reset()
        self.goal_discovery.reset()
        self.explorer.reset_context()
        self.world_model.reset()  # Keeps rules with confidence decay!
        self.sequence_memory.reset()
        self.episode_store.mark_level_boundary()
        self.confidence.reset()

        self._prev_grid = None
        self._prev_percept = None
        self._controllable_ids = set()
        self._tick = 0

    def _hard_reset(self):
        """Full reset for a new game."""
        self.perception.reset()
        self.world_model.hard_reset()
        self.ontology.reset()
        self.goal_discovery.hard_reset()
        self.explorer.hard_reset()
        self.episode_store.hard_reset()
        self.working_memory.clear()
        self.sequence_memory.hard_reset()
        self.confidence.reset()
        self.logger.clear()

        self._tick = 0
        self._prev_grid = None
        self._prev_percept = None
        self._last_action = 0
        self._last_info_gain = 0.0
        self._controllable_ids = set()
        self._total_actions = 0
        self._levels_completed = 0

    def _parse_observation(self, observation) -> Optional[np.ndarray]:
        """
        Parse observation into numpy grid.

        Handles:
        - np.ndarray directly (mock env)
        - FrameDataRaw from real SDK (grid in .frame[0])
        - dict with 'grid' key
        """
        if observation is None:
            return None
        if isinstance(observation, np.ndarray):
            return observation.astype(int)
        # Real SDK: FrameDataRaw has .frame attribute (list of numpy arrays)
        if hasattr(observation, 'frame') and observation.frame:
            grid = observation.frame[0]
            if isinstance(grid, np.ndarray):
                return grid.astype(int)
        if isinstance(observation, dict):
            if 'grid' in observation:
                return np.array(observation['grid'], dtype=int)
            if 'board' in observation:
                return np.array(observation['board'], dtype=int)
        try:
            return np.array(observation, dtype=int)
        except (ValueError, TypeError):
            return None

    def _extract_score(self, step_result) -> Optional[float]:
        """Extract score from step result (mock dict or SDK FrameDataRaw)."""
        if step_result is None:
            return None
        if isinstance(step_result, dict):
            return step_result.get('score', step_result.get('reward', None))
        # Real SDK: FrameDataRaw has levels_completed (use as proxy for score)
        if hasattr(step_result, 'levels_completed'):
            return float(step_result.levels_completed)
        return None

    def _check_level_complete(self, step_result) -> bool:
        if step_result is None:
            return False
        if isinstance(step_result, dict):
            return step_result.get('level_complete', False)
        # Real SDK: check if levels_completed changed
        if hasattr(step_result, 'levels_completed'):
            new_levels = step_result.levels_completed
            if new_levels > self._levels_completed:
                return True
        return False

    def _check_game_over(self, step_result) -> bool:
        if step_result is None:
            return False
        if isinstance(step_result, dict):
            return step_result.get('game_over', step_result.get('done', False))
        # Real SDK: GameState.FINISHED or GameState.WON
        if hasattr(step_result, 'state'):
            state_val = str(step_result.state)
            if 'FINISHED' in state_val or 'WON' in state_val or 'LOST' in state_val:
                if 'NOT_FINISHED' not in state_val:
                    return True
        return False

    def _state_hash(self, grid: np.ndarray) -> int:
        """Fast hash for dictionary lookups."""
        return hash(grid.tobytes())

    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        # Try default config
        default = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
        if default.exists():
            with open(default) as f:
                return yaml.safe_load(f)
        return {
            'max_rules': 10000,
            'ontology_window': 50,
            'info_gain_weight': 0.5,
            'wm_capacity': 7,
            'max_episodes': 5000,
            'ontology_check_interval': 50,
        }
