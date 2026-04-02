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
        # Try real SDK first, fall back to mock
        try:
            import arc_agi
            arcade = arc_agi.Arcade()
            env = arcade.make(game_id, render_mode="terminal" if render else None)
        except ImportError:
            from charith.mock_env import MockArcade
            arcade = MockArcade()
            env = arcade.make(game_id)

        self._hard_reset()

        for action_num in range(max_actions):
            # ===== OBSERVE =====
            observation = env.get_observation()
            grid = self._parse_observation(observation)
            if grid is None:
                break

            # ===== THE HUM (stripped) =====
            action = self._tick_cycle(grid)

            # ===== ACT =====
            step_result = env.step(action)

            # ===== PROCESS FEEDBACK =====
            score = self._extract_score(step_result)
            level_complete = self._check_level_complete(step_result)
            game_over = self._check_game_over(step_result)

            # ===== REWARD (Amendment 3: discriminating hypotheses) =====
            reward = self.goal_discovery.update(
                grid, action, score=score,
                level_complete=level_complete, game_over=game_over,
                percept_prev=self._prev_percept,
                percept_curr=self.perception.perceive(grid) if self._prev_percept else None,
            )

            # ===== UPDATE EXPLORATION =====
            self.explorer.update(
                context_hash=self._state_hash(grid),
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
        action = self.explorer.select_action(
            context_hash=s_hash,
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
        """Compute error at the object level."""
        if predicted is None:
            return PredictionError(
                predicted_grid=None, observed_grid=None,
                error_magnitude=1.0, error_cells=[],
                precision=0.0, weighted_error=0.0, is_novel=True
            )

        self.world_model._total_predictions += 1

        matches = 0
        total = max(len(predicted), len(actual), 1)
        for p_eff in predicted:
            for a_eff in actual:
                if (p_eff.object_color == a_eff.object_color
                        and p_eff.displacement == a_eff.displacement):
                    matches += 1
                    break

        accuracy = matches / total
        magnitude = 1.0 - accuracy
        precision = 0.5

        if magnitude < 0.01:
            self.world_model._correct_predictions += 1

        return PredictionError(
            predicted_grid=None, observed_grid=None,
            error_magnitude=magnitude, error_cells=[],
            precision=precision, weighted_error=magnitude * precision,
            is_novel=False
        )

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
        """Parse observation into numpy grid."""
        if observation is None:
            return None
        if isinstance(observation, np.ndarray):
            return observation
        if isinstance(observation, dict):
            if 'grid' in observation:
                return np.array(observation['grid'])
            if 'board' in observation:
                return np.array(observation['board'])
        try:
            return np.array(observation)
        except (ValueError, TypeError):
            return None

    def _extract_score(self, step_result) -> Optional[float]:
        """Extract score from step result."""
        if step_result is None:
            return None
        if isinstance(step_result, dict):
            return step_result.get('score', step_result.get('reward', None))
        if hasattr(step_result, 'score'):
            return step_result.score
        return None

    def _check_level_complete(self, step_result) -> bool:
        if step_result is None:
            return False
        if isinstance(step_result, dict):
            return step_result.get('level_complete', False)
        return getattr(step_result, 'level_complete', False)

    def _check_game_over(self, step_result) -> bool:
        if step_result is None:
            return False
        if isinstance(step_result, dict):
            return step_result.get('game_over', step_result.get('done', False))
        return getattr(step_result, 'done', False)

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
