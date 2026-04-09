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
from charith.full_stack.results import AttemptResult, LevelResult, GameResult
import time


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

        self.perception = CoreKnowledgePerception()

        # Causal engine persists across attempts/levels within a game
        self.table = ArcTableModel(num_actions=num_actions)
        self.error_analyzer = ArcErrorAnalyzer()

        self.explorer = Explorer(env, self.perception, self.table)
        self.hypothesizer = Hypothesizer(llm, num_actions=num_actions)
        self.verifier = Verifier(env, self.perception, self.table, self.error_analyzer)
        self.error_checker = ErrorChecker(self.table, self.error_analyzer, llm)
        self.planner = Planner(llm, self.table, max_plan_length=self.budgets.max_plan_length)
        self.executor = Executor(
            env, self.perception, self.table, self.error_analyzer,
            halt_threshold=self.budgets.consecutive_surprises_to_halt,
        )

    def play_level(self) -> LevelResult:
        """Play one level — up to max_attempts_per_level attempts."""
        attempts = []
        for _ in range(self.budgets.max_attempts_per_level):
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

    def play_game(self, game_id: str = "", max_levels: int = 5) -> GameResult:
        """
        Play multiple levels of the same game in sequence.

        Keeps the causal table and error analyzer intact across levels
        (mechanics often carry across levels). Resets per-attempt hypothesis
        state implicitly on each play_level() call.

        Stops early when a level fails completely (no completion after
        max_attempts_per_level attempts) or when max_levels is reached.
        """
        start = time.time()
        level_results: List[LevelResult] = []
        levels_completed = 0
        levels_attempted = 0
        stopped_reason = "max_levels_reached"

        for level_idx in range(max_levels):
            levels_attempted += 1
            lr = self.play_level()
            level_results.append(lr)
            if lr.completed:
                levels_completed += 1
            else:
                stopped_reason = "level_failed"
                break

        return GameResult(
            game_id=game_id,
            levels_completed=levels_completed,
            levels_attempted=levels_attempted,
            level_results=level_results,
            total_actions=sum(lr.total_actions for lr in level_results),
            total_llm_calls=sum(lr.total_llm_calls for lr in level_results),
            wall_time_sec=time.time() - start,
            stopped_reason=stopped_reason,
        )

    def _play_attempt(self) -> AttemptResult:
        llm_calls_before = self._current_llm_call_count()
        actions_before = self.table.total_observations

        hypotheses = []
        goal = ""
        verified = []
        evidence = []
        expansions_triggered = []

        phase_reached = 1
        for _cycle in range(self.budgets.max_expansion_cycles_per_attempt):
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
                continue
            else:
                break

        # Phase 5
        phase_reached = 5
        state_obs = self.env.get_observation()
        state_grid = state_obs.frame[0]
        state_percept = self.perception.perceive(state_grid)

        # Identify the controllable by finding the object that actually moved
        # during exploration. If that fails, fall back to "smallest non-
        # background object" as the likely sprite.
        controllable_id = self._identify_controllable(evidence, state_percept)
        target_id = self._identify_target(state_percept, controllable_id)

        state_desc = self._build_state_description(
            state_percept, state_grid, controllable_id, target_id
        )
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

    def _build_state_description(self, percept, grid, controllable_id=None, target_id=None) -> str:
        """
        Build a spatial description of the current scene for the planner.

        Includes grid size, total object count, and per-object
        (color, centroid, size). When controllable_id / target_id are known
        from exploration, highlight them explicitly so the LLM plans paths
        between the right two objects.
        """
        try:
            h, w = grid.shape[:2]
        except Exception:
            h, w = 0, 0

        lines = [f"Grid size: {h}x{w}", f"Total objects: {percept.object_count}"]

        objs = list(percept.objects) if percept and percept.objects else []

        # Highlight controllable + target at the top of the list
        controllable = next((o for o in objs if o.object_id == controllable_id), None)
        target = next((o for o in objs if o.object_id == target_id), None)

        if controllable is not None:
            cr, cc = controllable.centroid
            lines.append(
                f"CONTROLLABLE: object_id={controllable.object_id} color={controllable.color} "
                f"centroid=(row={int(round(cr))}, col={int(round(cc))}) size={controllable.size}"
            )
        if target is not None:
            tr, tc = target.centroid
            lines.append(
                f"TARGET (best guess): object_id={target.object_id} color={target.color} "
                f"centroid=(row={int(round(tr))}, col={int(round(tc))}) size={target.size}"
            )
            if controllable is not None:
                dr = int(round(tr - cr))
                dc = int(round(tc - cc))
                lines.append(
                    f"DISTANCE from controllable to target: "
                    f"delta_row={dr} delta_col={dc}"
                )

        lines.append("Other objects:")
        # Sort rest by size, cap to 15 for prompt budget
        others = [o for o in objs if o.object_id not in {controllable_id, target_id}]
        others.sort(key=lambda o: -o.size)
        for o in others[:15]:
            r, c = o.centroid
            lines.append(
                f"  object_id={o.object_id} color={o.color} "
                f"centroid=(row={int(round(r))}, col={int(round(c))}) size={o.size}"
            )
        if len(others) > 15:
            lines.append(f"  ... and {len(others) - 15} more")

        return "\n".join(lines)

    def _identify_controllable(self, evidence, current_percept):
        """
        Find which object moved during Phase 1 exploration by running the
        object tracker on each action's before/after percepts. The object
        with the largest cross-frame displacement is the controllable.

        Falls back to the smallest non-background object if exploration
        didn't yield a clear mover (e.g., nothing moved, or all frames
        identical).
        """
        from charith.perception.object_tracker import ObjectTracker

        tracker = ObjectTracker()
        best_id = None
        best_disp = 0.0

        for ev in evidence:
            if ev.percept_before is None or ev.percept_after is None:
                continue
            try:
                matches = tracker.match(ev.percept_before.objects, ev.percept_after.objects)
            except Exception:
                continue
            before_by_id = {o.object_id: o for o in ev.percept_before.objects}
            after_by_id = {o.object_id: o for o in ev.percept_after.objects}
            for before_id, after_id in matches:
                b = before_by_id.get(before_id)
                a = after_by_id.get(after_id)
                if b is None or a is None:
                    continue
                dr = a.centroid[0] - b.centroid[0]
                dc = a.centroid[1] - b.centroid[1]
                disp = (dr * dr + dc * dc) ** 0.5
                if disp > best_disp:
                    best_disp = disp
                    best_id = after_id

        if best_id is not None and best_disp >= 0.5:
            return best_id

        # Fallback: smallest non-background object in the current scene
        objs = list(current_percept.objects) if current_percept.objects else []
        if not objs:
            return None
        objs.sort(key=lambda o: o.size)  # smallest first
        # Skip the largest (background-ish) if there are many
        return objs[0].object_id if objs else None

    def _identify_target(self, percept, controllable_id):
        """
        Target heuristic: the largest non-controllable, non-background
        object. Background is assumed to be the largest object overall.
        The LLM can override this in its reasoning if wrong.
        """
        objs = list(percept.objects) if percept.objects else []
        if not objs:
            return None
        objs.sort(key=lambda o: -o.size)  # largest first
        # Skip largest (background) and any object that matches the controllable
        for o in objs[1:]:
            if o.object_id != controllable_id:
                return o.object_id
        return None
