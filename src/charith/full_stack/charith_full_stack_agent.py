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

    def _play_attempt(self) -> AttemptResult:
        llm_calls_before = self._current_llm_call_count()
        actions_before = self.table.total_observations

        hypotheses = []
        goal = ""
        verified = []
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
