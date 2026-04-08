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
        if len(valid_prev) >= 2:
            try:
                kr = stats.kruskal(*valid_prev)
                p = float(kr.pvalue)
                # scipy returns NaN when all values across all groups are identical
                if np.isnan(p):
                    p = 1.0
                result["kruskal_p"] = p
                result["kruskal_fires"] = p < 0.05
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
        if len(valid_act) >= 2:
            try:
                kr_a = stats.kruskal(*valid_act)
                p = float(kr_a.pvalue)
                if np.isnan(p):
                    p = 1.0
                result["kruskal_cell_p"] = p
                result["kruskal_cell_fires"] = p < 0.05
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
