"""Graduation exam: does the agent pass probing for a given reality?"""
from typing import List, Dict
from charith.descartes.probes import ProbeResult


def graduation_exam(probe_results: List[ProbeResult],
                    pass_threshold: float = 0.8) -> Dict:
    """Check if agent passes graduation for a curriculum stage.

    Args:
        probe_results: list of ProbeResult for all mandatory features
        pass_threshold: fraction of features that must pass (default 80%)
    Returns:
        dict with 'passed', 'score', 'failed_features', 'passed_features'
    """
    n_passed = sum(1 for r in probe_results if r.passed)
    n_total = len(probe_results)
    score = n_passed / max(n_total, 1)

    return {
        "passed": score >= pass_threshold,
        "score": score,
        "n_passed": n_passed,
        "n_total": n_total,
        "passed_features": [
            r.feature_name for r in probe_results if r.passed
        ],
        "failed_features": [
            r.feature_name for r in probe_results if not r.passed
        ],
        "details": {
            r.feature_name: {
                "r2": r.r2_trained,
                "delta_r2": r.delta_r2,
                "p": r.p_value,
                "passed": r.passed,
            }
            for r in probe_results
        },
    }
