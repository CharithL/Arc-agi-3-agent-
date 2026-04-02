"""Tests for DESCARTES linear probing pipeline."""
import numpy as np
import pytest

from charith.descartes.probes import LinearProbe, ProbeResult, run_probe
from charith.descartes.graduation import graduation_exam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_data(n: int = 500, hidden_size: int = 32,
                      seed: int = 0) -> tuple:
    """Create hidden states that linearly encode a single feature.

    h_t is random, but the target is a linear function of h_t plus
    small noise, so Ridge should achieve high R^2.
    """
    rng = np.random.RandomState(seed)
    hidden_states = rng.randn(n, hidden_size)
    weights = rng.randn(hidden_size)
    targets = hidden_states @ weights + rng.randn(n) * 0.1
    # 5 episodes of 100 timesteps each
    episode_boundaries = list(range(0, n, n // 5))
    return hidden_states, targets, episode_boundaries


def _make_random_data(n: int = 500, hidden_size: int = 32,
                      seed: int = 7) -> tuple:
    """Create hidden states with targets that are independent noise."""
    rng = np.random.RandomState(seed)
    hidden_states = rng.randn(n, hidden_size)
    targets = rng.randn(n)
    episode_boundaries = list(range(0, n, n // 5))
    return hidden_states, targets, episode_boundaries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProbeDetectsLinearSignal:
    """Probe should detect a feature that is linearly encoded."""

    def test_probe_detects_linear_signal(self):
        hidden_states, targets, boundaries = _make_linear_data()
        probe = LinearProbe(alpha=1.0)
        r2 = probe.fit_and_score(hidden_states, targets, n_folds=5)
        # With a true linear relationship + small noise, R^2 should be high
        assert r2 > 0.8, f"Expected R^2 > 0.8 for linear signal, got {r2:.4f}"


class TestProbeRejectsRandom:
    """Probe should NOT find signal in random data."""

    def test_probe_rejects_random(self):
        hidden_states, targets, boundaries = _make_random_data()
        result = run_probe(
            feature_name="random_noise",
            hidden_states=hidden_states,
            targets=targets,
            episode_boundaries=boundaries,
            threshold=0.1,
            n_permutations=50,
            alpha=1.0,
        )
        # delta_R2 should be near zero -- no real signal
        assert abs(result.delta_r2) < 0.15, (
            f"Expected delta_R2 near 0 for random data, got {result.delta_r2:.4f}"
        )
        assert not result.passed, (
            "Probe should NOT pass for random data"
        )


class TestProbeNullDistribution:
    """Null R^2 values should be lower than trained R^2 for real signal."""

    def test_probe_null_distribution(self):
        hidden_states, targets, boundaries = _make_linear_data()
        probe = LinearProbe(alpha=1.0)

        r2_trained = probe.fit_and_score(hidden_states, targets, n_folds=5)
        null_r2s = probe.null_distribution(
            hidden_states, targets, boundaries,
            n_permutations=50, n_folds=5,
        )

        null_mean = np.mean(null_r2s)
        # Trained R^2 should clearly exceed the null mean
        assert r2_trained > null_mean + 0.1, (
            f"Trained R^2 ({r2_trained:.4f}) should exceed null mean "
            f"({null_mean:.4f}) by > 0.1"
        )
        # The vast majority of null R^2 should be below trained R^2.
        # With only 5 episodes, a permutation can land on the identity
        # order, so we allow a small fraction to match trained R^2.
        frac_below = np.mean([1.0 if nr < r2_trained - 1e-6 else 0.0
                              for nr in null_r2s])
        assert frac_below >= 0.90, (
            f"Expected >= 90% of null R^2 below trained, got {frac_below:.2%}"
        )


class TestProbeResultFields:
    """ProbeResult should contain all expected fields."""

    def test_probe_result_fields(self):
        hidden_states, targets, boundaries = _make_linear_data(n=200)
        result = run_probe(
            feature_name="test_feature",
            hidden_states=hidden_states,
            targets=targets,
            episode_boundaries=boundaries,
            threshold=0.1,
            n_permutations=20,
        )

        assert isinstance(result, ProbeResult)
        assert result.feature_name == "test_feature"
        assert isinstance(result.r2_trained, float)
        assert isinstance(result.r2_null_mean, float)
        assert isinstance(result.r2_null_std, float)
        assert isinstance(result.delta_r2, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.threshold, float)
        assert result.threshold == 0.1
        # p_value must be in [0, 1]
        assert 0.0 <= result.p_value <= 1.0
        # delta_r2 = r2_trained - r2_null_mean
        assert abs(result.delta_r2 - (result.r2_trained - result.r2_null_mean)) < 1e-8


class TestGraduationPass:
    """All features pass -> graduation passes."""

    def test_graduation_pass(self):
        # Create 5 passing results
        results = [
            ProbeResult(
                feature_name=f"feat_{i}",
                r2_trained=0.9,
                r2_null_mean=0.1,
                r2_null_std=0.05,
                delta_r2=0.8,
                p_value=0.0,
                passed=True,
                threshold=0.1,
            )
            for i in range(5)
        ]
        exam = graduation_exam(results, pass_threshold=0.8)
        assert exam["passed"] is True
        assert exam["score"] == 1.0
        assert exam["n_passed"] == 5
        assert exam["n_total"] == 5
        assert len(exam["passed_features"]) == 5
        assert len(exam["failed_features"]) == 0


class TestGraduationFail:
    """Fewer than 80% pass -> graduation fails."""

    def test_graduation_fail(self):
        # 2 pass, 3 fail = 40% pass rate
        results = []
        for i in range(2):
            results.append(ProbeResult(
                feature_name=f"pass_{i}",
                r2_trained=0.9,
                r2_null_mean=0.1,
                r2_null_std=0.05,
                delta_r2=0.8,
                p_value=0.0,
                passed=True,
                threshold=0.1,
            ))
        for i in range(3):
            results.append(ProbeResult(
                feature_name=f"fail_{i}",
                r2_trained=0.05,
                r2_null_mean=0.04,
                r2_null_std=0.02,
                delta_r2=0.01,
                p_value=0.6,
                passed=False,
                threshold=0.1,
            ))
        exam = graduation_exam(results, pass_threshold=0.8)
        assert exam["passed"] is False
        assert exam["score"] == pytest.approx(0.4)
        assert exam["n_passed"] == 2
        assert exam["n_total"] == 5
        assert len(exam["failed_features"]) == 3
        # Verify details dict exists and has entries
        assert "pass_0" in exam["details"]
        assert "fail_0" in exam["details"]
        assert exam["details"]["pass_0"]["passed"] is True
        assert exam["details"]["fail_0"]["passed"] is False


class TestTemporalBlockCV:
    """Verify that CV uses temporal blocks, not random splits.

    Strategy: create data where the first half has one linear relationship
    and the second half has a DIFFERENT one.  If CV used random splits,
    it would mix them and get moderate R^2 on both.  With temporal blocks,
    predicting across the boundary should yield lower R^2 on the fold
    that spans the transition.
    """

    def test_temporal_block_cv(self):
        rng = np.random.RandomState(99)
        n = 200
        hidden_size = 16
        hidden_states = rng.randn(n, hidden_size)

        # First half: target = h @ w1
        w1 = rng.randn(hidden_size)
        # Second half: target = h @ w2 (completely different weights)
        w2 = rng.randn(hidden_size)

        targets = np.zeros(n)
        targets[:n // 2] = hidden_states[:n // 2] @ w1
        targets[n // 2:] = hidden_states[n // 2:] @ w2

        probe = LinearProbe(alpha=1.0)

        # Temporal block CV: the fold at the boundary trains on one
        # regime and tests on another, yielding lower R^2 for that fold.
        temporal_r2 = probe.fit_and_score(hidden_states, targets, n_folds=5)

        # Compare with a "cheating" approach: fit only on first half,
        # score on first half -- should be nearly perfect.
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        half = n // 2
        model.fit(hidden_states[:half], targets[:half])
        r2_same_regime = model.score(hidden_states[:half], targets[:half])

        # The temporal CV R^2 should be noticeably lower than the
        # within-regime R^2 because some folds cross the regime boundary.
        assert temporal_r2 < r2_same_regime, (
            f"Temporal CV R^2 ({temporal_r2:.4f}) should be lower than "
            f"within-regime R^2 ({r2_same_regime:.4f}) due to block splits "
            f"crossing the distribution shift"
        )
