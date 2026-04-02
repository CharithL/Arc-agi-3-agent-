"""DESCARTES linear probes on GRU hidden states.

Methodology:
1. Collect (h_t, feature_t) pairs from gameplay
2. Train Ridge regression: h_t -> feature_t
3. Compute cross-validated R^2 (temporal block CV, NOT random split)
4. Compute null distribution via episode-shuffled permutation
5. Decision: delta_R2 > threshold AND p < 0.05 -> feature IS encoded
"""
import numpy as np
from sklearn.linear_model import Ridge
from dataclasses import dataclass
from typing import List


@dataclass
class ProbeResult:
    """Result of one linear probe."""
    feature_name: str
    r2_trained: float          # Cross-validated R^2
    r2_null_mean: float        # Mean R^2 from null distribution
    r2_null_std: float         # Std of null R^2
    delta_r2: float            # r2_trained - r2_null_mean
    p_value: float             # Fraction of null R^2 >= r2_trained
    passed: bool               # delta_r2 > threshold AND p < 0.05
    threshold: float           # The threshold used


class LinearProbe:
    """Ridge regression probe for one feature."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit_and_score(self, hidden_states: np.ndarray,
                      targets: np.ndarray,
                      n_folds: int = 5) -> float:
        """Train probe with temporal block CV and return mean R^2.

        Uses temporal blocks (NOT random splits) because hidden states
        are temporally autocorrelated. Random splits would leak.

        Args:
            hidden_states: [N, hidden_size] array
            targets: [N] array of ground truth values
            n_folds: number of temporal blocks
        Returns:
            Mean cross-validated R^2
        """
        n = len(hidden_states)
        if n < n_folds * 2:
            return 0.0

        # Temporal block split (sequential chunks)
        block_size = n // n_folds
        r2_scores = []

        for fold in range(n_folds):
            test_start = fold * block_size
            test_end = test_start + block_size if fold < n_folds - 1 else n

            test_idx = list(range(test_start, test_end))
            train_idx = [i for i in range(n) if i not in test_idx]

            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            X_train = hidden_states[train_idx]
            y_train = targets[train_idx]
            X_test = hidden_states[test_idx]
            y_test = targets[test_idx]

            model = Ridge(alpha=self.alpha)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            r2_scores.append(r2)

        return float(np.mean(r2_scores)) if r2_scores else 0.0

    def null_distribution(self, hidden_states: np.ndarray,
                          targets: np.ndarray,
                          episode_boundaries: List[int],
                          n_permutations: int = 100,
                          n_folds: int = 5) -> List[float]:
        """Build null distribution by shuffling targets across episodes.

        CRITICAL: shuffle across episodes, not within.
        This controls for temporal autocorrelation within episodes.

        Args:
            hidden_states: [N, hidden_size]
            targets: [N]
            episode_boundaries: list of indices where episodes start
            n_permutations: number of shuffles
            n_folds: CV folds
        Returns:
            List of R^2 values under null hypothesis
        """
        null_r2s = []

        # Build episode chunks
        boundaries = sorted(episode_boundaries) + [len(targets)]
        episodes = []
        for i in range(len(boundaries) - 1):
            episodes.append(targets[boundaries[i]:boundaries[i + 1]])

        rng = np.random.RandomState(42)

        for _ in range(n_permutations):
            # Shuffle episode order (keep within-episode structure)
            shuffled_episodes = list(episodes)
            rng.shuffle(shuffled_episodes)
            shuffled_targets = np.concatenate(shuffled_episodes)

            # Truncate/pad to match length
            if len(shuffled_targets) > len(targets):
                shuffled_targets = shuffled_targets[:len(targets)]
            elif len(shuffled_targets) < len(targets):
                pad = np.zeros(len(targets) - len(shuffled_targets))
                shuffled_targets = np.concatenate([shuffled_targets, pad])

            r2 = self.fit_and_score(hidden_states, shuffled_targets, n_folds)
            null_r2s.append(r2)

        return null_r2s


def run_probe(feature_name: str,
              hidden_states: np.ndarray,
              targets: np.ndarray,
              episode_boundaries: List[int],
              threshold: float = 0.1,
              n_permutations: int = 100,
              alpha: float = 1.0) -> ProbeResult:
    """Run a complete DESCARTES probe for one feature.

    Args:
        feature_name: name of the feature being probed
        hidden_states: [N, hidden_size] GRU hidden states
        targets: [N] ground truth feature values
        episode_boundaries: start indices of each episode
        threshold: minimum delta_R2 to pass
        n_permutations: number of null permutations
        alpha: Ridge regression alpha
    Returns:
        ProbeResult with decision
    """
    probe = LinearProbe(alpha=alpha)

    # Trained R^2
    r2_trained = probe.fit_and_score(hidden_states, targets)

    # Null distribution
    null_r2s = probe.null_distribution(
        hidden_states, targets, episode_boundaries, n_permutations
    )

    r2_null_mean = float(np.mean(null_r2s))
    r2_null_std = float(np.std(null_r2s))
    delta_r2 = r2_trained - r2_null_mean
    p_value = float(np.mean([1.0 if nr >= r2_trained else 0.0
                             for nr in null_r2s]))

    passed = delta_r2 > threshold and p_value < 0.05

    return ProbeResult(
        feature_name=feature_name,
        r2_trained=r2_trained,
        r2_null_mean=r2_null_mean,
        r2_null_std=r2_null_std,
        delta_r2=delta_r2,
        p_value=p_value,
        passed=passed,
        threshold=threshold,
    )
