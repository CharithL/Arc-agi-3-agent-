"""Base class for synthetic reality environments."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np


class SyntheticReality(ABC):
    """Base class for all synthetic realities.

    Interface matches ARC-AGI-3 SDK:
    - reset() -> observation (numpy grid, int values 0-12)
    - step(action) -> (observation, reward, done, info)
    - get_ground_truth() -> dict of probing target values
    """

    def __init__(self, grid_size: int = 16, n_actions: int = 4, max_steps: int = 200):
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.max_steps = max_steps
        self._grid = np.zeros((grid_size, grid_size), dtype=int)
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation grid."""
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action, return (observation, reward, done, info)."""
        pass

    @abstractmethod
    def get_ground_truth(self) -> Dict[str, float]:
        """Return ground truth values for DESCARTES probing.

        Keys are feature names (e.g., 'controllable_relative_row'),
        values are floats (normalized 0-1 for positions, 0/1 for booleans).
        """
        pass

    def _render_grid(self) -> np.ndarray:
        """Return a copy of the current grid."""
        return self._grid.copy()
