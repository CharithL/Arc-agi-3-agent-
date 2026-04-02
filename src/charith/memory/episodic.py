"""Hash-indexed episode store."""
from dataclasses import dataclass
from typing import List
import numpy as np
import hashlib

@dataclass
class Episode:
    state_hash: int
    action: int
    next_state_hash: int
    error: float
    tick: int
    level: int = 0

class EpisodeStore:
    def __init__(self, max_episodes: int = 5000):
        self._max = max_episodes
        self._episodes: List[Episode] = []
        self._level_boundaries: List[int] = []
        self._current_level: int = 0

    def record(self, state: np.ndarray, action: int,
               next_state: np.ndarray, error: float, tick: int):
        s_hash = int(hashlib.md5(state.tobytes()).hexdigest()[:16], 16)
        ns_hash = int(hashlib.md5(next_state.tobytes()).hexdigest()[:16], 16)
        ep = Episode(s_hash, action, ns_hash, error, tick, self._current_level)
        self._episodes.append(ep)
        if len(self._episodes) > self._max:
            self._episodes = self._episodes[-self._max:]

    def mark_level_boundary(self):
        self._level_boundaries.append(len(self._episodes))
        self._current_level += 1

    @property
    def count(self) -> int:
        return len(self._episodes)

    def hard_reset(self):
        self._episodes.clear()
        self._level_boundaries.clear()
        self._current_level = 0
