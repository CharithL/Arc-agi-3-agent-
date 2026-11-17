"""Episode recording & replay (guide ground rule 6 / §2.6).

Records one real episode per game (grids, actions, available actions, score per
tick) so tests and the S10 ablation dry-run can run fully offline. Files are
gzipped when large.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

_GZIP_THRESHOLD = 5 * 1024 * 1024  # 5 MB


@dataclass
class EpisodeRecorder:
    game_id: str
    path: str
    ticks: List[dict] = field(default_factory=list)

    def add(self, tick: int, grid, action, available_actions, score: int) -> None:
        self.ticks.append({
            "tick": tick,
            "grid": np.asarray(grid, dtype=int).tolist(),
            "action_taken": int(action),
            "available_actions": [int(a) for a in available_actions],
            "score": int(score),
        })

    def to_dict(self) -> dict:
        return {"game_id": self.game_id, "ticks": self.ticks}

    def save(self) -> str:
        """Write the episode to ``path`` (gzip if the JSON exceeds 5 MB)."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        blob = json.dumps(self.to_dict())
        if len(blob.encode("utf-8")) > _GZIP_THRESHOLD:
            gz_path = self.path if self.path.endswith(".gz") else self.path + ".gz"
            with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                fh.write(blob)
            return gz_path
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(blob)
        return self.path


def load_episode(path: str) -> dict:
    """Load an episode fixture. Grids are returned as numpy arrays."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    for t in data["ticks"]:
        t["grid"] = np.array(t["grid"], dtype=int)
    return data
