"""Fast state hashing utilities."""
import numpy as np
import hashlib

def state_hash(grid: np.ndarray) -> int:
    return int(hashlib.md5(grid.tobytes()).hexdigest()[:16], 16)
