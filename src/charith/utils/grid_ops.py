"""Grid manipulation utilities."""
import numpy as np

def grid_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a != b

def grid_entropy(grid: np.ndarray) -> float:
    unique, counts = np.unique(grid, return_counts=True)
    probs = counts / counts.sum()
    return -float(np.sum(probs * np.log2(probs + 1e-10)))
