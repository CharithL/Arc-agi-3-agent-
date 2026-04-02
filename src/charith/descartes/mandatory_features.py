"""Mandatory features that each synthetic reality should impose."""
from dataclasses import dataclass
from typing import List


@dataclass
class MandatoryFeature:
    name: str
    description: str
    threshold: float  # minimum delta_R2 to pass
    method: str = "ridge_r2"  # or "CKA_similarity" or "negative_probe"


MAZE_FEATURES: List[MandatoryFeature] = [
    MandatoryFeature(
        "controllable_relative_row",
        "Normalized row of controllable",
        0.1,
    ),
    MandatoryFeature(
        "controllable_relative_col",
        "Normalized col of controllable",
        0.1,
    ),
    MandatoryFeature(
        "distance_to_goal",
        "Normalized euclidean distance to goal",
        0.1,
    ),
    MandatoryFeature(
        "wall_adjacent_up",
        "Wall directly above controllable?",
        0.05,
    ),
    MandatoryFeature(
        "wall_adjacent_down",
        "Wall directly below?",
        0.05,
    ),
    MandatoryFeature(
        "wall_adjacent_left",
        "Wall directly left?",
        0.05,
    ),
    MandatoryFeature(
        "wall_adjacent_right",
        "Wall directly right?",
        0.05,
    ),
]

ROLE_FEATURES: List[MandatoryFeature] = [
    MandatoryFeature(
        "controllable_role_encoding",
        "Color-invariant controllable repr",
        0.7,
        "CKA_similarity",
    ),
    MandatoryFeature(
        "goal_role_encoding",
        "Color-invariant goal repr",
        0.7,
        "CKA_similarity",
    ),
    MandatoryFeature(
        "color_invariance",
        "Color NOT encoded (negative probe)",
        0.3,
        "negative_probe",
    ),
]

FEATURES_BY_REALITY = {
    "maze_reality": MAZE_FEATURES,
    "role_reality": ROLE_FEATURES,
}
