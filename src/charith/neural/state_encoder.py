"""StateEncoder: StructuredPercept -> fixed-size tensor for GRU input."""

import math

import torch
import numpy as np
from typing import Optional, Set

from charith.perception.core_knowledge import StructuredPercept

MAX_OBJECTS = 20
N_COLORS = 13   # colors 0-12
N_ACTIONS = 8

# Per-object: one-hot color (13) + relative_row (1) + relative_col (1)
#           + log_size (1) + is_controllable (1) = 17
OBJECT_FEATURE_SIZE = N_COLORS + 4

# Global: background one-hot (13) + object_count (1) + unique_colors (1)
#       + h_sym (1) + v_sym (1) + action one-hot (8) = 25
GLOBAL_FEATURE_SIZE = N_COLORS + 4 + N_ACTIONS

D_INPUT = MAX_OBJECTS * OBJECT_FEATURE_SIZE + GLOBAL_FEATURE_SIZE  # 365


def encode(
    percept: StructuredPercept,
    action: int,
    controllable_ids: Optional[Set[int]] = None,
    grid_dims: Optional[tuple] = None,
) -> torch.Tensor:
    """Encode a StructuredPercept + action into a fixed-size tensor.

    Args:
        percept: The structured percept from the perception pipeline.
        action: Integer action index (0 .. N_ACTIONS-1).
        controllable_ids: Set of object_ids considered controllable.
        grid_dims: (rows, cols) of the grid; defaults to percept.grid_dims.

    Returns:
        Tensor of shape [D_INPUT] (365), dtype float32.
    """
    if controllable_ids is None:
        controllable_ids = set()
    if grid_dims is None:
        grid_dims = percept.grid_dims

    rows, cols = grid_dims
    grid_area = rows * cols

    # Sort objects by size (largest first) for deterministic ordering
    sorted_objects = sorted(percept.objects, key=lambda o: o.size, reverse=True)

    # Encode per-object features (up to MAX_OBJECTS)
    object_features = torch.zeros(MAX_OBJECTS * OBJECT_FEATURE_SIZE, dtype=torch.float32)

    for i, obj in enumerate(sorted_objects[:MAX_OBJECTS]):
        offset = i * OBJECT_FEATURE_SIZE

        # One-hot color (13 values)
        color_idx = min(obj.color, N_COLORS - 1)
        object_features[offset + color_idx] = 1.0

        # Relative row: centroid[0] / rows
        object_features[offset + N_COLORS] = obj.centroid[0] / rows if rows > 0 else 0.0

        # Relative col: centroid[1] / cols
        object_features[offset + N_COLORS + 1] = obj.centroid[1] / cols if cols > 0 else 0.0

        # Normalized log size: log(num_cells + 1) / log(grid_area)
        if grid_area > 1:
            object_features[offset + N_COLORS + 2] = math.log(obj.size + 1) / math.log(grid_area)
        else:
            object_features[offset + N_COLORS + 2] = 0.0

        # Is controllable
        object_features[offset + N_COLORS + 3] = 1.0 if obj.object_id in controllable_ids else 0.0

    # Encode global features
    global_features = torch.zeros(GLOBAL_FEATURE_SIZE, dtype=torch.float32)

    # Background color one-hot (13 values)
    bg_idx = min(percept.background_color, N_COLORS - 1)
    global_features[bg_idx] = 1.0

    # Normalized object count
    global_features[N_COLORS] = percept.object_count / MAX_OBJECTS

    # Normalized unique colors
    global_features[N_COLORS + 1] = len(percept.unique_colors) / N_COLORS

    # Symmetry flags
    global_features[N_COLORS + 2] = 1.0 if percept.symmetry.get("h_symmetric", False) else 0.0
    global_features[N_COLORS + 3] = 1.0 if percept.symmetry.get("v_symmetric", False) else 0.0

    # Action one-hot (8 values)
    action_idx = min(action, N_ACTIONS - 1)
    global_features[N_COLORS + 4 + action_idx] = 1.0

    # Concatenate: object features + global features
    return torch.cat([object_features, global_features])
