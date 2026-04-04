"""Percept Tokenizer: StructuredPercept -> D_model vector for Transformer.

CRITICAL: Must include CHANGE features — what changed since last tick.
The change features are the most important signal for in-context learning.
"""
import torch
import torch.nn as nn
import numpy as np
from charith.perception.core_knowledge import StructuredPercept

D_MODEL = 256
MAX_OBJECTS = 20


class PerceptTokenizer(nn.Module):
    """Converts StructuredPercept into a D_model dimensional token.

    Encodes:
    - Per-object features (color, position, size, controllable) for up to 20 objects
    - Global features (background, counts, symmetry)
    - CHANGE features (cells changed, objects appeared/disappeared/moved)

    The raw features (~113 dims) are projected to D_model (256) by a learned MLP.
    """

    def __init__(self, d_model=256):
        super().__init__()
        # Per-object: color_norm(1) + rel_row(1) + rel_col(1) + log_size(1) + is_ctrl(1) = 5
        # 20 objects * 5 = 100
        # Global: bg_color(1) + obj_count(1) + n_colors(1) + h_sym(1) + v_sym(1) = 5
        # Change: cells_changed(1) + n_appeared(1) + n_disappeared(1) + n_moved(1)
        #         + ctrl_moved(1) + ctrl_dx(1) + ctrl_dy(1) + any_color_change(1) = 8
        # Total raw: 100 + 5 + 8 = 113
        raw_size = MAX_OBJECTS * 5 + 5 + 8

        self.projection = nn.Sequential(
            nn.Linear(raw_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self._prev_percept = None
        self._raw_size = raw_size

    def forward(self, percept: StructuredPercept,
                grid_dims: tuple,
                controllable_ids: set = None,
                prev_percept: StructuredPercept = None) -> torch.Tensor:
        """Convert percept to token vector.

        Args:
            percept: current StructuredPercept
            grid_dims: (rows, cols) for normalization
            controllable_ids: set of controllable object IDs
            prev_percept: previous percept for computing changes
        Returns:
            token: [D_model] tensor
        """
        if controllable_ids is None:
            controllable_ids = set()

        features = []

        # Per-object features (sorted by size, zero-pad)
        objects = sorted(percept.objects, key=lambda o: o.size, reverse=True)[:MAX_OBJECTS]
        for obj in objects:
            features.extend([
                obj.color / 12.0,
                obj.centroid[0] / max(grid_dims[0], 1),
                obj.centroid[1] / max(grid_dims[1], 1),
                np.log1p(obj.size) / 10.0,
                1.0 if obj.object_id in controllable_ids else 0.0,
            ])
        # Zero-pad remaining object slots
        while len(features) < MAX_OBJECTS * 5:
            features.append(0.0)

        # Global features
        features.extend([
            percept.background_color / 12.0,
            percept.object_count / 20.0,
            len(percept.unique_colors) / 13.0 if isinstance(percept.unique_colors, set) else percept.unique_colors / 13.0,
            float(percept.symmetry.get('h_symmetric', False)),
            float(percept.symmetry.get('v_symmetric', False)),
        ])

        # Change features (THE MOST IMPORTANT for in-context learning)
        if prev_percept is not None:
            diff = (percept.raw_grid != prev_percept.raw_grid)
            cells_changed = float(np.sum(diff)) / max(percept.raw_grid.size, 1)

            prev_ids = {o.object_id for o in prev_percept.objects}
            curr_ids = {o.object_id for o in percept.objects}
            n_appeared = len(curr_ids - prev_ids) / 20.0
            n_disappeared = len(prev_ids - curr_ids) / 20.0

            # Count moved objects (centroid changed)
            n_moved = 0
            ctrl_moved = 0.0
            ctrl_dx = 0.0
            ctrl_dy = 0.0
            prev_map = {o.object_id: o for o in prev_percept.objects}
            for obj in percept.objects:
                if obj.object_id in prev_map:
                    p = prev_map[obj.object_id]
                    if abs(obj.centroid[0] - p.centroid[0]) > 0.5 or abs(obj.centroid[1] - p.centroid[1]) > 0.5:
                        n_moved += 1
                        if obj.object_id in controllable_ids:
                            ctrl_moved = 1.0
                            ctrl_dx = (obj.centroid[0] - p.centroid[0]) / max(grid_dims[0], 1)
                            ctrl_dy = (obj.centroid[1] - p.centroid[1]) / max(grid_dims[1], 1)

            # Color changes
            any_color_change = 0.0
            for obj in percept.objects:
                if obj.object_id in prev_map and obj.color != prev_map[obj.object_id].color:
                    any_color_change = 1.0
                    break

            features.extend([
                cells_changed, n_appeared, n_disappeared,
                n_moved / 20.0, ctrl_moved, ctrl_dx, ctrl_dy, any_color_change
            ])
        else:
            features.extend([0.0] * 8)  # No change info for first tick

        x = torch.tensor(features, dtype=torch.float32)
        # Move to same device as model weights
        x = x.to(next(self.parameters()).device)
        return self.projection(x)

    def reset(self):
        """Reset for new episode."""
        self._prev_percept = None
