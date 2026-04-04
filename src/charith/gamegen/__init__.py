"""Procedural game generator for Path 4 meta-learning.

Generates diverse games with randomized mechanics so the Transformer
must learn game rules through in-context learning.
"""

from .mechanics import (
    CardinalMove, IceSliding, MirroredMove,
    ColorCycle, Rotation, ColorChanger, KeyDoor,
)
from .win_conditions import ReachPosition, MatchState, CollectAll
from .grid_builder import GridBuilder
from .generator import GameSpec, ProceduralGame, GameGenerator
from .validator import validate_solvable

__all__ = [
    'CardinalMove', 'IceSliding', 'MirroredMove',
    'ColorCycle', 'Rotation', 'ColorChanger', 'KeyDoor',
    'ReachPosition', 'MatchState', 'CollectAll',
    'GridBuilder',
    'GameSpec', 'ProceduralGame', 'GameGenerator',
    'validate_solvable',
]
