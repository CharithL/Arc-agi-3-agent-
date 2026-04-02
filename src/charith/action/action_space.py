"""ARC-AGI-3 action mapping -- adapt when SDK is available."""
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    USE = 4
    SECONDARY = 5
    WAIT = 6
    CONFIRM = 7


N_ACTIONS = len(Action)
