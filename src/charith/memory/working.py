"""Capacity-limited working memory (7 slots, per Miller's Law)."""
from collections import OrderedDict
from typing import Any, Optional

class WorkingMemory:
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._slots: OrderedDict = OrderedDict()

    def store(self, key: str, value: Any):
        if key in self._slots:
            self._slots.move_to_end(key)
            self._slots[key] = value
        else:
            if len(self._slots) >= self.capacity:
                self._slots.popitem(last=False)  # Remove oldest
            self._slots[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        return self._slots.get(key)

    @property
    def size(self) -> int:
        return len(self._slots)

    def clear(self):
        self._slots.clear()
