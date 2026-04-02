"""Structured logging for agent analysis."""
import time
from typing import Any, Dict, List

class AgentLogger:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def log(self, event_type: str, data: Any, tick: int = 0):
        self._events.append({
            'tick': tick, 'event': event_type,
            'data': data, 'time': time.time(),
        })

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def clear(self):
        self._events.clear()
