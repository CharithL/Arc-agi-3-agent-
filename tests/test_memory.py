"""Tests for memory modules (working, episodic) and utils."""
import numpy as np
import pytest

from charith.memory.working import WorkingMemory
from charith.memory.episodic import EpisodeStore


class TestWorkingMemory:
    def test_working_memory_capacity(self):
        """Store 10 items in capacity-7 -> size==7, oldest evicted."""
        wm = WorkingMemory(capacity=7)
        for i in range(10):
            wm.store(f"key_{i}", i)
        assert wm.size == 7
        # Oldest 3 (key_0, key_1, key_2) should be evicted
        assert wm.retrieve("key_0") is None
        assert wm.retrieve("key_1") is None
        assert wm.retrieve("key_2") is None
        # Newest 7 should remain
        for i in range(3, 10):
            assert wm.retrieve(f"key_{i}") == i

    def test_working_memory_clear(self):
        """Clear empties all slots."""
        wm = WorkingMemory(capacity=7)
        for i in range(5):
            wm.store(f"key_{i}", i)
        assert wm.size == 5
        wm.clear()
        assert wm.size == 0
        assert wm.retrieve("key_0") is None


class TestEpisodicStore:
    def test_episodic_store_record_and_query(self):
        """Record an episode -> count==1."""
        store = EpisodeStore()
        state = np.array([[1, 2], [3, 4]])
        next_state = np.array([[5, 6], [7, 8]])
        store.record(state, action=0, next_state=next_state, error=0.1, tick=1)
        assert store.count == 1

    def test_episodic_store_level_boundary(self):
        """Mark level boundary -> recorded."""
        store = EpisodeStore()
        state = np.array([[1, 2], [3, 4]])
        next_state = np.array([[5, 6], [7, 8]])
        store.record(state, action=0, next_state=next_state, error=0.1, tick=1)
        store.mark_level_boundary()
        store.record(state, action=1, next_state=next_state, error=0.2, tick=2)
        assert store.count == 2
        assert store._current_level == 1
        assert len(store._level_boundaries) == 1

    def test_episodic_store_max_capacity(self):
        """Store 10 episodes in max-5 -> count==5."""
        store = EpisodeStore(max_episodes=5)
        for i in range(10):
            state = np.array([[i, i + 1], [i + 2, i + 3]])
            next_state = np.array([[i + 4, i + 5], [i + 6, i + 7]])
            store.record(state, action=i % 4, next_state=next_state,
                         error=0.1 * i, tick=i)
        assert store.count == 5
