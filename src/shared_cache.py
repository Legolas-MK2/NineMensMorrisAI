"""
Lock-free shared memory hash table for minimax move caching.

Maps Zobrist position hash (int64) -> best action (int16).

Design:
- Open-addressing with linear probing, power-of-2 table size
- Two parallel numpy arrays in shared memory: keys[] and actions[]
- Lock-free: concurrent reads/writes are safe because wrong cached actions
  are validated against legal moves before use (benign races)
- Writing action before key means a partial write looks like a cache miss
"""

import numpy as np
from multiprocessing import shared_memory
from typing import Optional


class SharedMoveCache:
    """
    Fixed-size open-addressing hash table in shared memory.

    Entry layout (10 bytes each):
      - keys[i]:    int64  (Zobrist hash, 0 = empty sentinel)
      - actions[i]: int16  (best action index)

    Total memory = size_gb GB split as:
      first  n*8 bytes → keys array
      next   n*2 bytes → actions array
    """

    _SHM_NAME = "nmm_move_cache"

    def __init__(self, size_gb: float, create: bool = True):
        bytes_per_entry = 10  # int64 key + int16 action
        total_bytes = int(size_gb * (1024 ** 3))
        n = total_bytes // bytes_per_entry

        # Round down to largest power-of-2 that fits
        self._n = 1 << max(1, n.bit_length() - 1)
        self._mask = self._n - 1
        self._size_bytes = self._n * bytes_per_entry

        if create:
            self._shm = self._create_shm(self._size_bytes)
        else:
            self._shm = shared_memory.SharedMemory(name=self._SHM_NAME, create=False)

        # Two parallel arrays backed by the same shared memory buffer
        self._keys = np.ndarray(
            (self._n,), dtype=np.int64, buffer=self._shm.buf, offset=0
        )
        self._acts = np.ndarray(
            (self._n,), dtype=np.int16, buffer=self._shm.buf, offset=self._n * 8
        )

        # Note: Linux zero-fills new shm pages on demand (POSIX guarantee),
        # so explicit initialization is unnecessary and forces immediate
        # physical page allocation for the entire segment (can cause SIGBUS
        # if /dev/shm is undersized). Keys default to 0 = empty sentinel.

        self._is_creator = create
        gb_actual = self._size_bytes / (1024 ** 3)
        print(f"  SharedMoveCache: {gb_actual:.1f} GB → {self._n:,} slots")

    @classmethod
    def _create_shm(cls, size_bytes: int):
        try:
            return shared_memory.SharedMemory(
                create=True, size=size_bytes, name=cls._SHM_NAME
            )
        except FileExistsError:
            # Stale shared memory from a previous crashed run — clean it up
            try:
                old = shared_memory.SharedMemory(name=cls._SHM_NAME, create=False)
                old.close()
                old.unlink()
            except Exception:
                pass
            return shared_memory.SharedMemory(
                create=True, size=size_bytes, name=cls._SHM_NAME
            )

    @classmethod
    def attach(cls, size_gb: float) -> "SharedMoveCache":
        """Attach to the existing shared memory (call once per worker process)."""
        return cls(size_gb=size_gb, create=False)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: int) -> Optional[int]:
        """Return cached action for key, or None on miss. key=0 is never stored."""
        key = key & 0x7FFFFFFFFFFFFFFF  # Fit into signed int64 (Zobrist uses unsigned 64-bit)
        if key == 0:
            return None
        idx = key & self._mask
        for _ in range(16):
            k = int(self._keys[idx])
            if k == key:
                return int(self._acts[idx])
            if k == 0:
                return None
            idx = (idx + 1) & self._mask
        return None

    def __setitem__(self, key: int, action: int):
        """Store action for key. Silently skips if probe limit exceeded."""
        key = key & 0x7FFFFFFFFFFFFFFF  # Fit into signed int64 (Zobrist uses unsigned 64-bit)
        if key == 0:
            return
        idx = key & self._mask
        for _ in range(16):
            k = int(self._keys[idx])
            if k == 0 or k == key:
                # Write action before key so a torn write looks like a miss
                self._acts[idx] = np.int16(action)
                self._keys[idx] = np.int64(key)
                return
            idx = (idx + 1) & self._mask
        # Probe limit hit — table is locally dense; skip this entry

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Detach from shared memory (call in every process on exit)."""
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        """Destroy the shared memory segment (call once, from creator only)."""
        if self._is_creator:
            try:
                self._shm.unlink()
            except Exception:
                pass
