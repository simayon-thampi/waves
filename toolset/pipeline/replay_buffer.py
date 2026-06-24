import collections
import threading
from typing import Optional, List, Tuple
from toolset.processing.channel_response import ChannelResponse
from toolset.cs_utils.cs_subevent import SubeventResults

class ReplayBuffer:
    """
    A thread-safe ring buffer for storing raw, unprocessed ChannelResponse sweeps
    and their optional associated SubeventResults.
    """
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._buffer = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, cr: ChannelResponse, subevent: Optional[SubeventResults] = None):
        """Append a raw ChannelResponse and its optional subevent metadata."""
        with self._lock:
            self._buffer.append((cr, subevent))

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def get_all(self) -> List[Tuple[ChannelResponse, Optional[SubeventResults]]]:
        """Return a copy of the current buffer items as a list of tuples."""
        with self._lock:
            return list(self._buffer)

    def size(self) -> int:
        """Return the current number of items in the buffer."""
        with self._lock:
            return len(self._buffer)
