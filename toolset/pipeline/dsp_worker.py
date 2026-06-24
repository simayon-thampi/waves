"""
pipeline/dsp_worker.py
======================
Background DSP thread that runs all range estimators off the Tkinter main
thread.

Design
------
Two ``queue.Queue`` instances decouple the GUI from computation:

* **input_queue** — ``ChannelResponse`` objects pushed by the GUI immediately
  after building them from live / historical data.  Items are the fully-formed
  ``ChannelResponse`` dataclass; no lock is needed because ``queue.Queue.put``
  and ``queue.Queue.get`` are thread-safe by design.

* **output_queue** — ``(procedure_counter, List[EstimatorResult])`` tuples produced by the worker
  and consumed by the GUI via ``root.after(0, _drain_dsp_queue)``.

Sentinel convention (consistent with ``producer_worker``):
  ``input_queue.put(None)`` signals the worker to stop gracefully.

Why ``queue.Queue`` and not ``threading.Event`` / ``tkinter.StringVar``
-----------------------------------------------------------------------
* ``threading.Event`` only signals *that* a result is ready, not *what* it is.
  You would still need a shared variable for the result itself, requiring an
  explicit lock.
* ``tkinter.StringVar`` can only hold a string; it cannot carry the full
  ``EstimatorResult`` with its numpy array diagnostics, and — critically —
  tracing a StringVar fires the callback on the thread that set it, not on the
  Tk main thread, which is unsafe for Tk widget updates.
* ``queue.Queue`` is designed for exactly this pattern: a producer puts an
  immutable-at-handoff object; the consumer calls ``get_nowait()`` which is
  atomic and needs no lock.  Tk's ``root.after(0, callback)`` guarantees the
  drain callback runs on the main thread, making all widget updates safe.

Drop-in for ``_LAG_SKIP_COUNT``
--------------------------------
The ``_LAG_SKIP_COUNT = 3`` guard exists to prevent back-to-back expensive
estimator calls from stacking up and starving the Tkinter mainloop.  Once
estimators run in this background thread, that cost is gone.  The constant is
kept but reduced from 3 → 1 in ``cs_viewer.py`` (see comment there); it now
only guards against data-source bursts, not computation latency.
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional

from toolset.estimators.base import Estimator, EstimatorResult
from toolset.processing.channel_response import ChannelResponse


class DSPWorker:
    """
    Daemon worker that runs a list of :class:`~toolset.estimators.base.Estimator`
    instances on every incoming :class:`~toolset.processing.channel_response.ChannelResponse`.

    Usage
    -----
    ::

        worker = DSPWorker(
            estimators=[PhaseSlopeEstimator(), IFFTEstimator(), MUSICEstimator()],
        )
        # run.py connects the queues to the viewer:
        viewer.set_dsp_queues(worker.input_queue, worker.output_queue)
        worker.start()   # starts daemon thread

    The worker processes items strictly in order (one CR at a time) and emits
    a single ``(procedure_counter, List[EstimatorResult])`` tuple to
    ``output_queue`` per CR received.

    If the input queue grows faster than the estimators can process (e.g. MUSIC
    on weak hardware), items accumulate in the queue.  The worker always
    processes the *latest* item by draining stale entries before committing to
    one — effectively a 1-deep "latest value" behaviour with bounded memory.
    """

    # Maximum number of CRs to skip when the input queue is backed up.
    # After skipping, the worker processes the most-recently-arrived item.
    _MAX_SKIP = 4

    def __init__(
        self,
        estimators: List[Estimator],
        *,
        input_queue:  Optional[queue.Queue] = None,
        output_queue: Optional[queue.Queue] = None,
    ) -> None:
        """
        Parameters
        ----------
        estimators : list of Estimator
            Ordered list of estimator adapters to run on each ChannelResponse.
            Results are posted as a list in the same order.
        input_queue : queue.Queue, optional
            If not provided, one is created automatically (``maxsize=0``).
        output_queue : queue.Queue, optional
            If not provided, one is created automatically (``maxsize=0``).
        """
        self.estimators: List[Estimator] = list(estimators)
        self.input_queue:  queue.Queue = input_queue  or queue.Queue()
        self.output_queue: queue.Queue = output_queue or queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn and start the daemon worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent
        self._thread = threading.Thread(
            target=self._run,
            name="DSPWorker",
            daemon=True,   # exits automatically when the main thread exits
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop by posting the sentinel ``None``."""
        self.input_queue.put(None)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Main loop.  Blocks on ``input_queue.get()`` and processes each
        ``ChannelResponse``.  Exits when the sentinel ``None`` is received.

        Backpressure handling
        ~~~~~~~~~~~~~~~~~~~~~
        If the queue has accumulated more than ``_MAX_SKIP`` items (because
        MUSIC is slow and the data source is fast), the worker drains surplus
        items in a tight loop and only processes the newest one.  This ensures
        the GUI always shows the most-recent data instead of playing catch-up
        through a stale backlog.
        """
        while True:
            cr: Optional[ChannelResponse] = self.input_queue.get()

            # Sentinel → shutdown
            if cr is None:
                break

            # Drain backlog: keep consuming until the queue is empty or we've
            # skipped _MAX_SKIP items, always keeping the last non-None item.
            skipped = 0
            while skipped < self._MAX_SKIP:
                try:
                    next_item = self.input_queue.get_nowait()
                except queue.Empty:
                    break
                if next_item is None:
                    # Sentinel arrived mid-drain — honour it immediately.
                    self.input_queue.put(None)   # re-post so outer loop sees it
                    break
                cr = next_item
                skipped += 1

            # Run all estimators on the most-recent ChannelResponse.
            results: List[EstimatorResult] = []
            for est in self.estimators:
                try:
                    results.append(est(cr))
                except Exception as exc:  # noqa: BLE001 — never crash the worker
                    import traceback
                    print(f"[DSPWorker] {est.name} raised {type(exc).__name__}: {exc}")
                    traceback.print_exc()

            # Post the result bundle atomically.  queue.Queue.put is thread-safe;
            # the GUI reads with get_nowait() on the Tk main thread.
            # The procedure_counter is bound here (at computation time) so the
            # session logger always writes the counter that was actually processed,
            # regardless of where the GUI's live counter is when the drain fires.
            if results:
                self.output_queue.put((cr.procedure_counter, results))
