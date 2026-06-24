"""
pipeline/session_logger.py
==========================
Manages a per-session log folder and writes distance estimates to CSV.

Folder layout
-------------
::

    log/
    └── 20260601_121003/
        ├── initiator.txt      ← raw UART bytes (written by UartDataSource)
        ├── reflector.txt      ← raw UART bytes (written by UartDataSource)
        ├── session.log        ← console stdout + stderr  (optional)
        └── distances.csv      ← one row per DSP result bundle

CSV columns
-----------
timestamp_s, procedure_counter, phase_slope_m, ifft_m, music_m

* ``timestamp_s`` — seconds since the session started (float, 3 d.p.)
* Values are ``nan`` when an estimator did not produce a valid estimate.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

from toolset.estimators.base import EstimatorResult


class SessionLogger:
    """
    Creates the session folder and manages all log files for one run.

    Parameters
    ----------
    log_root : str
        Parent directory for all sessions (default ``"log"``).
    enable_session_log : bool
        When ``True``, tee stdout + stderr to ``session.log``.
    timestamp : str, optional
        ``YYYYMMDD_HHMMSS`` string; generated from ``datetime.now()`` if omitted.
    """

    # Ordered column headers for distances.csv
    _CSV_HEADER = [
        "timestamp_s",
        "procedure_counter",
        "phase_slope_m",
        "ifft_m",
        "music_m",
        "wls_m",
        "wls_ema_m",
    ]

    def __init__(
        self,
        log_root: str = "log",
        *,
        enable_session_log: bool = False,
        timestamp: Optional[str] = None,
    ) -> None:
        self.timestamp: str = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir: str = os.path.join(log_root, self.timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        self._start_time: float = time.monotonic()

        # --- distances.csv ------------------------------------------------
        csv_path = os.path.join(self.session_dir, "distances.csv")
        self._csv_fh = open(csv_path, "w", newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_writer.writerow(self._CSV_HEADER)
        print(f"Distance logging: {csv_path}")

        # --- session.log (tee) --------------------------------------------
        self._session_log_fh = None
        if enable_session_log:
            log_path = os.path.join(self.session_dir, "session.log")
            self._session_log_fh = open(log_path, "w", buffering=1)
            sys.stdout = _TeeWriter(sys.stdout, self._session_log_fh)
            sys.stderr = _TeeWriter(sys.stderr, self._session_log_fh)
            print(f"Session logging: {log_path}")

    # ------------------------------------------------------------------
    # Paths for UART log files (passed to UartDataSource.enable_logging)
    # ------------------------------------------------------------------

    @property
    def initiator_log_path(self) -> str:
        return os.path.join(self.session_dir, "initiator.txt")

    @property
    def reflector_log_path(self) -> str:
        return os.path.join(self.session_dir, "reflector.txt")

    # ------------------------------------------------------------------
    # CSV writer — called from the Tk thread via _drain_dsp_queue hook
    # ------------------------------------------------------------------

    def log_dsp_results(
        self,
        procedure_counter: int,
        results: List[EstimatorResult],
    ) -> None:
        """
        Append one row to ``distances.csv``.

        Parameters
        ----------
        procedure_counter : int
            The BLE CS procedure counter for this measurement.
        results : list of EstimatorResult
            The bundle emitted by :class:`~toolset.pipeline.dsp_worker.DSPWorker`.
            Estimators are matched by ``estimator_name``; unknown names are
            silently ignored.
        """
        # Build name → distance_m map
        dist: dict[str, float] = {}
        for r in results:
            dist[r.estimator_name] = r.distance_m

        elapsed = time.monotonic() - self._start_time

        def _fmt(val: float) -> str:
            return "" if math.isnan(val) else f"{val:.4f}"

        self._csv_writer.writerow([
            f"{elapsed:.3f}",
            procedure_counter,
            _fmt(dist.get("Phase Slope", math.nan)),
            _fmt(dist.get("IFFT",        math.nan)),
            _fmt(dist.get("MUSIC",       math.nan)),
            _fmt(dist.get("Weighted LS", math.nan)),
            _fmt(dist.get("Weighted LS (EMA)", math.nan)),
        ])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles; restore sys streams."""
        if self._csv_fh and not self._csv_fh.closed:
            self._csv_fh.flush()
            self._csv_fh.close()

        if self._session_log_fh:
            sys.stdout = sys.stdout._original
            sys.stderr = sys.stderr._original
            self._session_log_fh.close()
            self._session_log_fh = None


# ---------------------------------------------------------------------------
# Helper: tee writer
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Writes to both the original stream and a log file simultaneously."""

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        self._log.write(data)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def fileno(self):
        return self._original.fileno()
