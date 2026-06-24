"""
estimators/base.py
==================
Abstract base class for all range estimators and the shared result type.

Design rules
------------
* Zero GUI imports — safe to instantiate in any context (pytest, worker thread).
* Timing is always measured here in ``__call__``, never inside concrete subclasses.
* Confidence is a 0-1 float placeholder; subclasses may override it in
  ``estimate()`` when meaningful metrics are available.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolset.processing.channel_response import ChannelResponse


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

import math

@dataclass
class EstimatorResult:
    """
    Output of a single estimator invocation.

    Attributes
    ----------
    distance_m : float
        Estimated one-way distance in metres.  ``float('nan')`` when the
        estimator cannot produce a valid estimate.
    confidence : float
        Placeholder quality metric in [0, 1].  Subclasses may populate this
        with SNR-derived or eigenvalue-ratio-derived values in the future.
    diagnostics : dict
        Estimator-specific intermediate products (spectra, slopes, …).
        Keys and value types are defined by each concrete subclass.
    estimator_name : str
        Human-readable identifier (e.g. ``"Phase Slope"``).
    latency_ms : float
        Wall-clock time taken by the ``estimate()`` call, in milliseconds.
        Measured externally by ``Estimator.__call__``; not set by the
        concrete implementation.
    """

    distance_m:     float
    confidence:     float
    diagnostics:    dict  = field(default_factory=dict)
    estimator_name: str   = ""
    latency_ms:     float = 0.0

    def __post_init__(self):
        # Dynamically compute confidence score if residual_rms is provided
        if "residual_rms" in self.diagnostics:
            rms = self.diagnostics["residual_rms"]
            if not math.isnan(rms):
                self.confidence = float(1.0 / (1.0 + rms))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Estimator(ABC):
    """
    Abstract range estimator.

    Subclasses implement :meth:`estimate`.  Callers should use
    ``__call__`` so that wall-clock timing is recorded automatically.

    Example
    -------
    >>> est = PhaseSlopeEstimator()
    >>> result = est(cr)          # EstimatorResult with latency_ms filled
    >>> print(result.distance_m)
    """

    # Subclasses should set this to a short human-readable string.
    name: str = "Unnamed Estimator"

    @abstractmethod
    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        """
        Run the estimation algorithm on *cr* and return a result.

        The ``latency_ms`` field of the returned :class:`EstimatorResult`
        is **not** expected to be populated here — it will be set by
        :meth:`__call__`.

        Parameters
        ----------
        cr : ChannelResponse
            Frequency-domain channel snapshot to estimate distance from.

        Returns
        -------
        EstimatorResult
        """

    # ------------------------------------------------------------------
    # Timing wrapper — do NOT override
    # ------------------------------------------------------------------

    def __call__(self, cr: "ChannelResponse") -> EstimatorResult:
        """
        Invoke :meth:`estimate` and stamp ``latency_ms`` on the result.

        Parameters
        ----------
        cr : ChannelResponse

        Returns
        -------
        EstimatorResult
            Same object returned by :meth:`estimate`, with ``latency_ms``
            set to the wall-clock duration of the call.
        """
        t0 = time.perf_counter()
        result = self.estimate(cr)
        result.latency_ms = (time.perf_counter() - t0) * 1e3
        result.estimator_name = self.name
        return result
