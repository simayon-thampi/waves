"""
estimators/ema.py
=================
Generic Exponential Moving Average (EMA) wrapper for any range estimator.

Wraps an existing :class:`~toolset.estimators.base.Estimator` and applies
a single-pole IIR low-pass filter to its ``distance_m`` output:

    ema_t = α · raw_t + (1 − α) · ema_{t-1}

where ``α ∈ (0, 1]`` is the smoothing factor.  Smaller α = more smoothing
(slower step response); larger α = faster tracking (less smoothing).

Typical choices
---------------
* α = 0.10 → heavy smoothing, ~9 samples time constant
* α = 0.20 → moderate smoothing, ~4 samples time constant
* α = 0.50 → light smoothing, ~1 sample time constant

The filter is **reset** when the inner estimator returns ``nan`` (e.g. RANSAC
failure), so the EMA picks up cleanly on the next valid sample rather than
bleeding stale state across a gap.

The ``estimator_name`` of the wrapped result is preserved with an
``" (EMA)"`` suffix so the session logger and GUI identify it separately.

Thread-safety
-------------
The EMA state (``_ema``) is mutated only inside :meth:`estimate`, which is
called exclusively from the ``DSPWorker`` daemon thread — no lock needed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from toolset.estimators.base import Estimator, EstimatorResult

if TYPE_CHECKING:
    from toolset.processing.channel_response import ChannelResponse


class EMAEstimator(Estimator):
    """
    EMA smoothing wrapper around any :class:`Estimator`.

    Parameters
    ----------
    inner : Estimator
        The underlying estimator whose ``distance_m`` output is smoothed.
    alpha : float
        Smoothing factor in ``(0, 1]``.  ``alpha=1.0`` is a pass-through
        (no smoothing).  Default ``0.2``.
    name_suffix : str
        Appended to the inner estimator's name.  Default ``" (EMA)"``.
    """

    def __init__(
        self,
        inner: Estimator,
        alpha: float = 0.2,
        name_suffix: str = " (EMA)",
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        self.inner = inner
        self.alpha = alpha
        self.name = inner.name + name_suffix

        # EMA state — None means "not yet initialised" (seed on first valid sample)
        self._ema: Optional[float] = None

    # ------------------------------------------------------------------
    # Estimator protocol
    # ------------------------------------------------------------------

    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        """
        Run the inner estimator then apply EMA to its ``distance_m``.

        The returned :class:`EstimatorResult` inherits all diagnostics from
        the inner result and adds:

        * ``"raw_distance_m"`` — the unsmoothed inner estimate
        * ``"ema_alpha"`` — the configured smoothing factor
        """
        inner_result = self.inner.estimate(cr)
        raw = inner_result.distance_m

        if math.isnan(raw):
            # Reset filter on invalid sample so stale state doesn't persist.
            self._ema = None
            smoothed = math.nan
        elif self._ema is None:
            # Seed: first valid sample initialises the filter with no lag.
            self._ema = raw
            smoothed = raw
        else:
            self._ema = self.alpha * raw + (1.0 - self.alpha) * self._ema
            smoothed = self._ema

        diagnostics = dict(inner_result.diagnostics)
        diagnostics["raw_distance_m"] = raw
        diagnostics["ema_alpha"] = self.alpha

        return EstimatorResult(
            distance_m=smoothed,
            confidence=inner_result.confidence,
            diagnostics=diagnostics,
        )

    def reset(self) -> None:
        """Manually reset the EMA state (e.g. when starting a new session)."""
        self._ema = None
