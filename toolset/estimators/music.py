"""
estimators/music.py
===================
Thin adapter: wraps the existing MUSIC pseudo-spectrum algorithm.

Algorithm ownership
-------------------
All maths live in ``toolset.processing.cs_music``.  This module only
bridges between ``ChannelResponse`` and the legacy dict-based API.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from toolset.estimators.base import Estimator, EstimatorResult
from toolset.processing.cs_music import compute_music_spectrum, calculate_distance_from_music

if TYPE_CHECKING:
    from toolset.processing.channel_response import ChannelResponse


class MUSICEstimator(Estimator):
    """
    Range estimator using the MUSIC pseudo-spectrum algorithm.

    Converts ``ChannelResponse`` into the ``Dict[channel, float]`` format
    expected by :func:`~toolset.processing.cs_music.compute_music_spectrum`,
    then delegates to :func:`~toolset.processing.cs_music.calculate_distance_from_music`.
    """

    name = "MUSIC"

    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        """
        Estimate distance from the MUSIC pseudo-spectrum peak.

        Parameters
        ----------
        cr : ChannelResponse
            Must have ``n_channels >= 4`` (MUSIC minimum).  Both ``phase_rad``
            and ``amplitude_db`` are used.

        Returns
        -------
        EstimatorResult
            ``distance_m`` is ``float('nan')`` when MUSIC cannot run (fewer
            than 4 channels).  ``diagnostics`` contains ``"delays_ns"`` and
            ``"pseudo_spectrum"`` arrays (or ``None`` on failure).
        """
        # Build the legacy dicts from ChannelResponse arrays.
        phase_data: dict[int, float] = {
            int(ch): float(ph)
            for ch, ph in zip(cr.channels, cr.phase_rad)
        }
        amplitude_data: dict[int, float] = {
            int(ch): float(amp)
            for ch, amp in zip(cr.channels, cr.amplitude_db)
        }

        delays_ns, pseudo_spectrum = compute_music_spectrum(phase_data, amplitude_data)

        if delays_ns is not None:
            distance = calculate_distance_from_music(delays_ns, pseudo_spectrum)
        else:
            distance = math.nan

        return EstimatorResult(
            distance_m=distance,
            confidence=0.0,
            diagnostics={"delays_ns": delays_ns, "pseudo_spectrum": pseudo_spectrum},
        )
