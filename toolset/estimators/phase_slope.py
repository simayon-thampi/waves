"""
estimators/phase_slope.py
==========================
Thin adapter: wraps the existing Phase Slope algorithm.

Algorithm ownership
-------------------
All maths live in ``toolset.processing.cs_phase_slope``.  This module only
bridges between ``ChannelResponse`` and the legacy dict-based API, then
delegates to :func:`calculate_distance_from_phase_slope`.

``ChannelResponse.phase_rad`` already stores the unwrapped, /2-referenced
one-way channel phase (identical to what ``calculate_phase_slope_data()``
produces from raw subevents), so we build the equivalent
``Dict[channel_index, phase_rad]`` directly from the dataclass fields.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from toolset.estimators.base import Estimator, EstimatorResult
from toolset.processing.cs_phase_slope import calculate_distance_from_phase_slope
from toolset.preprocess.config import PreprocessorConfig

if TYPE_CHECKING:
    from toolset.processing.channel_response import ChannelResponse


class PhaseSlopeEstimator(Estimator):
    """
    Range estimator using the Phase Slope method on RANSAC inlier tones only.
    """

    name = "Phase Slope"

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config if config is not None else PreprocessorConfig()

    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        from toolset.preprocess.pipeline import run_ransac
        
        # Run RANSAC first to isolate robust inliers
        best_slope, best_intercept, inlier_mask = run_ransac(
            cr.channels,
            cr.phase_rad,
            n_iterations=self.config.ransac_n_iterations,
            threshold_rad=self.config.ransac_inlier_threshold_rad,
            min_sample_size=self.config.ransac_min_sample_size
        )
        
        # Filter ChannelResponse to inlier subset
        cr_inliers = cr[inlier_mask]
        
        if cr_inliers.n_channels < 2:
            return EstimatorResult(
                distance_m=math.nan,
                confidence=0.0,
                diagnostics={"inlier_mask": inlier_mask}
            )

        # Build the dict that calculate_distance_from_phase_slope() expects on the inliers.
        phase_slope_data: dict[int, float] = {
            int(ch): float(ph)
            for ch, ph in zip(cr_inliers.channels, cr_inliers.phase_rad)
        }

        distance = calculate_distance_from_phase_slope(phase_slope_data)

        return EstimatorResult(
            distance_m=distance if distance is not None else math.nan,
            confidence=0.0,
            diagnostics={
                "phase_slope_data": phase_slope_data,
                "inlier_mask": inlier_mask,
                "best_slope": best_slope,
                "best_intercept": best_intercept,
            },
        )
