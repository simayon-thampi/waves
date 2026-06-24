"""
estimators/weighted_ls.py
==========================
Weighted Least Squares range estimator.

Fits a weighted straight line  φ(f) = slope·f + intercept  to the
unwrapped per-tone phase on the RANSAC inlier subset.  The weight of
each tone is the linear amplitude raised to *exponent* (default 2.0),
so tones with strong SNR dominate the fit while noisy / faded tones
are suppressed.

Distance is derived from the slope via:

    d = −slope · c / (2π)

where c is the speed of light.  The sign convention matches the rest
of the pipeline (positive distance for a causal path).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from toolset.estimators.base import Estimator, EstimatorResult
from toolset.constants import SPEED_OF_LIGHT
from toolset.preprocess.config import PreprocessorConfig

if TYPE_CHECKING:
    from toolset.processing.channel_response import ChannelResponse


class WeightedLSEstimator(Estimator):
    """
    Weighted Least Squares range estimator on RANSAC inlier subset.

    Fits a straight line to the unwrapped phase weighted by power values.

    Parameters
    ----------
    exponent : float
        Linear-amplitude weight exponent.  ``2.0`` ≈ power weighting,
        ``0.0`` reduces to unweighted ordinary LS.
    config : PreprocessorConfig, optional
        RANSAC hyper-parameters.  Defaults to ``PreprocessorConfig()``.
    """

    name = "Weighted LS"

    def __init__(self, exponent: float = 2.0, config: Optional[PreprocessorConfig] = None):
        self.exponent = exponent
        self.config = config if config is not None else PreprocessorConfig()

    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        from toolset.preprocess.pipeline import run_ransac

        # Run RANSAC first
        best_slope, best_intercept, inlier_mask = run_ransac(
            cr.channels,
            cr.phase_rad,
            n_iterations=self.config.ransac_n_iterations,
            threshold_rad=self.config.ransac_inlier_threshold_rad,
            min_sample_size=self.config.ransac_min_sample_size,
        )

        cr_inliers = cr[inlier_mask]

        if cr_inliers.n_channels < 2:
            return EstimatorResult(
                distance_m=math.nan,
                confidence=0.0,
                diagnostics={"inlier_mask": inlier_mask},
            )

        freqs = (2402 + cr_inliers.channels.astype(np.float64)) * 1e6
        phases = cr_inliers.phase_rad.astype(np.float64)

        if cr_inliers.weights is not None:
            w = cr_inliers.weights.astype(np.float64)
        else:
            w = np.ones_like(phases)

        w = w ** self.exponent
        w_sum = np.sum(w)
        if w_sum <= 0:
            w = np.ones_like(phases)
            w_sum = len(phases)
        w = w / w_sum

        f_bar = np.sum(w * freqs)
        p_bar = np.sum(w * phases)
        num = np.sum(w * (freqs - f_bar) * (phases - p_bar))
        den = np.sum(w * (freqs - f_bar) ** 2)

        if den == 0:
            return EstimatorResult(distance_m=math.nan, confidence=0.0)

        slope = num / den
        intercept = p_bar - slope * f_bar
        distance = -slope * SPEED_OF_LIGHT / (2.0 * np.pi)

        fit_vals = slope * freqs + intercept
        residuals = phases - fit_vals
        residual_rms = float(np.sqrt(np.sum(w * (residuals ** 2))))

        return EstimatorResult(
            distance_m=float(distance),
            confidence=float(1.0 / (1.0 + residual_rms)),
            diagnostics={
                "residual_rms": residual_rms,
                "residuals": residuals,
                "fit_vals": fit_vals,
                "inlier_mask": inlier_mask,
                "best_slope": best_slope,
                "best_intercept": best_intercept,
            },
        )
