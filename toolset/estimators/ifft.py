"""
estimators/ifft.py
==================
Thin adapter: wraps the existing IFFT impulse-response algorithm.

Algorithm ownership
-------------------
All maths live in ``toolset.processing.cs_ifft``.  This module only
bridges between ``ChannelResponse`` and the legacy dict-based API.
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


class IFFTEstimator(Estimator):
    """
    Range estimator using complex IFFT-based direct-path delay extraction.
    """

    name = "IFFT"

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config if config is not None else PreprocessorConfig()

    def estimate(self, cr: "ChannelResponse") -> EstimatorResult:
        if len(cr.channels) < 2:
            return EstimatorResult(
                distance_m=math.nan,
                confidence=0.0,
                diagnostics={
                    "t_ns": None,
                    "magnitude": None,
                    "is_multipath": False
                }
            )

        from toolset.preprocess.pipeline import run_ransac
        
        # STAGE A: Run RANSAC first to get inliers
        best_slope, best_intercept, inlier_mask = run_ransac(
            cr.channels,
            cr.phase_rad,
            n_iterations=self.config.ransac_n_iterations,
            threshold_rad=self.config.ransac_inlier_threshold_rad,
            min_sample_size=self.config.ransac_min_sample_size
        )
        
        n_fft = 1024
        H = np.zeros(n_fft, dtype=complex)
        
        if len(cr.channels) > 0:
            ch_min = cr.channels[0]
            # Apply Hann window to inlier positions
            hann = np.hanning(len(cr.channels))
            for idx in range(len(cr.channels)):
                if inlier_mask[idx]:
                    ch = cr.channels[idx]
                    H[ch - ch_min] = (10.0 ** (cr.amplitude_db[idx] / 20.0)) * np.exp(1j * cr.phase_rad[idx]) * hann[idx]

            # Compute h(t) = ifft(H)
            h = np.fft.ifft(H)
            magnitude = np.abs(h)
            
            # Delay axis: tau[n] = n / (N_fft * 1e6) * 1e9 = n * (1000.0 / 1024.0) ns
            t_ns = np.arange(n_fft) * (1000.0 / float(n_fft))
            
            # Peak 1 (direct path): highest peak in [0, 20 ns]
            mask1 = (t_ns >= 0.0) & (t_ns <= self.config.ifft_direct_path_max_ns)
            # Peak 2 (reflected path): highest peak in [20 ns, 100 ns]
            mask2 = (t_ns > self.config.ifft_direct_path_max_ns) & (t_ns <= 100.0)
            
            peak1_val, peak2_val = 0.0, 0.0
            peak1_ns, peak2_ns = 0.0, 0.0
            
            if np.any(mask1):
                idx1_in_mask = np.argmax(magnitude[mask1])
                peak1_idx = np.where(mask1)[0][idx1_in_mask]
                peak1_ns = float(t_ns[peak1_idx])
                peak1_val = float(magnitude[peak1_idx])
                
            if np.any(mask2):
                idx2_in_mask = np.argmax(magnitude[mask2])
                peak2_idx = np.where(mask2)[0][idx2_in_mask]
                peak2_ns = float(t_ns[peak2_idx])
                peak2_val = float(magnitude[peak2_idx])
                
            # If peak 2 amplitude > 0.5 * peak 1 amplitude: flag as multipath
            is_multipath = False
            if peak1_val > 0.0 and peak2_val > self.config.ifft_multipath_ratio_threshold * peak1_val:
                is_multipath = True
                
            d_ifft = peak1_ns * SPEED_OF_LIGHT / 1e9
        else:
            d_ifft = math.nan
            t_ns = np.arange(n_fft) * (1000.0 / float(n_fft))
            magnitude = np.zeros(n_fft)
            is_multipath = False
            
        return EstimatorResult(
            distance_m=d_ifft,
            confidence=0.0,
            diagnostics={
                "t_ns": t_ns,
                "magnitude": magnitude,
                "inlier_mask": inlier_mask,
                "best_slope": best_slope,
                "best_intercept": best_intercept,
                "is_multipath": is_multipath,
            }
        )
