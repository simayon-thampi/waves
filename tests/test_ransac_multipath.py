import math
import numpy as np
import pytest

from toolset.processing.channel_response import ChannelResponse
from toolset.preprocess.pipeline import run_ransac
from toolset.estimators.phase_slope import PhaseSlopeEstimator
from toolset.estimators.ifft import IFFTEstimator
from toolset.preprocess.config import PreprocessorConfig
from toolset.constants import SPEED_OF_LIGHT


def test_ransac_clean_los():
    """
    Test 1: Verify RANSAC recovers clean LOS slope within 5% and tags >80% inliers.
    """
    n_channels = 73
    channels = np.arange(n_channels)
    
    # 1.0 m direct path delay: one-way tau = d / c
    d_true = 1.0
    tau_true = d_true / SPEED_OF_LIGHT
    
    # φ[k] = -2π * f[k] * tau
    f_rel = channels * 1e6  # 1 MHz spacing
    phase_clean = -2.0 * np.pi * f_rel * tau_true
    
    # Add minor white noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, n_channels)
    phase_noisy = phase_clean + noise
    
    slope, intercept, inliers = run_ransac(
        channels=channels,
        phases=phase_noisy,
        n_iterations=100,
        threshold_rad=0.3,
        min_sample_size=6
    )
    
    # Compute the recovered distance
    # slope is rad / channel. Since channel spacing is 1 MHz:
    # d = -slope * c / (2 * pi * 1e6)
    d_recovered = -slope * SPEED_OF_LIGHT / (2.0 * np.pi * 1e6)
    
    assert abs(d_recovered - d_true) < 0.05
    assert np.sum(inliers) / n_channels > 0.80


def test_ransac_multipath_rejection():
    """
    Test 2: Verify RANSAC isolates direct path (1 m) from severe localized multipath distortion,
    flagging the corrupted tones as outliers.
    """
    n_channels = 73
    channels = np.arange(n_channels)
    
    # 1.0 m direct path delay: one-way tau = d / c
    d_true = 1.0
    tau_true = d_true / SPEED_OF_LIGHT
    
    # φ[k] = -2π * f[k] * tau
    f_rel = channels * 1e6  # 1 MHz spacing
    phase_clean = -2.0 * np.pi * f_rel * tau_true
    
    # Add minor white noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.02, n_channels)
    phase_noisy = phase_clean + noise
    
    # Corrupt a localized region of channels (e.g., 20 to 45) with strong multipath phase pulling
    # (e.g., adding a peak deviation of 0.8 rad)
    corrupt_mask = (channels >= 20) & (channels <= 45)
    phase_noisy[corrupt_mask] += 0.8
    
    threshold_rad = 0.25
    slope, intercept, inliers = run_ransac(
        channels=channels,
        phases=phase_noisy,
        n_iterations=200,
        threshold_rad=threshold_rad,
        min_sample_size=6
    )
    
    d_recovered = -slope * SPEED_OF_LIGHT / (2.0 * np.pi * 1e6)
    
    # The direct path slope must be successfully recovered within 5% margin
    assert abs(d_recovered - d_true) < 0.05
    
    # All corrupted channels must be flagged as outliers
    for idx in range(n_channels):
        if corrupt_mask[idx]:
            assert not inliers[idx]


def test_ifft_direct_path_detection():
    """
    Test 3: Verify IFFT detects a synthetic direct-path delay of 3.3 ns within 1 ns.
    """
    n_channels = 73
    channels = np.arange(n_channels)
    
    # Direct path delay: 3.3 ns (corresponds to ~1.0 m one-way distance)
    tau_direct = 3.33  # ns
    
    # Construct complex ChannelResponse
    amplitude_db = np.ones(n_channels) * -40.0
    # phase profile: φ[k] = -2π * f_rel * tau
    f_rel = channels * 1e6
    phase_rad = -2.0 * np.pi * f_rel * (tau_direct * 1e-9)
    
    iq_per_path = np.zeros((n_channels, 1), dtype=complex)
    for idx in range(n_channels):
        iq_per_path[idx, 0] = (10.0 ** (amplitude_db[idx] / 20.0)) * np.exp(1j * phase_rad[idx])
        
    cr = ChannelResponse(
        channels=channels,
        iq_per_path=iq_per_path,
        amplitude_db=amplitude_db,
        phase_rad=phase_rad,
        quality_flags=np.zeros(n_channels, dtype=np.uint8),
        procedure_counter=10,
        role="initiator",
        timestamp=1234567.89,
        weights=np.ones(n_channels)
    )
    
    config = PreprocessorConfig(
        ransac_n_iterations=100,
        ransac_inlier_threshold_rad=0.3,
        ransac_min_sample_size=6,
        ifft_direct_path_max_ns=20.0,
        ifft_multipath_ratio_threshold=0.5
    )
    
    estimator = IFFTEstimator(config=config)
    res = estimator.estimate(cr)
    
    assert not math.isnan(res.distance_m)
    
    # Convert recovered distance back to delay: tau = d / c
    tau_recovered = (res.distance_m / SPEED_OF_LIGHT) * 1e9
    
    assert abs(tau_recovered - tau_direct) < 1.0
