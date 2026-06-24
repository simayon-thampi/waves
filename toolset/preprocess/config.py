from dataclasses import dataclass
from enum import Enum

class UnwrapStrategy(str, Enum):
    NUMPY = "numpy"
    ITOH = "itoh"
    WEIGHTED = "weighted"

@dataclass
class PreprocessorConfig:
    # Stage 1: CFO Correction
    enable_cfo: bool = True
    cfo_threshold_hz: float = 10.0  # Skip CFO correction if below this threshold
    
    # Stage 2: Bad Tone Rejection
    enable_rejection: bool = True
    amplitude_dip_threshold_db: float = 20.0  # Reject if amp is more than this below median
    phase_discontinuity_threshold: float = 1.5707963267948966  # pi/2 discontinuity outlier threshold
    min_channels_for_rejection: int = 5
    
    # Stage 3: Phase Unwrap
    unwrap_strategy: UnwrapStrategy = UnwrapStrategy.WEIGHTED
    
    # Stage 4: Amplitude-Based Phase Weighting
    enable_weighting: bool = True
    
    # Stage 5: MRC Combining
    enable_mrc: bool = True

    # Spatio-Temporal Smoothing
    smoothing_window_size: int = 8

    # Scene Classification Thresholds
    null_depth_los_threshold: float = 10.0
    null_depth_nlos_threshold: float = 20.0
    residual_rms_los_threshold: float = 0.3
    reject_fraction_los_threshold: float = 0.15
    reject_fraction_nlos_threshold: float = 0.4

    # RANSAC Settings
    ransac_n_iterations: int = 100
    ransac_inlier_threshold_rad: float = 0.2
    ransac_min_sample_size: int = 6
    ifft_direct_path_max_ns: float = 20.0
    ifft_multipath_ratio_threshold: float = 0.5
