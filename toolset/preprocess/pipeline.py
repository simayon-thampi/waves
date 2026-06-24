import copy
from typing import Optional
import numpy as np

from toolset.processing.channel_response import ChannelResponse, QUALITY_UNAVAILABLE
from toolset.cs_utils.cs_subevent import SubeventResults
from toolset.preprocess.config import PreprocessorConfig, UnwrapStrategy

class Preprocessor:
    """
    RF DSP Preprocessing Pipeline for BLE Channel Sounding.
    Cleans raw ChannelResponse sweeps before they are consumed by estimators.
    """
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config if config is not None else PreprocessorConfig()
        
    def preprocess(self, cr: ChannelResponse, subevent: Optional[SubeventResults] = None) -> ChannelResponse:
        """
        Runs all enabled preprocessor stages in order:
        1. CFO Correction
        2. Bad Tone Rejection
        3. Phase Unwrap (numpy, Itoh, or weighted)
        4. Amplitude-based phase weighting
        5. MRC combining (multi-path)
        
        Returns a new cleaned ChannelResponse object.
        """
        channels = cr.channels.copy()
        iq_per_path = cr.iq_per_path.copy()
        amplitude_db = cr.amplitude_db.copy()
        phase_rad = cr.phase_rad.copy()
        quality_flags = cr.quality_flags.copy()
        
        procedure_counter = cr.procedure_counter
        role = cr.role
        timestamp = cr.timestamp
        weights = cr.weights.copy() if cr.weights is not None else None

        # -------------------------------------------------------------
        # Stage 1: CFO Correction
        # -------------------------------------------------------------
        if self.config.enable_cfo:
            cfo_ppm = None
            if subevent is not None:
                if getattr(subevent, 'measured_freq_offset', None) is not None:
                    cfo_ppm = subevent.measured_freq_offset * 0.01
                else:
                    # Extract from Mode 0 steps
                    from toolset.cs_utils.cs_step import CSStepMode0
                    mode0_steps = [step for step in subevent.steps if isinstance(step, CSStepMode0) and getattr(step, 'measured_freq_offset', None) is not None]
                    if mode0_steps:
                        cfo_ppm = mode0_steps[0].measured_freq_offset * 0.01

            if cfo_ppm is not None:
                freqs_hz = (2402 + channels.astype(np.float64)) * 1e6
                # CFO in Hz per channel
                cfo_hz_per_chan = cfo_ppm * 1e-6 * freqs_hz
                
                # CFO skip criterion: skip if absolute frequency offset is below threshold
                if np.mean(np.abs(cfo_hz_per_chan)) >= self.config.cfo_threshold_hz:
                    # φ_corrected[k] = φ[k] - 2π * CFO_Hz / f[k]
                    # Since CFO_Hz / f[k] = cfo_ppm * 1e-6:
                    # phase_correction = 2 * pi * cfo_ppm * 1e-6
                    phase_correction = 2.0 * np.pi * cfo_ppm * 1e-6
                    phase_rad = (phase_rad - phase_correction).astype(np.float32)
                    
                    # Apply corresponding phase rotation to complex IQ data to maintain consistency
                    # The two-way correction factor is 2.0
                    iq_per_path = (iq_per_path * np.exp(-1j * 2.0 * phase_correction)).astype(np.complex64)

        # -------------------------------------------------------------
        # Stage 2: Bad Tone Rejection
        # -------------------------------------------------------------
        if self.config.enable_rejection and len(channels) >= self.config.min_channels_for_rejection:
            rejected_mask = np.zeros(len(channels), dtype=bool)
            
            # Criterion A: ToneQualityIndicator == LOW for >50% of paths
            if subevent is not None:
                from toolset.cs_utils.cs_step import CSStepMode2, ToneQualityIndicator
                for i, ch in enumerate(channels):
                    low_quality_count = 0
                    total_paths = 0
                    for step in subevent.steps:
                        if isinstance(step, CSStepMode2) and step.channel == ch and step.tones:
                            valid_tones = [t for t in step.tones if getattr(t, 'quality', None) is not None]
                            total_paths += len(valid_tones)
                            low_quality_count += sum(1 for t in valid_tones if t.quality >= ToneQualityIndicator.TONE_QUALITY_LOW)
                    if total_paths > 0 and (low_quality_count / total_paths) > 0.5:
                        rejected_mask[i] = True
            else:
                # Fallback to worst quality per channel in ChannelResponse
                # If worst quality is LOW (2) or UNAVAILABLE (3)
                rejected_mask = rejected_mask | (quality_flags >= 2)

            # Criterion B: Amplitude dip more than threshold below median
            median_amp = np.median(amplitude_db)
            amp_dip_mask = amplitude_db < (median_amp - self.config.amplitude_dip_threshold_db)
            rejected_mask = rejected_mask | amp_dip_mask
            
            # Criterion C: Phase discontinuity outlier
            if len(channels) >= 2:
                diffs = np.diff(phase_rad)
                chan_diffs = np.diff(channels)
                normalized_diffs = diffs / chan_diffs
                expected_slope = np.median(normalized_diffs)
                
                for k in range(1, len(phase_rad)):
                    actual_diff = phase_rad[k] - phase_rad[k-1]
                    spacing = channels[k] - channels[k-1]
                    discontinuity = actual_diff - expected_slope * spacing
                    # Wrap discontinuity to [-pi, pi]
                    discontinuity = (discontinuity + np.pi) % (2.0 * np.pi) - np.pi
                    if np.abs(discontinuity) > self.config.phase_discontinuity_threshold:
                        rejected_mask[k] = True

            # Mark in quality_flags, but avoid rejecting all channels
            if not np.all(rejected_mask):
                quality_flags[rejected_mask] = QUALITY_UNAVAILABLE

        # -------------------------------------------------------------
        # Stage 3: Phase Unwrap
        # -------------------------------------------------------------
        if len(channels) > 1:
            mrc_phasor = np.sum(iq_per_path, axis=1)
            wrapped_phase = np.angle(mrc_phasor)
            
            def wrap_difference(d):
                return (d + np.pi) % (2.0 * np.pi) - np.pi
                
            strategy = self.config.unwrap_strategy
            if strategy == UnwrapStrategy.NUMPY:
                unwrapped = np.unwrap(wrapped_phase)
            elif strategy == UnwrapStrategy.ITOH:
                unwrapped = np.zeros_like(wrapped_phase)
                unwrapped[0] = wrapped_phase[0]
                for k in range(1, len(wrapped_phase)):
                    diff = wrapped_phase[k] - wrapped_phase[k-1]
                    unwrapped[k] = unwrapped[k-1] + wrap_difference(diff)
            elif strategy == UnwrapStrategy.WEIGHTED:
                linear_amp = 10.0 ** (amplitude_db / 20.0)
                anchor_idx = int(np.argmax(linear_amp))
                
                unwrapped = np.zeros_like(wrapped_phase)
                unwrapped[anchor_idx] = wrapped_phase[anchor_idx]
                
                # Propagate right
                for k in range(anchor_idx + 1, len(wrapped_phase)):
                    diff = wrapped_phase[k] - wrapped_phase[k-1]
                    unwrapped[k] = unwrapped[k-1] + wrap_difference(diff)
                    
                # Propagate left
                for k in range(anchor_idx - 1, -1, -1):
                    diff = wrapped_phase[k] - wrapped_phase[k+1]
                    unwrapped[k] = unwrapped[k+1] + wrap_difference(diff)
            else:
                unwrapped = np.unwrap(wrapped_phase)
                
            # Divide unwrapped phase by 2.0 and subtract offset so phase_rad[0] == 0
            phase_rad_f64 = unwrapped / 2.0
            phase_rad_f64 -= phase_rad_f64[0]
            phase_rad = phase_rad_f64.astype(np.float32)

        # -------------------------------------------------------------
        # Stage 4: Amplitude-Based Phase Weighting
        # -------------------------------------------------------------
        if self.config.enable_weighting:
            linear_amp = 10.0 ** (amplitude_db / 20.0)
            power = linear_amp ** 2
            power_sum = np.sum(power)
            
            if power_sum > 0.0 and not np.isnan(power_sum):
                weights = (power / power_sum).astype(np.float32)
            else:
                weights = (np.ones(len(channels)) / len(channels)).astype(np.float32)

        # -------------------------------------------------------------
        # Stage 5: MRC Combining
        # -------------------------------------------------------------
        if self.config.enable_mrc:
            n_ch, n_paths = iq_per_path.shape
            if n_paths > 1:
                # First path acts as reference path h_i[k]
                h = iq_per_path[:, 0]
                h_power_sum = (np.abs(h)**2) * n_paths
                
                mrc_combined_phasor = np.zeros(n_ch, dtype=np.complex64)
                for path_idx in range(n_paths):
                    mrc_combined_phasor += np.conj(h) * iq_per_path[:, path_idx]
                    
                safe_mask = h_power_sum > 1e-12
                z_mrc = np.zeros(n_ch, dtype=np.complex64)
                z_mrc[safe_mask] = mrc_combined_phasor[safe_mask] / h_power_sum[safe_mask]
                
                # Duplicate the combined MRC signal scaled by paths so coherent sum matches z_mrc
                for path_idx in range(n_paths):
                    iq_per_path[:, path_idx] = z_mrc / float(n_paths)

        return ChannelResponse(
            channels=channels,
            iq_per_path=iq_per_path,
            amplitude_db=amplitude_db,
            phase_rad=phase_rad,
            quality_flags=quality_flags,
            procedure_counter=procedure_counter,
            role=role,
            timestamp=timestamp,
            weights=weights,
        )


def run_ransac(
    channels: np.ndarray,
    phases: np.ndarray,
    n_iterations: int = 100,
    threshold_rad: float = 0.3,
    min_sample_size: int = 6
) -> tuple[float, float, np.ndarray]:
    """
    RANSAC robust line fitting algorithm.
    Fits a straight line φ = a * x + b where x is the channel index.
    
    Parameters
    ----------
    channels : ndarray
        The channel indices.
    phases : ndarray
        The unwrapped phases (radians).
    n_iterations : int
        Number of RANSAC iterations.
    threshold_rad : float
        The inlier threshold in radians.
    min_sample_size : int
        Minimum number of points to sample for fitting.
        
    Returns
    -------
    best_slope : float
        Slope in radians per channel.
    best_intercept : float
        Intercept in radians.
    best_inliers : ndarray
        Boolean mask of inlier channels.
    """
    n_total = len(channels)
    if n_total < min_sample_size:
        # Fallback: if not enough points, fit a basic line using all points
        if n_total >= 2:
            slope, intercept = np.polyfit(channels, phases, 1)
            return float(slope), float(intercept), np.ones(n_total, dtype=bool)
        else:
            return 0.0, 0.0, np.ones(n_total, dtype=bool)

    best_slope = 0.0
    best_intercept = 0.0
    best_inliers = np.zeros(n_total, dtype=bool)
    max_inliers_count = -1

    # Seeded RNG for perfectly reproducible and deterministic tests
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        sample_indices = rng.choice(n_total, size=min_sample_size, replace=False)
        sample_x = channels[sample_indices]
        sample_y = phases[sample_indices]
        
        # Fit a line to the chosen sample
        slope, intercept = np.polyfit(sample_x, sample_y, 1)
        
        # Compute residuals across all points
        fit_y = slope * channels + intercept
        residuals = np.abs(phases - fit_y)
        
        # Determine inliers
        inliers = residuals < threshold_rad
        inlier_count = np.sum(inliers)
        
        if inlier_count > max_inliers_count:
            max_inliers_count = inlier_count
            best_inliers = inliers
            best_slope = slope
            best_intercept = intercept

    # Refit using all consensus inliers for maximum accuracy
    if max_inliers_count >= 2:
        inlier_indices = np.where(best_inliers)[0]
        best_slope, best_intercept = np.polyfit(channels[inlier_indices], phases[inlier_indices], 1)
        
    return float(best_slope), float(best_intercept), best_inliers
