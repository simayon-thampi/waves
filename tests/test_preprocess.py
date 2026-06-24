import numpy as np

from toolset.processing.channel_response import ChannelResponse, Role, QUALITY_HIGH, QUALITY_UNAVAILABLE
from toolset.preprocess.config import PreprocessorConfig, UnwrapStrategy
from toolset.preprocess.pipeline import Preprocessor

def test_stage2_bad_tone_rejection():
    """
    Tests Stage 2 (Bad Tone Rejection) using synthetic ChannelResponse data.
    Ensures that deep amplitude fades and phase discontinuity outliers are flagged correctly.
    """
    n_ch = 15
    channels = np.arange(n_ch, dtype=np.int32)
    
    # Flat amplitude: -60 dBm
    amplitude_db = -60.0 * np.ones(n_ch, dtype=np.float32)
    # Perfect linear phase response: -0.1 radians per channel
    expected_slope = -0.1
    phase_rad = (expected_slope * channels).astype(np.float32)
    
    # Coherent complex phasors
    mag = 2048.0 * (10.0 ** (amplitude_db / 20.0))
    iq_per_path = (mag[:, np.newaxis] * np.exp(1j * 2.0 * phase_rad[:, np.newaxis])).astype(np.complex64)
    
    quality_flags = np.zeros(n_ch, dtype=np.uint8)  # QUALITY_HIGH
    
    # Inject Anomaly A: Deep amplitude dip at channel index 5 (-85 dBm, 25 dB below median)
    amplitude_db[5] = -85.0
    mag[5] = 2048.0 * (10.0 ** (-85.0 / 20.0))
    iq_per_path[5, 0] = mag[5] * np.exp(1j * 2.0 * phase_rad[5])
    
    # Inject Anomaly B: Phase discontinuity outlier at channel index 10 (add pi to phase)
    phase_rad[10] += np.pi
    iq_per_path[10, 0] = mag[10] * np.exp(1j * 2.0 * phase_rad[10])
    
    cr = ChannelResponse(
        channels=channels,
        iq_per_path=iq_per_path,
        amplitude_db=amplitude_db,
        phase_rad=phase_rad,
        quality_flags=quality_flags,
        procedure_counter=1,
        role=Role.COMBINED,
        timestamp=12345.6
    )
    
    config = PreprocessorConfig(
        enable_cfo=False,
        enable_rejection=True,
        amplitude_dip_threshold_db=20.0,
        phase_discontinuity_threshold=1.5,
        unwrap_strategy=UnwrapStrategy.NUMPY,
        enable_weighting=False,
        enable_mrc=False
    )
    preprocessor = Preprocessor(config)
    cleaned_cr = preprocessor.preprocess(cr)
    
    # Assertions
    # Channels 5 (amplitude dip) and 10 (phase discontinuity outlier) must be rejected
    assert cleaned_cr.quality_flags[5] == QUALITY_UNAVAILABLE
    assert cleaned_cr.quality_flags[10] == QUALITY_UNAVAILABLE
    
    # Channel 2 should remain clean and not rejected
    assert cleaned_cr.quality_flags[2] == QUALITY_HIGH

def test_stage3_phase_unwrap():
    """
    Tests Stage 3 (Phase Unwrap) comparing NumPy, Itoh, and Weighted unwrap strategies.
    Verifies that all three strategies successfully unwrap a clean wrapped phase, and
    demonstrates the weighted strategy's robust anchoring against error propagation.
    """
    n_ch = 10
    channels = np.arange(n_ch, dtype=np.int32)
    
    # Perfect linear phase sweep that crosses multiple wrap boundaries
    expected_slope = 2.0
    true_phase = expected_slope * channels  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    # Wrapped phase (mod 2*pi)
    wrapped_phase = (true_phase + np.pi) % (2.0 * np.pi) - np.pi
    
    amplitude_db = -60.0 * np.ones(n_ch, dtype=np.float32)
    mag = 2048.0 * (10.0 ** (amplitude_db / 20.0))
    iq_per_path = (mag[:, np.newaxis] * np.exp(1j * wrapped_phase[:, np.newaxis])).astype(np.complex64)
    
    # 1. Clean Sweep: Verify all strategies recover the true phase slope
    for strategy in [UnwrapStrategy.NUMPY, UnwrapStrategy.ITOH, UnwrapStrategy.WEIGHTED]:
        cr = ChannelResponse(
            channels=channels,
            iq_per_path=iq_per_path,
            amplitude_db=amplitude_db,
            phase_rad=wrapped_phase.astype(np.float32),
            quality_flags=np.zeros(n_ch, dtype=np.uint8),
            procedure_counter=1,
            role=Role.COMBINED,
            timestamp=12345.6
        )
        config = PreprocessorConfig(
            enable_cfo=False,
            enable_rejection=False,
            unwrap_strategy=strategy,
            enable_weighting=False,
            enable_mrc=False
        )
        preprocessor = Preprocessor(config)
        cleaned_cr = preprocessor.preprocess(cr)
        
        # Verify slope (change per channel should be exactly 1.0 since phase_rad = unwrapped / 2)
        diffs = np.diff(cleaned_cr.phase_rad)
        np.testing.assert_allclose(diffs, 1.0, atol=1e-5)

    # 2. Noisy Sweep: Inject a phase error at a low amplitude channel near the start (index 1)
    amplitude_db_noisy = -60.0 * np.ones(n_ch, dtype=np.float32)
    amplitude_db_noisy[1] = -90.0  # deep fade
    amplitude_db_noisy[5] = -50.0  # high amplitude anchor
    
    wrapped_phase_noisy = wrapped_phase.copy()
    wrapped_phase_noisy[1] += np.pi  # large phase jump on noisy channel
    
    mag_noisy = 2048.0 * (10.0 ** (amplitude_db_noisy / 20.0))
    iq_per_path_noisy = (mag_noisy[:, np.newaxis] * np.exp(1j * wrapped_phase_noisy[:, np.newaxis])).astype(np.complex64)
    
    cr_noisy = ChannelResponse(
        channels=channels,
        iq_per_path=iq_per_path_noisy,
        amplitude_db=amplitude_db_noisy,
        phase_rad=wrapped_phase_noisy.astype(np.float32),
        quality_flags=np.zeros(n_ch, dtype=np.uint8),
        procedure_counter=1,
        role=Role.COMBINED,
        timestamp=12345.6
    )
    
    # Execute with Standard sequential unwrap (NUMPY)
    preprocessor_numpy = Preprocessor(PreprocessorConfig(
        enable_cfo=False, enable_rejection=False, unwrap_strategy=UnwrapStrategy.NUMPY, enable_weighting=False, enable_mrc=False
    ))
    cleaned_numpy = preprocessor_numpy.preprocess(cr_noisy)
    
    # Execute with Amplitude-Weighted unwrap (anchored at index 5)
    preprocessor_weighted = Preprocessor(PreprocessorConfig(
        enable_cfo=False, enable_rejection=False, unwrap_strategy=UnwrapStrategy.WEIGHTED, enable_weighting=False, enable_mrc=False
    ))
    cleaned_weighted = preprocessor_weighted.preprocess(cr_noisy)
    
    # The weighted unwrap propagates outwards from anchor index 5, recovering pristine phase slope (differences of 1.0) at 5-9
    diffs_weighted = np.diff(cleaned_weighted.phase_rad[5:])
    np.testing.assert_allclose(diffs_weighted, 1.0, atol=1e-5)
