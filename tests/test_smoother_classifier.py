import numpy as np
import math

from toolset.processing.channel_response import ChannelResponse, Role, QUALITY_HIGH, QUALITY_UNAVAILABLE
from toolset.preprocess.config import PreprocessorConfig
from toolset.preprocess.temporal_smoother import TemporalSmoother
from toolset.preprocess.scene_classifier import SceneClassifier, SCENE_LOS, SCENE_MULTIPATH, SCENE_NLOS

def test_temporal_smoother_math():
    """
    Analytically verifies coherent complex averaging of TemporalSmoother.
    Sweep 1: amp = -50 dBm, phase = 0 rad
    Sweep 2: amp = -50 dBm, phase = pi/2 rad (0.5 * pi)
    
    Coherent average phasor:
      H1 = 10^(-2.5) * exp(0*j) = 10^(-2.5)
      H2 = 10^(-2.5) * exp(j * pi/2) = 10^(-2.5) * j
      H_avg = 10^(-2.5) * (1 + j)/2
      
      |H_avg| = 10^(-2.5) * sqrt(2)/2 = 10^(-2.5) / sqrt(2)
      Amp_avg = 20 * log10(|H_avg|) = -50 - 20 * log10(sqrt(2)) = -53.0103 dBm
      Phase_avg = angle(H_avg) = pi/4 = 0.785398 rad
    """
    channels = np.array([0], dtype=np.int32)
    iq = np.ones((1, 1), dtype=np.complex64)
    
    cr1 = ChannelResponse(
        channels=channels,
        iq_per_path=iq,
        amplitude_db=np.array([-50.0], dtype=np.float32),
        phase_rad=np.array([0.0], dtype=np.float32),
        quality_flags=np.array([QUALITY_HIGH], dtype=np.uint8),
        procedure_counter=1,
        role=Role.COMBINED,
        timestamp=100.0
    )
    
    cr2 = ChannelResponse(
        channels=channels,
        iq_per_path=iq,
        amplitude_db=np.array([-50.0], dtype=np.float32),
        phase_rad=np.array([0.5 * np.pi], dtype=np.float32),
        quality_flags=np.array([QUALITY_HIGH], dtype=np.uint8),
        procedure_counter=2,
        role=Role.COMBINED,
        timestamp=100.1
    )
    
    smoother = TemporalSmoother(window_size=2)
    
    # Push Sweep 1
    res1 = smoother.process(cr1)
    assert res1.metadata["active_size"] == 1
    assert abs(res1.amplitude_db[0] - (-50.0)) < 1e-4
    assert abs(res1.phase_rad[0] - 0.0) < 1e-4
    
    # Push Sweep 2 (should smooth coherently)
    res2 = smoother.process(cr2)
    assert res2.metadata["active_size"] == 2
    
    # Verify smoothed math
    expected_amp = -50.0 - 20.0 * np.log10(np.sqrt(2.0))
    expected_phase = np.pi / 4.0
    
    assert abs(res2.amplitude_db[0] - expected_amp) < 1e-4
    assert abs(res2.phase_rad[0] - expected_phase) < 1e-4
    
    # Verify N=1 exact pass-through
    smoother_n1 = TemporalSmoother(window_size=1)
    res_n1 = smoother_n1.process(cr2)
    assert res_n1.metadata["window_size"] == 1
    assert abs(res_n1.amplitude_db[0] - (-50.0)) < 1e-4
    assert abs(res_n1.phase_rad[0] - (0.5 * np.pi)) < 1e-4

def test_scene_classifier_scenes():
    """Verifies that the SceneClassifier correctly maps inputs to LOS, Multipath, and NLOS scenes."""
    config = PreprocessorConfig()
    classifier = SceneClassifier(config)
    
    # 1. Clean LOS Sweep
    # low null depth, low residual, low reject fraction
    channels = np.arange(10, dtype=np.int32)
    amp_los = -50.0 * np.ones(10, dtype=np.float32)  # null depth = 0
    phase_los = (0.1 * channels).astype(np.float32)  # perfect linear fit, residual = 0
    cr_los = ChannelResponse(
        channels=channels,
        iq_per_path=np.ones((10, 1), dtype=np.complex64),
        amplitude_db=amp_los,
        phase_rad=phase_los,
        quality_flags=np.zeros(10, dtype=np.uint8),
        procedure_counter=1,
        role=Role.COMBINED,
        timestamp=0.0
    )
    assert classifier.classify(cr_los) == SCENE_LOS
    
    # 2. Moderate Multipath Sweep
    # null depth is 18 dB (between 10 dB and 20 dB thresholds), or poor fit
    amp_mp = -50.0 * np.ones(10, dtype=np.float32)
    amp_mp[5] = -68.0  # dip of 18 dB
    cr_mp = ChannelResponse(
        channels=channels,
        iq_per_path=np.ones((10, 1), dtype=np.complex64),
        amplitude_db=amp_mp,
        phase_rad=phase_los,
        quality_flags=np.zeros(10, dtype=np.uint8),
        procedure_counter=2,
        role=Role.COMBINED,
        timestamp=0.0
    )
    assert classifier.classify(cr_mp) == SCENE_MULTIPATH

    # 3. Heavily Corrupted NLOS Sweep
    # null depth is 25 dB (>20 dB), reject fraction is 50% (>40%)
    amp_nlos = -50.0 * np.ones(10, dtype=np.float32)
    amp_nlos[2] = -75.0  # dip of 25 dB
    flags_nlos = np.zeros(10, dtype=np.uint8)
    flags_nlos[:5] = QUALITY_UNAVAILABLE  # 50% rejected
    cr_nlos = ChannelResponse(
        channels=channels,
        iq_per_path=np.ones((10, 1), dtype=np.complex64),
        amplitude_db=amp_nlos,
        phase_rad=phase_los,
        quality_flags=flags_nlos,
        procedure_counter=3,
        role=Role.COMBINED,
        timestamp=0.0
    )
    assert classifier.classify(cr_nlos) == SCENE_NLOS

if __name__ == "__main__":
    test_temporal_smoother_math()
    test_scene_classifier_scenes()
    print("SMOOTHER AND CLASSIFIER TESTS PASSED!")
