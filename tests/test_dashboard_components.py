import numpy as np
import math
import threading
import time

from toolset.processing.channel_response import ChannelResponse, Role
from toolset.pipeline.replay_buffer import ReplayBuffer
from toolset.estimators.base import EstimatorResult
from toolset.gui.inspection_dashboard import WeightedLSEstimator

def test_replay_buffer_ring_properties():
    """Verify that ReplayBuffer is a thread-safe ring buffer with configurable max size."""
    buf = ReplayBuffer(max_size=5)
    
    channels = np.array([1, 2, 3], dtype=np.int32)
    iq = np.ones((3, 1), dtype=np.complex64)
    amp = np.zeros(3, dtype=np.float32)
    phase = np.zeros(3, dtype=np.float32)
    
    # Fill the buffer beyond capacity
    for i in range(10):
        cr = ChannelResponse(
            channels=channels,
            iq_per_path=iq,
            amplitude_db=amp,
            phase_rad=phase,
            quality_flags=np.zeros(3, dtype=np.uint8),
            procedure_counter=i,
            role=Role.COMBINED,
            timestamp=float(i)
        )
        buf.append(cr, None)
        
    assert buf.size() == 5
    items = buf.get_all()
    # Should contain counters 5, 6, 7, 8, 9
    assert items[0][0].procedure_counter == 5
    assert items[-1][0].procedure_counter == 9

def test_replay_buffer_concurrency():
    """Verify that concurrent appends/reads to ReplayBuffer are thread-safe and do not crash."""
    buf = ReplayBuffer(max_size=100)
    channels = np.array([1, 2, 3], dtype=np.int32)
    iq = np.ones((3, 1), dtype=np.complex64)
    amp = np.zeros(3, dtype=np.float32)
    phase = np.zeros(3, dtype=np.float32)
    
    def writer_thread():
        for i in range(500):
            cr = ChannelResponse(
                channels=channels,
                iq_per_path=iq,
                amplitude_db=amp,
                phase_rad=phase,
                quality_flags=np.zeros(3, dtype=np.uint8),
                procedure_counter=i,
                role=Role.COMBINED,
                timestamp=0.0
            )
            buf.append(cr, None)
            time.sleep(0.001)

    threads = [threading.Thread(target=writer_thread) for _ in range(5)]
    for t in threads:
        t.start()
        
    # Read concurrently
    for _ in range(100):
        items = buf.get_all()
        assert len(items) <= 100
        time.sleep(0.002)

    for t in threads:
        t.join()

    assert buf.size() == 100

def test_estimator_result_dynamic_confidence():
    """Verify EstimatorResult calculates confidence score dynamically from residual_rms."""
    # Without residual_rms: defaults to the initialization value
    res1 = EstimatorResult(distance_m=5.0, confidence=0.7)
    assert res1.confidence == 0.7

    # With residual_rms in diagnostics: 1 / (1 + rms)
    res2 = EstimatorResult(
        distance_m=5.0,
        confidence=0.0,
        diagnostics={"residual_rms": 0.25}
    )
    assert res2.confidence == 0.8  # 1 / (1 + 0.25) = 0.8

    # Handing NaN gracefully
    res3 = EstimatorResult(
        distance_m=5.0,
        confidence=0.5,
        diagnostics={"residual_rms": float('nan')}
    )
    assert res3.confidence == 0.5

def test_weighted_ls_estimator_math():
    """Verify that WeightedLSEstimator runs and computes linear fit correctly."""
    channels = np.array([0, 1, 2, 3], dtype=np.int32)
    
    # Expected slope: -0.5 rad per channel, 1 MHz spacing
    # With 2402 MHz base freq, freq steps are 1 MHz apart
    # phase_rad = slope * (2402 + channel) * 1e6
    # Let's define a clean linear phase sweep
    phases = np.array([0.0, -0.5, -1.0, -1.5], dtype=np.float32)
    amplitude_db = -50.0 * np.ones(4, dtype=np.float32)
    mag = 2048.0 * (10.0 ** (amplitude_db / 20.0))
    iq_per_path = (mag[:, np.newaxis] * np.exp(1j * phases[:, np.newaxis])).astype(np.complex64)
    
    cr = ChannelResponse(
        channels=channels,
        iq_per_path=iq_per_path,
        amplitude_db=amplitude_db,
        phase_rad=phases,
        quality_flags=np.zeros(4, dtype=np.uint8),
        procedure_counter=1,
        role=Role.COMBINED,
        timestamp=0.0
    )
    
    # Weighted LS estimator with uniform weights (exponent=0)
    wls = WeightedLSEstimator(exponent=0.0)
    res = wls.estimate(cr)
    
    # Assertions
    assert not math.isnan(res.distance_m)
    # The perfect linear fit should yield residual_rms extremely close to 0, confidence close to 1.0
    assert res.confidence > 0.99
    assert res.diagnostics["residual_rms"] < 1e-5

def test_incremental_statistics_math():
    """Verify that incremental statistics (RMSE, Bias, Jitter) are computed correctly."""
    estimates = [4.0, 6.0, 5.0]
    true_dist = 5.0
    
    n = 0
    sum_err = 0.0
    sum_sq_err = 0.0
    sum_est = 0.0
    sum_sq_est = 0.0
    
    for d in estimates:
        err = d - true_dist
        n += 1
        sum_err += err
        sum_sq_err += err ** 2
        sum_est += d
        sum_sq_est += d ** 2
        
    bias = sum_err / n
    rmse = math.sqrt(sum_sq_err / n)
    var = (sum_sq_est / n) - (sum_est / n) ** 2
    jitter = math.sqrt(max(0.0, var))
    
    assert abs(bias - 0.0) < 1e-7
    assert abs(rmse - math.sqrt(2.0 / 3.0)) < 1e-7
    assert abs(jitter - math.sqrt(2.0 / 3.0)) < 1e-7

if __name__ == "__main__":
    test_replay_buffer_ring_properties()
    test_replay_buffer_concurrency()
    test_estimator_result_dynamic_confidence()
    test_weighted_ls_estimator_math()
    test_incremental_statistics_math()
    print("DASHBOARD COMPONENT TESTS PASSED!")
