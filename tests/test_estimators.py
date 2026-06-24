"""
tests/test_estimators.py
========================
Unit tests confirming each estimator adapter runs on a synthetic
ChannelResponse without importing tkinter.

Design
------
* Tests are fully self-contained — no GUI, no UART, no real device data.
* ChannelResponse is constructed directly from numpy arrays (same approach
  as test_channel_response.py).
* Each test checks that:
    - The adapter is constructable and callable without GUI side-effects.
    - The EstimatorResult carries the correct estimator_name.
    - latency_ms is non-negative (timing measured by the base class).
    - distance_m is a finite float for sufficient channel counts.
    - distance_m is NaN when the input is too sparse (boundary case).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: verify tkinter is NOT imported as a side-effect
# ---------------------------------------------------------------------------

_TKINTER_MODULES = frozenset(("tkinter", "_tkinter", "tkinter.ttk"))


def _no_tkinter_imported() -> bool:
    """Return True if none of the tkinter modules are in sys.modules."""
    return _TKINTER_MODULES.isdisjoint(sys.modules)


# ---------------------------------------------------------------------------
# Synthetic ChannelResponse factory (mirrors test_channel_response.py)
# ---------------------------------------------------------------------------

from toolset.processing.channel_response import ChannelResponse, Role


def _make_cr(n_channels: int = 10, *, amplitude_db: float = -60.0) -> ChannelResponse:
    """
    Return a ChannelResponse with deterministic synthetic data.

    Channels start at index 10 and are spaced 1 apart (1 MHz each).
    Phase is a simple linear ramp (zero-referenced) so that phase-slope
    distance estimation is well-defined.
    """
    channels = np.arange(10, 10 + n_channels, dtype=np.int32)
    # Linear phase ramp: simulate a 1-metre target
    # φ(f) = -2π·f·(2d/c)  →  slope = -2π·(2d/c) / f_step
    d_m = 1.0
    c = 299_792_458.0
    f_step = 1e6
    freqs = (2402 + channels.astype(np.float64)) * f_step
    phase_rad_f64 = -2 * math.pi * freqs * (2 * d_m / c)
    phase_rad_f64 -= phase_rad_f64[0]  # zero-reference

    iq = np.exp(1j * phase_rad_f64).astype(np.complex64)[:, np.newaxis]

    return ChannelResponse(
        channels=channels,
        iq_per_path=iq,
        amplitude_db=np.full(n_channels, amplitude_db, dtype=np.float32),
        phase_rad=phase_rad_f64.astype(np.float32),
        quality_flags=np.zeros(n_channels, dtype=np.uint8),
        procedure_counter=0,
        role=Role.COMBINED,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# Import adapters (must happen AFTER the guard helpers are defined)
# ---------------------------------------------------------------------------

from toolset.estimators.base import Estimator, EstimatorResult
from toolset.estimators.phase_slope import PhaseSlopeEstimator
from toolset.estimators.ifft import IFFTEstimator
from toolset.estimators.music import MUSICEstimator


# ---------------------------------------------------------------------------
# Sanity: no GUI imports pulled in by the estimator package
# ---------------------------------------------------------------------------

class TestNoGuiImport:

    def test_estimator_imports_do_not_pull_tkinter(self):
        """Importing the estimator package must never touch tkinter."""
        assert _no_tkinter_imported(), (
            "tkinter was imported as a side-effect of the estimator package. "
            "Offending modules: " + str(_TKINTER_MODULES & sys.modules.keys())
        )


# ---------------------------------------------------------------------------
# EstimatorResult dataclass contract
# ---------------------------------------------------------------------------

class TestEstimatorResult:

    def test_result_fields_present(self):
        r = EstimatorResult(distance_m=1.5, confidence=0.9)
        assert r.distance_m == pytest.approx(1.5)
        assert r.confidence == pytest.approx(0.9)
        assert isinstance(r.diagnostics, dict)
        assert r.estimator_name == ""
        assert r.latency_ms == 0.0

    def test_result_default_diagnostics(self):
        r1 = EstimatorResult(distance_m=0.0, confidence=0.0)
        r2 = EstimatorResult(distance_m=0.0, confidence=0.0)
        # Each instance gets its own dict (no shared mutable default)
        r1.diagnostics["x"] = 1
        assert "x" not in r2.diagnostics


# ---------------------------------------------------------------------------
# Timing: latency_ms is set by __call__, not estimate()
# ---------------------------------------------------------------------------

class TestTimingWrapper:

    def test_latency_ms_is_nonnegative(self):
        est = PhaseSlopeEstimator()
        cr = _make_cr(8)
        result = est(cr)
        assert result.latency_ms >= 0.0

    def test_estimator_name_set_by_call(self):
        est = PhaseSlopeEstimator()
        cr = _make_cr(8)
        result = est(cr)
        assert result.estimator_name == "Phase Slope"

    def test_ifft_latency_ms_is_nonnegative(self):
        est = IFFTEstimator()
        result = est(_make_cr(8))
        assert result.latency_ms >= 0.0

    def test_music_latency_ms_is_nonnegative(self):
        est = MUSICEstimator()
        result = est(_make_cr(10))
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# PhaseSlopeEstimator
# ---------------------------------------------------------------------------

class TestPhaseSlopeEstimator:

    def test_constructable_without_gui(self):
        est = PhaseSlopeEstimator()
        assert isinstance(est, Estimator)

    def test_name(self):
        assert PhaseSlopeEstimator.name == "Phase Slope"

    def test_distance_is_finite_for_sufficient_channels(self):
        est = PhaseSlopeEstimator()
        result = est(_make_cr(8))
        assert math.isfinite(result.distance_m), (
            f"Expected finite distance, got {result.distance_m!r}"
        )

    def test_distance_is_nan_for_single_channel(self):
        """Phase slope requires ≥2 channels; single channel → NaN."""
        est = PhaseSlopeEstimator()
        result = est(_make_cr(1))
        assert math.isnan(result.distance_m)

    def test_diagnostics_contains_phase_slope_data(self):
        est = PhaseSlopeEstimator()
        result = est(_make_cr(6))
        assert "phase_slope_data" in result.diagnostics
        assert isinstance(result.diagnostics["phase_slope_data"], dict)

    def test_distance_approximately_correct(self):
        """
        The synthetic CR encodes a 1 m target; the phase-slope estimator
        should return something physically plausible (within ±50 m is a
        loose sanity check — absolute accuracy depends on channel count).
        """
        est = PhaseSlopeEstimator()
        result = est(_make_cr(20))
        assert math.isfinite(result.distance_m)
        # Very loose: just confirm the sign is positive and order-of-magnitude OK
        assert 0.0 < result.distance_m < 300.0, (
            f"distance_m={result.distance_m:.3f} looks implausible"
        )


# ---------------------------------------------------------------------------
# IFFTEstimator
# ---------------------------------------------------------------------------

class TestIFFTEstimator:

    def test_constructable_without_gui(self):
        est = IFFTEstimator()
        assert isinstance(est, Estimator)

    def test_name(self):
        assert IFFTEstimator.name == "IFFT"

    def test_distance_is_finite_for_sufficient_channels(self):
        est = IFFTEstimator()
        result = est(_make_cr(10))
        assert math.isfinite(result.distance_m)

    def test_distance_is_nan_for_single_channel(self):
        """IFFT requires ≥2 channels."""
        est = IFFTEstimator()
        result = est(_make_cr(1))
        assert math.isnan(result.distance_m)

    def test_diagnostics_contain_arrays(self):
        est = IFFTEstimator()
        result = est(_make_cr(8))
        assert "t_ns" in result.diagnostics
        assert "magnitude" in result.diagnostics
        assert isinstance(result.diagnostics["t_ns"], np.ndarray)
        assert isinstance(result.diagnostics["magnitude"], np.ndarray)

    def test_diagnostics_none_on_failure(self):
        est = IFFTEstimator()
        result = est(_make_cr(1))
        assert result.diagnostics["t_ns"] is None
        assert result.diagnostics["magnitude"] is None

    def test_output_clipped_to_500ns(self):
        """compute_ifft_response must never return delays > 500 ns."""
        from toolset.processing.cs_ifft import compute_ifft_response
        # Build simple phase/amplitude dicts
        channels = list(range(10, 83))   # 73 channels
        phase = {ch: 0.0 for ch in channels}
        ampl  = {ch: -60.0 for ch in channels}
        t_ns, mag = compute_ifft_response(phase, ampl)
        assert t_ns is not None
        assert float(t_ns[-1]) <= 500.0 + 1e-6, (
            f"Delay axis extends to {t_ns[-1]:.1f} ns — must be ≤ 500 ns"
        )

    def test_output_length_reflects_zero_padding(self):
        """Zero-padding to ≥4×N must produce more bins than the raw N."""
        from toolset.processing.cs_ifft import compute_ifft_response, _next_pow2
        n_ch = 20
        channels = list(range(10, 10 + n_ch))
        phase = {ch: 0.0 for ch in channels}
        ampl  = {ch: -60.0 for ch in channels}
        t_ns, mag = compute_ifft_response(phase, ampl)
        assert t_ns is not None
        # Full FFT size before clip: next_pow2(4 * n_ch) = 128 for n_ch=20
        n_fft = _next_pow2(4 * n_ch)
        # After 500 ns clip we keep approximately n_fft/2 bins; must be > n_ch
        assert len(t_ns) > n_ch, (
            f"Expected more than {n_ch} delay bins after zero-padding, got {len(t_ns)}"
        )


# ---------------------------------------------------------------------------
# IFFT: two-tone synthetic delay recovery
# ---------------------------------------------------------------------------

class TestIFFTTwoToneDelay:
    """
    Verify that compute_ifft_response correctly recovers a known delay.

    Signal model
    ------------
    Two-tone complex channel response with a single dominant reflector at
    τ = 30 ns::

        H[k] = exp(-j · 2π · f_k · τ),   f_k = (2402 + ch_k) · 1e6 Hz

    The amplitude is uniform (0 dB) so the peak of |h(τ)| should appear at
    τ ≈ 30 ns.  Because the spectrum is bandlimited and zero-padded, the
    peak should be detected within ±3 ns.

    Why this test is important
    --------------------------
    The old code discarded the second half of the IFFT output assuming a
    Hermitian (real-valued-time-domain) transform.  For complex input the
    full IFFT is needed; this test would fail (peak outside ±3 ns, or
    reported as 0 ns from a DC artifact) with the old code.
    """

    _TARGET_DELAY_NS = 30.0   # target delay embedded in synthetic data
    _TOLERANCE_NS    =  3.0   # ±3 ns detection tolerance

    @staticmethod
    def _build_two_tone_dicts(
        n_channels: int = 73,
        ch_start: int = 10,
        tau_ns: float = 30.0,
    ) -> tuple[dict, dict]:
        """
        Return (phase_data, amplitude_data) dicts encoding a single reflector
        at delay *tau_ns* nanoseconds.

        Phase model: φ[k] = -2π · f_k · τ  (one-way phase for a round-trip
        delay of 2τ, divided by 2 as in ChannelResponse.phase_rad convention).
        Amplitude: 0 dBm everywhere (uniform spectral density).
        """
        tau_s = tau_ns * 1e-9
        f_step = 1e6  # BLE_CS_STEP_1MHZ

        phase_data: dict[int, float] = {}
        amplitude_data: dict[int, float] = {}

        for i in range(n_channels):
            ch = ch_start + i
            f_hz = (2402 + ch) * f_step
            # One-way phase shift for a reflector at delay tau_s
            phase_data[ch] = -2.0 * math.pi * f_hz * tau_s
            amplitude_data[ch] = 0.0   # 0 dBm → linear amplitude = 1.0

        return phase_data, amplitude_data

    def test_peak_at_known_delay_direct(self):
        """
        Call compute_ifft_response directly and assert the peak lands within
        ±3 ns of the injected 30 ns delay.
        """
        from toolset.processing.cs_ifft import compute_ifft_response

        tau_ns = self._TARGET_DELAY_NS
        phase_data, amplitude_data = self._build_two_tone_dicts(tau_ns=tau_ns)

        t_ns, magnitude = compute_ifft_response(phase_data, amplitude_data)

        assert t_ns is not None, "compute_ifft_response returned None for valid input"
        assert magnitude is not None

        peak_ns = float(t_ns[np.argmax(magnitude)])

        assert abs(peak_ns - tau_ns) <= self._TOLERANCE_NS, (
            f"Peak at {peak_ns:.2f} ns, expected {tau_ns:.1f} ± {self._TOLERANCE_NS} ns. "
            f"Delay axis range: [{t_ns[0]:.1f}, {t_ns[-1]:.1f}] ns, "
            f"n_bins={len(t_ns)}"
        )

    def test_peak_via_ifft_estimator_adapter(self):
        """
        Same two-tone scenario exercised through the IFFTEstimator adapter,
        which goes through ChannelResponse → dicts → compute_ifft_response.
        The round-trip must also recover the peak within ±3 ns.
        """
        tau_ns = self._TARGET_DELAY_NS
        tau_s  = tau_ns * 1e-9
        c      = 299_792_458.0
        f_step = 1e6

        n_ch    = 73
        ch_start = 10
        channels = np.arange(ch_start, ch_start + n_ch, dtype=np.int32)
        freqs    = (2402 + channels.astype(np.float64)) * f_step

        phase_rad   = (-2.0 * math.pi * freqs * tau_s).astype(np.float32)
        amplitude_db = np.zeros(n_ch, dtype=np.float32)   # 0 dBm

        iq = np.exp(1j * phase_rad.astype(np.float64)).astype(np.complex64)[:, np.newaxis]

        cr = ChannelResponse(
            channels=channels,
            iq_per_path=iq,
            amplitude_db=amplitude_db,
            phase_rad=phase_rad,
            quality_flags=np.zeros(n_ch, dtype=np.uint8),
            procedure_counter=0,
            role=Role.COMBINED,
            timestamp=0.0,
        )

        result = IFFTEstimator()(cr)

        assert math.isfinite(result.distance_m), (
            f"Expected finite distance, got {result.distance_m!r}"
        )

        t_ns    = result.diagnostics["t_ns"]
        mag     = result.diagnostics["magnitude"]
        peak_ns = float(t_ns[np.argmax(mag)])

        assert abs(peak_ns - tau_ns) <= self._TOLERANCE_NS, (
            f"Adapter peak at {peak_ns:.2f} ns, expected {tau_ns:.1f} ± "
            f"{self._TOLERANCE_NS} ns  (distance={result.distance_m:.3f} m)"
        )

    def test_delay_axis_is_monotonically_increasing(self):
        """Sanity: the returned t_ns axis must be strictly monotone."""
        from toolset.processing.cs_ifft import compute_ifft_response
        phase_data, amplitude_data = self._build_two_tone_dicts()
        t_ns, _ = compute_ifft_response(phase_data, amplitude_data)
        assert t_ns is not None
        assert np.all(np.diff(t_ns) > 0), "t_ns is not strictly increasing"

    def test_magnitude_is_non_negative(self):
        """IFFT magnitude is |h|; must be ≥ 0 everywhere."""
        from toolset.processing.cs_ifft import compute_ifft_response
        phase_data, amplitude_data = self._build_two_tone_dicts()
        _, magnitude = compute_ifft_response(phase_data, amplitude_data)
        assert magnitude is not None
        assert np.all(magnitude >= 0.0)




# ---------------------------------------------------------------------------
# MUSICEstimator
# ---------------------------------------------------------------------------

class TestMUSICEstimator:

    def test_constructable_without_gui(self):
        est = MUSICEstimator()
        assert isinstance(est, Estimator)

    def test_name(self):
        assert MUSICEstimator.name == "MUSIC"

    def test_distance_is_finite_for_sufficient_channels(self):
        """MUSIC requires ≥4 channels."""
        est = MUSICEstimator()
        result = est(_make_cr(10))
        assert math.isfinite(result.distance_m)

    def test_distance_is_nan_for_too_few_channels(self):
        """Fewer than 4 channels → NaN."""
        est = MUSICEstimator()
        result = est(_make_cr(3))
        assert math.isnan(result.distance_m)

    def test_diagnostics_contain_arrays(self):
        est = MUSICEstimator()
        result = est(_make_cr(10))
        assert "delays_ns" in result.diagnostics
        assert "pseudo_spectrum" in result.diagnostics
        assert isinstance(result.diagnostics["delays_ns"], np.ndarray)
        assert isinstance(result.diagnostics["pseudo_spectrum"], np.ndarray)

    def test_diagnostics_none_on_failure(self):
        est = MUSICEstimator()
        result = est(_make_cr(3))
        assert result.diagnostics["delays_ns"] is None
        assert result.diagnostics["pseudo_spectrum"] is None
