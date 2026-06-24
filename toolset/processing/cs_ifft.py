"""
processing/cs_ifft.py
=====================
IFFT-based delay-domain range estimator.

Signal model
------------
The BLE CS channel response measured across N frequencies spaced Δf = 1 MHz apart
is modelled as a complex-valued transfer function::

    H[k] = A[k] · exp(j·φ[k]),   k = 0 … N-1

where A[k] is the linear amplitude and φ[k] is the unwrapped one-way phase at
channel k.  The delay-domain impulse response is obtained by the N-point IDFT::

    h[n] = (1/N) Σ_k  H[k] · exp(j·2π·k·n/N)

Because H[k] is **complex** (not Hermitian-symmetric), ``np.fft.irfft`` must
**not** be used — it silently assumes the input is Hermitian and produces a
real-only output with half the useful length.  ``np.fft.ifft`` is the correct
transform.

The delay axis for bin n is::

    τ[n] = n / (N_fft · Δf)

with the unambiguous range running from 0 to 1/Δf = 1 μs.  For Δf = 1 MHz and
N = 73 channels, the *unaliased* maximum delay is 500 ns (the channel-spacing
Nyquist limit), so only bins 0 … floor(N_fft/2) need to be inspected.

Zero-padding to N_fft ≥ 4·N improves delay resolution by interpolation without
adding information.  A Hann window applied before the IFFT suppresses sidelobes
that would otherwise bias the peak location.

Previous bug (now fixed)
------------------------
The old code called ``compute_ifft_response()`` with N_fft = N (no zero-padding,
no windowing), then the GUI caller discarded the second half of the output on the
assumption the spectrum was real-valued.  For complex H[k] the second half is
**not** a mirror and contains valid negative-frequency delay bins wrapped to
[500 ns, 1 μs).  Discarding it did not cause wrong peaks for single dominant
paths (the peak is at n=n_peak < N/2) but it degraded resolution and misled
every consumer into thinking the output was only half as long as it should be.
The fix is applied entirely inside this module; no GUI code was changed.
"""

from typing import Dict, Optional
import numpy as np
from toolset.constants import SPEED_OF_LIGHT, BLE_CS_STEP_1MHZ

# Maximum unambiguous delay for 1 MHz channel spacing (= 1/Δf / 2 = 500 ns).
# Targets beyond this range alias back into [0, 500 ns), so we clip there.
_MAX_DELAY_NS = 500.0


def _next_pow2(n: int) -> int:
    """Return the smallest power of two that is ≥ n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def compute_ifft_response(
    phase_data: Dict[int, float],
    amplitude_data: Dict[int, float],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (t_ns, magnitude) delay-domain arrays from complex channel data.

    Returns ``(None, None)`` when fewer than 2 common channels are available.

    Parameters
    ----------
    phase_data : dict
        Mapping of BLE CS channel index → unwrapped one-way phase (rad).
    amplitude_data : dict
        Mapping of BLE CS channel index → amplitude (dBm or dB-relative).

    Returns
    -------
    t_ns : ndarray, shape (M,)
        Delay axis clipped to [0, _MAX_DELAY_NS] ns.
    magnitude : ndarray, shape (M,)
        |h(τ)| — magnitude of the delay-domain impulse response.

    Notes
    -----
    Processing steps:

    1. Build the complex channel vector H[k] = A[k]·exp(j·φ[k]) on the
       contiguous grid [ch_min, ch_max].  Missing channels are zero-filled
       (equivalent to a rectangular spectral window with no signal there).
    2. Apply a Hann window across the N occupied bins to suppress sidelobes.
       The window is applied to the *filled* grid positions only so that the
       zero-filled gaps are not windowed to non-zero values.
    3. Zero-pad to N_fft = next_pow2(4·N) for ×4 delay resolution by
       sinc interpolation.
    4. Compute h = ifft(H_padded) using the full complex IDFT.
       Do **not** use irfft — H is not Hermitian.
    5. The delay axis τ[n] = n / (N_fft·Δf).  Return only bins where
       τ ≤ _MAX_DELAY_NS (the unambiguous half-range for 1 MHz spacing).
    """
    common_channels = sorted(set(phase_data) & set(amplitude_data))
    if len(common_channels) < 2:
        return None, None

    ch_min, ch_max = common_channels[0], common_channels[-1]
    n_grid = ch_max - ch_min + 1          # contiguous channel grid size
    f_step = BLE_CS_STEP_1MHZ             # 1e6 Hz

    # ---- Step 1: build complex spectrum on the contiguous grid ---------------
    spectrum = np.zeros(n_grid, dtype=complex)
    for ch in common_channels:
        amplitude_linear = 10.0 ** (amplitude_data[ch] / 20.0)
        spectrum[ch - ch_min] = amplitude_linear * np.exp(1j * phase_data[ch])

    # ---- Step 2: Hann window on occupied positions only ----------------------
    # Build a mask of which grid bins are actually populated.
    occupied = np.zeros(n_grid, dtype=bool)
    for ch in common_channels:
        occupied[ch - ch_min] = True

    # Hann window sized to the number of real measurements (N_ch), but
    # placed at the occupied grid positions — keeps zero-filled gaps at 0.
    n_ch = len(common_channels)
    hann = np.hanning(n_ch)               # shape (N_ch,)
    spectrum[occupied] *= hann

    # ---- Step 3: zero-pad to next power-of-2 ≥ 4·N_grid --------------------
    n_fft = _next_pow2(max(4 * n_grid, 8))
    spectrum_padded = np.zeros(n_fft, dtype=complex)
    spectrum_padded[:n_grid] = spectrum   # zero-pad at the high end

    # ---- Step 4: full complex IFFT ------------------------------------------
    # np.fft.ifft is correct for complex H; irfft would silently drop the
    # imaginary part of each bin and assume Hermitian symmetry.
    h = np.fft.ifft(spectrum_padded)
    magnitude = np.abs(h)                 # shape (n_fft,)

    # ---- Step 5: build delay axis and clip to unambiguous range -------------
    # τ[n] = n / (N_fft · Δf) — in nanoseconds
    t_ns_full = np.arange(n_fft) / (n_fft * f_step) * 1e9

    # Keep only τ ∈ [0, _MAX_DELAY_NS].
    # The unambiguous range ends at 1/Δf = 1000 ns; beyond 500 ns the delays
    # wrap and are ambiguous, so we stop at 500 ns.
    clip_mask = t_ns_full <= _MAX_DELAY_NS
    t_ns = t_ns_full[clip_mask]
    magnitude = magnitude[clip_mask]

    return t_ns, magnitude


def calculate_distance_from_ifft(
    t_ns: np.ndarray,
    magnitude: np.ndarray,
) -> float:
    """
    Return the distance (m) corresponding to the IFFT magnitude peak.

    The search is performed over the full provided range (the caller of
    ``compute_ifft_response`` is responsible for clipping to [0, 500 ns]).

    Parameters
    ----------
    t_ns : ndarray
        Delay axis in nanoseconds.
    magnitude : ndarray
        Magnitude of the delay-domain impulse response, same length as t_ns.

    Returns
    -------
    float
        Estimated one-way distance in metres: d = c · τ_peak.
    """
    peak_ns = float(t_ns[np.argmax(magnitude)])
    return peak_ns * SPEED_OF_LIGHT / 1e9
