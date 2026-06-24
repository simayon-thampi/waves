"""
channel_response.py
===================
Intermediate representation between the BLE CS subevent parser and every
range / sensing estimator.

Design contract
---------------
* Pure numpy — no GUI, no serial, no parser imports required at load() time.
* All arrays are indexed on axis-0 by channel, sorted in ascending order.
* Boolean indexing returns a new ChannelResponse restricted to selected channels.
* Serialises to / from a plain .npz file — no pickling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from math import log
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    # Only imported for type hints so that load() stays GUI/parser-free.
    from toolset.cs_utils.cs_subevent import SubeventResults


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class Role(IntEnum):
    """Indicates which device side(s) contributed to this ChannelResponse."""
    INITIATOR = 0
    REFLECTOR = 1
    COMBINED  = 2   # built from a matched initiator + reflector pair


# ---------------------------------------------------------------------------
# Quality flag constants (mirrors ToneQualityIndicator.value)
# ---------------------------------------------------------------------------

QUALITY_HIGH        = np.uint8(0)
QUALITY_MEDIUM      = np.uint8(1)
QUALITY_LOW         = np.uint8(2)
QUALITY_UNAVAILABLE = np.uint8(3)


# ---------------------------------------------------------------------------
# ChannelResponse
# ---------------------------------------------------------------------------

@dataclass
class ChannelResponse:
    """
    Frequency-domain channel snapshot for a single BLE CS procedure.

    All arrays share axis-0 length **N_channels**, sorted in ascending
    channel-index order.

    Attributes
    ----------
    channels : ndarray, shape (N_ch,), int32
        BLE CS channel indices.  Channel *n* maps to (2402 + n) MHz.

    iq_per_path : ndarray, shape (N_ch, N_paths), complex64
        Combined (initiator × reflector) phasor per antenna path per channel.
        ``arg(iq_per_path[i, p]) == φ_ini[i,p] + φ_ref[i,p]``; device-local
        oscillator offsets cancel in the /2 used to derive ``phase_rad``.
        Magnitude equals ``|IQ_ini| × |IQ_ref|`` before normalisation.

    amplitude_db : ndarray, shape (N_ch,), float32
        Per-channel signal amplitude in dBm, averaged (power domain) across
        initiator and reflector, referenced to ``reference_power_level``.
        Formula: ``20·log10(|IQ| / 2048) + RPL``, then averaged in dB.

    phase_rad : ndarray, shape (N_ch,), float32
        Unwrapped channel-propagation phase in radians, referenced to the
        first channel (offset subtracted so phase_rad[0] == 0).
        Derived: ``np.unwrap(angle(coherent_sum_paths(iq_per_path))) / 2``.

    quality_flags : ndarray, shape (N_ch,), uint8
        Per-channel quality.  Value is the *worst* ToneQualityIndicator
        across all antenna paths and both devices:
        0=HIGH, 1=MEDIUM, 2=LOW, 3=UNAVAILABLE.
        Use for masking: ``cr[cr.quality_flags == QUALITY_HIGH]``.

    procedure_counter : int
        BLE CS procedure counter from the subevent header.

    role : Role
        COMBINED for objects built by ``from_subevent_pair()``.

    timestamp : float
        UNIX timestamp (seconds) recorded at construction time.
    """

    channels:          np.ndarray   # (N_ch,)          int32
    iq_per_path:       np.ndarray   # (N_ch, N_paths)  complex64
    amplitude_db:      np.ndarray   # (N_ch,)          float32
    phase_rad:         np.ndarray   # (N_ch,)          float32
    quality_flags:     np.ndarray   # (N_ch,)          uint8
    procedure_counter: int
    role:              Role
    timestamp:         float
    weights:           Optional[np.ndarray] = None  # (N_ch,) float32 amplitude-based weights

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        """Number of channels in this snapshot."""
        return int(self.channels.shape[0])

    @property
    def n_paths(self) -> int:
        """Number of antenna paths per channel."""
        return int(self.iq_per_path.shape[1]) if self.iq_per_path.ndim == 2 else 0

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Centre frequency (Hz) for each channel, shape (N_ch,), float64."""
        return (2402 + self.channels.astype(np.float64)) * 1e6

    # ------------------------------------------------------------------
    # Boolean indexing
    # ------------------------------------------------------------------

    def __getitem__(self, mask: np.ndarray) -> "ChannelResponse":
        """
        Return a new ChannelResponse restricted to channels where *mask* is True.

        Parameters
        ----------
        mask : ndarray bool, shape (N_ch,)
            Boolean selection array aligned to ``self.channels``.

        Returns
        -------
        ChannelResponse
            New object with all per-channel arrays indexed by *mask*.
            Scalar fields (procedure_counter, role, timestamp) are copied
            unchanged.

        Examples
        --------
        >>> good   = cr[cr.quality_flags == QUALITY_HIGH]
        >>> strong = cr[cr.amplitude_db > -70.0]
        >>> both   = cr[(cr.quality_flags == QUALITY_HIGH) & (cr.amplitude_db > -70.0)]
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_channels,):
            raise IndexError(
                f"Mask shape {mask.shape} does not match n_channels={self.n_channels}"
            )
        return ChannelResponse(
            channels=self.channels[mask].copy(),
            iq_per_path=self.iq_per_path[mask].copy(),
            amplitude_db=self.amplitude_db[mask].copy(),
            phase_rad=self.phase_rad[mask].copy(),
            quality_flags=self.quality_flags[mask].copy(),
            procedure_counter=self.procedure_counter,
            role=self.role,
            timestamp=self.timestamp,
            weights=self.weights[mask].copy() if self.weights is not None else None,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise to a numpy .npz archive.

        Complex IQ is split into real/imaginary float32 arrays because not
        all numpy versions round-trip complex64 through savez reliably.

        Parameters
        ----------
        path : str
            Destination path (.npz extension appended automatically if absent).
        """
        kwargs = {
            "channels": self.channels,
            "iq_real": self.iq_per_path.real.astype(np.float32),
            "iq_imag": self.iq_per_path.imag.astype(np.float32),
            "amplitude_db": self.amplitude_db,
            "phase_rad": self.phase_rad,
            "quality_flags": self.quality_flags,
            "procedure_counter": np.array(self.procedure_counter, dtype=np.int32),
            "role": np.array(int(self.role), dtype=np.int32),
            "timestamp": np.array(self.timestamp, dtype=np.float64),
        }
        if self.weights is not None:
            kwargs["weights"] = self.weights
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path: str) -> "ChannelResponse":
        """
        Deserialise from a .npz archive written by :meth:`save`.

        No parser or GUI modules are imported; safe to call from any context.

        Parameters
        ----------
        path : str
            Path to the .npz file.

        Returns
        -------
        ChannelResponse
        """
        d = np.load(path)
        iq = (d["iq_real"] + 1j * d["iq_imag"]).astype(np.complex64)
        weights = d["weights"] if "weights" in d else None
        return cls(
            channels=d["channels"].astype(np.int32),
            iq_per_path=iq,
            amplitude_db=d["amplitude_db"].astype(np.float32),
            phase_rad=d["phase_rad"].astype(np.float32),
            quality_flags=d["quality_flags"].astype(np.uint8),
            procedure_counter=int(d["procedure_counter"]),
            role=Role(int(d["role"])),
            timestamp=float(d["timestamp"]),
            weights=weights,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _empty(cls, procedure_counter: int, timestamp: float) -> "ChannelResponse":
        """Return a zero-channel ChannelResponse (used when no common channels exist)."""
        return cls(
            channels=np.empty(0, dtype=np.int32),
            iq_per_path=np.empty((0, 0), dtype=np.complex64),
            amplitude_db=np.empty(0, dtype=np.float32),
            phase_rad=np.empty(0, dtype=np.float32),
            quality_flags=np.empty(0, dtype=np.uint8),
            procedure_counter=procedure_counter,
            role=Role.COMBINED,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Factory: from subevent pair
    # ------------------------------------------------------------------

    @classmethod
    def from_subevent_pair(
        cls,
        initiator: "SubeventResults",
        reflector: "SubeventResults",
        *,
        timestamp: Optional[float] = None,
    ) -> "ChannelResponse":
        """
        Build a ChannelResponse from a matched initiator / reflector pair.

        Algorithm
        ---------
        1.  Extract per-path complex IQ phasors from all Mode-2 steps on each
            side, using only non-extension-slot tones.  When a channel appears
            in multiple steps within one subevent the step with the best mean
            tone quality is kept.
        2.  Retain only channels present on **both** sides.
        3.  Normalise path count to the global minimum (truncate if unequal).
        4.  Combined phasor per path:
                iq_combined[ch, p] = iq_ini[ch, p]  ×  iq_ref[ch, p]
            Phase adds (φ_ini + φ_ref), so device oscillator offsets cancel
            in the /2 step below.
        5.  MRC phase per channel:
                φ_mrc[ch] = angle( Σ_p  iq_combined[ch, p] )
            Coherent summation across paths (equal weights; each path
            contributes exp(jφ) proportional to its magnitude).
        6.  Unwrap φ_mrc across channels sorted by index, then divide by 2
            (one-way channel propagation phase), then subtract offset so that
            phase_rad[0] == 0.
        7.  Amplitude per side:
                amp_dBm = 20·log10( |IQ_mrc| / 2048 ) + RPL
            Averaged in dB across initiator and reflector (power-domain avg).
        8.  quality_flag per channel = max(worst_ini, worst_ref) where
            worst = max(ToneQualityIndicator.value) across paths.

        Parameters
        ----------
        initiator : SubeventResults
            Parsed CS subevent from the initiator device.
        reflector : SubeventResults
            Parsed CS subevent from the reflector device (same procedure counter).
        timestamp : float, optional
            Override construction timestamp (default: ``time.time()``).

        Returns
        -------
        ChannelResponse
            Empty (n_channels == 0) if no common Mode-2 channels with valid
            tones exist on both sides.
        """
        # Delayed import: keeps load() / __getitem__ free of parser deps.
        from toolset.cs_utils.cs_step import (
            CSStepMode2,
            ToneQualityIndicator,
            ToneQualityIndicatorExtensionSlot,
        )

        ts = timestamp if timestamp is not None else time.time()

        # ---- Step 1: per-channel, per-path IQ extraction ---------------
        ini_by_ch = _extract_mode2_iq(
            initiator.steps,
            initiator.reference_power_level,
            CSStepMode2,
            ToneQualityIndicator,
            ToneQualityIndicatorExtensionSlot,
        )
        ref_by_ch = _extract_mode2_iq(
            reflector.steps,
            reflector.reference_power_level,
            CSStepMode2,
            ToneQualityIndicator,
            ToneQualityIndicatorExtensionSlot,
        )

        # ---- Step 2: common channels ------------------------------------
        common_channels: List[int] = sorted(set(ini_by_ch) & set(ref_by_ch))
        if not common_channels:
            return cls._empty(
                procedure_counter=initiator.procedure_counter, timestamp=ts
            )

        # ---- Step 3: uniform path count (global minimum) ---------------
        n_paths = min(
            min(ini_by_ch[ch]["n_paths"] for ch in common_channels),
            min(ref_by_ch[ch]["n_paths"] for ch in common_channels),
        )
        if n_paths == 0:
            return cls._empty(
                procedure_counter=initiator.procedure_counter, timestamp=ts
            )

        N_ch = len(common_channels)
        iq_combined   = np.zeros((N_ch, n_paths), dtype=np.complex64)
        amplitude_db  = np.zeros(N_ch,            dtype=np.float32)
        quality_flags = np.zeros(N_ch,            dtype=np.uint8)

        for i, ch in enumerate(common_channels):
            ini_d = ini_by_ch[ch]
            ref_d = ref_by_ch[ch]

            # Truncate to global n_paths
            iq_i = ini_d["iq"][:n_paths]   # (n_paths,) complex64
            iq_r = ref_d["iq"][:n_paths]   # (n_paths,) complex64

            # ---- Step 4: combined phasor --------------------------------
            iq_combined[i] = iq_i * iq_r

            # ---- Step 7: amplitude (dB average across sides) ------------
            # Each side: 20·log10(|IQ_mrc| / 2048) + RPL
            mrc_ini = float(np.abs(np.sum(iq_i)))
            mrc_ref = float(np.abs(np.sum(iq_r)))
            amp_ini = _mag_to_dbm(mrc_ini, ini_d["rpl_dbm"])
            amp_ref = _mag_to_dbm(mrc_ref, ref_d["rpl_dbm"])
            amplitude_db[i] = np.float32(_avg_dbm(amp_ini, amp_ref))

            # ---- Step 8: quality flag (worst across paths and sides) ----
            quality_flags[i] = np.uint8(
                max(ini_d["worst_quality"], ref_d["worst_quality"])
            )

        # ---- Step 5: MRC phase per channel (coherent sum over paths) ---
        mrc_phasor = np.sum(iq_combined, axis=1)   # (N_ch,) complex64

        # ---- Step 6: unwrap then /2 then zero-reference ----------------
        raw_phase = np.angle(mrc_phasor).astype(np.float64)
        unwrapped = np.unwrap(raw_phase)
        phase_rad_f64 = unwrapped / 2.0
        phase_rad_f64 -= phase_rad_f64[0]           # reference to first channel

        return cls(
            channels=np.array(common_channels, dtype=np.int32),
            iq_per_path=iq_combined,
            amplitude_db=amplitude_db,
            phase_rad=phase_rad_f64.astype(np.float32),
            quality_flags=quality_flags,
            procedure_counter=initiator.procedure_counter,
            role=Role.COMBINED,
            timestamp=ts,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------

def _mag_to_dbm(magnitude: float, rpl_dbm: float) -> float:
    """Convert a raw IQ magnitude to dBm using the reference power level."""
    if magnitude <= 0.0:
        return float("-inf")
    return 20.0 * log(magnitude / 2048.0, 10) + rpl_dbm


def _avg_dbm(a_dbm: float, b_dbm: float) -> float:
    """Power-domain average of two dBm values."""
    a_mw = 10.0 ** (a_dbm / 10.0)
    b_mw = 10.0 ** (b_dbm / 10.0)
    return 10.0 * log((a_mw + b_mw) / 2.0, 10)


def _extract_mode2_iq(
    steps,
    rpl_dbm: float,
    CSStepMode2,
    ToneQualityIndicator,
    ToneQualityIndicatorExtensionSlot,
) -> Dict[int, dict]:
    """
    Walk *steps* and collect per-channel Mode-2 IQ data.

    Returns
    -------
    dict mapping channel_index -> {
        "iq":            np.ndarray (N_paths,) complex64   — one phasor per path
        "n_paths":       int
        "worst_quality": int   — max ToneQualityIndicator.value across paths
        "rpl_dbm":       float — passed through for amplitude computation
    }

    When a channel appears in multiple steps (firmware sends duplicates),
    the step whose tones have the lowest worst-quality value (i.e. the best
    quality) is kept.
    """
    by_ch: Dict[int, dict] = {}

    for step in steps:
        if not isinstance(step, CSStepMode2):
            continue
        if not step.tones:
            continue

        # Collect only real measurement tones (exclude unused extension slots)
        valid_tones = [
            t for t in step.tones
            if t.quality_extension_slot
            != ToneQualityIndicatorExtensionSlot.TONE_EXTENSION_NOT_EXPECTED
        ]
        if not valid_tones:
            continue

        # Build per-path complex phasor array
        iq_paths = np.array(
            [complex(t.pct_i, t.pct_q) for t in valid_tones],
            dtype=np.complex64,
        )

        worst_quality = int(
            max(t.quality.value for t in valid_tones)
        )

        ch = step.channel
        # Keep this step only if channel unseen or this step has better quality
        if ch not in by_ch or worst_quality < by_ch[ch]["worst_quality"]:
            by_ch[ch] = {
                "iq":            iq_paths,
                "n_paths":       len(valid_tones),
                "worst_quality": worst_quality,
                "rpl_dbm":       rpl_dbm,
            }

    return by_ch
