"""
tests/test_channel_response.py
================================
Unit tests for toolset.processing.channel_response.

All tests are self-contained — no UART, no GUI, no real SubeventResults
required.  Synthetic data is constructed directly so that every numerical
assertion is traceable by hand.
"""

import math
import tempfile
import os

import numpy as np
import pytest

from toolset.processing.channel_response import (
    ChannelResponse,
    Role,
    QUALITY_HIGH,
    QUALITY_MEDIUM,
    QUALITY_LOW,
    QUALITY_UNAVAILABLE,
    _mag_to_dbm,
    _avg_dbm,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic ChannelResponse objects
# ---------------------------------------------------------------------------

def _make_cr(
    n_channels: int = 8,
    n_paths: int = 2,
    *,
    procedure_counter: int = 0,
    role: Role = Role.COMBINED,
    timestamp: float = 0.0,
    quality_value: int = 0,
) -> ChannelResponse:
    """Return a ChannelResponse populated with deterministic synthetic data."""
    channels = np.arange(10, 10 + n_channels, dtype=np.int32)

    # IQ: unit magnitude, phase = 2π * channel_idx / n_channels per path
    angles = 2.0 * np.pi * np.arange(n_channels) / max(n_channels, 1)
    iq = np.exp(1j * angles[:, None]).astype(np.complex64)   # (N_ch, 1)
    iq = np.repeat(iq, n_paths, axis=1)                      # (N_ch, N_paths)

    amplitude_db = np.full(n_channels, -60.0, dtype=np.float32)

    # phase_rad matches the known angles (offset-subtracted)
    phase_rad = (angles - angles[0]).astype(np.float32)

    quality_flags = np.full(n_channels, quality_value, dtype=np.uint8)

    return ChannelResponse(
        channels=channels,
        iq_per_path=iq,
        amplitude_db=amplitude_db,
        phase_rad=phase_rad,
        quality_flags=quality_flags,
        procedure_counter=procedure_counter,
        role=role,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# 1. Shape contract
# ---------------------------------------------------------------------------

class TestShapeContract:

    def test_basic_shapes(self):
        N_ch, N_paths = 12, 3
        cr = _make_cr(N_ch, N_paths)

        assert cr.channels.shape       == (N_ch,)
        assert cr.iq_per_path.shape    == (N_ch, N_paths)
        assert cr.amplitude_db.shape   == (N_ch,)
        assert cr.phase_rad.shape      == (N_ch,)
        assert cr.quality_flags.shape  == (N_ch,)

    def test_dtypes(self):
        cr = _make_cr(8, 2)
        assert cr.channels.dtype      == np.int32
        assert cr.iq_per_path.dtype   == np.complex64
        assert cr.amplitude_db.dtype  == np.float32
        assert cr.phase_rad.dtype     == np.float32
        assert cr.quality_flags.dtype == np.uint8

    def test_n_channels_property(self):
        cr = _make_cr(7, 2)
        assert cr.n_channels == 7

    def test_n_paths_property(self):
        cr = _make_cr(5, 4)
        assert cr.n_paths == 4

    def test_frequencies_hz(self):
        cr = _make_cr(3, 1)
        # channels are [10, 11, 12] → [2412, 2413, 2414] MHz
        expected = np.array([2412e6, 2413e6, 2414e6])
        np.testing.assert_array_equal(cr.frequencies_hz, expected)


# ---------------------------------------------------------------------------
# 2. Known phase / amplitude values
# ---------------------------------------------------------------------------

class TestKnownValues:

    def test_phase_values_match_construction(self):
        """phase_rad[i] == 2π·i/N (offset-subtracted)."""
        N = 8
        cr = _make_cr(N, 2)
        expected = np.array(
            [2.0 * math.pi * i / N - 0.0 for i in range(N)],
            dtype=np.float32,
        )
        np.testing.assert_allclose(cr.phase_rad, expected, rtol=1e-5)

    def test_phase_first_channel_is_zero(self):
        cr = _make_cr(10, 2)
        assert cr.phase_rad[0] == pytest.approx(0.0, abs=1e-6)

    def test_amplitude_values(self):
        cr = _make_cr(5, 2)
        np.testing.assert_allclose(cr.amplitude_db, -60.0, rtol=1e-5)

    def test_quality_flags_value(self):
        cr = _make_cr(6, 2, quality_value=int(QUALITY_MEDIUM))
        assert np.all(cr.quality_flags == QUALITY_MEDIUM)

    def test_procedure_counter_preserved(self):
        cr = _make_cr(4, 1, procedure_counter=42)
        assert cr.procedure_counter == 42

    def test_role_preserved(self):
        cr = _make_cr(4, 1, role=Role.COMBINED)
        assert cr.role == Role.COMBINED

    def test_timestamp_preserved(self):
        cr = _make_cr(4, 1, timestamp=1234567.89)
        assert cr.timestamp == pytest.approx(1234567.89)


# ---------------------------------------------------------------------------
# 3. Boolean indexing
# ---------------------------------------------------------------------------

class TestBooleanIndexing:

    def test_all_true_mask_returns_full_copy(self):
        cr = _make_cr(8, 2)
        mask = np.ones(8, dtype=bool)
        cr2 = cr[mask]
        assert cr2.n_channels == 8
        np.testing.assert_array_equal(cr2.channels, cr.channels)

    def test_all_false_mask_returns_empty(self):
        cr = _make_cr(8, 2)
        mask = np.zeros(8, dtype=bool)
        cr2 = cr[mask]
        assert cr2.n_channels == 0
        assert cr2.iq_per_path.shape == (0, 2)

    def test_partial_mask_correct_channels(self):
        cr = _make_cr(8, 2)
        mask = cr.quality_flags == QUALITY_HIGH   # all True here
        cr2 = cr[mask]
        assert cr2.n_channels == 8

    def test_quality_filter_high_only(self):
        """Channels with MEDIUM quality are excluded by HIGH-only filter."""
        cr = _make_cr(6, 2, quality_value=int(QUALITY_HIGH))
        # Manually degrade channels 1, 3, 5
        cr.quality_flags[[1, 3, 5]] = QUALITY_MEDIUM
        good = cr[cr.quality_flags == QUALITY_HIGH]
        assert good.n_channels == 3
        np.testing.assert_array_equal(good.channels, cr.channels[[0, 2, 4]])

    def test_amplitude_filter(self):
        cr = _make_cr(8, 2)
        cr.amplitude_db[[0, 2, 4, 6]] = np.float32(-80.0)  # below threshold
        strong = cr[cr.amplitude_db > -70.0]
        assert strong.n_channels == 4

    def test_combined_mask(self):
        cr = _make_cr(8, 2)
        cr.quality_flags[0] = QUALITY_LOW
        cr.amplitude_db[7]  = np.float32(-90.0)
        mask = (cr.quality_flags == QUALITY_HIGH) & (cr.amplitude_db > -70.0)
        cr2 = cr[mask]
        assert cr2.n_channels == 6    # channels 1-6 pass both filters

    def test_indexing_returns_new_object(self):
        """Mutation of indexed result does not affect original."""
        cr = _make_cr(4, 2)
        cr2 = cr[np.ones(4, dtype=bool)]
        cr2.amplitude_db[:] = -999.0
        assert not np.any(cr.amplitude_db == -999.0)

    def test_mask_shape_mismatch_raises(self):
        cr = _make_cr(4, 2)
        with pytest.raises(IndexError):
            cr[np.ones(5, dtype=bool)]

    def test_scalar_fields_preserved_after_mask(self):
        cr = _make_cr(5, 2, procedure_counter=77, timestamp=9.9)
        cr2 = cr[np.array([True, False, True, False, True])]
        assert cr2.procedure_counter == 77
        assert cr2.timestamp == pytest.approx(9.9)
        assert cr2.role == Role.COMBINED

    def test_iq_shape_preserved_after_mask(self):
        cr = _make_cr(6, 3)
        cr2 = cr[np.array([True, True, False, True, False, False])]
        assert cr2.iq_per_path.shape == (3, 3)   # 3 channels kept, 3 paths


# ---------------------------------------------------------------------------
# 4. Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:

    def _round_trip(self, cr: ChannelResponse) -> ChannelResponse:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test_cr")
            cr.save(path)
            return ChannelResponse.load(path + ".npz")

    def test_round_trip_shapes(self):
        cr = _make_cr(10, 2)
        cr2 = self._round_trip(cr)
        assert cr2.channels.shape      == cr.channels.shape
        assert cr2.iq_per_path.shape   == cr.iq_per_path.shape
        assert cr2.amplitude_db.shape  == cr.amplitude_db.shape
        assert cr2.phase_rad.shape     == cr.phase_rad.shape
        assert cr2.quality_flags.shape == cr.quality_flags.shape

    def test_round_trip_values(self):
        cr = _make_cr(8, 2)
        cr2 = self._round_trip(cr)
        np.testing.assert_array_equal(cr2.channels, cr.channels)
        np.testing.assert_allclose(cr2.phase_rad,     cr.phase_rad,     rtol=1e-5)
        np.testing.assert_allclose(cr2.amplitude_db,  cr.amplitude_db,  rtol=1e-5)
        np.testing.assert_array_equal(cr2.quality_flags, cr.quality_flags)

    def test_round_trip_complex_iq(self):
        cr = _make_cr(6, 3)
        cr2 = self._round_trip(cr)
        np.testing.assert_allclose(cr2.iq_per_path.real, cr.iq_per_path.real, atol=1e-6)
        np.testing.assert_allclose(cr2.iq_per_path.imag, cr.iq_per_path.imag, atol=1e-6)

    def test_round_trip_scalars(self):
        cr = _make_cr(5, 2, procedure_counter=123, role=Role.INITIATOR, timestamp=5555.5)
        cr2 = self._round_trip(cr)
        assert cr2.procedure_counter == 123
        assert cr2.role              == Role.INITIATOR
        assert cr2.timestamp         == pytest.approx(5555.5)

    def test_round_trip_empty(self):
        cr = ChannelResponse._empty(procedure_counter=0, timestamp=0.0)
        cr2 = self._round_trip(cr)
        assert cr2.n_channels == 0


# ---------------------------------------------------------------------------
# 5. Empty ChannelResponse
# ---------------------------------------------------------------------------

class TestEmpty:

    def test_empty_shape(self):
        cr = ChannelResponse._empty(procedure_counter=7, timestamp=0.0)
        assert cr.n_channels == 0
        assert cr.channels.shape      == (0,)
        assert cr.iq_per_path.shape   == (0, 0)
        assert cr.amplitude_db.shape  == (0,)
        assert cr.phase_rad.shape     == (0,)
        assert cr.quality_flags.shape == (0,)

    def test_empty_indexing(self):
        cr = ChannelResponse._empty(procedure_counter=0, timestamp=0.0)
        cr2 = cr[np.array([], dtype=bool)]
        assert cr2.n_channels == 0


# ---------------------------------------------------------------------------
# 6. Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_mag_to_dbm_known_value(self):
        # mag = 2048 → 20·log10(1) + rpl = 0 + rpl
        assert _mag_to_dbm(2048.0, -16.0) == pytest.approx(-16.0, abs=1e-6)

    def test_mag_to_dbm_zero_returns_neginf(self):
        assert _mag_to_dbm(0.0, 0.0) == float("-inf")

    def test_avg_dbm_equal_values(self):
        # Average of (x, x) should equal x
        assert _avg_dbm(-50.0, -50.0) == pytest.approx(-50.0, abs=1e-6)

    def test_avg_dbm_symmetry(self):
        assert _avg_dbm(-40.0, -60.0) == pytest.approx(_avg_dbm(-60.0, -40.0), abs=1e-9)


# ---------------------------------------------------------------------------
# 7. from_subevent_pair — synthetic SubeventResults
# ---------------------------------------------------------------------------

def _make_fake_subevent(channels: list[int], n_paths: int, procedure_counter: int = 0,
                        rpl: int = -16, quality: int = 0):
    """
    Build a minimal synthetic SubeventResults with Mode-2 steps only.
    Tones have unit magnitude and incrementing phase per path.
    """
    # Local imports to avoid requiring the full parser at module import time.
    from toolset.cs_utils.cs_subevent import SubeventResults, ProcedureDoneStatus, SubeventDoneStatus, ProcedureAbortReason, SubeventAbortReason
    from toolset.cs_utils.cs_step import (
        CSStepMode2, CSMode, ToneData,
        ToneQualityIndicator, ToneQualityIndicatorExtensionSlot,
    )

    steps = []
    for ch in channels:
        tones = []
        for p in range(n_paths):
            angle = 2.0 * math.pi * ch / 80.0 + p * 0.1
            i_val = int(2047 * math.cos(angle))
            q_val = int(2047 * math.sin(angle))
            tones.append(ToneData(
                pct_i=i_val,
                pct_q=q_val,
                quality=ToneQualityIndicator(quality),
                quality_extension_slot=ToneQualityIndicatorExtensionSlot.NOT_TONE_EXTENSION_SLOT,
            ))
        steps.append(CSStepMode2(
            mode=CSMode.MODE_2,
            channel=ch,
            antenna_permutation_index=0,
            tones=tones,
        ))

    return SubeventResults(
        procedure_counter=procedure_counter,
        reference_power_level=rpl,
        procedure_done_status=ProcedureDoneStatus.PROC_ALL_RESULTS_COMPLETED,
        subevent_done_status=SubeventDoneStatus.SUBEVENT_ALL_RESULTS_COMPLETED,
        procedure_abort_reason=ProcedureAbortReason.PROC_NO_ABORT,
        subevent_abort_reason=SubeventAbortReason.SUBEVENT_NO_ABORT,
        num_steps_reported=len(steps),
        steps=steps,
    )


class TestFromSubeventPair:

    def test_shape_from_pair(self):
        channels = [10, 11, 12, 13, 14]
        ini = _make_fake_subevent(channels, n_paths=2, procedure_counter=1)
        ref = _make_fake_subevent(channels, n_paths=2, procedure_counter=1)
        cr = ChannelResponse.from_subevent_pair(ini, ref)

        assert cr.n_channels            == len(channels)
        assert cr.n_paths               == 2
        assert cr.channels.shape        == (len(channels),)
        assert cr.iq_per_path.shape     == (len(channels), 2)
        assert cr.amplitude_db.shape    == (len(channels),)
        assert cr.phase_rad.shape       == (len(channels),)
        assert cr.quality_flags.shape   == (len(channels),)

    def test_procedure_counter_from_initiator(self):
        channels = [20, 21, 22]
        ini = _make_fake_subevent(channels, n_paths=1, procedure_counter=7)
        ref = _make_fake_subevent(channels, n_paths=1, procedure_counter=7)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.procedure_counter == 7

    def test_role_is_combined(self):
        channels = [5, 6, 7]
        ini = _make_fake_subevent(channels, n_paths=1)
        ref = _make_fake_subevent(channels, n_paths=1)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.role == Role.COMBINED

    def test_channels_sorted(self):
        ini = _make_fake_subevent([15, 10, 12], n_paths=1)
        ref = _make_fake_subevent([10, 15, 12], n_paths=1)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert list(cr.channels) == [10, 12, 15]

    def test_phase_first_channel_is_zero(self):
        channels = list(range(10, 20))
        ini = _make_fake_subevent(channels, n_paths=2)
        ref = _make_fake_subevent(channels, n_paths=2)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.phase_rad[0] == pytest.approx(0.0, abs=1e-5)

    def test_no_common_channels_returns_empty(self):
        ini = _make_fake_subevent([10, 11], n_paths=1)
        ref = _make_fake_subevent([20, 21], n_paths=1)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.n_channels == 0

    def test_partial_overlap_keeps_common_only(self):
        ini = _make_fake_subevent([10, 11, 12, 13], n_paths=1)
        ref = _make_fake_subevent([11, 12, 14, 15], n_paths=1)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert list(cr.channels) == [11, 12]

    def test_path_count_truncated_to_minimum(self):
        """If initiator has 3 paths but reflector has 2, result has 2."""
        channels = [10, 11, 12]
        ini = _make_fake_subevent(channels, n_paths=3)
        ref = _make_fake_subevent(channels, n_paths=2)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.n_paths == 2

    def test_quality_flag_worst_wins(self):
        """If one side is MEDIUM quality, the flag should be MEDIUM."""
        channels = [30, 31, 32]
        ini = _make_fake_subevent(channels, n_paths=1, quality=int(QUALITY_HIGH))
        ref = _make_fake_subevent(channels, n_paths=1, quality=int(QUALITY_MEDIUM))
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert np.all(cr.quality_flags == QUALITY_MEDIUM)

    def test_quality_filter_after_from_pair(self):
        """Boolean indexing works correctly on a from_subevent_pair result."""
        channels = [10, 11, 12, 13, 14]
        ini = _make_fake_subevent(channels, n_paths=1, quality=int(QUALITY_HIGH))
        ref = _make_fake_subevent(channels, n_paths=1, quality=int(QUALITY_HIGH))
        cr  = ChannelResponse.from_subevent_pair(ini, ref)

        # Manually degrade two channels post-construction
        cr.quality_flags[[1, 3]] = QUALITY_LOW
        good = cr[cr.quality_flags == QUALITY_HIGH]
        assert good.n_channels == 3
        assert list(good.channels) == [10, 12, 14]

    def test_iq_dtype_is_complex64(self):
        channels = [40, 41, 42]
        ini = _make_fake_subevent(channels, n_paths=2)
        ref = _make_fake_subevent(channels, n_paths=2)
        cr  = ChannelResponse.from_subevent_pair(ini, ref)
        assert cr.iq_per_path.dtype == np.complex64
