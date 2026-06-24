import collections
import copy
from typing import List, Optional
import numpy as np
from toolset.processing.channel_response import ChannelResponse

class TemporalSmoother:
    """
    Spatio-Temporal Smoother for BLE Channel Sounding sweeps.
    Performs coherent (complex) averaging over a sliding window of N raw sweeps.
    """
    def __init__(self, window_size: int = 3):
        self.window_size = max(1, window_size)
        self.window = collections.deque(maxlen=self.window_size)
        
    def set_window_size(self, window_size: int):
        self.window_size = max(1, window_size)
        # Resize deque while keeping existing elements if possible
        self.window = collections.deque(self.window, maxlen=self.window_size)
        
    def clear(self):
        self.window.clear()
        
    def process(self, cr: ChannelResponse) -> ChannelResponse:
        """
        Pushes a new raw ChannelResponse into the window and returns
        the coherently averaged (smoothed) ChannelResponse.
        """
        if self.window_size == 1:
            # Reproduce single-subevent behavior exactly
            # We can use shallow copy or deepcopy to avoid mutating original
            cr_copy = copy.copy(cr)
            cr_copy.metadata = {
                "smoothed": False,
                "window_size": 1,
                "active_size": 1
            }
            return cr_copy
            
        self.window.append(cr)
        active_size = len(self.window)
        
        # Coherent average:
        # H_i[k] = 10^(amplitude_db/20) * exp(j * phase_rad)
        ref_cr = self.window[0]
        channels = ref_cr.channels.copy()
        
        sum_iq = np.zeros_like(ref_cr.iq_per_path, dtype=np.complex128)
        sum_linear_h = np.zeros_like(ref_cr.phase_rad, dtype=np.complex128)
        
        for item in self.window:
            if len(item.channels) != len(channels) or not np.array_equal(item.channels, channels):
                # Channel set mismatch: flush and start over with current sweep
                self.window.clear()
                self.window.append(cr)
                cr_copy = copy.copy(cr)
                cr_copy.metadata = {
                    "smoothed": True,
                    "window_size": self.window_size,
                    "active_size": 1
                }
                return cr_copy
                
            sum_iq += item.iq_per_path.astype(np.complex128)
            
            # Construct complex phasor
            lin_amp = 10.0 ** (item.amplitude_db.astype(np.float64) / 20.0)
            phasor = lin_amp * np.exp(1j * item.phase_rad.astype(np.float64))
            sum_linear_h += phasor

        avg_iq = (sum_iq / active_size).astype(np.complex64)
        avg_h = sum_linear_h / active_size
        
        # Smooth amplitude in dB
        avg_mag = np.abs(avg_h)
        avg_mag = np.maximum(avg_mag, 1e-10)  # avoid log10(0)
        avg_amp = (20.0 * np.log10(avg_mag)).astype(np.float32)
        
        # Smooth phase angle in radians
        avg_phase = np.angle(avg_h).astype(np.float32)
        
        latest_cr = self.window[-1]
        
        smoothed = ChannelResponse(
            channels=channels,
            iq_per_path=avg_iq,
            amplitude_db=avg_amp,
            phase_rad=avg_phase,
            quality_flags=latest_cr.quality_flags.copy(),
            procedure_counter=latest_cr.procedure_counter,
            role=latest_cr.role,
            timestamp=latest_cr.timestamp,
            weights=latest_cr.weights.copy() if latest_cr.weights is not None else None
        )
        
        smoothed.metadata = {
            "smoothed": True,
            "window_size": self.window_size,
            "active_size": active_size
        }
        return smoothed
