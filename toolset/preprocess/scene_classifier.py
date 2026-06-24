import math
import numpy as np
from typing import Optional

from toolset.processing.channel_response import ChannelResponse, QUALITY_UNAVAILABLE
from toolset.preprocess.config import PreprocessorConfig

# Scene classification labels
SCENE_LOS = 0
SCENE_MULTIPATH = 1
SCENE_NLOS = 2

SCENE_NAMES = {
    SCENE_LOS: "LOS",
    SCENE_MULTIPATH: "MULTIPATH",
    SCENE_NLOS: "NLOS"
}

class SceneClassifier:
    """
    Scene Classifier for BLE Channel Sounding environments.
    Classifies a preprocessed ChannelResponse into LOS, MULTIPATH, or NLOS scenes.
    """
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config if config is not None else PreprocessorConfig()
        
    def classify(self, cr: ChannelResponse) -> int:
        """
        Classifies the ChannelResponse and returns the scene label integer (0, 1, or 2).
        """
        if len(cr.channels) == 0:
            return SCENE_MULTIPATH

        # Feature 1: Null depth
        min_amp = np.min(cr.amplitude_db)
        median_amp = np.median(cr.amplitude_db)
        null_depth_db = float(median_amp - min_amp)

        # Feature 2: Phase residual RMS after linear fit
        ch = cr.channels
        phases = cr.phase_rad
        if len(ch) >= 2:
            freqs = (2402 + ch.astype(np.float64)) * 1e6
            slope, intercept = np.polyfit(freqs, phases, 1)
            fit_line = slope * freqs + intercept
            residual_rms = float(np.std(phases - fit_line))
        else:
            residual_rms = 0.0

        # Feature 3: Fraction of rejected tones
        rejected_count = np.sum(cr.quality_flags == QUALITY_UNAVAILABLE)
        reject_fraction = float(rejected_count / len(cr.channels)) if len(cr.channels) > 0 else 0.0

        # Decision logic based on configurable thresholds
        los_cond = (
            null_depth_db < self.config.null_depth_los_threshold and
            residual_rms < self.config.residual_rms_los_threshold and
            reject_fraction < self.config.reject_fraction_los_threshold
        )
        
        nlos_cond = (
            null_depth_db > self.config.null_depth_nlos_threshold and
            reject_fraction > self.config.reject_fraction_nlos_threshold
        )

        if los_cond:
            scene = SCENE_LOS
        elif nlos_cond:
            scene = SCENE_NLOS
        else:
            scene = SCENE_MULTIPATH

        return scene
