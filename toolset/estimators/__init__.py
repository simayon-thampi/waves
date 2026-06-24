"""
estimators
==========
Range-estimation adapters for BLE Channel Sounding.

Each adapter wraps an existing algorithm module as a thin, GUI-free class
that accepts a :class:`~toolset.processing.channel_response.ChannelResponse`
and returns an :class:`EstimatorResult`.
"""

from .base import Estimator, EstimatorResult
from .phase_slope import PhaseSlopeEstimator
from .ifft import IFFTEstimator
from .music import MUSICEstimator
from .weighted_ls import WeightedLSEstimator
from .ema import EMAEstimator

__all__ = [
    "Estimator",
    "EstimatorResult",
    "PhaseSlopeEstimator",
    "IFFTEstimator",
    "MUSICEstimator",
    "WeightedLSEstimator",
    "EMAEstimator",
]
