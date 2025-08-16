"""Abstract interfaces for spatial audio feature extraction."""

from .audio_loader import IAudioLoader
from .feature_extractor import IFeatureExtractor
from .result_saver import IResultSaver

__all__ = ["IAudioLoader", "IFeatureExtractor", "IResultSaver"]
