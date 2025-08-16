"""Base class for feature extractors."""

import time
from abc import abstractmethod
from typing import Any

import librosa
import numpy as np

from embeddings_create.interfaces.feature_extractor import IFeatureExtractor
from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.extraction_result import FeatureResult
from embeddings_create.models.feature_config import FeatureConfig, FeatureType


class BaseFeatureExtractor(IFeatureExtractor):
    """Base implementation for feature extractors.

    This class provides common functionality for all feature extractors,
    following the Template Method pattern and DRY principle. Concrete
    extractors only need to implement the specific extraction logic.
    """

    def __init__(self, feature_type: FeatureType):
        """Initialize base extractor.

        Args:
            feature_type: The type of feature this extractor handles
        """
        self._feature_type = feature_type

    def extract(self, audio_data: AudioData, config: FeatureConfig) -> FeatureResult:
        """Extract spatial features from audio data.

        This method implements the Template Method pattern, providing
        the overall extraction workflow while delegating specific
        extraction logic to concrete implementations.
        """
        if not self.validate_audio_data(audio_data):
            raise ValueError(f"Audio data is not valid for {self.name} extractor")

        if not self.validate_config(config):
            raise ValueError(f"Configuration is not valid for {self.name} extractor")

        start_time = time.time()

        try:
            features = self._extract_features(audio_data, config)
            metadata = self._generate_metadata(audio_data, config, features)

            extraction_time = time.time() - start_time

            return FeatureResult(
                feature_type=self._feature_type,
                features=features,
                metadata=metadata,
                extraction_time=extraction_time,
                config_used=self._config_to_dict(config),
            )

        except Exception as e:
            raise RuntimeError(f"Feature extraction failed for {self.name}: {e!s}") from e

    def validate_audio_data(self, audio_data: AudioData) -> bool:
        """Validate audio data for this extractor."""
        if audio_data.signal is None:
            return False

        if audio_data.channels != self.required_channels:
            return False

        if audio_data.sample_rate <= 0:
            return False

        return True

    def validate_config(self, config: FeatureConfig) -> bool:
        """Validate configuration for this extractor."""
        if config.feature_type != self._feature_type:
            return False

        if config.n_fft <= 0 or config.hop_length <= 0:
            return False

        if config.n_mels <= 0:
            return False

        return True

    @abstractmethod
    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract features - to be implemented by concrete classes.

        Args:
            audio_data: The input audio data
            config: Extraction configuration

        Returns:
            Extracted features as numpy array
        """

    def _generate_metadata(
        self, audio_data: AudioData, config: FeatureConfig, features: np.ndarray
    ) -> dict[str, Any]:
        """Generate metadata for extraction result.

        Args:
            audio_data: Input audio data
            config: Extraction configuration
            features: Extracted features

        Returns:
            Dictionary containing metadata
        """
        return {
            "input_shape": audio_data.signal.shape,
            "output_shape": features.shape,
            "sample_rate": audio_data.sample_rate,
            "duration": audio_data.duration,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "n_mels": config.n_mels,
            "use_pcen": config.use_pcen,
            "extractor": self.name,
        }

    def _config_to_dict(self, config: FeatureConfig) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Args:
            config: Feature configuration

        Returns:
            Dictionary representation of configuration
        """
        return {
            "feature_type": config.feature_type.value,
            "n_fft": config.n_fft,
            "hop_length": config.hop_length,
            "n_mels": config.n_mels,
            "use_pcen": config.use_pcen,
            "sample_rate": config.sample_rate,
            "parameters": config.parameters,
        }

    def _compute_stft(self, signal: np.ndarray, config: FeatureConfig) -> np.ndarray:
        """Compute Short-Time Fourier Transform.

        Args:
            signal: Input signal
            config: Configuration with STFT parameters

        Returns:
            Complex STFT coefficients
        """
        return librosa.stft(
            signal,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )

    def _compute_mel_spectrogram(self, stft: np.ndarray, config: FeatureConfig) -> np.ndarray:
        """Compute mel-scale spectrogram.

        Args:
            stft: STFT coefficients
            config: Configuration with mel parameters

        Returns:
            Mel-scale spectrogram
        """
        magnitude = np.abs(stft)

        mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=0.0,
            fmax=config.sample_rate // 2,
        )

        mel_spec = np.dot(mel_basis, magnitude)

        if config.use_pcen:
            mel_spec = librosa.pcen(mel_spec)
        else:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec
