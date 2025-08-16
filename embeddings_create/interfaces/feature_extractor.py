"""Abstract interface for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any

from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.extraction_result import FeatureResult
from embeddings_create.models.feature_config import FeatureConfig


class IFeatureExtractor(ABC):
    """Abstract interface for spatial audio feature extractors.

    This interface follows the Interface Segregation Principle by defining
    only the essential methods needed for feature extraction. Each concrete
    implementation will handle a specific type of spatial audio feature.
    """

    @abstractmethod
    def extract(self, audio_data: AudioData, config: FeatureConfig) -> FeatureResult:
        """Extract spatial features from audio data.

        Args:
            audio_data: The input audio data to process
            config: Configuration parameters for extraction

        Returns:
            FeatureResult containing extracted features and metadata

        Raises:
            ValueError: If audio_data or config is invalid
            RuntimeError: If extraction fails
        """

    @abstractmethod
    def validate_audio_data(self, audio_data: AudioData) -> bool:
        """Validate that audio data is compatible with this extractor.

        Args:
            audio_data: Audio data to validate

        Returns:
            True if audio data is valid for this extractor
        """

    @abstractmethod
    def validate_config(self, config: FeatureConfig) -> bool:
        """Validate that configuration is compatible with this extractor.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid for this extractor
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this feature extractor."""

    @property
    @abstractmethod
    def required_channels(self) -> int:
        """Get the required number of audio channels for this extractor."""

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration parameters for this extractor.

        Returns:
            Dictionary with default configuration parameters
        """
        return {}
