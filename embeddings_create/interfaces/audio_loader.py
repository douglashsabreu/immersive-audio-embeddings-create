"""Abstract interface for audio loading."""

from abc import ABC, abstractmethod
from pathlib import Path

from embeddings_create.models.audio_data import AudioData


class IAudioLoader(ABC):
    """Abstract interface for audio loading operations.

    This interface follows the Interface Segregation Principle by defining
    only audio loading related methods, separating concerns from feature
    extraction logic.
    """

    @abstractmethod
    def load_audio(self, file_path: Path) -> AudioData:
        """Load audio data from a file.

        Args:
            file_path: Path to the audio file

        Returns:
            AudioData object containing loaded audio and metadata

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or unsupported
            RuntimeError: If loading fails
        """

    @abstractmethod
    def load_audio_batch(self, file_paths: list[Path]) -> list[AudioData]:
        """Load multiple audio files in batch.

        Args:
            file_paths: List of paths to audio files

        Returns:
            List of AudioData objects

        Raises:
            ValueError: If any audio file is invalid
        """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Get list of supported audio formats.

        Returns:
            List of supported file extensions (e.g., ['.wav', '.mp3'])
        """

    @abstractmethod
    def validate_file(self, file_path: Path) -> bool:
        """Validate if file can be loaded by this loader.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file can be loaded
        """
