"""Audio loading utility implementation."""

import concurrent.futures
from pathlib import Path

import librosa
import soundfile as sf

from embeddings_create.interfaces.audio_loader import IAudioLoader
from embeddings_create.models.audio_data import AudioData


class AudioLoader(IAudioLoader):
    """Concrete implementation of audio loading functionality.

    This class handles loading audio files from disk and converting them
    to the internal AudioData format. It supports multiple audio formats
    and provides both single and batch loading capabilities.

    This implementation follows the Single Responsibility Principle by
    focusing only on audio loading operations.
    """

    def __init__(self, target_sample_rate: int | None = None, mono: bool = False):
        """Initialize audio loader.

        Args:
            target_sample_rate: Target sample rate for loaded audio (None to keep original)
            mono: Whether to convert stereo to mono
        """
        self._target_sample_rate = target_sample_rate
        self._mono = mono
        self._supported_formats = [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"]

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
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not self.validate_file(file_path):
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")

        try:
            signal, sample_rate = librosa.load(
                str(file_path), sr=self._target_sample_rate, mono=self._mono, res_type="soxr_hq"
            )

            if signal.ndim == 1 and not self._mono:
                signal = signal.reshape(1, -1)

            return AudioData(
                signal=signal,
                sample_rate=sample_rate,
                file_path=file_path,
                duration=(
                    len(signal) / sample_rate if signal.ndim == 1 else signal.shape[1] / sample_rate
                ),
                channels=1 if signal.ndim == 1 else signal.shape[0],
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {file_path}: {e!s}") from e

    def load_audio_batch(self, file_paths: list[Path]) -> list[AudioData]:
        """Load multiple audio files in batch.

        Args:
            file_paths: List of paths to audio files

        Returns:
            List of AudioData objects

        Raises:
            ValueError: If any audio file is invalid
        """
        audio_data_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(self.load_audio, path): path for path in file_paths}

            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    audio_data = future.result()
                    audio_data_list.append(audio_data)
                except Exception as e:
                    raise ValueError(f"Failed to load {path}: {e!s}") from e

        audio_data_list.sort(key=lambda x: str(x.file_path))
        return audio_data_list

    def supported_formats(self) -> list[str]:
        """Get list of supported audio formats.

        Returns:
            List of supported file extensions
        """
        return self._supported_formats.copy()

    def validate_file(self, file_path: Path) -> bool:
        """Validate if file can be loaded by this loader.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file can be loaded
        """
        if not file_path.exists():
            return False

        if file_path.suffix.lower() not in self._supported_formats:
            return False

        try:
            info = sf.info(str(file_path))
            return bool(info.frames > 0 and info.samplerate > 0)
        except Exception:
            return False
