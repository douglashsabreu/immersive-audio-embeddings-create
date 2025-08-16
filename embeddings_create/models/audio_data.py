"""Audio data model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioData:
    """Represents audio data with metadata.

    This class encapsulates audio signal data along with its metadata,
    following the single responsibility principle by only handling
    audio data representation.
    """

    signal: np.ndarray
    sample_rate: int
    file_path: Path | None = None
    duration: float | None = None
    channels: int | None = None

    def __post_init__(self) -> None:
        """Initialize computed properties after object creation."""
        if self.duration is None and self.signal is not None:
            self.duration = len(self.signal) / self.sample_rate

        if self.channels is None and self.signal is not None:
            self.channels = self.signal.shape[0] if self.signal.ndim > 1 else 1

    @property
    def is_binaural(self) -> bool:
        """Check if audio data is binaural (2 channels)."""
        return self.channels == 2

    @property
    def is_foa(self) -> bool:
        """Check if audio data is First Order Ambisonic (4 channels)."""
        return self.channels == 4

    @property
    def samples(self) -> int:
        """Get total number of samples."""
        return int(len(self.signal) if self.signal.ndim == 1 else self.signal.shape[1])
