"""Feature extraction configuration models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureType(Enum):
    """Types of spatial audio features that can be extracted."""

    SALSA = "salsa"
    SALSA_LITE = "salsa_lite"
    INTENSITY_DOA = "intensity_doa"
    DIFFUSENESS = "diffuseness"
    IACC = "iacc"


@dataclass
class FeatureConfig:
    """Configuration for feature extraction.

    This class follows the single responsibility principle by only
    handling configuration data for feature extraction parameters.
    """

    feature_type: FeatureType
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    use_pcen: bool = True
    sample_rate: int = 44100
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


@dataclass
class ExtractionConfig:
    """Configuration for the entire extraction process.

    Manages multiple feature configurations and global settings
    following the single responsibility principle.
    """

    feature_configs: list[FeatureConfig]
    output_directory: str | None = None
    save_intermediate: bool = False
    parallel_processing: bool = True
    max_workers: int | None = None
    verbose: bool = False

    def get_config_by_type(self, feature_type: FeatureType) -> FeatureConfig | None:
        """Get configuration for specific feature type."""
        for config in self.feature_configs:
            if config.feature_type == feature_type:
                return config
        return None
