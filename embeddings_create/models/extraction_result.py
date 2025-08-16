"""Extraction result models."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .feature_config import FeatureType


@dataclass
class FeatureResult:
    """Result of a single feature extraction.

    Encapsulates the output of a feature extractor along with metadata,
    following the single responsibility principle.
    """

    feature_type: FeatureType
    features: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_time: float | None = None
    config_used: dict[str, Any] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape of extracted features."""
        return tuple(self.features.shape)

    @property
    def size(self) -> int:
        """Get total size of extracted features."""
        return int(self.features.size)


@dataclass
class ExtractionResult:
    """Complete result of spatial audio feature extraction.

    Contains all extracted features and metadata from a single audio file,
    following the single responsibility principle for result aggregation.
    """

    source_file: Path
    features: dict[FeatureType, FeatureResult] = field(default_factory=dict)
    total_extraction_time: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str | None = None

    def add_feature_result(self, result: FeatureResult) -> None:
        """Add a feature extraction result."""
        self.features[result.feature_type] = result

    def get_feature(self, feature_type: FeatureType) -> FeatureResult | None:
        """Get specific feature result."""
        return self.features.get(feature_type)

    def has_feature(self, feature_type: FeatureType) -> bool:
        """Check if specific feature was extracted."""
        return feature_type in self.features

    @property
    def extracted_features(self) -> list[FeatureType]:
        """Get list of successfully extracted features."""
        return list(self.features.keys())

    @property
    def is_successful(self) -> bool:
        """Check if extraction was successful."""
        return self.success and len(self.features) > 0
