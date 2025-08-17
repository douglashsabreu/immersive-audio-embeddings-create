"""Pipeline interfaces following SOLID principles."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class IFeatureExtractor(ABC):
    """Interface for feature extraction operations."""

    @abstractmethod
    def extract_features(
        self,
        input_dir: Path,
        output_dir: Path,
        max_files: Optional[int] = None,
        file_pattern: str = "*.wav",
    ) -> Dict[str, Any]:
        """Extract features from audio files.

        Returns:
            Dictionary with extraction results and statistics
        """
        pass


class IDatasetOrganizer(ABC):
    """Interface for dataset organization operations."""

    @abstractmethod
    def organize_dataset(self, features_dir: Path, output_file: Path) -> Dict[str, Any]:
        """Organize features into structured dataset.

        Returns:
            Dictionary with organization results and statistics
        """
        pass


class IEmbeddingGenerator(ABC):
    """Interface for embedding generation operations."""

    @abstractmethod
    def generate_embeddings(
        self,
        dataset_file: Path,
        output_file: Path,
        embedding_dim: int = 128,
        model_path: Optional[Path] = None,
        generate_visualizations: bool = True,
        viz_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate embeddings from organized dataset.

        Returns:
            Dictionary with generation results and statistics
        """
        pass


class IPipelineOrchestrator(ABC):
    """Interface for complete pipeline orchestration."""

    @abstractmethod
    def run_complete_pipeline(
        self,
        input_dir: Path,
        features_dir: Path,
        dataset_file: Path,
        embeddings_file: Path,
        max_files: Optional[int] = None,
        embedding_dim: int = 128,
        visualize: bool = True,
    ) -> Dict[str, Any]:
        """Run complete pipeline from audio to embeddings.

        Returns:
            Dictionary with complete pipeline results
        """
        pass


class ILogger(ABC):
    """Interface for logging operations."""

    @abstractmethod
    def info(self, message: str) -> None:
        """Log info message."""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def success(self, message: str) -> None:
        """Log success message."""
        pass
