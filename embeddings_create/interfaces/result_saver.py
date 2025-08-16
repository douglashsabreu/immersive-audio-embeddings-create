"""Abstract interface for saving extraction results."""

from abc import ABC, abstractmethod
from pathlib import Path

from embeddings_create.models.extraction_result import ExtractionResult


class IResultSaver(ABC):
    """Abstract interface for saving feature extraction results.

    This interface follows the Interface Segregation Principle by defining
    only result saving operations, keeping concerns separate from extraction
    and loading logic.
    """

    @abstractmethod
    def save_result(self, result: ExtractionResult, output_path: Path) -> bool:
        """Save a single extraction result.

        Args:
            result: The extraction result to save
            output_path: Path where to save the result

        Returns:
            True if saving was successful

        Raises:
            IOError: If saving fails
            ValueError: If result or output_path is invalid
        """

    @abstractmethod
    def save_batch_results(
        self, results: list[ExtractionResult], output_directory: Path
    ) -> list[bool]:
        """Save multiple extraction results.

        Args:
            results: List of extraction results to save
            output_directory: Directory where to save results

        Returns:
            List of boolean values indicating success for each result

        Raises:
            IOError: If directory creation or saving fails
        """

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Get list of supported output formats.

        Returns:
            List of supported file extensions (e.g., ['.npz', '.json'])
        """

    @abstractmethod
    def create_output_filename(self, source_file: Path, feature_type: str) -> str:
        """Create output filename for a specific feature type.

        Args:
            source_file: Original audio file path
            feature_type: Type of feature extracted

        Returns:
            Generated output filename
        """
