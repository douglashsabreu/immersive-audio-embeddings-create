"""Result saving utility implementation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from embeddings_create.interfaces.result_saver import IResultSaver
from embeddings_create.models.extraction_result import ExtractionResult


class ResultSaver(IResultSaver):
    """Concrete implementation of result saving functionality.

    This class handles saving feature extraction results to disk in
    various formats. It provides both single and batch saving capabilities
    with proper error handling and metadata preservation.

    This implementation follows the Single Responsibility Principle by
    focusing only on result saving operations.
    """

    def __init__(self, default_format: str = "npz"):
        """Initialize result saver.

        Args:
            default_format: Default output format ('npz', 'json', 'both')
        """
        self._default_format = default_format
        self._supported_formats = [".npz", ".json"]

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
        if not result.is_successful:
            raise ValueError("Cannot save unsuccessful extraction result")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self._default_format in ["npz", "both"]:
                self._save_as_npz(result, output_path.with_suffix(".npz"))

            if self._default_format in ["json", "both"]:
                self._save_as_json(result, output_path.with_suffix(".json"))

            return True

        except Exception as e:
            raise OSError(f"Failed to save result to {output_path}: {e!s}") from e

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
            IOError: If directory creation fails
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        success_list = []

        for result in results:
            try:
                if result.is_successful:
                    output_filename = self.create_output_filename(result.source_file, "features")
                    output_path = output_directory / output_filename
                    success = self.save_result(result, output_path)
                    success_list.append(success)
                else:
                    success_list.append(False)

            except Exception:
                success_list.append(False)

        return success_list

    def supported_formats(self) -> list[str]:
        """Get list of supported output formats.

        Returns:
            List of supported file extensions
        """
        return self._supported_formats.copy()

    def create_output_filename(self, source_file: Path, feature_type: str) -> str:
        """Create output filename for a specific feature type.

        Args:
            source_file: Original audio file path
            feature_type: Type of feature extracted

        Returns:
            Generated output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = source_file.stem
        return f"{base_name}_{feature_type}_{timestamp}"

    def _save_as_npz(self, result: ExtractionResult, output_path: Path) -> None:
        """Save result as NPZ format.

        Args:
            result: Extraction result to save
            output_path: Output file path with .npz extension
        """
        save_dict = {
            "source_file": str(result.source_file),
            "timestamp": result.timestamp.isoformat(),
            "total_extraction_time": result.total_extraction_time,
            "success": result.success,
        }

        for feature_type, feature_result in result.features.items():
            prefix = feature_type.value
            save_dict[f"{prefix}_features"] = feature_result.features
            save_dict[f"{prefix}_metadata"] = self._serialize_metadata(feature_result.metadata)
            save_dict[f"{prefix}_extraction_time"] = feature_result.extraction_time
            if feature_result.config_used:
                save_dict[f"{prefix}_config"] = self._serialize_metadata(feature_result.config_used)

        np.savez_compressed(output_path, **save_dict)

    def _save_as_json(self, result: ExtractionResult, output_path: Path) -> None:
        """Save result as JSON format (metadata only).

        Args:
            result: Extraction result to save
            output_path: Output file path with .json extension
        """
        json_dict = {
            "source_file": str(result.source_file),
            "timestamp": result.timestamp.isoformat(),
            "total_extraction_time": result.total_extraction_time,
            "success": result.success,
            "features": {},
        }

        for feature_type, feature_result in result.features.items():
            feature_info = {
                "shape": feature_result.shape,
                "size": feature_result.size,
                "extraction_time": feature_result.extraction_time,
                "metadata": feature_result.metadata,
                "config_used": feature_result.config_used,
            }
            json_dict["features"][feature_type.value] = feature_info

        with open(output_path, "w") as f:
            json.dump(json_dict, f, indent=2, default=str)

    def _serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Serialize metadata for NPZ storage.

        Args:
            metadata: Metadata dictionary

        Returns:
            Serialized metadata dictionary
        """
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            else:
                serialized[key] = value
        return serialized
