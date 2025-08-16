"""Main processor for spatial audio feature extraction."""

import concurrent.futures
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from embeddings_create.factories.extractor_factory import ExtractorFactory
from embeddings_create.factories.loader_factory import LoaderFactory
from embeddings_create.factories.saver_factory import SaverFactory
from embeddings_create.interfaces.audio_loader import IAudioLoader
from embeddings_create.interfaces.result_saver import IResultSaver
from embeddings_create.models.extraction_result import ExtractionResult
from embeddings_create.models.feature_config import ExtractionConfig, FeatureConfig, FeatureType

if TYPE_CHECKING:
    from embeddings_create.interfaces.feature_extractor import IFeatureExtractor


class SpatialAudioProcessor:
    """Main processor for spatial audio feature extraction.

    This class coordinates the entire feature extraction pipeline,
    following the Single Responsibility Principle by focusing on
    orchestrating the extraction process. It uses dependency injection
    to follow the Dependency Inversion Principle.
    """

    def __init__(
        self, audio_loader: IAudioLoader | None = None, result_saver: IResultSaver | None = None
    ):
        """Initialize the spatial audio processor.

        Args:
            audio_loader: Custom audio loader (optional)
            result_saver: Custom result saver (optional)
        """
        self._audio_loader = audio_loader or LoaderFactory.create_loader()
        self._result_saver = result_saver or SaverFactory.create_saver()
        self._extractors: dict[FeatureType, IFeatureExtractor] = {}

    def configure_extractors(self, feature_configs: list[FeatureConfig]) -> None:
        """Configure feature extractors based on provided configurations.

        Args:
            feature_configs: List of feature configurations
        """
        self._extractors.clear()

        for config in feature_configs:
            if config.feature_type not in self._extractors:
                extractor = ExtractorFactory.create_extractor(config.feature_type)
                self._extractors[config.feature_type] = extractor

    def process_single_file(
        self, file_path: Path, extraction_config: ExtractionConfig
    ) -> ExtractionResult:
        """Process a single audio file.

        Args:
            file_path: Path to the audio file
            extraction_config: Configuration for extraction

        Returns:
            ExtractionResult containing all extracted features

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If configuration is invalid
            RuntimeError: If processing fails
        """
        if extraction_config.verbose:
            print(f"Processing: {file_path}")

        start_time = time.time()
        result = ExtractionResult(source_file=file_path)

        try:
            audio_data = self._audio_loader.load_audio(file_path)

            self.configure_extractors(extraction_config.feature_configs)

            for config in extraction_config.feature_configs:
                try:
                    extractor = self._extractors[config.feature_type]

                    if not extractor.validate_audio_data(audio_data):
                        if extraction_config.verbose:
                            print(f"  Skipping {extractor.name}: incompatible audio data")
                        continue

                    feature_result = extractor.extract(audio_data, config)
                    result.add_feature_result(feature_result)

                    if extraction_config.verbose:
                        print(f"  Extracted {extractor.name}: {feature_result.shape}")

                except Exception as e:
                    if extraction_config.verbose:
                        print(f"  Failed {config.feature_type.value}: {e!s}")
                    continue

            result.total_extraction_time = time.time() - start_time

            if extraction_config.output_directory:
                output_dir = Path(extraction_config.output_directory)
                self._result_saver.save_result(result, output_dir / file_path.stem)

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.total_extraction_time = time.time() - start_time

            if extraction_config.verbose:
                print(f"  Error processing {file_path}: {e!s}")

            return result

    def process_directory(
        self, directory_path: Path, extraction_config: ExtractionConfig
    ) -> list[ExtractionResult]:
        """Process all audio files in a directory.

        Args:
            directory_path: Path to directory containing audio files
            extraction_config: Configuration for extraction

        Returns:
            List of ExtractionResult objects

        Raises:
            NotADirectoryError: If directory_path is not a directory
        """
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        audio_files = self._find_audio_files(directory_path)

        if not audio_files:
            print(f"No audio files found in {directory_path}")
            return []

        return self.process_batch(audio_files, extraction_config)

    def process_batch(
        self, file_paths: list[Path], extraction_config: ExtractionConfig
    ) -> list[ExtractionResult]:
        """Process a batch of audio files.

        Args:
            file_paths: List of paths to audio files
            extraction_config: Configuration for extraction

        Returns:
            List of ExtractionResult objects
        """
        if extraction_config.verbose:
            print(f"Processing {len(file_paths)} files...")

        results = []

        if extraction_config.parallel_processing and len(file_paths) > 1:
            results = self._process_parallel(file_paths, extraction_config)
        else:
            results = self._process_sequential(file_paths, extraction_config)

        if extraction_config.output_directory:
            successful_results = [r for r in results if r.is_successful]
            if successful_results:
                output_dir = Path(extraction_config.output_directory)
                self._result_saver.save_batch_results(successful_results, output_dir)

        return results

    def _process_sequential(
        self, file_paths: list[Path], extraction_config: ExtractionConfig
    ) -> list[ExtractionResult]:
        """Process files sequentially with progress bar.

        Args:
            file_paths: List of paths to process
            extraction_config: Extraction configuration

        Returns:
            List of extraction results
        """
        results = []

        progress_bar = (
            tqdm(file_paths, desc="Processing files") if extraction_config.verbose else file_paths
        )

        for file_path in progress_bar:
            result = self.process_single_file(file_path, extraction_config)
            results.append(result)

        return results

    def _process_parallel(
        self, file_paths: list[Path], extraction_config: ExtractionConfig
    ) -> list[ExtractionResult]:
        """Process files in parallel with thread pool.

        Args:
            file_paths: List of paths to process
            extraction_config: Extraction configuration

        Returns:
            List of extraction results
        """
        max_workers = extraction_config.max_workers or min(4, len(file_paths))

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_file, path, extraction_config): path
                for path in file_paths
            }

            progress_bar = (
                tqdm(
                    concurrent.futures.as_completed(future_to_path),
                    total=len(file_paths),
                    desc="Processing files",
                )
                if extraction_config.verbose
                else concurrent.futures.as_completed(future_to_path)
            )

            for future in progress_bar:
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = ExtractionResult(source_file=path)
                    error_result.success = False
                    error_result.error_message = str(e)
                    results.append(error_result)

        results.sort(key=lambda x: str(x.source_file))
        return results

    def _find_audio_files(self, directory: Path) -> list[Path]:
        """Find all audio files in directory and subdirectories.

        Args:
            directory: Directory to search

        Returns:
            List of audio file paths
        """
        audio_extensions = self._audio_loader.supported_formats()
        audio_files: list[Path] = []

        for ext in audio_extensions:
            audio_files.extend(directory.rglob(f"*{ext}"))
            audio_files.extend(directory.rglob(f"*{ext.upper()}"))

        return sorted(set(audio_files))

    def get_extraction_summary(self, results: list[ExtractionResult]) -> dict[str, Any]:
        """Generate a summary of extraction results.

        Args:
            results: List of extraction results

        Returns:
            Summary dictionary with statistics
        """
        total_files = len(results)
        successful_files = sum(1 for r in results if r.is_successful)

        feature_counts = {}
        total_extraction_time = 0.0

        for result in results:
            if result.total_extraction_time:
                total_extraction_time += result.total_extraction_time

            for feature_type in result.extracted_features:
                if feature_type not in feature_counts:
                    feature_counts[feature_type] = 0
                feature_counts[feature_type] += 1

        return {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": total_files - successful_files,
            "success_rate": successful_files / total_files if total_files > 0 else 0,
            "feature_extraction_counts": {ft.value: count for ft, count in feature_counts.items()},
            "total_extraction_time": total_extraction_time,
            "average_time_per_file": total_extraction_time / total_files if total_files > 0 else 0,
        }
