"""Concrete implementations of pipeline interfaces following SOLID principles."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from ..interfaces.pipeline_interfaces import (
    IDatasetOrganizer,
    IEmbeddingGenerator,
    IFeatureExtractor,
    ILogger,
    IPipelineOrchestrator,
)
from ..scripts.batch_extract_features import batch_extract_features
from ..scripts.generate_embeddings import (
    compute_embedding_metrics,
    generate_embeddings_from_dataset,
    save_embeddings,
    visualize_embeddings_2d,
)
from ..scripts.reorganize_dataset import reorganize_dataset_with_4_classes


class ClickLogger(ILogger):
    """Concrete implementation of logger using Click."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def info(self, message: str) -> None:
        """Log info message."""
        if self.verbose:
            click.echo(message)

    def error(self, message: str) -> None:
        """Log error message."""
        click.echo(click.style(f"❌ {message}", fg="red"))

    def success(self, message: str) -> None:
        """Log success message."""
        if self.verbose:
            click.echo(click.style(f"✅ {message}", fg="green"))


class SpatialFeatureExtractor(IFeatureExtractor):
    """Concrete implementation for spatial feature extraction."""

    def __init__(self, logger: ILogger):
        self.logger = logger

    def extract_features(
        self,
        input_dir: Path,
        output_dir: Path,
        max_files: Optional[int] = None,
        file_pattern: str = "*.wav",
    ) -> Dict[str, Any]:
        """Extract features from audio files."""
        self.logger.info("🎵 SPATIAL AUDIO FEATURE EXTRACTION")
        self.logger.info("=" * 40)

        start_time = time.time()

        try:
            batch_extract_features(
                input_dir=input_dir,
                output_dir=output_dir,
                file_pattern=file_pattern,
                max_files=max_files,
            )

            elapsed = time.time() - start_time
            self.logger.success(f"Feature extraction completed in {elapsed:.1f}s")

            return {
                "success": True,
                "elapsed_time": elapsed,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "max_files": max_files,
            }

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {"success": False, "error": str(e), "elapsed_time": time.time() - start_time}


class FourClassDatasetOrganizer(IDatasetOrganizer):
    """Concrete implementation for 4-class dataset organization."""

    def __init__(self, logger: ILogger):
        self.logger = logger

    def organize_dataset(self, features_dir: Path, output_file: Path) -> Dict[str, Any]:
        """Organize features into structured dataset."""
        self.logger.info("📊 DATASET ORGANIZATION")
        self.logger.info("=" * 25)

        start_time = time.time()

        try:
            class_counts = reorganize_dataset_with_4_classes(features_dir, output_file)

            elapsed = time.time() - start_time
            self.logger.success(f"Dataset organized: {output_file}")

            return {
                "success": True,
                "elapsed_time": elapsed,
                "output_file": str(output_file),
                "class_counts": class_counts,
                "total_samples": sum(class_counts.values()) if class_counts else 0,
            }

        except Exception as e:
            self.logger.error(f"Dataset organization failed: {str(e)}")
            return {"success": False, "error": str(e), "elapsed_time": time.time() - start_time}


class NeuralEmbeddingGenerator(IEmbeddingGenerator):
    """Concrete implementation for neural embedding generation."""

    def __init__(self, logger: ILogger):
        self.logger = logger

    def generate_embeddings(
        self,
        dataset_file: Path,
        output_file: Path,
        embedding_dim: int = 128,
        model_path: Optional[Path] = None,
        generate_visualizations: bool = True,
        viz_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate embeddings from organized dataset."""
        self.logger.info("🧠 EMBEDDING GENERATION")
        self.logger.info("=" * 22)

        start_time = time.time()

        try:
            # Generate embeddings
            embeddings_dict = generate_embeddings_from_dataset(
                dataset_file=dataset_file, embedding_dim=embedding_dim, model_path=model_path
            )

            # Save embeddings
            save_embeddings(embeddings_dict, output_file)

            # Compute metrics
            metrics = compute_embedding_metrics(embeddings_dict)

            # Generate visualizations
            if generate_visualizations and viz_dir:
                self.logger.info("🎨 Generating visualizations...")
                viz_dir = Path(viz_dir)
                visualize_embeddings_2d(embeddings_dict, viz_dir, method="tsne")
                visualize_embeddings_2d(embeddings_dict, viz_dir, method="pca")

            elapsed = time.time() - start_time
            self.logger.success(f"Embeddings generated: {output_file}")

            return {
                "success": True,
                "elapsed_time": elapsed,
                "output_file": str(output_file),
                "embedding_dim": embedding_dim,
                "n_samples": embeddings_dict["n_samples"],
                "metrics": metrics,
                "visualizations_generated": generate_visualizations,
            }

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return {"success": False, "error": str(e), "elapsed_time": time.time() - start_time}


class CompletePipelineOrchestrator(IPipelineOrchestrator):
    """Concrete implementation for complete pipeline orchestration."""

    def __init__(
        self,
        feature_extractor: IFeatureExtractor,
        dataset_organizer: IDatasetOrganizer,
        embedding_generator: IEmbeddingGenerator,
        logger: ILogger,
    ):
        self.feature_extractor = feature_extractor
        self.dataset_organizer = dataset_organizer
        self.embedding_generator = embedding_generator
        self.logger = logger

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
        """Run complete pipeline from audio to embeddings."""
        self.logger.info("🚀 COMPLETE SPATIAL AUDIO EMBEDDINGS PIPELINE")
        self.logger.info("=" * 50)

        pipeline_start = time.time()
        results: Dict[str, Any] = {"success": True, "steps": {}}

        # Step 1: Extract features
        self.logger.info("\n1️⃣ EXTRACTING SPATIAL FEATURES...")
        extract_result = self.feature_extractor.extract_features(
            input_dir=input_dir, output_dir=features_dir, max_files=max_files
        )
        results["steps"]["feature_extraction"] = extract_result

        if not extract_result["success"]:
            results["success"] = False
            return results

        # Step 2: Organize dataset
        self.logger.info("\n2️⃣ ORGANIZING DATASET...")
        organize_result = self.dataset_organizer.organize_dataset(
            features_dir=features_dir, output_file=dataset_file
        )
        results["steps"]["dataset_organization"] = organize_result

        if not organize_result["success"]:
            results["success"] = False
            return results

        # Step 3: Generate embeddings
        self.logger.info("\n3️⃣ GENERATING EMBEDDINGS...")
        embedding_result = self.embedding_generator.generate_embeddings(
            dataset_file=dataset_file,
            output_file=embeddings_file,
            embedding_dim=embedding_dim,
            generate_visualizations=visualize,
            viz_dir=Path("output/visualizations") if visualize else None,
        )
        results["steps"]["embedding_generation"] = embedding_result

        if not embedding_result["success"]:
            results["success"] = False
            return results

        # Final summary
        total_elapsed = time.time() - pipeline_start
        results["total_elapsed_time"] = total_elapsed

        self.logger.info(f"\n🎉 PIPELINE COMPLETED!")
        self.logger.info(f"   ⏱️  Total time: {total_elapsed:.1f}s")
        self.logger.info(f"   📊 Embeddings: {embeddings_file}")
        self.logger.info(f"   📈 Visualizations: output/visualizations/")
        self.logger.info(f"   🎯 Ready for ML applications!")

        return results
