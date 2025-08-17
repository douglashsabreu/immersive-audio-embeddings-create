#!/usr/bin/env python3
"""Spatial Audio Embeddings Generator.

Complete pipeline for generating embeddings from binaural spatial audio recordings.
Supports dummy head recordings and extracts 128D embeddings for ML applications.

Architecture follows SOLID principles with dependency injection and Factory pattern.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from embeddings_create.factories.pipeline_factory import PipelineComponentFactory

# Setup module path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))


@click.group()
@click.version_option()
def main() -> None:
    """Spatial Audio Embeddings Generator.

    Complete pipeline for processing binaural audio recordings and generating
    high-quality 128D embeddings for machine learning applications.

    Supports 4 audio format classes:
    • 5.1+4h (Immersive) • 5.1 (Surround) • 2.0 (Stereo) • 1.0 (Mono)
    """


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    default="features_extracted",
    type=click.Path(path_type=Path),
    help="Directory to save extracted features (default: features_extracted)",
)
@click.option("--max-files", type=int, help="Limit number of files to process (for testing)")
@click.option("--pattern", default="*.wav", help="File pattern to match (default: *.wav)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def extract(
    input_dir: Path, output_dir: Path, max_files: Optional[int], pattern: str, verbose: bool
) -> None:
    """Extract spatial features from audio files.

    Processes binaural audio recordings and extracts 75 spatial features:
    - Interaural features (IPD, ITD, ILD)
    - Spatial correlation measures
    - Spectral differences
    - Pseudo-intensity vectors

    Example:
    python main.py extract audios_input/ -o features/ -v
    """
    logger = PipelineComponentFactory.create_logger(verbose=verbose)
    feature_extractor = PipelineComponentFactory.create_feature_extractor(logger=logger)

    result = feature_extractor.extract_features(
        input_dir=input_dir, output_dir=output_dir, max_files=max_files, file_pattern=pattern
    )

    if not result["success"]:
        raise click.ClickException(
            f"Feature extraction failed: {result.get('error', 'Unknown error')}"
        )


@main.command()
@click.argument("features_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    default="spatial_audio_dataset_4classes.npz",
    type=click.Path(path_type=Path),
    help="Output dataset file",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def organize(features_dir: Path, output: Path, verbose: bool) -> None:
    """Organize extracted features into 4-class dataset.

    Groups features by audio format:
    - 5.1+4h: Immersive (height channels)
    - 5.1: Traditional surround
    - 2.0: Stereo
    - 1.0: Mono

    Example:
    python main.py organize features_extracted/ -o dataset.npz -v
    """
    logger = PipelineComponentFactory.create_logger(verbose=verbose)
    dataset_organizer = PipelineComponentFactory.create_dataset_organizer(logger=logger)

    result = dataset_organizer.organize_dataset(features_dir=features_dir, output_file=output)

    if not result["success"]:
        raise click.ClickException(
            f"Dataset organization failed: {result.get('error', 'Unknown error')}"
        )


@main.command()
@click.argument("dataset_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    default="spatial_audio_embeddings.npz",
    type=click.Path(path_type=Path),
    help="Output embeddings file",
)
@click.option("--dim", default=128, type=int, help="Embedding dimension (default: 128)")
@click.option("--model", type=click.Path(path_type=Path), help="Path to trained model (optional)")
@click.option("--visualize/--no-visualize", default=True, help="Generate visualizations")
@click.option(
    "--viz-dir",
    default="output/visualizations",
    type=click.Path(path_type=Path),
    help="Visualizations directory",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def embeddings(
    dataset_file: Path,
    output: Path,
    dim: int,
    model: Optional[Path],
    visualize: bool,
    viz_dir: Path,
    verbose: bool,
) -> None:
    """Generate 128D embeddings from organized dataset.

    Uses neural encoder with Metal Performance Shaders (MPS) acceleration
    on Apple Silicon. Generates L2-normalized embeddings suitable for
    similarity analysis and classification.

    Example:
    python main.py embeddings dataset.npz -o embeddings.npz --visualize -v
    """
    logger = PipelineComponentFactory.create_logger(verbose=verbose)
    embedding_generator = PipelineComponentFactory.create_embedding_generator(logger=logger)

    result = embedding_generator.generate_embeddings(
        dataset_file=dataset_file,
        output_file=output,
        embedding_dim=dim,
        model_path=model,
        generate_visualizations=visualize,
        viz_dir=viz_dir if visualize else None,
    )

    if not result["success"]:
        raise click.ClickException(
            f"Embedding generation failed: {result.get('error', 'Unknown error')}"
        )


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--features-dir", default="features_extracted", type=click.Path(path_type=Path))
@click.option(
    "--dataset", default="spatial_audio_dataset_4classes.npz", type=click.Path(path_type=Path)
)
@click.option(
    "--embeddings", default="spatial_audio_embeddings.npz", type=click.Path(path_type=Path)
)
@click.option("--max-files", type=int, help="Limit files for testing")
@click.option("--embedding-dim", default=128, type=int, help="Embedding dimension")
@click.option("--visualize/--no-visualize", default=True, help="Generate visualizations")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def pipeline(
    input_dir: Path,
    features_dir: Path,
    dataset: Path,
    embeddings: Path,
    max_files: Optional[int],
    embedding_dim: int,
    visualize: bool,
    verbose: bool,
) -> None:
    """Complete pipeline: audio → features → dataset → embeddings.

    Runs the complete pipeline in sequence:
    1. Extract spatial features from audio files
    2. Organize into 4-class dataset
    3. Generate 128D embeddings with MPS acceleration
    4. Create visualizations and metrics

    Example:
    python main.py pipeline audios_input/ --verbose
    """
    pipeline_orchestrator = PipelineComponentFactory.create_complete_pipeline(verbose=verbose)

    result = pipeline_orchestrator.run_complete_pipeline(
        input_dir=input_dir,
        features_dir=features_dir,
        dataset_file=dataset,
        embeddings_file=embeddings,
        max_files=max_files,
        embedding_dim=embedding_dim,
        visualize=visualize,
    )

    if not result["success"]:
        failed_step = None
        for step_name, step_result in result.get("steps", {}).items():
            if not step_result.get("success", True):
                failed_step = step_name
                break

        error_msg = f"Pipeline failed at step: {failed_step}" if failed_step else "Pipeline failed"
        raise click.ClickException(error_msg)


if __name__ == "__main__":
    main()
