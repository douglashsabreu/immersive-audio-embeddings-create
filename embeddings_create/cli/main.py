"""Command line interface implementation."""

import json
from pathlib import Path

import click

from embeddings_create.core.processor import SpatialAudioProcessor
from embeddings_create.factories.extractor_factory import ExtractorFactory
from embeddings_create.models.feature_config import ExtractionConfig, FeatureConfig, FeatureType


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for extracted features",
)
@click.option(
    "--features",
    "-f",
    type=click.Choice([ft.value for ft in FeatureType], case_sensitive=False),
    multiple=True,
    default=[ft.value for ft in FeatureType],
    help="Feature types to extract (default: all)",
)
@click.option("--n-fft", type=int, default=1024, help="FFT window size (default: 1024)")
@click.option("--hop-length", type=int, default=512, help="Hop length for STFT (default: 512)")
@click.option("--n-mels", type=int, default=64, help="Number of mel bands (default: 64)")
@click.option("--sample-rate", type=int, default=44100, help="Target sample rate (default: 44100)")
@click.option(
    "--use-pcen/--no-pcen", default=True, help="Use PCEN instead of log-mel (default: True)"
)
@click.option(
    "--parallel/--no-parallel", default=True, help="Use parallel processing (default: True)"
)
@click.option("--max-workers", type=int, default=None, help="Maximum number of worker threads")
@click.option("--verbose/--quiet", "-v/-q", default=False, help="Verbose output")
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="JSON configuration file",
)
def main(
    input_path: Path,
    output_dir: Path | None,
    features: tuple,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    sample_rate: int,
    use_pcen: bool,
    parallel: bool,
    max_workers: int | None,
    verbose: bool,
    config_file: Path | None,
) -> None:
    """Extract spatial audio features from binaural audio files.

    INPUT_PATH can be either a single audio file or a directory containing audio files.

    The system will automatically detect the input type and extract the following
    spatial audio features:

    - SALSA: Log-mel spectrogram with spatial covariance features
    - SALSA-Lite: Lightweight version using inter-channel phase differences
    - Intensity DOA: Direction of arrival analysis from intensity vectors
    - Diffuseness: DirAC-based diffuseness measures
    - IACC: Interaural cross-correlation for binaural content

    Example usage:

        # Process single file
        python -m embeddings_create.cli.main audio.wav -o ./output -v

        # Process directory with specific features
        python -m embeddings_create.cli.main ./audio_dir -f salsa iacc -o ./output

        # Use configuration file
        python -m embeddings_create.cli.main ./audio_dir -c config.json
    """
    if config_file:
        extraction_config = _load_config_file(config_file)
    else:
        feature_configs = _create_feature_configs(
            features, n_fft, hop_length, n_mels, sample_rate, use_pcen
        )

        extraction_config = ExtractionConfig(
            feature_configs=feature_configs,
            output_directory=str(output_dir) if output_dir else None,
            parallel_processing=parallel,
            max_workers=max_workers,
            verbose=verbose,
        )

    processor = SpatialAudioProcessor()

    try:
        if input_path.is_file():
            if verbose:
                click.echo(f"Processing single file: {input_path}")

            result = processor.process_single_file(input_path, extraction_config)

            if result.is_successful:
                click.echo(f"✓ Successfully extracted {len(result.features)} feature types")
                if verbose:
                    for feature_type in result.extracted_features:
                        feature_result = result.get_feature(feature_type)
                        click.echo(f"  - {feature_type.value}: {feature_result.shape}")
            else:
                click.echo(f"✗ Failed to process {input_path}: {result.error_message}")

        elif input_path.is_dir():
            if verbose:
                click.echo(f"Processing directory: {input_path}")

            results = processor.process_directory(input_path, extraction_config)

            summary = processor.get_extraction_summary(results)

            click.echo(f"Processed {summary['total_files']} files:")
            click.echo(f"  ✓ Successful: {summary['successful_files']}")
            click.echo(f"  ✗ Failed: {summary['failed_files']}")
            click.echo(f"  Success rate: {summary['success_rate']:.1%}")

            if verbose and summary["feature_extraction_counts"]:
                click.echo("\nFeature extraction counts:")
                for feature_type, count in summary["feature_extraction_counts"].items():
                    click.echo(f"  - {feature_type}: {count}")

                click.echo(f"\nTotal processing time: {summary['total_extraction_time']:.2f}s")
                click.echo(f"Average time per file: {summary['average_time_per_file']:.2f}s")

        else:
            click.echo(f"Error: {input_path} is neither a file nor a directory")
            return

        if output_dir and verbose:
            click.echo(f"\nResults saved to: {output_dir}")

    except Exception as e:
        click.echo(f"Error: {e!s}")
        if verbose:
            import traceback

            traceback.print_exc()


def _create_feature_configs(
    features: tuple, n_fft: int, hop_length: int, n_mels: int, sample_rate: int, use_pcen: bool
) -> list[FeatureConfig]:
    """Create feature configurations from CLI arguments.

    Args:
        features: Tuple of feature type names
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        sample_rate: Target sample rate
        use_pcen: Whether to use PCEN

    Returns:
        List of FeatureConfig objects
    """
    configs = []

    for feature_name in features:
        try:
            feature_type = FeatureType(feature_name.lower())

            extractor_info = ExtractorFactory.get_extractor_info(feature_type)
            default_params = extractor_info["default_config"]

            config = FeatureConfig(
                feature_type=feature_type,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                sample_rate=sample_rate,
                use_pcen=use_pcen,
                parameters=default_params,
            )

            configs.append(config)

        except ValueError as e:
            click.echo(f"Warning: Skipping unknown feature type '{feature_name}': {e!s}")
            continue

    return configs


def _load_config_file(config_path: Path) -> ExtractionConfig:
    """Load extraction configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        ExtractionConfig object

    Raises:
        ValueError: If configuration file is invalid
    """
    try:
        with open(config_path) as f:
            config_dict = json.load(f)

        feature_configs = []
        for feature_config_dict in config_dict.get("features", []):
            feature_type = FeatureType(feature_config_dict["feature_type"])
            config = FeatureConfig(
                feature_type=feature_type,
                n_fft=feature_config_dict.get("n_fft", 1024),
                hop_length=feature_config_dict.get("hop_length", 512),
                n_mels=feature_config_dict.get("n_mels", 64),
                sample_rate=feature_config_dict.get("sample_rate", 44100),
                use_pcen=feature_config_dict.get("use_pcen", True),
                parameters=feature_config_dict.get("parameters", {}),
            )
            feature_configs.append(config)

        return ExtractionConfig(
            feature_configs=feature_configs,
            output_directory=config_dict.get("output_directory"),
            save_intermediate=config_dict.get("save_intermediate", False),
            parallel_processing=config_dict.get("parallel_processing", True),
            max_workers=config_dict.get("max_workers"),
            verbose=config_dict.get("verbose", False),
        )

    except Exception as e:
        raise ValueError(f"Invalid configuration file: {e!s}") from e


if __name__ == "__main__":
    main()
