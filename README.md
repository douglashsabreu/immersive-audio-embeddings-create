# Spatial Audio Feature Extraction System

A robust and extensible system for extracting spatial audio features from binaural audio content, built with SOLID principles and Clean Code practices.

## Features

The system extracts the following spatial audio features:

- **SALSA**: Log-mel multicanal features stacked with principal eigenvector of spatial covariance matrix
- **SALSA-Lite**: Lightweight variation using Inter-channel Phase Differences (IPD)
- **Intensity DOA**: Direction of Arrival analysis from intensity vectors with statistical moments
- **Diffuseness**: DirAC-based diffuseness measures for spatial content analysis
- **IACC**: Interaural Cross-Correlation for binaural spatial perception

## Installation

1. Install dependencies:
```bash
pip install -e .
```

## Usage

### Command Line Interface

Process a single audio file:
```bash
python main.py audio_file.wav -o ./output -v
```

Process a directory of audio files:
```bash
python main.py ./audio_directory -o ./output -v
```

Extract specific features:
```bash
python main.py ./audio_directory -f salsa iacc -o ./output
```

Use configuration file:
```bash
python main.py ./audio_directory -c embeddings-create/examples/config_example.json
```

### Python API

```python
from pathlib import Path
from embeddings_create.core.processor import SpatialAudioProcessor
from embeddings_create.models.feature_config import FeatureConfig, FeatureType, ExtractionConfig

# Configure feature extraction
feature_configs = [
    FeatureConfig(feature_type=FeatureType.SALSA),
    FeatureConfig(feature_type=FeatureType.IACC)
]

extraction_config = ExtractionConfig(
    feature_configs=feature_configs,
    output_directory="./output",
    verbose=True
)

# Process audio files
processor = SpatialAudioProcessor()
results = processor.process_directory(Path("./audio_input"), extraction_config)

# Get summary
summary = processor.get_extraction_summary(results)
print(f"Processed {summary['successful_files']}/{summary['total_files']} files")
```

## Architecture

The system follows SOLID principles with a clean, modular architecture:

- **Interfaces**: Abstract contracts for extensibility
- **Extractors**: Concrete implementations for each feature type  
- **Factories**: Pattern for creating extractors dynamically
- **Models**: Data structures with clear responsibilities
- **Core**: Main processing orchestration
- **CLI**: Command-line interface

## Configuration

Example configuration file (`config_example.json`):

```json
{
  "features": [
    {
      "feature_type": "salsa",
      "n_fft": 1024,
      "hop_length": 512,
      "n_mels": 64,
      "parameters": {
        "eps": 1e-8,
        "normalize_eigenvector": true
      }
    }
  ],
  "parallel_processing": true,
  "verbose": true
}
```

## Input Requirements

- **Binaural audio files** (2-channel) for IACC analysis
- **FOA (First Order Ambisonic) content** (4-channel) for spatial features
- Supported formats: WAV, MP3, FLAC, M4A, AAC, OGG

## Output

Results are saved in NPZ format containing:
- Extracted features as numpy arrays
- Metadata and configuration used
- Processing statistics and timing information

## Extending the System

Add new feature extractors by:

1. Implementing the `IFeatureExtractor` interface
2. Registering with `ExtractorFactory`
3. Adding corresponding `FeatureType` enum value

The system is designed to be easily extensible following the Open/Closed Principle.
