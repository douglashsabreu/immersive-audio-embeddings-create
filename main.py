"""Main entry point for spatial audio feature extraction system."""

import sys
from pathlib import Path

# Import the CLI main function
from cli.main import main

# Setup module path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir / "embeddings_create"))

if __name__ == "__main__":
    main()
