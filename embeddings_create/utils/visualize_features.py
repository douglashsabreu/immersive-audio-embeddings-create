"""Visualizer for extracted spatial audio features."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_features(npz_path: Path) -> dict[str, Any]:
    """Load features from NPZ file.

    Args:
        npz_path: Path to the NPZ file

    Returns:
        Dictionary containing the loaded features
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        return {key: data[key] for key in data}
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return {}


def print_feature_info(features: dict[str, Any]) -> None:
    """Print information about the loaded features.

    Args:
        features: Dictionary containing feature arrays
    """
    if not features:
        print("No features found!")
        return

    print(f"Found {len(features)} feature type(s):")
    print("-" * 50)

    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Data type: {value.dtype}")

            if np.issubdtype(value.dtype, np.number) and value.size > 0:
                print(f"  Min value: {value.min():.4f}")
                print(f"  Max value: {value.max():.4f}")
                print(f"  Mean: {value.mean():.4f}")
                print(f"  Std: {value.std():.4f}")
            elif value.size > 0:
                print(f"  Content: {value}")
            else:
                print("  Empty array")
        else:
            print(f"{key}: {type(value)} - {value}")
        print()


def visualize_features(
    features: dict[str, Any],
    output_dir: Path | None = None,
    selected_features: list[str] | None = None,
    separate_windows: bool = False,
) -> None:
    """Create visualizations for the features.

    Args:
        features: Dictionary containing feature arrays
        output_dir: Directory to save plots (optional)
        selected_features: List of specific features to visualize (optional)
        separate_windows: Show each feature in separate window (optional)
    """
    if not features:
        print("No features to visualize!")
        return

    numeric_features = [
        k
        for k, v in features.items()
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.size > 0
    ]

    if selected_features:
        available_numeric = set(numeric_features)
        selected_numeric = [f for f in selected_features if f in available_numeric]
        if not selected_numeric:
            print(f"None of the selected features {selected_features} are numeric arrays!")
            print(f"Available numeric features: {numeric_features}")
            return
        numeric_features = selected_numeric

    if not numeric_features:
        print("No numeric array features to visualize!")
        return

    if separate_windows:
        for key in numeric_features:
            value = features[key]
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            _plot_feature(ax, key, value)

            plt.tight_layout()

            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_path = output_dir / f"{key}_visualization.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to: {plot_path}")

            plt.show()
    else:
        n_features = len(numeric_features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        if n_features == 1:
            axes = [axes]

        for idx, key in enumerate(numeric_features):
            value = features[key]
            _plot_feature(axes[idx], key, value, show_colorbar=True)

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / "features_visualization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {plot_path}")

        plt.show()


def _plot_feature(ax, key: str, value: np.ndarray, show_colorbar: bool = True) -> None:
    """Plot a single feature on the given axis.

    Args:
        ax: Matplotlib axis to plot on
        key: Feature name
        value: Feature array
        show_colorbar: Whether to show colorbar for 2D plots
    """
    if value.ndim == 1:
        ax.plot(value)
        ax.set_title(f"{key} - Time Series")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")

    elif value.ndim == 2:
        im = ax.imshow(value, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{key} - Spectrogram")
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Feature Dimension")
        if show_colorbar:
            plt.colorbar(im, ax=ax)

    elif value.ndim == 3:
        value_2d = value.reshape(value.shape[0], -1)
        im = ax.imshow(value_2d, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{key} - Flattened 3D Feature")
        ax.set_xlabel("Flattened Dimensions")
        ax.set_ylabel("Time Frame")
        if show_colorbar:
            plt.colorbar(im, ax=ax)

    else:
        ax.text(
            0.5,
            0.5,
            f"{key}\nShape: {value.shape}\nDimensions > 3",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"{key} - High Dimensional")


def main() -> None:
    """Main function for feature visualization."""
    parser = argparse.ArgumentParser(description="Visualize extracted spatial audio features")
    parser.add_argument("npz_file", type=Path, help="Path to the NPZ file containing features")
    parser.add_argument(
        "--save-plots", "-s", type=Path, help="Directory to save visualization plots"
    )
    parser.add_argument("--no-plot", action="store_true", help="Only print info, don't show plots")
    parser.add_argument(
        "--features", "-f", nargs="+", help="Specific features to visualize (default: all)"
    )
    parser.add_argument(
        "--separate-windows", action="store_true", help="Show each feature in separate window"
    )
    parser.add_argument(
        "--list-features", "-l", action="store_true", help="List available features and exit"
    )

    args = parser.parse_args()

    if not args.npz_file.exists():
        print(f"Error: File {args.npz_file} does not exist!")
        return

    print(f"Loading features from: {args.npz_file}")
    features = load_features(args.npz_file)

    if args.list_features:
        print("Available features:")
        for key in features:
            feature_type = (
                "numeric array"
                if (
                    isinstance(features[key], np.ndarray)
                    and np.issubdtype(features[key].dtype, np.number)
                )
                else "other"
            )
            print(f"  - {key} ({feature_type})")
        return

    print_feature_info(features)

    if not args.no_plot:
        visualize_features(
            features,
            args.save_plots,
            selected_features=args.features,
            separate_windows=args.separate_windows,
        )


if __name__ == "__main__":
    main()
