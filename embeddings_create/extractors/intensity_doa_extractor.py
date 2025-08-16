"""Intensity Vector and Direction of Arrival (DOA) feature extractor."""

from typing import Any

import numpy as np

from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.feature_config import FeatureConfig, FeatureType

from .base_extractor import BaseFeatureExtractor


class IntensityDOAExtractor(BaseFeatureExtractor):
    """Intensity Vector and Direction of Arrival feature extractor.

    From FOA input, computes the active intensity vector I(t,f) to derive
    direction (azimuth, elevation) and magnitude. Aggregates DOA histograms (2D)
    and statistical moments (concentration, entropy, circular variance).

    These descriptors help differentiate between different spatial audio formats:
    - 5.1.4 shows peaks in elevation as well
    - 5.1 shows activity mainly on horizontal plane
    - 2.0/1.0 shows more concentrated distributions around frontal axis

    This implementation follows the Single Responsibility Principle by focusing
    only on intensity vector and DOA analysis.
    """

    def __init__(self) -> None:
        """Initialize Intensity DOA extractor."""
        super().__init__(FeatureType.INTENSITY_DOA)

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "Intensity_DOA"

    @property
    def required_channels(self) -> int:
        """Intensity DOA requires 4-channel FOA input."""
        return 4

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for Intensity DOA."""
        return {
            "azimuth_bins": 36,
            "elevation_bins": 18,
            "intensity_threshold": 1e-6,
            "circular_stats": True,
            "histogram_smoothing": True,
            "eps": 1e-12,
        }

    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract intensity vector and DOA features from FOA audio data.

        Args:
            audio_data: 4-channel FOA audio data (W, X, Y, Z channels)
            config: Extraction configuration

        Returns:
            Combined features including DOA histograms and statistical moments
        """
        if not audio_data.is_foa:
            raise ValueError("Intensity DOA extractor requires 4-channel FOA input")

        signal = audio_data.signal

        azimuth_bins = config.parameters.get("azimuth_bins", 36)
        elevation_bins = config.parameters.get("elevation_bins", 18)
        intensity_threshold = config.parameters.get("intensity_threshold", 1e-6)
        circular_stats = config.parameters.get("circular_stats", True)
        eps = config.parameters.get("eps", 1e-12)

        stft_channels = []
        for channel in range(4):
            stft = self._compute_stft(signal[channel], config)
            stft_channels.append(stft)

        stft_matrix = np.stack(stft_channels, axis=0)

        intensity_vectors, magnitudes = self._compute_intensity_vectors(stft_matrix, eps)

        azimuths, elevations = self._compute_doa_from_intensity(
            intensity_vectors, intensity_threshold
        )

        histogram_features = self._compute_doa_histograms(
            azimuths, elevations, magnitudes, azimuth_bins, elevation_bins
        )

        if circular_stats:
            statistical_features = self._compute_circular_statistics(
                azimuths, elevations, magnitudes
            )
            features = np.concatenate([histogram_features, statistical_features])
        else:
            features = histogram_features

        return features

    def _compute_intensity_vectors(
        self, stft_matrix: np.ndarray, eps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute active intensity vectors from FOA STFT.

        Args:
            stft_matrix: FOA STFT with shape (4, n_freqs, n_frames) [W, X, Y, Z]
            eps: Small value for numerical stability

        Returns:
            Tuple of (intensity_vectors, magnitudes)
            - intensity_vectors: shape (3, n_freqs, n_frames) [X, Y, Z components]
            - magnitudes: shape (n_freqs, n_frames)
        """
        W = stft_matrix[0]
        X = stft_matrix[1]
        Y = stft_matrix[2]
        Z = stft_matrix[3]

        energy = np.abs(W) ** 2 + eps

        Ix = np.real(W * np.conj(X))
        Iy = np.real(W * np.conj(Y))
        Iz = np.real(W * np.conj(Z))

        intensity_x = Ix / energy
        intensity_y = Iy / energy
        intensity_z = Iz / energy

        intensity_vectors = np.stack([intensity_x, intensity_y, intensity_z], axis=0)
        magnitudes = np.sqrt(intensity_x**2 + intensity_y**2 + intensity_z**2)

        return intensity_vectors, magnitudes

    def _compute_doa_from_intensity(
        self, intensity_vectors: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute DOA (azimuth, elevation) from intensity vectors.

        Args:
            intensity_vectors: Intensity vectors with shape (3, n_freqs, n_frames)
            threshold: Minimum intensity magnitude threshold

        Returns:
            Tuple of (azimuths, elevations) in radians
        """
        Ix, Iy, Iz = intensity_vectors

        magnitude = np.sqrt(Ix**2 + Iy**2 + Iz**2)

        valid_mask = magnitude > threshold

        azimuths = np.zeros_like(magnitude)
        elevations = np.zeros_like(magnitude)

        azimuths[valid_mask] = np.arctan2(Iy[valid_mask], Ix[valid_mask])

        horizontal_magnitude = np.sqrt(Ix**2 + Iy**2)
        elevations[valid_mask] = np.arctan2(
            Iz[valid_mask], horizontal_magnitude[valid_mask] + 1e-12
        )

        return azimuths, elevations

    def _compute_doa_histograms(
        self,
        azimuths: np.ndarray,
        elevations: np.ndarray,
        magnitudes: np.ndarray,
        azimuth_bins: int,
        elevation_bins: int,
    ) -> np.ndarray:
        """Compute 2D DOA histograms weighted by intensity magnitude.

        Args:
            azimuths: Azimuth angles in radians
            elevations: Elevation angles in radians
            magnitudes: Intensity magnitudes for weighting
            azimuth_bins: Number of azimuth histogram bins
            elevation_bins: Number of elevation histogram bins

        Returns:
            Flattened histogram features
        """
        azimuth_edges = np.linspace(-np.pi, np.pi, azimuth_bins + 1)
        elevation_edges = np.linspace(-np.pi / 2, np.pi / 2, elevation_bins + 1)

        flatten_azimuths = azimuths.flatten()
        flatten_elevations = elevations.flatten()
        flatten_magnitudes = magnitudes.flatten()

        hist_2d, _, _ = np.histogram2d(
            flatten_azimuths,
            flatten_elevations,
            bins=[azimuth_edges, elevation_edges],
            weights=flatten_magnitudes,
        )

        hist_azimuth, _ = np.histogram(
            flatten_azimuths, bins=azimuth_edges, weights=flatten_magnitudes
        )

        hist_elevation, _ = np.histogram(
            flatten_elevations, bins=elevation_edges, weights=flatten_magnitudes
        )

        hist_2d_normalized = hist_2d / (np.sum(hist_2d) + 1e-12)
        hist_azimuth_normalized = hist_azimuth / (np.sum(hist_azimuth) + 1e-12)
        hist_elevation_normalized = hist_elevation / (np.sum(hist_elevation) + 1e-12)

        return np.concatenate(
            [hist_2d_normalized.flatten(), hist_azimuth_normalized, hist_elevation_normalized]
        )

    def _compute_circular_statistics(
        self, azimuths: np.ndarray, elevations: np.ndarray, magnitudes: np.ndarray
    ) -> np.ndarray:
        """Compute circular statistical moments for DOA.

        Args:
            azimuths: Azimuth angles in radians
            elevations: Elevation angles in radians
            magnitudes: Intensity magnitudes for weighting

        Returns:
            Array of statistical features
        """
        features = []

        flatten_azimuths = azimuths.flatten()
        flatten_elevations = elevations.flatten()
        flatten_magnitudes = magnitudes.flatten()

        valid_mask = flatten_magnitudes > 1e-6

        if np.sum(valid_mask) > 10:
            valid_azimuths = flatten_azimuths[valid_mask]
            valid_elevations = flatten_elevations[valid_mask]
            valid_weights = flatten_magnitudes[valid_mask]

            azimuth_stats = self._circular_moments(valid_azimuths, valid_weights)
            elevation_stats = self._circular_moments(valid_elevations, valid_weights)

            features.extend(azimuth_stats)
            features.extend(elevation_stats)

            concentration = self._compute_concentration(
                valid_azimuths, valid_elevations, valid_weights
            )
            features.append(concentration)

            entropy = self._compute_doa_entropy(valid_azimuths, valid_elevations, valid_weights)
            features.append(entropy)

        else:
            features.extend([0.0] * 10)

        return np.array(features)

    def _circular_moments(self, angles: np.ndarray, weights: np.ndarray) -> list[float]:
        """Compute circular statistical moments.

        Args:
            angles: Circular angles in radians
            weights: Weights for each angle

        Returns:
            List of circular statistics [mean_direction, concentration, variance]
        """
        weights_normalized = weights / np.sum(weights)

        cos_sum = np.sum(weights_normalized * np.cos(angles))
        sin_sum = np.sum(weights_normalized * np.sin(angles))

        mean_direction = np.arctan2(sin_sum, cos_sum)

        resultant_length = np.sqrt(cos_sum**2 + sin_sum**2)

        circular_variance = 1 - resultant_length

        concentration = resultant_length / (1 - resultant_length + 1e-12)

        return [mean_direction, concentration, circular_variance]

    def _compute_concentration(
        self, azimuths: np.ndarray, elevations: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute 3D concentration measure.

        Args:
            azimuths: Azimuth angles
            elevations: Elevation angles
            weights: Weights for each direction

        Returns:
            3D concentration value
        """
        weights_normalized = weights / np.sum(weights)

        x = np.cos(elevations) * np.cos(azimuths)
        y = np.cos(elevations) * np.sin(azimuths)
        z = np.sin(elevations)

        mean_x = np.sum(weights_normalized * x)
        mean_y = np.sum(weights_normalized * y)
        mean_z = np.sum(weights_normalized * z)

        resultant_length = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)

        return float(resultant_length)

    def _compute_doa_entropy(
        self, azimuths: np.ndarray, elevations: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute entropy of DOA distribution.

        Args:
            azimuths: Azimuth angles
            elevations: Elevation angles
            weights: Weights for each direction

        Returns:
            Entropy value
        """
        hist_2d, _, _ = np.histogram2d(
            azimuths,
            elevations,
            bins=[36, 18],
            weights=weights,
            range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
        )

        hist_normalized = hist_2d / (np.sum(hist_2d) + 1e-12)

        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-12))

        return float(entropy)
