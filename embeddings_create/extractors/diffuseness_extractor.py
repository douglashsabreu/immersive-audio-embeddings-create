"""Diffuseness (DirAC) feature extractor."""

from typing import Any

import numpy as np

from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.feature_config import FeatureConfig, FeatureType

from .base_extractor import BaseFeatureExtractor


class DiffusenessExtractor(BaseFeatureExtractor):
    """Diffuseness feature extractor based on DirAC (Directional Audio Coding).

    Calculates diffuseness/coherence per frequency bin and aggregates statistics
    (mean, percentiles). Diffuseness tends to increase with immersive/reverberant
    content and can help separate wide bed layer mixes from narrow signals.

    Diffuseness is computed as 1 minus the magnitude of the intensity vector
    normalized by the energy, providing a measure of how diffuse vs directional
    the sound field is at each time-frequency point.

    This implementation follows the Single Responsibility Principle by focusing
    only on diffuseness computation and statistical analysis.
    """

    def __init__(self) -> None:
        """Initialize Diffuseness extractor."""
        super().__init__(FeatureType.DIFFUSENESS)

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "Diffuseness"

    @property
    def required_channels(self) -> int:
        """Diffuseness requires 4-channel FOA input."""
        return 4

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for Diffuseness."""
        return {
            "smoothing_time": 0.1,
            "smoothing_freq": 3,
            "percentiles": [5, 25, 50, 75, 95],
            "frequency_bands": 8,
            "eps": 1e-12,
        }

    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract diffuseness features from FOA audio data.

        Args:
            audio_data: 4-channel FOA audio data (W, X, Y, Z channels)
            config: Extraction configuration

        Returns:
            Diffuseness features including statistics and frequency band analysis
        """
        if not audio_data.is_foa:
            raise ValueError("Diffuseness extractor requires 4-channel FOA input")

        signal_data = audio_data.signal

        smoothing_time = config.parameters.get("smoothing_time", 0.1)
        smoothing_freq = config.parameters.get("smoothing_freq", 3)
        percentiles = config.parameters.get("percentiles", [5, 25, 50, 75, 95])
        frequency_bands = config.parameters.get("frequency_bands", 8)
        eps = config.parameters.get("eps", 1e-12)

        stft_channels = []
        for channel in range(4):
            stft = self._compute_stft(signal_data[channel], config)
            stft_channels.append(stft)

        stft_matrix = np.stack(stft_channels, axis=0)

        diffuseness_map = self._compute_diffuseness_map(stft_matrix, eps)

        if smoothing_time > 0 or smoothing_freq > 0:
            diffuseness_map = self._apply_smoothing(
                diffuseness_map,
                smoothing_time,
                smoothing_freq,
                config.sample_rate,
                config.hop_length,
            )

        global_stats = self._compute_global_statistics(diffuseness_map, percentiles)

        band_stats = self._compute_frequency_band_statistics(
            diffuseness_map, frequency_bands, percentiles
        )

        temporal_stats = self._compute_temporal_statistics(diffuseness_map)

        features = np.concatenate([global_stats, band_stats, temporal_stats])

        return features

    def _compute_diffuseness_map(self, stft_matrix: np.ndarray, eps: float) -> np.ndarray:
        """Compute diffuseness map from FOA STFT.

        Args:
            stft_matrix: FOA STFT with shape (4, n_freqs, n_frames) [W, X, Y, Z]
            eps: Small value for numerical stability

        Returns:
            Diffuseness map with shape (n_freqs, n_frames)
        """
        W = stft_matrix[0]
        X = stft_matrix[1]
        Y = stft_matrix[2]
        Z = stft_matrix[3]

        energy = np.abs(W) ** 2 + eps

        Ix = np.real(W * np.conj(X))
        Iy = np.real(W * np.conj(Y))
        Iz = np.real(W * np.conj(Z))

        intensity_magnitude = np.sqrt(Ix**2 + Iy**2 + Iz**2)

        intensity_magnitude_normalized = intensity_magnitude / (energy + eps)

        diffuseness = 1.0 - intensity_magnitude_normalized

        diffuseness = np.clip(diffuseness, 0.0, 1.0)

        return diffuseness

    def _apply_smoothing(
        self,
        diffuseness_map: np.ndarray,
        smoothing_time: float,
        smoothing_freq: int,
        sample_rate: int,
        hop_length: int,
    ) -> np.ndarray:
        """Apply temporal and frequency smoothing to diffuseness map.

        Args:
            diffuseness_map: Input diffuseness map
            smoothing_time: Time smoothing window in seconds
            smoothing_freq: Frequency smoothing window in bins
            sample_rate: Audio sample rate
            hop_length: STFT hop length

        Returns:
            Smoothed diffuseness map
        """
        smoothed = diffuseness_map.copy()

        if smoothing_freq > 0:
            kernel_freq = np.ones(smoothing_freq) / smoothing_freq
            for frame in range(smoothed.shape[1]):
                smoothed[:, frame] = np.convolve(smoothed[:, frame], kernel_freq, mode="same")

        if smoothing_time > 0:
            hop_time = hop_length / sample_rate
            time_window = int(smoothing_time / hop_time)
            if time_window > 1:
                kernel_time = np.ones(time_window) / time_window
                for freq in range(smoothed.shape[0]):
                    smoothed[freq, :] = np.convolve(smoothed[freq, :], kernel_time, mode="same")

        return smoothed

    def _compute_global_statistics(
        self, diffuseness_map: np.ndarray, percentiles: list[float]
    ) -> np.ndarray:
        """Compute global statistics of diffuseness map.

        Args:
            diffuseness_map: Diffuseness values
            percentiles: List of percentiles to compute

        Returns:
            Array of global statistics
        """
        flattened = diffuseness_map.flatten()

        stats = [np.mean(flattened), np.std(flattened), np.min(flattened), np.max(flattened)]

        for p in percentiles:
            stats.append(np.percentile(flattened, p))

        return np.array(stats)

    def _compute_frequency_band_statistics(
        self, diffuseness_map: np.ndarray, num_bands: int, percentiles: list[float]
    ) -> np.ndarray:
        """Compute statistics per frequency band.

        Args:
            diffuseness_map: Diffuseness values
            num_bands: Number of frequency bands
            percentiles: List of percentiles to compute

        Returns:
            Array of per-band statistics
        """
        n_freqs = diffuseness_map.shape[0]
        band_size = n_freqs // num_bands

        band_stats = []

        for band in range(num_bands):
            start_freq = band * band_size
            end_freq = min((band + 1) * band_size, n_freqs)

            band_data = diffuseness_map[start_freq:end_freq, :].flatten()

            stats = [np.mean(band_data), np.std(band_data)]

            for p in percentiles:
                stats.append(np.percentile(band_data, p))

            band_stats.extend(stats)

        return np.array(band_stats)

    def _compute_temporal_statistics(self, diffuseness_map: np.ndarray) -> np.ndarray:
        """Compute temporal evolution statistics.

        Args:
            diffuseness_map: Diffuseness values

        Returns:
            Array of temporal statistics
        """
        temporal_mean = np.mean(diffuseness_map, axis=0)

        stats = [
            np.mean(temporal_mean),
            np.std(temporal_mean),
            np.min(temporal_mean),
            np.max(temporal_mean),
        ]

        temporal_diff = np.diff(temporal_mean)
        stats.extend([np.mean(np.abs(temporal_diff)), np.std(temporal_diff)])

        zero_crossings = np.sum(np.diff(np.sign(temporal_diff - np.mean(temporal_diff))) != 0)
        stats.append(zero_crossings / len(temporal_diff))

        return np.array(stats)
