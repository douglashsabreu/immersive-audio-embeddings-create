"""Interaural Cross-Correlation (IACC) feature extractor."""

from typing import Any

import numpy as np
from scipy import signal

from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.feature_config import FeatureConfig, FeatureType

from .base_extractor import BaseFeatureExtractor


class IACCExtractor(BaseFeatureExtractor):
    """Interaural Cross-Correlation (IACC) feature extractor.

    Since the input audio is already binaural, this extractor directly measures
    IACC and band-IACC from the left and right channels. Lower IACC values
    typically indicate greater "width" or spatial envelopment.

    Different mix formats typically show:
    - 5.1.4 mixes: Lower IACC (greater width/envelopment)
    - 2.0/1.0 mixes: Higher IACC (more correlated between ears)

    This provides an additional feature for perceived "spaciousness" that
    complements the spatial features extracted from FOA content.

    This implementation follows the Single Responsibility Principle by focusing
    only on binaural correlation analysis.
    """

    def __init__(self):
        """Initialize IACC extractor."""
        super().__init__(FeatureType.IACC)

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "IACC"

    @property
    def required_channels(self) -> int:
        """IACC requires 2-channel binaural input."""
        return 2

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for IACC."""
        return {
            "window_size": 0.1,
            "overlap": 0.5,
            "max_lag_ms": 1.0,
            "frequency_bands": [
                (20, 200),
                (200, 500),
                (500, 1000),
                (1000, 2000),
                (2000, 4000),
                (4000, 8000),
            ],
            "normalize": True,
            "absolute_iacc": True,
        }

    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract IACC features from binaural audio data.

        Args:
            audio_data: 2-channel binaural audio data
            config: Extraction configuration

        Returns:
            IACC features including broadband and frequency-band specific measures
        """
        if not audio_data.is_binaural:
            raise ValueError("IACC extractor requires 2-channel binaural input")

        signal_data = audio_data.signal
        left_channel = signal_data[0]
        right_channel = signal_data[1]

        window_size = config.parameters.get("window_size", 0.1)
        overlap = config.parameters.get("overlap", 0.5)
        max_lag_ms = config.parameters.get("max_lag_ms", 1.0)
        frequency_bands = config.parameters.get(
            "frequency_bands",
            [(20, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)],
        )
        normalize = config.parameters.get("normalize", True)
        absolute_iacc = config.parameters.get("absolute_iacc", True)

        broadband_iacc = self._compute_broadband_iacc(
            left_channel,
            right_channel,
            audio_data.sample_rate,
            window_size,
            overlap,
            max_lag_ms,
            normalize,
            absolute_iacc,
        )

        band_iacc_features = []
        for low_freq, high_freq in frequency_bands:
            band_iacc = self._compute_band_iacc(
                left_channel,
                right_channel,
                audio_data.sample_rate,
                low_freq,
                high_freq,
                window_size,
                overlap,
                max_lag_ms,
                normalize,
                absolute_iacc,
            )
            band_iacc_features.extend(band_iacc)

        global_stats = self._compute_global_statistics(broadband_iacc)

        features = np.concatenate([global_stats, np.array(band_iacc_features)])

        return features

    def _compute_broadband_iacc(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        window_size: float,
        overlap: float,
        max_lag_ms: float,
        normalize: bool,
        absolute_iacc: bool,
    ) -> np.ndarray:
        """Compute broadband IACC over time.

        Args:
            left: Left channel signal
            right: Right channel signal
            sample_rate: Audio sample rate
            window_size: Analysis window size in seconds
            overlap: Overlap factor between windows
            max_lag_ms: Maximum lag for correlation in milliseconds
            normalize: Whether to normalize correlation
            absolute_iacc: Whether to take absolute value of IACC

        Returns:
            Array of IACC values over time
        """
        window_samples = int(window_size * sample_rate)
        hop_samples = int(window_samples * (1 - overlap))
        max_lag_samples = int(max_lag_ms * sample_rate / 1000)

        n_frames = 1 + (len(left) - window_samples) // hop_samples
        iacc_values = np.zeros(n_frames)

        for frame in range(n_frames):
            start = frame * hop_samples
            end = start + window_samples

            left_frame = left[start:end]
            right_frame = right[start:end]

            if normalize:
                left_frame = left_frame / (np.std(left_frame) + 1e-12)
                right_frame = right_frame / (np.std(right_frame) + 1e-12)

            correlation = np.correlate(left_frame, right_frame, mode="full")

            center = len(correlation) // 2
            start_lag = max(0, center - max_lag_samples)
            end_lag = min(len(correlation), center + max_lag_samples + 1)

            correlation_segment = correlation[start_lag:end_lag]

            if normalize:
                normalization = np.sqrt(np.sum(left_frame**2) * np.sum(right_frame**2))
                correlation_segment = correlation_segment / (normalization + 1e-12)

            max_correlation = (
                np.max(np.abs(correlation_segment))
                if absolute_iacc
                else np.max(correlation_segment)
            )
            iacc_values[frame] = max_correlation

        return iacc_values

    def _compute_band_iacc(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        low_freq: float,
        high_freq: float,
        window_size: float,
        overlap: float,
        max_lag_ms: float,
        normalize: bool,
        absolute_iacc: bool,
    ) -> list:
        """Compute IACC for a specific frequency band.

        Args:
            left: Left channel signal
            right: Right channel signal
            sample_rate: Audio sample rate
            low_freq: Lower frequency bound in Hz
            high_freq: Upper frequency bound in Hz
            window_size: Analysis window size in seconds
            overlap: Overlap factor between windows
            max_lag_ms: Maximum lag for correlation in milliseconds
            normalize: Whether to normalize correlation
            absolute_iacc: Whether to take absolute value of IACC

        Returns:
            List of band-specific IACC statistics
        """
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq / nyquist, 0.99)

        sos = signal.butter(4, [low_norm, high_norm], btype="band", output="sos")

        left_filtered = signal.sosfiltfilt(sos, left)
        right_filtered = signal.sosfiltfilt(sos, right)

        band_iacc = self._compute_broadband_iacc(
            left_filtered,
            right_filtered,
            sample_rate,
            window_size,
            overlap,
            max_lag_ms,
            normalize,
            absolute_iacc,
        )

        stats = [
            np.mean(band_iacc),
            np.std(band_iacc),
            np.min(band_iacc),
            np.max(band_iacc),
            np.percentile(band_iacc, 25),
            np.percentile(band_iacc, 75),
        ]

        return stats

    def _compute_global_statistics(self, iacc_values: np.ndarray) -> np.ndarray:
        """Compute global statistics of IACC values.

        Args:
            iacc_values: Array of IACC values over time

        Returns:
            Array of statistical features
        """
        stats = [
            np.mean(iacc_values),
            np.std(iacc_values),
            np.min(iacc_values),
            np.max(iacc_values),
            np.median(iacc_values),
            np.percentile(iacc_values, 5),
            np.percentile(iacc_values, 25),
            np.percentile(iacc_values, 75),
            np.percentile(iacc_values, 95),
        ]

        temporal_diff = np.diff(iacc_values)
        stats.extend([np.mean(np.abs(temporal_diff)), np.std(temporal_diff)])

        zero_crossings = np.sum(np.diff(np.sign(iacc_values - np.mean(iacc_values))) != 0)
        stats.append(zero_crossings / len(iacc_values))

        return np.array(stats)
