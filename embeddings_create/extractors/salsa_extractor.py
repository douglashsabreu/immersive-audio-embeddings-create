"""SALSA and SALSA-Lite feature extractors."""

from typing import Any

import numpy as np
from scipy.linalg import eigh

from embeddings_create.models.audio_data import AudioData
from embeddings_create.models.feature_config import FeatureConfig, FeatureType

from .base_extractor import BaseFeatureExtractor


class SalsaExtractor(BaseFeatureExtractor):
    """SALSA (Spatial Audio Log-mel Spectrogram with Attention) feature extractor.

    Extracts log-mel multicanal features stacked with the principal eigenvector
    of the spatial covariance matrix per T-F bin, providing directional cues
    aligned with the spectrum.

    This implementation follows the Single Responsibility Principle by focusing
    only on SALSA feature extraction.
    """

    def __init__(self) -> None:
        """Initialize SALSA extractor."""
        super().__init__(FeatureType.SALSA)

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "SALSA"

    @property
    def required_channels(self) -> int:
        """SALSA requires 4-channel FOA input."""
        return 4

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for SALSA."""
        return {"eps": 1e-8, "normalize_eigenvector": True, "stack_channels": True}

    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract SALSA features from FOA audio data.

        Args:
            audio_data: 4-channel FOA audio data
            config: Extraction configuration

        Returns:
            SALSA features with shape (n_mels * n_channels + n_freqs, n_frames)
        """
        if not audio_data.is_foa:
            raise ValueError("SALSA extractor requires 4-channel FOA input")

        signal = audio_data.signal
        eps = config.parameters.get("eps", self.get_default_config()["eps"])
        normalize_eigenvector = config.parameters.get(
            "normalize_eigenvector", self.get_default_config()["normalize_eigenvector"]
        )

        stft_channels = []
        mel_features = []

        for channel in range(4):
            stft = self._compute_stft(signal[channel], config)
            stft_channels.append(stft)

            mel_spec = self._compute_mel_spectrogram(stft, config)
            mel_features.append(mel_spec)

        stft_matrix = np.stack(stft_channels, axis=0)
        mel_matrix = np.stack(mel_features, axis=0)

        spatial_features = self._compute_spatial_covariance_features(
            stft_matrix, eps, normalize_eigenvector
        )

        n_channels, n_mels, n_frames = mel_matrix.shape
        spatial_features.shape[1]

        stacked_mel = mel_matrix.reshape(n_channels * n_mels, n_frames)

        salsa_features = np.vstack([stacked_mel, spatial_features])

        return salsa_features

    def _compute_spatial_covariance_features(
        self, stft_matrix: np.ndarray, eps: float, normalize: bool
    ) -> np.ndarray:
        """Compute spatial covariance features from multi-channel STFT.

        Args:
            stft_matrix: Multi-channel STFT with shape (n_channels, n_freqs, n_frames)
            eps: Small value to avoid numerical issues
            normalize: Whether to normalize the principal eigenvector

        Returns:
            Principal eigenvectors with shape (n_freqs, n_frames)
        """
        n_channels, n_freqs, n_frames = stft_matrix.shape
        spatial_features = np.zeros((n_freqs, n_frames), dtype=np.float32)

        for freq_bin in range(n_freqs):
            for time_frame in range(n_frames):
                stft_vec = stft_matrix[:, freq_bin, time_frame]

                covariance_matrix = np.outer(stft_vec, np.conj(stft_vec))
                covariance_matrix = covariance_matrix.real + eps * np.eye(n_channels)

                eigenvals, eigenvecs = eigh(covariance_matrix)

                principal_eigenvec = eigenvecs[:, -1].real

                if normalize:
                    norm = np.linalg.norm(principal_eigenvec)
                    if norm > eps:
                        principal_eigenvec = principal_eigenvec / norm

                spatial_features[freq_bin, time_frame] = np.mean(principal_eigenvec)

        return spatial_features


class SalsaLiteExtractor(BaseFeatureExtractor):
    """SALSA-Lite feature extractor.

    A lightweight variation of SALSA based on Inter-channel Phase Differences (IPD)
    instead of full covariance analysis. Much faster and still effective for
    spatial audio analysis.

    This implementation follows the Single Responsibility Principle by focusing
    only on SALSA-Lite feature extraction.
    """

    def __init__(self) -> None:
        """Initialize SALSA-Lite extractor."""
        super().__init__(FeatureType.SALSA_LITE)

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "SALSA-Lite"

    @property
    def required_channels(self) -> int:
        """SALSA-Lite requires 4-channel FOA input."""
        return 4

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for SALSA-Lite."""
        return {"reference_channel": 0, "phase_wrapping": True, "eps": 1e-8}

    def _extract_features(self, audio_data: AudioData, config: FeatureConfig) -> np.ndarray:
        """Extract SALSA-Lite features from FOA audio data.

        Args:
            audio_data: 4-channel FOA audio data
            config: Extraction configuration

        Returns:
            SALSA-Lite features with shape (n_mels * n_channels + n_ipd_features, n_frames)
        """
        if not audio_data.is_foa:
            raise ValueError("SALSA-Lite extractor requires 4-channel FOA input")

        signal = audio_data.signal
        ref_channel = config.parameters.get(
            "reference_channel", self.get_default_config()["reference_channel"]
        )
        phase_wrapping = config.parameters.get(
            "phase_wrapping", self.get_default_config()["phase_wrapping"]
        )
        eps = config.parameters.get("eps", self.get_default_config()["eps"])

        stft_channels = []
        mel_features = []

        for channel in range(4):
            stft = self._compute_stft(signal[channel], config)
            stft_channels.append(stft)

            mel_spec = self._compute_mel_spectrogram(stft, config)
            mel_features.append(mel_spec)

        stft_matrix = np.stack(stft_channels, axis=0)
        mel_matrix = np.stack(mel_features, axis=0)

        ipd_features = self._compute_ipd_features(stft_matrix, ref_channel, phase_wrapping, eps)

        n_channels, n_mels, n_frames = mel_matrix.shape
        stacked_mel = mel_matrix.reshape(n_channels * n_mels, n_frames)

        salsa_lite_features = np.vstack([stacked_mel, ipd_features])

        return salsa_lite_features

    def _compute_ipd_features(
        self, stft_matrix: np.ndarray, ref_channel: int, phase_wrapping: bool, eps: float
    ) -> np.ndarray:
        """Compute Inter-channel Phase Difference features.

        Args:
            stft_matrix: Multi-channel STFT with shape (n_channels, n_freqs, n_frames)
            ref_channel: Reference channel for phase differences
            phase_wrapping: Whether to apply phase wrapping
            eps: Small value to avoid numerical issues

        Returns:
            IPD features with shape (n_ipd_pairs, n_frames)
        """
        n_channels, n_freqs, n_frames = stft_matrix.shape

        ref_stft = stft_matrix[ref_channel]
        ipd_features = []

        for channel in range(n_channels):
            if channel == ref_channel:
                continue

            channel_stft = stft_matrix[channel]

            phase_diff = np.angle(channel_stft) - np.angle(ref_stft + eps)

            if phase_wrapping:
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

            ipd_feature = np.mean(phase_diff, axis=0)
            ipd_features.append(ipd_feature)

        return np.vstack(ipd_features)
