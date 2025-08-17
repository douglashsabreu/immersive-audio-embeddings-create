"""Adaptadores para extrair features espaciais de áudio binaural."""


import numpy as np
from scipy import signal

from ..models.audio_data import AudioData
from ..models.feature_config import FeatureConfig


class BinauralSpatialExtractor:
    """Extrator de features espaciais adaptado para áudio binaural."""

    def __init__(self) -> None:
        self.name = "Binaural_Spatial"

    def extract_binaural_spatial_features(
        self, audio_data: AudioData, config: FeatureConfig
    ) -> dict[str, np.ndarray]:
        """Extrai features espaciais de áudio binaural.

        Args:
            audio_data: Dados de áudio binaural (2 canais)
            config: Configuração de extração

        Returns:
            Dicionário com diferentes features espaciais
        """
        if not audio_data.is_binaural:
            raise ValueError("Requer entrada binaural (2 canais)")

        left_channel = audio_data.signal[0]
        right_channel = audio_data.signal[1]

        features = {}

        # 1. Features de diferença interaural (IPD/ITD)
        features["interaural_features"] = self._extract_interaural_features(
            left_channel, right_channel, audio_data.sample_rate, config
        )

        # 2. Features de correlação cruzada espacial
        features["spatial_correlation"] = self._extract_spatial_correlation(
            left_channel, right_channel, config
        )

        # 3. Features espectrais comparativas
        features["spectral_differences"] = self._extract_spectral_differences(
            left_channel, right_channel, config
        )

        # 4. Pseudo-intensidade (simulação baseada em diferenças L/R)
        features["pseudo_intensity"] = self._extract_pseudo_intensity(
            left_channel, right_channel, config
        )

        return features

    def _extract_interaural_features(
        self, left: np.ndarray, right: np.ndarray, sample_rate: int, config: FeatureConfig
    ) -> np.ndarray:
        """Extrai features de diferenças interaurais."""

        # STFT de ambos os canais
        f, t, left_stft = signal.stft(
            left, fs=sample_rate, nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
        )
        _, _, right_stft = signal.stft(
            right, fs=sample_rate, nperseg=config.n_fft, noverlap=config.n_fft - config.hop_length
        )

        # IPD - Diferenças de fase interaural
        ipd = np.angle(left_stft) - np.angle(right_stft)
        ipd_wrapped = np.angle(np.exp(1j * ipd))  # Wrap to [-π, π]

        # ILD - Diferenças de nível interaural
        left_mag = np.abs(left_stft)
        right_mag = np.abs(right_stft)
        ild = 20 * np.log10((left_mag + 1e-12) / (right_mag + 1e-12))

        # Estatísticas por banda de frequência
        n_bands = 8
        freq_bands = np.logspace(np.log10(f[1]), np.log10(f[-1]), n_bands + 1)

        features = []
        for i in range(n_bands):
            # Encontrar índices da banda
            f_low = freq_bands[i]
            f_high = freq_bands[i + 1]
            band_indices = (f >= f_low) & (f <= f_high)

            if np.any(band_indices):
                # Estatísticas IPD
                ipd_band = ipd_wrapped[band_indices, :]
                features.extend(
                    [
                        np.mean(ipd_band),
                        np.std(ipd_band),
                        np.mean(np.abs(ipd_band)),  # Coerência de fase
                    ]
                )

                # Estatísticas ILD
                ild_band = ild[band_indices, :]
                features.extend([np.mean(ild_band), np.std(ild_band)])

        return np.array(features)

    def _extract_spatial_correlation(
        self, left: np.ndarray, right: np.ndarray, config: FeatureConfig
    ) -> np.ndarray:
        """Extrai features de correlação espacial."""

        # Correlação cruzada normalizada em janelas temporais
        window_size = int(0.1 * config.sample_rate)  # 100ms windows
        hop_size = window_size // 2

        correlations = []
        for i in range(0, len(left) - window_size, hop_size):
            left_win = left[i : i + window_size]
            right_win = right[i : i + window_size]

            # Correlação cruzada
            correlation = np.corrcoef(left_win, right_win)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)

        correlations = np.array(correlations)

        # Estatísticas temporais
        features = [
            np.mean(correlations),
            np.std(correlations),
            np.median(correlations),
            np.percentile(correlations, 25),
            np.percentile(correlations, 75),
            np.min(correlations),
            np.max(correlations),
        ]

        return np.array(features)

    def _extract_spectral_differences(
        self, left: np.ndarray, right: np.ndarray, config: FeatureConfig
    ) -> np.ndarray:
        """Extrai features de diferenças espectrais."""

        # Spectrograma de cada canal
        f, t, left_spec = signal.spectrogram(
            left,
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )
        _, _, right_spec = signal.spectrogram(
            right,
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )

        # Diferenças espectrais
        spectral_diff = np.log(left_spec + 1e-12) - np.log(right_spec + 1e-12)

        # Centro espectral médio de cada canal
        left_centroid = np.sum(f[:, np.newaxis] * left_spec, axis=0) / np.sum(left_spec, axis=0)
        right_centroid = np.sum(f[:, np.newaxis] * right_spec, axis=0) / np.sum(right_spec, axis=0)

        features = [
            np.mean(spectral_diff),
            np.std(spectral_diff),
            np.mean(left_centroid - right_centroid),  # Diferença de centroide
            np.std(left_centroid - right_centroid),
        ]

        return np.array(features)

    def _extract_pseudo_intensity(
        self, left: np.ndarray, right: np.ndarray, config: FeatureConfig
    ) -> np.ndarray:
        """Simula vetor de intensidade baseado em diferenças L/R."""

        # Aproximação: intensidade horizontal baseada em diferenças L/R
        # Left = +1, Right = -1, Center = 0

        # STFT
        f, t, left_stft = signal.stft(
            left,
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )
        _, _, right_stft = signal.stft(
            right,
            fs=config.sample_rate,
            nperseg=config.n_fft,
            noverlap=config.n_fft - config.hop_length,
        )

        # Pseudo vetor de intensidade (só componente horizontal)
        left_energy = np.abs(left_stft) ** 2
        right_energy = np.abs(right_stft) ** 2
        total_energy = left_energy + right_energy + 1e-12

        # Normalizar para [-1, 1] (esquerda para direita)
        horizontal_intensity = (left_energy - right_energy) / total_energy

        # Estatísticas por banda de frequência
        n_bands = 6
        freq_bands = np.logspace(np.log10(f[1]), np.log10(f[-1]), n_bands + 1)

        features = []
        for i in range(n_bands):
            f_low = freq_bands[i]
            f_high = freq_bands[i + 1]
            band_indices = (f >= f_low) & (f <= f_high)

            if np.any(band_indices):
                intensity_band = horizontal_intensity[band_indices, :]

                features.extend(
                    [
                        np.mean(intensity_band),  # Posição média
                        np.std(intensity_band),  # Variabilidade espacial
                        np.mean(np.abs(intensity_band)),  # Lateralização média
                        np.percentile(np.abs(intensity_band), 95),  # Pico de lateralização
                    ]
                )

        return np.array(features)


def create_binaural_feature_extractor() -> BinauralSpatialExtractor:
    """Factory function para criar extrator binaural."""
    return BinauralSpatialExtractor()
