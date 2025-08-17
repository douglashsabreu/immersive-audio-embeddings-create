"""Processamento em batch para extrair features espaciais de todos os áudios."""

import glob
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ..extractors.binaural_extractor import BinauralSpatialExtractor
from ..models.audio_data import AudioData
from ..models.feature_config import FeatureConfig, FeatureType


def extract_features_from_file(
    audio_file: Path, extractor: BinauralSpatialExtractor, config: FeatureConfig
) -> Optional[Dict[str, np.ndarray]]:
    """Extrai features de um único arquivo de áudio.

    Args:
        audio_file: Caminho para o arquivo de áudio
        extractor: Instância do extrator binaural
        config: Configuração de extração

    Returns:
        Dicionário com features extraídas ou None se erro
    """
    try:
        # Carregar áudio
        signal_data, sample_rate = sf.read(audio_file)

        # Verificar se é binaural
        if signal_data.ndim == 1:
            print(f"⚠️  {audio_file.name}: Mono - convertendo para estéreo")
            signal_data = np.vstack([signal_data, signal_data])
        elif signal_data.ndim == 2:
            # Transpor se necessário (samples, channels) -> (channels, samples)
            if signal_data.shape[0] > signal_data.shape[1]:
                signal_data = signal_data.T
        else:
            print(f"❌ {audio_file.name}: Formato inválido - {signal_data.shape}")
            return None

        # Criar AudioData
        audio_data = AudioData(signal=signal_data, sample_rate=sample_rate, file_path=audio_file)

        # Atualizar sample_rate na config se diferente
        if sample_rate != config.sample_rate:
            config = FeatureConfig(
                feature_type=config.feature_type,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                sample_rate=sample_rate,
                use_pcen=config.use_pcen,
                parameters=config.parameters,
            )

        # Extrair features
        features = extractor.extract_binaural_spatial_features(audio_data, config)

        # Adicionar metadados
        features["source_file"] = np.array(str(audio_file))
        features["sample_rate"] = np.array(sample_rate)
        features["duration"] = np.array(audio_data.duration)
        features["channels"] = np.array(audio_data.channels)

        return features

    except Exception as e:
        print(f"❌ Erro em {audio_file.name}: {str(e)}")
        return None


def batch_extract_features(
    input_dir: Path, output_dir: Path, file_pattern: str = "*.wav", max_files: Optional[int] = None
) -> None:
    """Extrai features de todos os arquivos de áudio em batch.

    Args:
        input_dir: Diretório com arquivos de áudio
        output_dir: Diretório para salvar features
        file_pattern: Padrão dos arquivos (ex: "*.wav", "immersive*.wav")
        max_files: Limite máximo de arquivos (para teste)
    """
    print("🎵 EXTRAÇÃO EM BATCH DE FEATURES ESPACIAIS BINAURAL 🎵")
    print("=" * 60)

    # Preparar diretórios
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar arquivos
    audio_files = list(glob.glob(str(input_dir / file_pattern)))
    if max_files:
        audio_files = audio_files[:max_files]

    print(f"📁 Diretório: {input_dir}")
    print(f"🔍 Padrão: {file_pattern}")
    print(f"📊 Arquivos encontrados: {len(audio_files)}")

    if not audio_files:
        print("❌ Nenhum arquivo encontrado!")
        return

    # Configuração padrão
    config = FeatureConfig(
        feature_type=FeatureType.IACC,  # Placeholder
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        sample_rate=48000,  # Será ajustado por arquivo
        use_pcen=True,
        parameters={},
    )

    # Inicializar extrator
    extractor = BinauralSpatialExtractor()

    # Estatísticas
    successful = 0
    failed = 0
    total_features = 0
    start_time = time.time()

    # Processar arquivos
    print(f"\n🚀 Iniciando processamento...")

    for audio_file in tqdm(audio_files, desc="Processando"):
        audio_path = Path(audio_file)

        # Extrair features
        features = extract_features_from_file(audio_path, extractor, config)

        if features:
            # Salvar features
            output_file = output_dir / f"{audio_path.stem}_features.npz"

            # Combinar features em array único para embeddings
            feature_arrays = [
                features["interaural_features"],
                features["spatial_correlation"],
                features["spectral_differences"],
                features["pseudo_intensity"],
            ]
            combined_features = np.concatenate([f.flatten() for f in feature_arrays])

            np.savez_compressed(
                output_file,
                **features,  # Features individuais
                combined_features=combined_features,  # Para embeddings
                extraction_time=time.time(),
            )

            successful += 1
            total_features += len(combined_features)

        else:
            failed += 1

    # Estatísticas finais
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n✅ PROCESSAMENTO CONCLUÍDO!")
    print("=" * 40)
    print(f"✅ Sucessos: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"📊 Taxa de sucesso: {successful / len(audio_files) * 100:.1f}%")
    print(f"🔢 Total de features por arquivo: {total_features // max(successful, 1)}")
    print(f"⏱️  Tempo total: {elapsed:.1f}s")
    print(f"⚡ Média por arquivo: {elapsed / len(audio_files):.2f}s")
    print(f"💾 Features salvas em: {output_dir}")


def organize_features_for_embeddings(features_dir: Path, output_file: Path) -> None:
    """Organiza todas as features extraídas em um dataset para embeddings.

    Args:
        features_dir: Diretório com arquivos .npz de features
        output_file: Arquivo para salvar dataset organizado
    """
    print(f"\n📋 ORGANIZANDO FEATURES PARA EMBEDDINGS...")

    # Encontrar todos os arquivos de features
    feature_files = list(features_dir.glob("*_features.npz"))

    if not feature_files:
        print("❌ Nenhum arquivo de features encontrado!")
        return

    print(f"📊 Features encontradas: {len(feature_files)}")

    # Coletar todas as features
    all_features = []
    all_labels = []
    all_filenames = []

    for feature_file in tqdm(feature_files, desc="Organizando"):
        try:
            data = np.load(feature_file, allow_pickle=True)

            # Extrair features combinadas
            combined_features = data["combined_features"]
            all_features.append(combined_features)

            # Extrair label do nome do arquivo (immersive1, immersive2, etc.)
            filename = feature_file.stem.replace("_features", "")
            all_filenames.append(filename)

            # Extrair categoria (immersive vs surround)
            if "immersive" in filename:
                label = "immersive"
            elif "surround" in filename:
                label = "surround"
            else:
                label = "unknown"
            all_labels.append(label)

        except Exception as e:
            print(f"❌ Erro em {feature_file}: {e}")
            continue

    if not all_features:
        print("❌ Nenhuma feature válida encontrada!")
        return

    # Converter para arrays numpy
    features_matrix = np.array(all_features)  # Shape: (n_samples, n_features)
    labels_array = np.array(all_labels)
    filenames_array = np.array(all_filenames)

    print(f"✅ Dataset organizado:")
    print(f"   📊 Shape das features: {features_matrix.shape}")
    print(f"   🏷️  Labels únicos: {np.unique(labels_array)}")
    print(f"   📁 Exemplos por categoria:")

    for label in np.unique(labels_array):
        count = np.sum(labels_array == label)
        print(f"      - {label}: {count} exemplos")

    # Salvar dataset
    np.savez_compressed(
        output_file,
        features=features_matrix,
        labels=labels_array,
        filenames=filenames_array,
        feature_names=[
            "interaural_features",
            "spatial_correlation",
            "spectral_differences",
            "pseudo_intensity",
        ],
        n_features=features_matrix.shape[1],
        n_samples=features_matrix.shape[0],
    )

    print(f"💾 Dataset salvo em: {output_file}")


def main() -> None:
    """Função principal para processamento em batch."""

    # Configurações
    input_dir = Path("audios_input")
    features_dir = Path("features_extracted")
    dataset_file = Path("spatial_audio_dataset.npz")

    print("🎯 PIPELINE DE EXTRAÇÃO DE FEATURES ESPACIAIS")
    print("=" * 50)

    # Passo 1: Extrair features de todos os áudios
    print("\n1️⃣ EXTRAINDO FEATURES DE TODOS OS ÁUDIOS...")
    batch_extract_features(
        input_dir=input_dir,
        output_dir=features_dir,
        file_pattern="*.wav",
        max_files=None,  # Processar todos - remova ou ajuste para teste
    )

    # Passo 2: Organizar para embeddings
    print("\n2️⃣ ORGANIZANDO DATASET PARA EMBEDDINGS...")
    organize_features_for_embeddings(features_dir, dataset_file)

    print(f"\n🎉 PIPELINE CONCLUÍDO!")
    print(f"➡️  Próximos passos:")
    print(f"   1. Criar modelo de embeddings (CNN/Conformer)")
    print(f"   2. Treinar classificador supervisionado/auto-supervisionado")
    print(f"   3. Avaliar separação entre categorias (immersive vs surround)")


if __name__ == "__main__":
    main()
