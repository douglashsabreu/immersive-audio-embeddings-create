"""Reorganizar dataset com as 4 classes corretas de áudio espacial."""

import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


def extract_audio_category(filename: str) -> str:
    """Extrai categoria de áudio do nome do arquivo.
    
    Args:
        filename: Nome do arquivo (ex: immersive1_snippet001)
        
    Returns:
        Categoria: '5.1+4h', '5.1', '2.0', '1.0'
    """
    filename_lower = filename.lower()
    
    if filename_lower.startswith('immersive'):
        return '5.1+4h'  # Immersive = 5.1 + altura (4 canais altura)
    elif filename_lower.startswith('surround'):
        return '5.1'     # Surround = 5.1 tradicional
    elif filename_lower.startswith('stereo'):
        return '2.0'     # Stereo = 2 canais
    elif filename_lower.startswith('mono'):
        return '1.0'     # Mono = 1 canal
    else:
        return 'unknown'


def reorganize_dataset_with_4_classes(
    features_dir: Path, 
    output_file: Path
) -> Dict[str, int]:
    """Reorganiza dataset com 4 classes de formato de áudio.
    
    Args:
        features_dir: Diretório com arquivos de features
        output_file: Arquivo de saída para dataset reorganizado
    """
    print("🎯 REORGANIZANDO DATASET COM 4 CLASSES DE FORMATO")
    print("=" * 60)
    print("📋 Classes:")
    print("   🎧 5.1+4h: Immersive (5.1 + 4 canais de altura)")
    print("   🔊 5.1:    Surround (5.1 tradicional)")
    print("   🎵 2.0:    Stereo (2 canais)")
    print("   📻 1.0:    Mono (1 canal)")
    print()
    
    # Encontrar arquivos de features
    feature_files = list(features_dir.glob("*_features.npz"))
    print(f"📊 Features encontradas: {len(feature_files)}")
    
    if not feature_files:
        print("❌ Nenhum arquivo de features encontrado!")
        return {}
    
    # Coletar dados
    all_features = []
    all_labels = []
    all_filenames = []
    all_numeric_labels = []
    
    # Mapeamento para labels numéricos
    label_to_numeric = {
        '5.1+4h': 0,
        '5.1': 1, 
        '2.0': 2,
        '1.0': 3
    }
    
    print("🔄 Processando arquivos...")
    for feature_file in tqdm(feature_files):
        try:
            data = np.load(feature_file, allow_pickle=True)
            
            # Extrair features combinadas
            combined_features = data["combined_features"]
            all_features.append(combined_features)
            
            # Extrair filename base
            filename = feature_file.stem.replace("_features", "")
            all_filenames.append(filename)
            
            # Extrair categoria correta
            category = extract_audio_category(filename)
            all_labels.append(category)
            
            # Label numérico
            if category in label_to_numeric:
                all_numeric_labels.append(label_to_numeric[category])
            else:
                print(f"⚠️  Categoria desconhecida: {category} para {filename}")
                all_numeric_labels.append(-1)  # Unknowns
                
        except Exception as e:
            print(f"❌ Erro em {feature_file}: {e}")
            continue
    
    # Converter para arrays
    features_matrix = np.array(all_features)
    labels_array = np.array(all_labels)
    filenames_array = np.array(all_filenames)
    numeric_labels_array = np.array(all_numeric_labels)
    
    # Estatísticas
    print(f"\n✅ DATASET REORGANIZADO:")
    print(f"   📊 Shape das features: {features_matrix.shape}")
    print(f"   🏷️  Classes detectadas: {sorted(set(all_labels))}")
    print(f"   📁 Distribuição por classe:")
    
    class_counts = {}
    for label in sorted(set(all_labels)):
        count = np.sum(labels_array == label)
        class_counts[label] = count
        
        # Emoji por classe
        emoji = {
            '5.1+4h': '🎧',
            '5.1': '🔊', 
            '2.0': '🎵',
            '1.0': '📻',
            'unknown': '❓'
        }.get(label, '📄')
        
        print(f"      {emoji} {label}: {count} exemplos")
    
    # Verificar balanceamento
    total_known = sum(v for k, v in class_counts.items() if k != 'unknown')
    print(f"\n📈 ANÁLISE DO DATASET:")
    print(f"   📋 Total de exemplos: {len(all_features)}")
    print(f"   ✅ Classificados: {total_known}")
    print(f"   ❓ Desconhecidos: {class_counts.get('unknown', 0)}")
    
    if total_known > 0:
        print(f"   ⚖️  Balanceamento:")
        for label, count in class_counts.items():
            if label != 'unknown':
                percentage = (count / total_known) * 100
                print(f"      - {label}: {percentage:.1f}%")
    
    # Salvar dataset reorganizado
    np.savez_compressed(
        output_file,
        features=features_matrix,
        labels=labels_array,
        numeric_labels=numeric_labels_array,
        filenames=filenames_array,
        label_names=['5.1+4h', '5.1', '2.0', '1.0'],
        label_to_numeric=label_to_numeric,
        class_counts=class_counts,
        n_features=features_matrix.shape[1],
        n_samples=features_matrix.shape[0],
        n_classes=4
    )
    
    print(f"\n💾 Dataset salvo em: {output_file}")
    print(f"📊 Pronto para treinamento de classificador!")
    
    return class_counts


def analyze_class_distribution(dataset_file: Path) -> None:
    """Analisa distribuição das classes no dataset."""
    print(f"\n📈 ANÁLISE DETALHADA DA DISTRIBUIÇÃO:")
    
    data = np.load(dataset_file, allow_pickle=True)
    labels = data['labels']
    filenames = data['filenames']
    
    # Análise por fonte (immersive1, immersive2, etc.)
    source_analysis: Dict[str, Dict[int, int]] = {}
    for i, filename in enumerate(filenames):
        source = filename.split('_')[0]  # immersive1, surround2, etc.
        label = labels[i]
        
        if source not in source_analysis:
            source_analysis[source] = {}
        if label not in source_analysis[source]:
            source_analysis[source][label] = 0
        source_analysis[source][label] += 1
    
    print("📊 Distribuição por fonte:")
    for source in sorted(source_analysis.keys()):
        counts = source_analysis[source]
        main_label = max(counts, key=lambda x: counts[x])
        total = sum(counts.values())
        print(f"   {source}: {total} exemplos → {main_label}")


def main() -> None:
    """Reorganiza dataset com classificação correta."""
    features_dir = Path("features_extracted")
    dataset_file = Path("spatial_audio_dataset_4classes.npz")
    
    # Reorganizar
    class_counts = reorganize_dataset_with_4_classes(features_dir, dataset_file)
    
    # Análise detalhada
    if dataset_file.exists():
        analyze_class_distribution(dataset_file)
    
    print(f"\n🚀 PRÓXIMOS PASSOS:")
    print(f"   1. 🧠 Treinar modelo de embeddings: python train_classifier.py")
    print(f"   2. 📊 Avaliar separação entre as 4 classes")
    print(f"   3. 🎯 Criar classificador supervisionado/auto-supervisionado")


if __name__ == "__main__":
    main()
