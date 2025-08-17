"""Gera embeddings das features espaciais usando o modelo encoder."""

from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..embeddings.embedding_model import SpatialAudioEncoder


def generate_embeddings_from_dataset(
    dataset_file: Path, embedding_dim: int = 128, model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Gera embeddings a partir do dataset de features.

    Args:
        dataset_file: Arquivo .npz com features
        embedding_dim: Dimensão dos embeddings
        model_path: Caminho para modelo treinado (opcional)

    Returns:
        Dicionário com embeddings e metadados
    """
    print("🧠 GERANDO EMBEDDINGS DE FEATURES ESPACIAIS")
    print("=" * 50)

    # Detectar dispositivo (Metal/MPS para Mac M1/M2/M3)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Usando Metal Performance Shaders (MPS) - Mac Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Usando CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚡ Usando CPU")

    print(f"   📱 Dispositivo: {device}")

    # Carregar dataset
    print(f"📂 Carregando dataset: {dataset_file}")
    data = np.load(dataset_file, allow_pickle=True)

    features = data["features"]  # [n_samples, n_features]
    labels = data["labels"]
    filenames = data["filenames"]
    numeric_labels = data["numeric_labels"]

    print(f"📊 Dataset:")
    print(f"   🔢 Samples: {features.shape[0]}")
    print(f"   🎯 Features: {features.shape[1]}")
    print(f"   🏷️  Classes: {len(set(labels))}")

    # Criar modelo
    print(f"\n🏗️  Criando modelo encoder...")
    model = SpatialAudioEncoder(
        input_dim=features.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=(256, 512, 256),
        dropout_rate=0.1,
    )

    # Mover modelo para dispositivo
    model = model.to(device)

    # Carregar pesos se fornecido
    if model_path is not None and model_path.exists():
        print(f"📥 Carregando modelo treinado: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("🆕 Usando modelo com pesos iniciais (não treinado)")

    model.eval()

    # Converter para tensor e mover para dispositivo
    print(f"🔄 Convertendo features para tensor...")
    features_tensor = torch.FloatTensor(features).to(device)

    print(f"   📊 Input shape: {features_tensor.shape}")
    print(f"   🎯 Input device: {features_tensor.device}")

    # Gerar embeddings com medição de tempo
    print(f"⚡ Gerando embeddings no {device}...")

    import time

    start_time = time.time()

    with torch.no_grad():
        embeddings = model(features_tensor)

        # Sincronizar GPU se necessário
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # Mover embeddings de volta para CPU para salvar
    embeddings_np = embeddings.cpu().numpy()

    print(f"⚡ Tempo de inferência: {inference_time:.3f}s")
    print(f"   📈 Throughput: {len(features) / (inference_time):.0f} samples/sec")

    print(f"✅ Embeddings gerados:")
    print(f"   📊 Shape: {embeddings_np.shape}")
    print(f"   📏 Dimensão: {embedding_dim}D")
    print(f"   📐 L2 norm média: {np.linalg.norm(embeddings_np, axis=1).mean():.4f}")
    print(f"   📉 L2 norm std: {np.linalg.norm(embeddings_np, axis=1).std():.4f}")

    # Verificar normalização L2
    norms = np.linalg.norm(embeddings_np, axis=1)
    if np.allclose(norms, 1.0, atol=1e-6):
        print("   ✅ Embeddings corretamente L2-normalizados")
    else:
        print(
            f"   ⚠️  Embeddings não normalizados (norm range: {norms.min():.4f}-{norms.max():.4f})"
        )

    # Estatísticas por classe
    print(f"\n📈 ANÁLISE POR CLASSE:")
    for label in sorted(set(labels)):
        mask = labels == label
        class_embeddings = embeddings_np[mask]

        # Centróide da classe
        centroid = np.mean(class_embeddings, axis=0)

        # Distâncias intra-classe
        intra_distances = []
        for emb in class_embeddings:
            dist = np.linalg.norm(emb - centroid)
            intra_distances.append(dist)

        emoji = {"5.1+4h": "🎧", "5.1": "🔊", "2.0": "🎵", "1.0": "📻"}.get(label, "📄")

        print(f"   {emoji} {label}: {len(class_embeddings)} exemplos")
        print(f"      - Distância intra-classe média: {np.mean(intra_distances):.4f}")
        print(f"      - Distância intra-classe std: {np.std(intra_distances):.4f}")

    return {
        "embeddings": embeddings_np,
        "labels": labels,
        "numeric_labels": numeric_labels,
        "filenames": filenames,
        "features_original": features,
        "embedding_dim": embedding_dim,
        "n_samples": features.shape[0],
        "n_features": features.shape[1],
        "model_params": sum(p.numel() for p in model.parameters()),
    }


def save_embeddings(embeddings_dict: Dict[str, Any], output_file: Path) -> None:
    """Salva embeddings em arquivo .npz.

    Args:
        embeddings_dict: Dicionário com embeddings e metadados
        output_file: Arquivo de saída
    """
    print(f"\n💾 Salvando embeddings: {output_file}")

    np.savez_compressed(
        output_file,
        **embeddings_dict,
        creation_time=np.array(str(Path(__file__).name)),
        description="Spatial audio embeddings from binaural features",
    )

    print(f"✅ Embeddings salvos com sucesso!")
    print(f"   📊 Arquivo: {output_file}")
    print(f"   💿 Tamanho: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def visualize_embeddings_2d(embeddings_dict: Dict[str, Any], output_dir: Path, method: str = "tsne") -> None:
    """Visualiza embeddings em 2D usando t-SNE ou PCA.

    Args:
        embeddings_dict: Dicionário com embeddings
        output_dir: Diretório para salvar visualizações
        method: 'tsne' ou 'pca'
    """
    print(f"\n📊 CRIANDO VISUALIZAÇÃO 2D ({method.upper()})...")

    embeddings = embeddings_dict["embeddings"]
    labels = embeddings_dict["labels"]
    n_samples = embeddings.shape[0]

    # Redução de dimensionalidade
    if method == "tsne":
        # Ajustar perplexity para datasets pequenos
        perplexity = min(30, max(1, n_samples - 2)) if n_samples > 2 else 1
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        print(
            f"   📈 Variância explicada: {explained_var[0]:.3f} + {explained_var[1]:.3f} = {sum(explained_var):.3f}"
        )
    else:
        raise ValueError(f"Método {method} não suportado")

    # Plot
    plt.figure(figsize=(12, 8))

    # Cores e símbolos por classe
    colors = {"5.1+4h": "#FF6B35", "5.1": "#004E89", "2.0": "#00A896", "1.0": "#7209B7"}
    markers = {"5.1+4h": "o", "5.1": "s", "2.0": "^", "1.0": "D"}

    for label in sorted(set(labels)):
        mask = labels == label
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]

        plt.scatter(
            x,
            y,
            c=colors.get(label, "#gray"),
            marker=markers.get(label, "o"),
            s=60,
            alpha=0.7,
            label=f"{label} ({sum(mask)} exemplos)",
            edgecolors="black",
            linewidth=0.5,
        )

    plt.title(f"Embeddings de Áudio Espacial - {method.upper()}", fontsize=16, fontweight="bold")
    plt.xlabel(f"{method.upper()} Componente 1", fontsize=12)
    plt.ylabel(f"{method.upper()} Componente 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Salvar
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f"embeddings_{method}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"   💾 Visualização salva: {plot_file}")

    plt.show()


def compute_embedding_metrics(embeddings_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula métricas de qualidade dos embeddings.

    Args:
        embeddings_dict: Dicionário com embeddings

    Returns:
        Dicionário com métricas
    """
    print(f"\n📏 CALCULANDO MÉTRICAS DE QUALIDADE...")

    embeddings = embeddings_dict["embeddings"]
    labels = embeddings_dict["labels"]

    metrics = {}

    # 1. Separação inter-classe (silhouette score aproximado)
    from sklearn.metrics import silhouette_score

    silhouette = silhouette_score(embeddings, labels)
    metrics["silhouette_score"] = silhouette

    # 2. Distâncias intra vs inter-classe
    unique_labels = sorted(set(labels))
    intra_distances = []
    inter_distances = []

    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]

        # Intra-classe: distâncias dentro da classe
        centroid = np.mean(class_embeddings, axis=0)
        for emb in class_embeddings:
            intra_distances.append(np.linalg.norm(emb - centroid))

        # Inter-classe: distância para outras classes
        for other_label in unique_labels:
            if other_label != label:
                other_mask = labels == other_label
                other_embeddings = embeddings[other_mask]
                other_centroid = np.mean(other_embeddings, axis=0)
                inter_distances.append(np.linalg.norm(centroid - other_centroid))

    metrics["intra_class_distance_mean"] = np.mean(intra_distances)
    metrics["inter_class_distance_mean"] = np.mean(inter_distances)
    metrics["separation_ratio"] = np.mean(inter_distances) / np.mean(intra_distances)

    print(f"   📊 Silhouette Score: {silhouette:.4f}")
    print(f"   📏 Distância intra-classe média: {np.mean(intra_distances):.4f}")
    print(f"   📏 Distância inter-classe média: {np.mean(inter_distances):.4f}")
    print(f"   🎯 Razão de separação: {metrics['separation_ratio']:.4f}")

    return metrics


def main() -> None:
    """Gera embeddings do dataset de features espaciais."""

    # Arquivos
    dataset_file = Path("spatial_audio_dataset_4classes.npz")
    embeddings_file = Path("spatial_audio_embeddings.npz")
    visualizations_dir = Path("embeddings_visualizations")

    if not dataset_file.exists():
        print(f"❌ Dataset não encontrado: {dataset_file}")
        print("   Execute: python reorganize_dataset.py")
        return

    # Gerar embeddings
    embeddings_dict = generate_embeddings_from_dataset(
        dataset_file=dataset_file,
        embedding_dim=128,
        model_path=None,  # Sem modelo treinado por enquanto
    )

    # Salvar embeddings
    save_embeddings(embeddings_dict, embeddings_file)

    # Métricas de qualidade
    metrics = compute_embedding_metrics(embeddings_dict)

    # Visualizações
    visualize_embeddings_2d(embeddings_dict, visualizations_dir, method="tsne")
    visualize_embeddings_2d(embeddings_dict, visualizations_dir, method="pca")

    print(f"\n🎉 EMBEDDINGS GERADOS COM SUCESSO!")
    print(f"📊 Arquivo principal: {embeddings_file}")
    print(f"🎨 Visualizações: {visualizations_dir}/")
    print(f"🎯 Próximo: Use os embeddings para treinar classificadores externos!")


if __name__ == "__main__":
    main()
