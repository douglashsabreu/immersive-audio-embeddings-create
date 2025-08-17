"""Modelo de embedding para features espaciais de áudio binaural."""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAudioEncoder(nn.Module):
    """Encoder leve para converter features espaciais em embeddings.

    Baseado no planejamento:
    - CNN/Conformer leve
    - Global pooling (média+desvio+pico+entropia)
    - Saída L2-normalizada 128-256D
    """

    def __init__(
        self,
        input_dim: int = 75,  # Features espaciais binaural
        embedding_dim: int = 128,
        hidden_dims: Tuple[int, ...] = (256, 512, 256),
        dropout_rate: float = 0.1,
    ):
        """Inicializa o encoder.

        Args:
            input_dim: Dimensão das features de entrada
            embedding_dim: Dimensão do embedding final
            hidden_dims: Dimensões das camadas ocultas
            dropout_rate: Taxa de dropout
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Encoder CNN leve
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Projeção final para embedding
        self.projection = nn.Linear(prev_dim, embedding_dim)

        # Inicialização
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa pesos do modelo."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Features de entrada [batch_size, input_dim]

        Returns:
            Embeddings L2-normalizados [batch_size, embedding_dim]
        """
        # Encoder
        features = self.encoder(x)

        # Projeção
        embeddings = self.projection(features)

        # L2 normalização
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class GlobalPoolingEncoder(nn.Module):
    """Encoder com global pooling para features temporais.

    Para quando temos features que variam no tempo (T, F).
    Aplica pooling: média+desvio+pico+entropia.
    """

    def __init__(self, feature_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        """Inicializa encoder com global pooling.

        Args:
            feature_dim: Dimensão das features por frame
            embedding_dim: Dimensão do embedding final
            hidden_dim: Dimensão da camada oculta
        """
        super().__init__()

        # 4 estatísticas: mean, std, max, entropy
        pooled_dim = feature_dim * 4

        self.encoder = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def global_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica global pooling.

        Args:
            x: Features [batch_size, time, feature_dim]

        Returns:
            Pooled features [batch_size, feature_dim * 4]
        """
        # Mean pooling
        mean_pool = torch.mean(x, dim=1)

        # Std pooling
        std_pool = torch.std(x, dim=1)

        # Max pooling
        max_pool, _ = torch.max(x, dim=1)

        # Entropy pooling (aproximação)
        # Normalizar para probabilidades
        x_norm = F.softmax(x, dim=1)
        entropy = -torch.sum(x_norm * torch.log(x_norm + 1e-8), dim=1)

        # Concatenar
        pooled = torch.cat([mean_pool, std_pool, max_pool, entropy], dim=1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com global pooling.

        Args:
            x: Features [batch_size, time, feature_dim]

        Returns:
            Embeddings L2-normalizados [batch_size, embedding_dim]
        """
        # Global pooling
        pooled = self.global_pooling(x)

        # Encoder
        embeddings = self.encoder(pooled)

        # L2 normalização
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class ContrastiveLoss(nn.Module):
    """Loss contrastiva para aprendizado auto-supervisionado."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calcula loss contrastiva.

        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Labels [batch_size]

        Returns:
            Loss contrastiva
        """
        batch_size = embeddings.size(0)

        # Similaridade coseno
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Máscara para pares positivos (mesmo label)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (auto-similaridade)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Softmax
        exp_sim = torch.exp(sim_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        log_prob = sim_matrix - torch.log(sum_exp_sim)

        # Loss para pares positivos
        pos_mask = mask / torch.sum(mask, dim=1, keepdim=True).clamp(min=1e-8)
        loss = -torch.sum(pos_mask * log_prob, dim=1)

        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss para separação de clusters."""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Calcula triplet loss.

        Args:
            anchor: Embeddings âncora
            positive: Embeddings positivos (mesma classe)
            negative: Embeddings negativos (classe diferente)

        Returns:
            Triplet loss
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def create_embeddings_from_features(
    features_file: str, model_path: Optional[str] = None
) -> np.ndarray:
    """Cria embeddings a partir de features extraídas.

    Args:
        features_file: Arquivo .npz com features
        model_path: Caminho para modelo treinado

    Returns:
        Array de embeddings
    """
    # Carregar features
    data = np.load(features_file)
    features = data["features"]  # [n_samples, n_features]

    # Modelo
    model = SpatialAudioEncoder(input_dim=features.shape[1])

    if model_path:
        model.load_state_dict(torch.load(model_path))

    model.eval()

    # Converter para tensor
    features_tensor = torch.FloatTensor(features)

    # Gerar embeddings
    with torch.no_grad():
        embeddings = model(features_tensor)

    return embeddings.numpy()


if __name__ == "__main__":
    # Exemplo de uso
    print("🧠 MODELO DE EMBEDDINGS PARA FEATURES ESPACIAIS")
    print("=" * 50)

    # Parâmetros
    batch_size = 32
    input_dim = 75  # Features binaural
    embedding_dim = 128

    # Modelo
    model = SpatialAudioEncoder(input_dim=input_dim, embedding_dim=embedding_dim)

    # Features simuladas
    features = torch.randn(batch_size, input_dim)

    # Forward pass
    embeddings = model(features)

    print(f"✅ Modelo criado:")
    print(f"   📊 Input: {input_dim}D")
    print(f"   🎯 Output: {embedding_dim}D")
    print(f"   📏 Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🧮 Exemplo: {features.shape} → {embeddings.shape}")
    print(f"   📐 L2 norm: {torch.norm(embeddings[0]).item():.4f}")

    print(f"\n➡️  Próximos passos:")
    print(f"   1. Treinar com dataset extraído")
    print(f"   2. Usar ContrastiveLoss ou TripletLoss")
    print(f"   3. Avaliar separação immersive vs surround")
