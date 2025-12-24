"""
Arquitectura Transformer para clasificación de secuencias de landmarks LSM.

Incluye:
    - Feature Weights: Prioriza manos sobre cara (configurable en settings.py)
    - Regularización avanzada: Label Smoothing, LayerNorm, Dropout
    - Arquitectura optimizada para secuencias temporales
"""

import torch
import torch.nn as nn
import math

from src.config.settings import (
    INPUT_DIM, D_MODEL, N_HEADS, N_LAYERS, DROPOUT, NUM_CLASSES, MAX_SEQ_LEN,
    FEATURE_WEIGHTS
)

# =============================================================================
# FEATURE WEIGHTS - Priorizar manos sobre cara
# =============================================================================
# RTMPose-WholeBody: 133 keypoints × 2 (x,y) = 266 features
# Índices:
#   Cuerpo: 0-16 (17 puntos) → índices 0-33
#   Pies: 17-22 (6 puntos) → índices 34-45
#   Cara: 23-90 (68 puntos) → índices 46-181
#   Mano izquierda: 91-111 (21 puntos) → índices 182-223
#   Mano derecha: 112-132 (21 puntos) → índices 224-265

def create_feature_weights(input_dim: int = INPUT_DIM) -> torch.Tensor:
    """
    Crea pesos por feature para enfatizar manos y reducir importancia de cara.
    Los pesos se leen de settings.FEATURE_WEIGHTS
    
    Returns:
        Tensor de shape [input_dim] con pesos por feature
    """
    weights = torch.ones(input_dim)
    
    # Definir rangos de índices (cada keypoint tiene 2 valores: x, y)
    body_start, body_end = 0, 17 * 2           # 0-33
    feet_start, feet_end = 17 * 2, 23 * 2      # 34-45
    face_start, face_end = 23 * 2, 91 * 2      # 46-181
    left_hand_start, left_hand_end = 91 * 2, 112 * 2   # 182-223
    right_hand_start, right_hand_end = 112 * 2, 133 * 2  # 224-265
    
    # Asignar pesos desde settings
    weights[body_start:body_end] = FEATURE_WEIGHTS['body']
    weights[feet_start:feet_end] = FEATURE_WEIGHTS['feet']
    weights[face_start:face_end] = FEATURE_WEIGHTS['face']
    weights[left_hand_start:left_hand_end] = FEATURE_WEIGHTS['left_hand']
    weights[right_hand_start:right_hand_end] = FEATURE_WEIGHTS['right_hand']
    
    return weights


class FeatureWeightedEmbedding(nn.Module):
    """
    Embedding que aplica pesos por feature antes de la proyección.
    Permite al modelo enfocarse en manos mientras ignora la cara.
    """
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Pesos por feature (learnable pero inicializados con prioridad a manos)
        self.feature_weights = nn.Parameter(create_feature_weights(input_dim))
        
        # Proyección con LayerNorm para estabilidad
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),  # GELU suele funcionar mejor que ReLU en Transformers
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplicar pesos por feature
        x = x * self.feature_weights
        return self.projection(x)


class PositionalEncoding(nn.Module):
    """
    Inyecta información sobre el ORDEN de los cuadros.
    Sin esto, el modelo no sabría si la mano sube o baja.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSMTransformer(nn.Module):
    """
    Transformer Encoder para clasificación de secuencias de landmarks LSM.
    
    Arquitectura:
        1. Feature-Weighted Embedding: landmarks (266) -> d_model (128)
           - Pesos por feature: manos × 2.5, cara × 0.1
        2. Positional Encoding: agrega información temporal
        3. Transformer Encoder: procesa la secuencia
        4. Global Average Pooling: resume la secuencia
        5. Classification Head: produce logits para cada clase
    """
    
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_classes: int = NUM_CLASSES,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Feature-Weighted Embedding
        self.input_embedding = FeatureWeightedEmbedding(input_dim, d_model, dropout)
        
        # 2. Positional Encoding con dropout
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. Transformer Encoder con pre-LN (más estable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            activation='gelu',
            norm_first=True  # Pre-LN para mejor estabilidad
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # Final LayerNorm
        )
        
        # 4. Classification Head con regularización
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 1.5),  # Mayor dropout antes del clasificador
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [Batch, Seq_Len, Input_Dim] - Secuencia de landmarks
            
        Returns:
            [Batch, Num_Classes] - Logits de clasificación
        """
        # Embedding con pesos por feature
        x = self.input_embedding(x)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_weights(self) -> torch.Tensor:
        """Retorna los pesos de features aprendidos."""
        return self.input_embedding.feature_weights.data


def create_model(device: str = 'cuda') -> LSMTransformer:
    """Factory function para crear el modelo."""
    model = LSMTransformer()
    model = model.to(device)
    print(f"📊 Modelo creado con {model.count_parameters():,} parámetros")
    return model
