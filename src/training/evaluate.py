"""
Evaluación del modelo: métricas, matriz de confusión, reportes.
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import (
    PROCESSED_DATA_DIR, MLRUNS_DIR, CLASSES, MAX_SEQ_LEN
)
from src.models.transformer import LSMTransformer
from src.training.train import LSMDataset


def load_best_model(device: str = 'cuda') -> LSMTransformer:
    """Carga el mejor modelo guardado."""
    model = LSMTransformer().to(device)
    
    model_path = MLRUNS_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def evaluate_model(model, loader, device):
    """Evalúa el modelo y retorna predicciones y etiquetas."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Matriz guardada en {save_path}")
    
    plt.show()


def main():
    """Punto de entrada para evaluación."""
    print("=" * 50)
    print("📊 Evaluación del Modelo")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    try:
        model = load_best_model(device)
        print("✅ Modelo cargado")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Cargar dataset completo
    dataset = LSMDataset(PROCESSED_DATA_DIR)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Evaluar
    preds, labels = evaluate_model(model, loader, device)
    
    # Reporte
    print("\n📋 Classification Report:")
    print(classification_report(labels, preds, target_names=CLASSES))
    
    # Matriz de confusión
    plot_confusion_matrix(labels, preds, MLRUNS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
