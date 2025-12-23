"""
Script Maestro de Entrenamiento LSM-Core v2.

Mejoras:
    - Feature Weights: Prioriza manos sobre cara en el modelo
    - Oversampling: Balancea clases minoritarias
    - Label Smoothing: Previene sobreconfianza
    - Regularización avanzada: Dropout, Weight Decay, Gradient Clipping
    - Early Stopping + LR Scheduling
    
Uso:
    python -m src.training.train
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from collections import Counter

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.pytorch

# Proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    MLRUNS_DIR,
    INPUT_DIM,
    MAX_SEQ_LEN,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    ensure_dirs
)
from src.models.transformer import LSMTransformer

# =============================================
# CONFIGURACIÓN MEJORADA
# =============================================
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']
NUM_CLASSES = len(CLASS_NAMES)

# Hiperparámetros optimizados
TRAIN_CONFIG = {
    # Datos
    'epochs': 150,              # Más épocas con early stopping
    'batch_size': 32,           # Batch más grande
    'test_size': 0.2,
    'max_seq_len': MAX_SEQ_LEN,
    
    # Modelo
    'input_dim': INPUT_DIM,
    'num_classes': NUM_CLASSES,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,              # Una capa más
    'dropout': 0.4,             # Mayor dropout
    
    # Optimización
    'learning_rate': 3e-4,      # LR más alto
    'weight_decay': 1e-4,       # Mayor regularización L2
    'label_smoothing': 0.1,     # Previene sobreconfianza
    'gradient_clip': 1.0,
    
    # Early stopping
    'patience': 20,             # Más paciencia
    'min_delta': 0.001,
}


# =============================================
# DATASET CON AUGMENTACIÓN
# =============================================
class LSMDataset(Dataset):
    """Dataset para secuencias de keypoints LSM con augmentación."""
    
    def __init__(self, sequences, labels, max_len=MAX_SEQ_LEN, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        label = self.labels[idx]
        
        # Augmentación (solo en entrenamiento)
        if self.augment:
            seq = self._augment(seq)
        
        # Truncar o hacer padding
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        elif len(seq) < self.max_len:
            padding = np.zeros((self.max_len - len(seq), seq.shape[1]))
            seq = np.vstack([seq, padding])
        
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]
    
    def _augment(self, seq):
        """Augmentación simple para regularización."""
        # 1. Ruido gaussiano
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.02, seq.shape)
            # Solo añadir ruido a valores no-cero
            mask = seq != 0
            seq = seq + noise * mask
        
        # 2. Escalado temporal (acelerar/ralentizar)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            indices = np.linspace(0, len(seq)-1, int(len(seq) * scale))
            indices = np.clip(indices.astype(int), 0, len(seq)-1)
            seq = seq[indices]
        
        # 3. Dropout de frames
        if np.random.random() < 0.2:
            drop_mask = np.random.random(len(seq)) > 0.1
            if drop_mask.sum() > 10:  # Mantener al menos 10 frames
                seq = seq[drop_mask]
        
        return seq


def load_dataset(data_dir: Path):
    """Carga todos los .npy del directorio."""
    sequences = []
    labels = []
    
    print("📂 Cargando dataset...")
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠️ Clase no encontrada: {class_name}")
            continue
        
        npy_files = list(class_dir.glob("*.npy"))
        print(f"  {class_name}: {len(npy_files)} archivos")
        
        for npy_file in npy_files:
            try:
                data = np.load(str(npy_file))
                if data.shape[1] == INPUT_DIM:
                    sequences.append(data)
                    labels.append(class_idx)
            except Exception as e:
                pass  # Ignorar archivos corruptos
    
    print(f"✅ Total: {len(sequences)} muestras")
    return sequences, labels


def create_balanced_sampler(labels):
    """
    Crea un sampler que balancea las clases durante el entrenamiento.
    Clases minoritarias se muestrean más frecuentemente.
    """
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


# =============================================
# ENTRENAMIENTO
# =============================================
def train_epoch(model, loader, criterion, optimizer, device, config):
    """Entrena una época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Genera y guarda matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - LSM Transformer v2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return str(save_path)


def plot_training_curves(history, save_path):
    """Genera curvas de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', color='blue')
    axes[1].plot(history['val_acc'], label='Validation', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    """Punto de entrada principal."""
    print("=" * 70)
    print("🚀 LSM-Core Training Pipeline v2")
    print("   - Feature Weights (Manos > Cara)")
    print("   - Oversampling + Label Smoothing")
    print("   - Regularización avanzada")
    print("=" * 70)
    
    ensure_dirs()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Cargar datos
    sequences, labels = load_dataset(PROCESSED_DATA_DIR)
    
    if len(sequences) == 0:
        print("❌ No hay datos para entrenar.")
        return
    
    # Mostrar distribución
    label_counts = Counter(labels)
    print("\n📊 Distribución de clases:")
    for i, name in enumerate(CLASS_NAMES):
        count = label_counts.get(i, 0)
        print(f"   {name}: {count}")
    
    # Split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=TRAIN_CONFIG['test_size'],
        stratify=labels,
        random_state=42
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    
    # Datasets
    train_dataset = LSMDataset(X_train, y_train, TRAIN_CONFIG['max_seq_len'], augment=True)
    val_dataset = LSMDataset(X_val, y_val, TRAIN_CONFIG['max_seq_len'], augment=False)
    
    # Sampler para balancear clases
    train_sampler = create_balanced_sampler(y_train)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        sampler=train_sampler,  # Usa sampler en vez de shuffle
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TRAIN_CONFIG['batch_size'], 
        shuffle=False
    )
    
    # Modelo
    model = LSMTransformer(
        input_dim=TRAIN_CONFIG['input_dim'],
        num_classes=TRAIN_CONFIG['num_classes'],
        d_model=TRAIN_CONFIG['d_model'],
        nhead=TRAIN_CONFIG['n_heads'],
        num_layers=TRAIN_CONFIG['n_layers'],
        dropout=TRAIN_CONFIG['dropout'],
        max_seq_len=TRAIN_CONFIG['max_seq_len']
    ).to(device)
    
    print(f"\n🧠 Modelo: {model.count_parameters():,} parámetros")
    
    # Loss con Label Smoothing (previene sobreconfianza)
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG['label_smoothing'])
    
    # Optimizer con weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Scheduler: Reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, verbose=True
    )
    
    # =============================================
    # MLflow
    # =============================================
    mlflow_db = MLRUNS_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("LSM_Core_Training_v2")
    
    print("\n" + "=" * 70)
    print("📈 Iniciando entrenamiento...")
    print("=" * 70)
    
    with mlflow.start_run(run_name=f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parámetros
        mlflow.log_params(TRAIN_CONFIG)
        mlflow.log_param("class_names", CLASS_NAMES)
        mlflow.log_param("class_distribution", dict(label_counts))
        
        best_val_acc = 0
        patience_counter = 0
        best_model_path = MLRUNS_DIR / "best_model.pth"
        
        # Historia para gráficas
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(TRAIN_CONFIG['epochs']):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, TRAIN_CONFIG
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            
            # Scheduler
            scheduler.step(val_acc)
            
            # Guardar historia
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log métricas
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print
            print(f"Epoch {epoch+1:3d}/{TRAIN_CONFIG['epochs']} | "
                  f"Train: {train_loss:.4f} / {train_acc:.4f} | "
                  f"Val: {val_loss:.4f} / {val_acc:.4f}")
            
            # Checkpointing
            if val_acc > best_val_acc + TRAIN_CONFIG['min_delta']:
                best_val_acc = val_acc
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': TRAIN_CONFIG
                }, best_model_path)
                
                print(f"  ✅ Best model saved (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_CONFIG['patience']:
                    print(f"\n⏹️ Early stopping en epoch {epoch+1}")
                    break
        
        # =============================================
        # EVALUACIÓN FINAL
        # =============================================
        print("\n" + "=" * 70)
        print("📊 Evaluación Final")
        print("=" * 70)
        
        # Cargar mejor modelo
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluar
        _, final_acc, final_preds, final_labels = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"\n🎯 Mejor Val Accuracy: {best_val_acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
        
        # Matriz de confusión
        cm_path = MLRUNS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(final_labels, final_preds, cm_path)
        print(f"\n📊 Matriz de confusión: {cm_path}")
        
        # Curvas de entrenamiento
        curves_path = MLRUNS_DIR / "training_curves.png"
        plot_training_curves(history, curves_path)
        print(f"📈 Curvas de entrenamiento: {curves_path}")
        
        # Feature weights finales
        feature_weights = model.get_feature_weights()
        print(f"\n🔧 Feature Weights (promedio por región):")
        print(f"   Cuerpo: {feature_weights[0:34].mean().item():.3f}")
        print(f"   Pies:   {feature_weights[34:46].mean().item():.3f}")
        print(f"   Cara:   {feature_weights[46:182].mean().item():.3f}")
        print(f"   Manos:  {feature_weights[182:266].mean().item():.3f}")
        
        # Log artefactos
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(curves_path))
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(__file__)
        
        # Log modelo
        mlflow.pytorch.log_model(model, "model")
        
        # Métricas finales
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("epochs_trained", len(history['train_loss']))
        
        print("\n" + "=" * 70)
        print("✅ Entrenamiento completado")
        print(f"📁 Artefactos en: {MLRUNS_DIR}")
        print("=" * 70)


if __name__ == "__main__":
    main()
