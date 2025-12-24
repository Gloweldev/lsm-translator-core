"""
Script Maestro de Entrenamiento LSM-Core v2.

Toda la configuración se importa desde src/config/settings.py
No modificar hyperparámetros aquí, solo en settings.py

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
    # Rutas
    PROCESSED_DATA_DIR,
    MLRUNS_DIR,
    # Modelo
    INPUT_DIM,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DROPOUT,
    MAX_SEQ_LEN,
    # Training
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    LABEL_SMOOTHING,
    EPOCHS,
    PATIENCE,
    MIN_DELTA,
    TEST_SIZE,
    # Clases
    CLASS_NAMES,
    NUM_CLASSES,
    # MLflow
    MLFLOW_EXPERIMENT_NAME,
    get_mlflow_tracking_uri,
    # Utils
    ensure_dirs
)
from src.models.transformer import LSMTransformer


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
            mask = seq != 0
            seq = seq + noise * mask
        
        # 2. Escalado temporal
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            indices = np.linspace(0, len(seq)-1, int(len(seq) * scale))
            indices = np.clip(indices.astype(int), 0, len(seq)-1)
            seq = seq[indices]
        
        # 3. Dropout de frames
        if np.random.random() < 0.2:
            drop_mask = np.random.random(len(seq)) > 0.1
            if drop_mask.sum() > 10:
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
            except Exception:
                pass
    
    print(f"✅ Total: {len(sequences)} muestras")
    return sequences, labels


def create_balanced_sampler(labels):
    """Crea sampler que balancea clases durante entrenamiento."""
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


# =============================================
# ENTRENAMIENTO
# =============================================
def train_epoch(model, loader, criterion, optimizer, device):
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
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
    plt.title('Matriz de Confusión - LSM Transformer')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history, save_path):
    """Genera curvas de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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
    # Construir config dict para logging
    config = {
        'input_dim': INPUT_DIM,
        'num_classes': NUM_CLASSES,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'max_seq_len': MAX_SEQ_LEN,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'gradient_clip': GRADIENT_CLIP,
        'label_smoothing': LABEL_SMOOTHING,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'min_delta': MIN_DELTA,
        'test_size': TEST_SIZE,
    }
    
    print("=" * 70)
    print("🚀 LSM-Core Training Pipeline")
    print("=" * 70)
    print(f"📊 Config desde settings.py:")
    print(f"   Modelo: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")
    print(f"   Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"   Clases: {CLASS_NAMES}")
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
        print(f"   {name}: {label_counts.get(i, 0)}")
    
    # Split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=42
    )
    
    print(f"\n📊 Split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Datasets
    train_dataset = LSMDataset(X_train, y_train, MAX_SEQ_LEN, augment=True)
    val_dataset = LSMDataset(X_val, y_val, MAX_SEQ_LEN, augment=False)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=create_balanced_sampler(y_train),
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Modelo
    model = LSMTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    print(f"\n🧠 Modelo: {model.count_parameters():,} parámetros")
    
    # Loss con Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, verbose=True
    )
    
    # =============================================
    # MLflow
    # =============================================
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"\n📈 MLflow: {get_mlflow_tracking_uri()}")
    print("=" * 70)
    
    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(config)
        mlflow.log_param("class_names", CLASS_NAMES)
        
        best_val_acc = 0
        patience_counter = 0
        best_model_path = MLRUNS_DIR / "best_model.pth"
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            
            scheduler.step(val_acc)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.4f} / {train_acc:.4f} | "
                  f"Val: {val_loss:.4f} / {val_acc:.4f}")
            
            # Checkpointing
            if val_acc > best_val_acc + MIN_DELTA:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Obtener run_id actual de MLflow
                run_id = mlflow.active_run().info.run_id
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config,
                    'run_id': run_id  # Para identificar versión del modelo
                }, best_model_path)
                
                print(f"  ✅ Best model saved (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n⏹️ Early stopping en epoch {epoch+1}")
                    break
        
        # =============================================
        # EVALUACIÓN FINAL
        # =============================================
        print("\n" + "=" * 70)
        print("📊 Evaluación Final")
        print("=" * 70)
        
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
        
        print(f"\n🎯 Mejor Val Accuracy: {best_val_acc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
        
        # Gráficas
        cm_path = MLRUNS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(final_labels, final_preds, cm_path)
        
        curves_path = MLRUNS_DIR / "training_curves.png"
        plot_training_curves(history, curves_path)
        
        # Feature weights
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
        mlflow.pytorch.log_model(model, "model")
        
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("epochs_trained", len(history['train_loss']))
        
        print("\n" + "=" * 70)
        print("✅ Entrenamiento completado")
        print(f"📁 Modelo: {best_model_path}")
        print(f"📊 MLflow UI: mlflow ui --backend-store-uri {get_mlflow_tracking_uri()}")
        print("=" * 70)


if __name__ == "__main__":
    main()
