"""
Análisis de Confusión del Modelo.

Evalúa el modelo contra todo el dataset procesado y genera:
- Matriz de confusión visual
- Reporte de confusiones clase por clase
- Análisis específico de confusiones A-B

Uso:
    python -m src.analysis.confusion_analysis
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    MLRUNS_DIR,
    INPUT_DIM,
    MAX_SEQ_LEN,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    CLASS_NAMES,
    get_latest_processed_dir
)
from src.models.transformer import LSMTransformer


def load_dataset(data_dir: Path):
    """Carga todo el dataset procesado."""
    sequences = []
    labels = []
    filenames = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ Directorio no encontrado: {class_dir}")
            continue
        
        for npy_file in class_dir.glob("*.npy"):
            try:
                seq = np.load(npy_file)
                sequences.append(seq)
                labels.append(class_idx)
                filenames.append(f"{class_name}/{npy_file.name}")
            except Exception as e:
                print(f"Error cargando {npy_file}: {e}")
    
    return sequences, labels, filenames


def prepare_sequence(seq, max_len=MAX_SEQ_LEN):
    """Prepara secuencia con padding y duración."""
    duration = min(len(seq), max_len)
    
    if len(seq) > max_len:
        seq = seq[:max_len]
    elif len(seq) < max_len:
        padding = np.zeros((max_len - len(seq), seq.shape[1]))
        seq = np.vstack([seq, padding])
    
    return seq, duration


def evaluate_model(model, sequences, labels, device):
    """Evalúa modelo en todo el dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seq, label in tqdm(zip(sequences, labels), total=len(sequences), desc="Evaluando"):
            prepared_seq, duration = prepare_sequence(seq)
            
            tensor = torch.FloatTensor(prepared_seq).unsqueeze(0).to(device)
            dur_tensor = torch.LongTensor([duration]).to(device)
            
            logits = model(tensor, duration=dur_tensor)
            probs = F.softmax(logits, dim=1)
            
            pred = probs.argmax(dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_confusions(y_true, y_pred, filenames):
    """Analiza confusiones detalladamente."""
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("📊 MATRIZ DE CONFUSIÓN")
    print("=" * 60)
    
    # Imprimir matriz en texto
    header = "          " + "  ".join([f"{c:>6}" for c in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    
    for i, class_name in enumerate(CLASS_NAMES):
        row = f"{class_name:>8} |"
        for j in range(len(CLASS_NAMES)):
            value = cm[i, j]
            if i == j:
                row += f"  [{value:>4}]"  # Diagonal
            elif value > 0:
                row += f"   {value:>4} "  # Confusión
            else:
                row += f"      ."
        print(row)
    
    print("\n" + "=" * 60)
    print("🔍 ANÁLISIS DE CONFUSIONES POR CLASE")
    print("=" * 60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n{class_name.upper()} (Total: {total}, Precisión: {accuracy:.1%})")
        
        # Mostrar confusiones
        confusions = []
        for j, other_class in enumerate(CLASS_NAMES):
            if i != j and cm[i, j] > 0:
                confusions.append((other_class, cm[i, j], cm[i, j] / total))
        
        if confusions:
            confusions.sort(key=lambda x: -x[1])
            for other, count, pct in confusions:
                print(f"  → Confundido con {other}: {count} veces ({pct:.1%})")
        else:
            print("  ✅ Sin confusiones significativas")
    
    # Análisis específico A-B
    print("\n" + "=" * 60)
    print("🔴 ANÁLISIS ESPECÍFICO A ↔ B")
    print("=" * 60)
    
    a_idx = CLASS_NAMES.index('a') if 'a' in CLASS_NAMES else -1
    b_idx = CLASS_NAMES.index('b') if 'b' in CLASS_NAMES else -1
    
    if a_idx >= 0 and b_idx >= 0:
        a_as_b = cm[a_idx, b_idx]
        b_as_a = cm[b_idx, a_idx]
        a_total = cm[a_idx].sum()
        b_total = cm[b_idx].sum()
        
        print(f"\n  A clasificada como B: {a_as_b}/{a_total} ({a_as_b/a_total:.1%})")
        print(f"  B clasificada como A: {b_as_a}/{b_total} ({b_as_a/b_total:.1%})")
        
        # Listar archivos con errores A→B
        print(f"\n  📁 Archivos A mal clasificados como B:")
        error_count = 0
        for idx, (true_label, pred_label, filename) in enumerate(zip(y_true, y_pred, filenames)):
            if true_label == a_idx and pred_label == b_idx:
                print(f"     - {filename}")
                error_count += 1
                if error_count >= 10:
                    remaining = a_as_b - 10
                    if remaining > 0:
                        print(f"     ... y {remaining} más")
                    break
    
    return cm


def plot_confusion_matrix(cm, save_path):
    """Genera visualización de matriz de confusión."""
    plt.figure(figsize=(10, 8))
    
    # Normalizar para porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear heatmap
    sns.heatmap(
        cm_normalized,
        annot=cm,  # Mostrar valores absolutos
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Proporción'}
    )
    
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title('Matriz de Confusión', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\n📊 Matriz guardada en: {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis de Confusión del Modelo")
    parser.add_argument("-d", "--data", type=str,
                       help="Path to processed data directory (should match training data)")
    args = parser.parse_args()
    
    # Determinar directorio de datos
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = PROCESSED_DATA_DIR.parent / args.data
    else:
        # Por defecto: usar la última versión procesada
        data_path = get_latest_processed_dir()
    
    print("=" * 60)
    print("🔍 Análisis de Confusión del Modelo")
    print("=" * 60)
    print(f"📂 Datos: {data_path}")
    
    if not data_path.exists():
        print(f"❌ Directorio no encontrado: {data_path}")
        return
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Dispositivo: {device}")
    
    model_path = MLRUNS_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"❌ No se encontró: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model = LSMTransformer(
        input_dim=config.get('input_dim', INPUT_DIM),
        num_classes=len(CLASS_NAMES),
        d_model=config.get('d_model', D_MODEL),
        nhead=config.get('nhead', N_HEADS),
        num_layers=config.get('num_layers', N_LAYERS)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', '?')
    run_id = checkpoint.get('run_id', 'N/A')
    val_acc = checkpoint.get('val_acc', 0)
    
    print(f"✅ Modelo cargado:")
    print(f"   📁 Path: {model_path.name}")
    print(f"   🔖 Run ID: {run_id}")
    print(f"   📊 Epoch: {epoch}, Val Acc: {val_acc:.4f}")
    
    # Cargar dataset
    print(f"\n📂 Cargando dataset desde: {data_path}")
    sequences, labels, filenames = load_dataset(data_path)
    print(f"   Total muestras: {len(sequences)}")
    
    # Evaluar
    print("\n🧠 Evaluando modelo...")
    y_pred, y_true, probs = evaluate_model(model, sequences, labels, device)
    
    # Métricas generales
    accuracy = (y_pred == y_true).mean()
    print(f"\n📈 Precisión global: {accuracy:.2%}")
    
    # Análisis detallado
    cm = analyze_confusions(y_true, y_pred, filenames)
    
    # Guardar matriz visual
    output_path = PROCESSED_DATA_DIR.parent / "confusion_matrix.png"
    plot_confusion_matrix(cm, output_path)
    
    # Classification report
    print("\n" + "=" * 60)
    print("📋 REPORTE DE CLASIFICACIÓN")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
