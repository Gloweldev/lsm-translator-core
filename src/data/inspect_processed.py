"""
Inspector de Tensores Procesados.

Valida los archivos .npy generados por el preprocessor:
    1. Dimensión correcta (266 features)
    2. No sean puros ceros
    3. Longitud de secuencia adecuada
    4. Sin valores NaN/Inf
    5. Análisis de FPS de videos originales

Genera reporte y class weights para entrenamiento balanceado.

Uso:
    python -m src.data.inspect_processed
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, INPUT_DIM, MIN_SEQ_LEN, MAX_SEQ_LEN, CLASS_NAMES

# Configuración de validación
EXPECTED_FEATURES = INPUT_DIM  # 266 (133 * 2)
OLD_FORMAT_DIM = 258  # MediaPipe format (obsoleto)


def validate_tensor(filepath: Path) -> dict:
    """
    Valida un archivo .npy.
    
    Returns:
        Dict con resultados de validación
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'frames': 0,
        'features': 0,
        'variance': 0.0,
        'has_nan': False,
        'has_inf': False,
        'is_zeros': False
    }
    
    try:
        data = np.load(str(filepath))
        
        # Forma
        result['frames'] = data.shape[0]
        result['features'] = data.shape[1] if len(data.shape) > 1 else 0
        
        # 1. Verificar dimensión de features
        if result['features'] != EXPECTED_FEATURES:
            if result['features'] == OLD_FORMAT_DIM:
                result['errors'].append(f"Formato viejo MediaPipe ({OLD_FORMAT_DIM})")
            else:
                result['errors'].append(f"Dimensión incorrecta: {result['features']} (esperado: {EXPECTED_FEATURES})")
            result['valid'] = False
        
        # 2. Verificar NaN/Inf
        result['has_nan'] = np.any(np.isnan(data))
        result['has_inf'] = np.any(np.isinf(data))
        
        if result['has_nan']:
            result['errors'].append("Contiene valores NaN")
            result['valid'] = False
        if result['has_inf']:
            result['errors'].append("Contiene valores Inf")
            result['valid'] = False
        
        # 3. Verificar muerte cerebral (todo ceros)
        result['variance'] = np.var(data)
        result['is_zeros'] = result['variance'] < 1e-10
        
        if result['is_zeros']:
            result['errors'].append("Tensor vacío (varianza = 0)")
            result['valid'] = False
        
        # 4. Verificar longitud de secuencia
        if result['frames'] < MIN_SEQ_LEN:
            result['warnings'].append(f"Muy corto: {result['frames']} frames")
        elif result['frames'] > MAX_SEQ_LEN:
            result['warnings'].append(f"Muy largo: {result['frames']} frames")
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Error de lectura: {str(e)}")
    
    return result


def get_video_fps(video_path: Path) -> tuple:
    """
    Obtiene FPS y frame count de un video.
    
    Returns:
        (fps, frame_count) o (None, None) si hay error
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Validar datos
        if fps <= 0 or fps > 120:
            fps = None
        if frame_count <= 0:
            frame_count = None
            
        return fps, frame_count
    except:
        return None, None


def analyze_raw_videos_fps():
    """
    Analiza FPS de todos los videos en el directorio raw.
    
    Returns:
        Dict con estadísticas de FPS
    """
    print("\n" + "=" * 70)
    print("🎬 ANÁLISIS DE FPS DE VIDEOS ORIGINALES")
    print("=" * 70)
    
    if not RAW_DATA_DIR.exists():
        print(f"❌ No existe directorio raw: {RAW_DATA_DIR}")
        return None
    
    fps_values = []
    frame_counts = []
    fps_distribution = defaultdict(int)  # FPS -> count
    class_fps = defaultdict(list)  # class -> [fps, fps, ...]
    video_details = []  # [(path, fps, frames), ...]
    
    total_videos = 0
    errors = 0
    
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        if not class_dir.exists():
            continue
        
        for video_file in class_dir.glob("*.mp4"):
            total_videos += 1
            fps, frames = get_video_fps(video_file)
            
            if fps is not None:
                fps_values.append(fps)
                fps_rounded = round(fps)
                fps_distribution[fps_rounded] += 1
                class_fps[class_name].append(fps)
                video_details.append((video_file.name, class_name, fps, frames))
                
                if frames is not None:
                    frame_counts.append(frames)
            else:
                errors += 1
    
    if not fps_values:
        print("❌ No se pudieron leer los videos")
        return None
    
    fps_arr = np.array(fps_values)
    
    # Estadísticas
    print(f"\n📂 Directorio: {RAW_DATA_DIR}")
    print(f"📹 Total videos escaneados: {total_videos}")
    print(f"✅ Videos leídos correctamente: {len(fps_values)}")
    print(f"❌ Videos con error: {errors}")
    
    print("\n" + "-" * 70)
    print("📊 DISTRIBUCIÓN DE FPS")
    print("-" * 70)
    
    for fps_val in sorted(fps_distribution.keys()):
        count = fps_distribution[fps_val]
        pct = count / len(fps_values) * 100
        bar = "█" * min(50, int(pct / 2))
        print(f"  {fps_val:3d} FPS | {count:4d} videos ({pct:5.1f}%) | {bar}")
    
    print("\n" + "-" * 70)
    print("📈 ESTADÍSTICAS DE FPS")
    print("-" * 70)
    print(f"  Mínimo:    {fps_arr.min():6.1f} FPS")
    print(f"  Máximo:    {fps_arr.max():6.1f} FPS")
    print(f"  Promedio:  {fps_arr.mean():6.1f} FPS")
    print(f"  Mediana:   {np.median(fps_arr):6.1f} FPS")
    print(f"  Moda:      {max(fps_distribution, key=fps_distribution.get)} FPS")
    print(f"  Std:       {fps_arr.std():6.1f}")
    
    # FPS por clase
    print("\n" + "-" * 70)
    print("📊 FPS PROMEDIO POR CLASE")
    print("-" * 70)
    for cls in sorted(class_fps.keys()):
        fps_list = class_fps[cls]
        avg_fps = np.mean(fps_list)
        min_fps = np.min(fps_list)
        max_fps = np.max(fps_list)
        print(f"  {cls:10s} | Avg: {avg_fps:5.1f} | Min: {min_fps:5.1f} | Max: {max_fps:5.1f} | Videos: {len(fps_list)}")
    
    # Recomendación
    print("\n" + "-" * 70)
    print("🎯 RECOMENDACIÓN PARA INFERENCIA")
    print("-" * 70)
    most_common_fps = max(fps_distribution, key=fps_distribution.get)
    print(f"  FPS más común: {most_common_fps} FPS")
    print(f"  Configurar TRAINING_FPS = {most_common_fps} en ipad_demo.py")
    
    if fps_arr.std() > 5:
        print("\n  ⚠️ ADVERTENCIA: Alta variación en FPS de videos de entrenamiento")
        print("     El modelo puede tener inconsistencias temporales.")
    
    # Histograma de FPS
    print("\n" + "-" * 70)
    print("📊 Generando histograma de FPS...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de FPS
    axes[0].hist(fps_values, bins=range(int(min(fps_values))-1, int(max(fps_values))+3), 
                 edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(fps_values), color='green', linestyle='-', 
                   label=f'Mean ({np.mean(fps_values):.0f})')
    axes[0].axvline(30, color='red', linestyle='--', label='30 FPS (target)')
    axes[0].set_xlabel('FPS')
    axes[0].set_ylabel('Cantidad de Videos')
    axes[0].set_title('Distribución de FPS en Videos de Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pie chart de FPS distribution
    labels = [f"{fps} FPS" for fps in sorted(fps_distribution.keys())]
    sizes = [fps_distribution[fps] for fps in sorted(fps_distribution.keys())]
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Proporción por FPS')
    
    output_path = PROCESSED_DATA_DIR.parent / "fps_analysis.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Guardado en: {output_path}")
    plt.close()
    
    return {
        'fps_values': fps_values,
        'fps_distribution': dict(fps_distribution),
        'mean_fps': float(fps_arr.mean()),
        'most_common_fps': most_common_fps
    }


def calculate_class_weights(class_counts: dict) -> dict:
    """
    Calcula pesos para balancear clases desbalanceadas.
    Fórmula: Total / (Num_Clases * Count_Clase)
    """
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = {}
    for cls, count in class_counts.items():
        if count > 0:
            weights[cls] = total / (num_classes * count)
        else:
            weights[cls] = 1.0
    
    return weights


def run_inspection():
    """Ejecuta la inspección completa."""
    print("=" * 70)
    print("🔍 Inspector de Tensores Procesados")
    print("=" * 70)
    print(f"📂 Directorio: {PROCESSED_DATA_DIR}")
    print(f"📏 Features esperados: {EXPECTED_FEATURES}")
    print("=" * 70)
    
    if not PROCESSED_DATA_DIR.exists():
        print("❌ El directorio no existe. Ejecuta primero el preprocessor.")
        return
    
    # Recolectar datos
    class_counts = defaultdict(int)
    frame_lengths = []
    errors = []
    warnings = []
    total_files = 0
    valid_files = 0
    
    # Escanear
    for class_dir in sorted(PROCESSED_DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        for npy_file in class_dir.glob("*.npy"):
            total_files += 1
            result = validate_tensor(npy_file)
            
            if result['valid']:
                valid_files += 1
                class_counts[class_name] += 1
                frame_lengths.append(result['frames'])
            else:
                for err in result['errors']:
                    errors.append(f"{class_name}/{npy_file.name}: {err}")
            
            for warn in result['warnings']:
                warnings.append(f"{class_name}/{npy_file.name}: {warn}")
    
    # =============================================
    # REPORTE
    # =============================================
    print("\n" + "=" * 70)
    print("📊 RESUMEN GENERAL")
    print("=" * 70)
    print(f"  Total archivos escaneados: {total_files}")
    print(f"  Archivos válidos:          {valid_files}")
    print(f"  Archivos con errores:      {total_files - valid_files}")
    print(f"  Total clases:              {len(class_counts)}")
    
    # Distribución de clases
    print("\n" + "-" * 70)
    print("📈 DISTRIBUCIÓN DE CLASES")
    print("-" * 70)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        bar = "█" * min(50, int(count / max(class_counts.values()) * 50))
        print(f"  {cls:20s} | {count:4d} | {bar}")
    
    # Class weights
    print("\n" + "-" * 70)
    print("⚖️ CLASS WEIGHTS (para entrenamiento balanceado)")
    print("-" * 70)
    weights = calculate_class_weights(class_counts)
    
    # Formato para copiar
    sorted_classes = sorted(class_counts.keys())
    print("\n  # Copiar a settings.py o train.py:")
    print(f"  CLASS_NAMES = {sorted_classes}")
    weight_list = [round(weights[cls], 4) for cls in sorted_classes]
    print(f"  CLASS_WEIGHTS = {weight_list}")
    
    print("\n  # O como tensor de PyTorch:")
    print(f"  weights = torch.tensor({weight_list})")
    
    # Estadísticas de longitud
    if frame_lengths:
        frame_arr = np.array(frame_lengths)
        print("\n" + "-" * 70)
        print("📏 ESTADÍSTICAS DE LONGITUD (frames)")
        print("-" * 70)
        print(f"  Mínimo:    {frame_arr.min():6d}")
        print(f"  Máximo:    {frame_arr.max():6d}")
        print(f"  Promedio:  {frame_arr.mean():6.1f}")
        print(f"  Mediana:   {np.median(frame_arr):6.1f}")
        print(f"  Std:       {frame_arr.std():6.1f}")
        
        short = np.sum(frame_arr < MIN_SEQ_LEN)
        long = np.sum(frame_arr > MAX_SEQ_LEN)
        print(f"\n  Videos < {MIN_SEQ_LEN} frames: {short}")
        print(f"  Videos > {MAX_SEQ_LEN} frames: {long}")
    
    # Warnings
    if warnings:
        print("\n" + "-" * 70)
        print(f"⚠️ ADVERTENCIAS ({len(warnings)})")
        print("-" * 70)
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... y {len(warnings) - 10} más")
    
    # Errores
    if errors:
        print("\n" + "-" * 70)
        print(f"❌ ERRORES ({len(errors)})")
        print("-" * 70)
        for e in errors[:15]:
            print(f"  - {e}")
        if len(errors) > 15:
            print(f"  ... y {len(errors) - 15} más")
    
    # Histograma de frames
    if frame_lengths:
        print("\n" + "-" * 70)
        print("📊 Generando histograma de frames...")
        
        plt.figure(figsize=(10, 6))
        plt.hist(frame_lengths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(MIN_SEQ_LEN, color='red', linestyle='--', label=f'Min ({MIN_SEQ_LEN})')
        plt.axvline(MAX_SEQ_LEN, color='orange', linestyle='--', label=f'Max ({MAX_SEQ_LEN})')
        plt.axvline(np.mean(frame_lengths), color='green', linestyle='-', label=f'Mean ({np.mean(frame_lengths):.0f})')
        plt.xlabel('Número de Frames')
        plt.ylabel('Cantidad de Videos')
        plt.title('Distribución de Longitud de Videos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar
        output_path = PROCESSED_DATA_DIR.parent / "frame_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Guardado en: {output_path}")
        plt.close()
    
    # =============================================
    # ANÁLISIS DE FPS
    # =============================================
    fps_result = analyze_raw_videos_fps()
    
    print("\n" + "=" * 70)
    print("✅ Inspección completada")
    print("=" * 70)


if __name__ == "__main__":
    run_inspection()

