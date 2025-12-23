"""
Configuración global del proyecto LSM-Core.
Todas las rutas, hyperparámetros y constantes van aquí.
"""

import os
from pathlib import Path

# -----------------------------------------------------------
# RUTAS DEL PROYECTO (Dinámicas)
# -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MLRUNS_DIR = EXPERIMENTS_DIR / "mlruns"

MODELS_DIR = PROJECT_ROOT / "models"

# -----------------------------------------------------------
# CONFIGURACIÓN RTMPose-WholeBody
# -----------------------------------------------------------
RTMPOSE_MODEL = 'rtmpose-l_8xb32-270e_coco-wholebody-384x288'
RTMPOSE_INPUT_SIZE = (384, 288)

# 133 keypoints: Cuerpo(17) + Pies(6) + Cara(68) + Manos(42)
KEYPOINTS_PER_FRAME = 133
VALUES_PER_KEYPOINT = 2  # x, y
INPUT_DIM = KEYPOINTS_PER_FRAME * VALUES_PER_KEYPOINT  # 266

# -----------------------------------------------------------
# CONFIGURACIÓN DE EXTRACCIÓN (CRÍTICO)
# -----------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.50  # Puntos con score < 0.5 se ponen a (0,0)

# OneEuroFilter params
FILTER_MIN_CUTOFF = 0.1   # Suavizado en reposo
FILTER_BETA = 0.009       # Reduce lag en movimiento

# Índices de puntos de referencia para normalización (COCO-WholeBody)
# Caderas: 11 (left_hip), 12 (right_hip)
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# -----------------------------------------------------------
# CONFIGURACIÓN DEL MODELO TRANSFORMER
# -----------------------------------------------------------
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

MAX_SEQ_LEN = 90
MIN_SEQ_LEN = 15
SEQ_LEN = MAX_SEQ_LEN

# -----------------------------------------------------------
# HYPERPARÁMETROS DE ENTRENAMIENTO
# -----------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
WEIGHT_DECAY = 1e-5
PATIENCE = 10
TRAIN_SPLIT = 0.8

# -----------------------------------------------------------
# CLASES (Dinámicas)
# -----------------------------------------------------------
CLASSES = []
NUM_CLASSES = 5  # Default, actualizar con get_num_classes()

def load_classes():
    """Carga clases desde las carpetas en dataset/raw/."""
    global CLASSES, NUM_CLASSES
    if RAW_DATA_DIR.exists():
        CLASSES = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
        NUM_CLASSES = len(CLASSES) if CLASSES else 5
    return CLASSES

def get_num_classes() -> int:
    load_classes()
    return NUM_CLASSES

# -----------------------------------------------------------
# INFERENCIA EN TIEMPO REAL
# -----------------------------------------------------------
INFERENCE_CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 5

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------
def ensure_dirs():
    """Crea directorios necesarios."""
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MLRUNS_DIR, MODELS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def get_class_index(class_name: str) -> int:
    load_classes()
    return CLASSES.index(class_name)

def get_class_name(index: int) -> str:
    load_classes()
    return CLASSES[index]

def print_config():
    """Imprime configuración."""
    load_classes()
    print("=" * 50)
    print("⚙️ LSM-Core Config")
    print("=" * 50)
    print(f"📁 Project: {PROJECT_ROOT}")
    print(f"📂 Raw: {RAW_DATA_DIR}")
    print(f"📦 Processed: {PROCESSED_DATA_DIR}")
    print(f"🎯 RTMPose: {RTMPOSE_MODEL}")
    print(f"📊 Input Dim: {INPUT_DIM} ({KEYPOINTS_PER_FRAME} kp × {VALUES_PER_KEYPOINT})")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"📈 Classes ({len(CLASSES)}): {CLASSES}")
    print("=" * 50)

load_classes()

if __name__ == "__main__":
    print_config()
