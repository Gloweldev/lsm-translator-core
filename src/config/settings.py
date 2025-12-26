"""
Configuración global del proyecto LSM-Core.
Todas las rutas, hyperparámetros y constantes van aquí.

IMPORTANTE: Este es el ÚNICO lugar para configurar el proyecto.
Los scripts de entrenamiento e inferencia importan de aquí.
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
# Opciones disponibles en mmpose:
# - rtmpose-m (rápido, ~35 FPS)
# - rtmpose-l (balanceado, ~25 FPS) ← Recomendado
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
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# -----------------------------------------------------------
# CLASES Y PESOS
# -----------------------------------------------------------
# Clases conocidas
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']
NUM_CLASSES = len(CLASS_NAMES)

# Class weights para balancear entrenamiento (inverso a frecuencia)
# Generados por inspect_processed.py - clases minoritarias tienen mayor peso
CLASS_WEIGHTS = [1.2515, 1.2294, 1.3933, 1.1117, 0.5649]

# Feature weights por región de keypoints (0-133)
# Controla importancia de cada parte del cuerpo
FEATURE_WEIGHTS = {
    'body': 1.0,      # Cuerpo (indices 0-16)
    'feet': 0.3,      # Pies (indices 17-22)
    'face': 0.1,      # Cara (indices 23-90)
    'left_hand': 2.5, # Mano izquierda (91-111)
    'right_hand': 2.5 # Mano derecha (112-132)
}

# Actualizar desde carpetas (si existen)
def load_classes():
    """Carga clases desde las carpetas en dataset/raw/."""
    global CLASS_NAMES, NUM_CLASSES
    if RAW_DATA_DIR.exists():
        folders = sorted([d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
        if folders:
            CLASS_NAMES = folders
            NUM_CLASSES = len(CLASS_NAMES)
    return CLASS_NAMES

def get_num_classes() -> int:
    load_classes()
    return NUM_CLASSES

def get_class_index(class_name: str) -> int:
    load_classes()
    return CLASS_NAMES.index(class_name)

def get_class_name(index: int) -> str:
    load_classes()
    return CLASS_NAMES[index]

# -----------------------------------------------------------
# CONFIGURACIÓN DEL MODELO TRANSFORMER
# -----------------------------------------------------------
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3            # 3 capas para mejor rendimiento
DROPOUT = 0.4           # Mayor dropout contra overfitting

MAX_SEQ_LEN = 90
MIN_SEQ_LEN = 15

# -----------------------------------------------------------
# HYPERPARÁMETROS DE ENTRENAMIENTO
# -----------------------------------------------------------
# Datos
BATCH_SIZE = 32
TEST_SIZE = 0.2

# Optimización
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4     # L2 regularization
GRADIENT_CLIP = 1.0

# Regularización
LABEL_SMOOTHING = 0.1   # Previene sobreconfianza

# Early stopping
EPOCHS = 150
PATIENCE = 20
MIN_DELTA = 0.001

# -----------------------------------------------------------
# INFERENCIA EN TIEMPO REAL
# -----------------------------------------------------------
INFERENCE_CONFIDENCE_THRESHOLD = 0.85
STABILITY_FRAMES = 5
MIN_BUFFER_FRAMES = 30  # Mínimo para empezar a predecir

# -----------------------------------------------------------
# MLFLOW
# -----------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "LSM_Core_Training"

def get_mlflow_tracking_uri() -> str:
    """Retorna URI de MLflow compatible con Windows."""
    mlflow_db = MLRUNS_DIR / "mlflow.db"
    return f"sqlite:///{mlflow_db}"

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------
def ensure_dirs():
    """Crea directorios necesarios."""
    dirs = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MLRUNS_DIR, MODELS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

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
    print(f"📊 Input Dim: {INPUT_DIM}")
    print(f"🧠 Model: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}")
    print(f"📈 Classes ({NUM_CLASSES}): {CLASS_NAMES}")
    print(f"🔧 Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("=" * 50)

# Cargar clases al importar
load_classes()

if __name__ == "__main__":
    print_config()
