# Model Card: LSM Translator v1.0 Baseline

## 📋 Metadata

| Campo | Valor |
|-------|-------|
| **Versión** | v1.0.0-baseline |
| **Fecha** | 2025-12-26 |
| **Run ID MLflow** | cfba5d7d1126436b830ac5efd6cccc39 |
| **Estado** | ✅ Estable / Impecable en inferencia |
| **Datos Procesados** | `processed_v20251226_130002` |

---

## 🐍 Entorno y Dependencias

### Sistema
- **OS:** Windows 10/11
- **Python:** 3.11.14 (Anaconda)
- **CUDA:** 12.1
- **Conda Env:** `lsm-ai`

### Dependencias Principales

```txt
# Core ML
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121

# Pose Estimation
mmpose==1.3.2
mmdet==3.2.0
mmcv==2.1.0
mmengine==0.10.7

# Data Science
numpy==1.26.4
pandas==2.3.3
scipy==1.16.3
scikit-learn==1.8.0

# Computer Vision
opencv-python==4.11.0.86
pillow==12.0.0

# Visualization
matplotlib==3.10.8
seaborn==0.13.2

# MLOps
mlflow==3.8.0

# Utils
tqdm==4.65.2
pyyaml==6.0.3
```

### Instalación Rápida

```bash
# 1. Crear entorno conda
conda create -n lsm-ai python=3.11 -y
conda activate lsm-ai

# 2. PyTorch con CUDA 12.1
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. MMPose y dependencias
pip install -U openmim
mim install mmengine mmcv mmdet mmpose

# 4. Resto de dependencias
pip install numpy pandas scipy scikit-learn opencv-python pillow matplotlib seaborn mlflow tqdm pyyaml
```

---

## 🎯 Clases Soportadas

```python
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']
NUM_CLASSES = 5
```

---

## 📊 Métricas de Rendimiento

### Validation Set (20% de 1045 muestras)
| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| a     | 0.76      | 0.77   | 0.76     | 33      |
| b     | 0.58      | 0.74   | 0.65     | 34      |
| c     | 0.83      | 0.58   | 0.68     | 30      |
| hola  | 0.88      | 0.98   | 0.93     | 38      |
| nada  | 1.00      | 0.94   | 0.97     | 74      |

**Accuracy Global:** ~83-85%  
**Macro Avg F1:** ~0.80  
**Weighted Avg F1:** ~0.84

### Inferencia Tiempo Real
- ✅ Funciona correctamente con iPad/webcam
- ✅ Detección de reposo (manos abajo) funcional
- ✅ Buffer inteligente con captura por gestos

---

## 🏗️ Arquitectura del Modelo

```python
LSMTransformer(
    input_dim=266,      # 133 keypoints × 2 (x, y)
    num_classes=5,
    d_model=128,
    nhead=4,
    num_layers=3,
    dropout=0.5,
    max_seq_len=90
)
```

### Feature Weights (Learnable)
```python
FEATURE_WEIGHTS = {
    'body': 0.5,        # Cuerpo (0-12)
    'feet': 0.0,        # Pies IGNORADOS (13-22)
    'face': 0.1,        # Cara (23-90)
    'left_hand': 3.0,   # Mano izquierda (91-111)
    'right_hand': 3.0   # Mano derecha (112-132)
}
```

---

## 🔧 Configuración de Entrenamiento

```python
# Hiperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.15
EPOCHS = 150 (early stopping ~60-80)

# Data Augmentation
- Zero lower body (100%)
- Gaussian noise (70%, std=0.01-0.05)
- Frame dropout (30%, 5-15%)
- Spatial jitter (40%, ±0.05)
- Scale augmentation (30%, 0.9x-1.1x)
- Keypoint dropout (20%, 10-30%)
- Horizontal flip: DESHABILITADO
```

---

## 📦 Datos de Entrenamiento

| Clase | Muestras | Proporción |
|-------|----------|------------|
| a     | 167      | 16.0%      |
| b     | 170      | 16.3%      |
| c     | 150      | 14.4%      |
| hola  | 188      | 18.0%      |
| nada  | 370      | 35.4%      |
| **Total** | **1045** | **100%** |

### Preprocesamiento Aplicado
- RTMPose RTMW-x (133 keypoints)
- OneEuroFilter suavizado
- Normalización a centro de caderas
- Hand validation (wrist proximity, confidence)
- Zero out piernas/pies

---

## 🚀 Cómo Usar

### Cargar el modelo
```python
import torch
from src.models.transformer import LSMTransformer
from src.config.settings import INPUT_DIM, CLASS_NAMES

checkpoint = torch.load('models/releases/v1_0_baseline/lsm_v1_0_baseline.pth')
config = checkpoint['config']

model = LSMTransformer(
    input_dim=config['input_dim'],
    num_classes=config['num_classes'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inferencia
```bash
# Tiempo real con iPad/webcam
python -m src.inference.ipad_demo

# Evaluación en videos
python -m src.inference.video_demo
```

---

## ⚠️ Limitaciones Conocidas

1. **Confusión A↔B:** ~21% de confusión mutua (señas similares)
2. **Confusión C→B:** ~35% de C clasificada como B
3. **Señas estáticas:** Menor precisión que señas con movimiento
4. **Generalización:** Mejor con signantes vistos en entrenamiento

---

## 📁 Archivos en este Release

```
models/releases/v1_0_baseline/
├── lsm_v1_0_baseline.pth   # Modelo principal (renombrado)
├── best_model.pth           # Copia de backup
└── model_card.md            # Este archivo
```

---

## 🔗 Referencias

- **Datos procesados:** `dataset/processed_v20251226_130002/`
- **Changelog:** `docs/CHANGELOG_2025-12-26_v2.md`
- **MLflow Run:** `mlruns/1/cfba5d7d1126436b830ac5efd6cccc39/`

---

## ✅ Verificado Por

- Fecha: 2025-12-26 23:42 CST
- Estado: Probado exitosamente en inferencia tiempo real
- Notas: Golden Master - NO SOBRESCRIBIR
