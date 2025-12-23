# LSM-Core: Traductor de Lengua de Señas Mexicana

Sistema de reconocimiento de Lengua de Señas Mexicana usando deep learning.

## Estructura del Proyecto

```
lsm-core/
├── dataset/           # Data Lake
│   ├── raw/           # Videos .mp4 originales
│   └── processed/     # Tensores .npy para entrenamiento
├── experiments/       # MLflow (métricas y modelos)
├── src/               # Código fuente
│   ├── config/        # Configuraciones
│   ├── extraction/    # Pose estimation (RTMPose)
│   ├── models/        # Arquitectura Transformer
│   ├── training/      # Entrenamiento y evaluación
│   ├── inference/     # Predicción en tiempo real
│   └── utils/         # Utilidades
└── requirements.txt
```

## Instalación

```bash
conda create -n lsm-ai python=3.11
conda activate lsm-ai
pip install -r requirements.txt
```

## Uso

1. **Preprocesar videos:**
   ```bash
   python -m src.extraction.preprocessor
   ```

2. **Entrenar modelo:**
   ```bash
   python -m src.training.train
   ```

3. **Demo en tiempo real:**
   ```bash
   python -m src.inference.realtime_demo
   ```
