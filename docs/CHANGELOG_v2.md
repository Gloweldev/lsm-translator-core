# LSM-Core v2 - Documentación de Cambios

**Fecha:** 2025-12-23  
**Versión:** 2.0  
**Accuracy:** 97.9%

---

## 📊 Resumen de Resultados

| Métrica | v1 | v2 |
|---------|----|----|
| Val Accuracy | ~50% | **97.9%** |
| Precision (macro) | N/A | 98% |
| Recall (macro) | N/A | 99% |

---

## 🧠 Cambios en el Transformer

### Archivo: `src/models/transformer.py`

#### 1. Feature-Weighted Embedding
Nuevo módulo que prioriza regiones del cuerpo:

```python
# Pesos por región de keypoints
Cuerpo (0-16):      × 1.0  (normal)
Pies (17-22):       × 0.3  (poco relevante)
Cara (23-90):       × 0.1  (ignorar)
Mano izq (91-111):  × 2.5  (muy importante)
Mano der (112-132): × 2.5  (muy importante)
```

Los pesos son **aprendibles** durante el entrenamiento.

#### 2. Arquitectura Mejorada
- **GELU** en lugar de ReLU (mejor gradientes)
- **Pre-LN** (LayerNorm antes de attention, más estable)
- **3 capas** de Transformer (antes 2)
- **LayerNorm adicional** en clasificador

#### 3. Regularización
- Dropout aumentado en clasificador (× 1.5)
- LayerNorm en embedding

---

## 📈 Cambios en Entrenamiento

### Archivo: `src/training/train.py`

#### 1. Anti-Overfitting

| Técnica | Valor | Propósito |
|---------|-------|-----------|
| Label Smoothing | 0.1 | Previene sobreconfianza |
| Dropout | 0.4 | Regularización |
| Weight Decay | 1e-4 | L2 regularization |
| Early Stopping | patience=20 | Detiene si no mejora |

#### 2. Balanceo de Clases
**WeightedRandomSampler** para oversampling:
- Clases minoritarias (b, c) se muestrean más frecuentemente
- Elimina sesgo hacia "nada" (clase mayoritaria)

#### 3. Data Augmentation
```python
# Aplicado solo en entrenamiento
- Ruido gaussiano (σ=0.02)
- Escalado temporal (0.8× - 1.2×)
- Dropout de frames (10%)
```

#### 4. Hiperparámetros Optimizados
```python
epochs: 150
batch_size: 32
learning_rate: 3e-4
n_layers: 3
dropout: 0.4
```

---

## 🔧 Cambios en Inferencia

### Archivos: `src/inference/ipad_demo.py`, `video_demo.py`

#### Carga Dinámica de Configuración
El modelo ahora lee la configuración del checkpoint:

```python
checkpoint = torch.load(model_path)
config = checkpoint.get('config', {})

model = LSMTransformer(
    input_dim=config.get('input_dim', INPUT_DIM),
    num_classes=config.get('num_classes', 5),
    d_model=config.get('d_model', 128),
    num_layers=config.get('n_layers', 3),
    ...
)
```

Esto garantiza compatibilidad con cualquier versión del modelo.

---

## 📁 Estructura de Archivos

```
src/
├── models/
│   └── transformer.py      # Transformer v2 con Feature Weights
├── training/
│   └── train.py            # Entrenamiento con oversampling
├── inference/
│   ├── ipad_demo.py        # Demo tiempo real
│   └── video_demo.py       # Validación con videos
└── extraction/
    └── preprocessor.py     # Pipeline RTMPose → .npy
```

---

## 🚀 Uso

### Entrenar
```bash
python -m src.training.train
```

### Validar con videos
```bash
python -m src.inference.video_demo
```

### Inferencia en tiempo real
```bash
python -m src.inference.ipad_demo
```

---

## 📊 Artefactos Generados

| Archivo | Descripción |
|---------|-------------|
| `experiments/mlruns/best_model.pth` | Modelo entrenado |
| `experiments/mlruns/confusion_matrix.png` | Matriz de confusión |
| `experiments/mlruns/training_curves.png` | Curvas loss/accuracy |
| `experiments/mlruns/mlflow.db` | Base de datos MLflow |

---

## 🔮 Próximos Pasos Sugeridos

1. **Más datos** - Agregar más videos de clases minoritarias
2. **Más clases** - Expandir vocabulario de señas
3. **Exportar modelo** - ONNX para producción
4. **API REST** - Servir modelo en servidor
