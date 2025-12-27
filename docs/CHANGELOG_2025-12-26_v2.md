# Changelog v2.0.0 - Hand Validation & Anti-Overfitting

**Fecha:** 2025-12-26  
**Autor:** Claude + User  
**Run ID MLflow:** `cfba5d7d1126436b830ac5efd6cccc39`  
**Datos Procesados:** `processed_v20251226_130002`  
**Modelo Release:** `models/releases/v1_0_baseline/lsm_v1_0_baseline.pth`

---

## 🎯 Resumen Ejecutivo

Esta versión implementa mejoras significativas en:
1. **Validación de manos** - Elimina keypoints "alucinados"
2. **Data augmentation agresiva** - Reduce overfitting
3. **Enfoque en manos** - Ignora partes irrelevantes del cuerpo
4. **Detección de reposo mejorada** - Funciona con cualquier complexión

---

## 📊 Resultados Finales

| Clase | Precisión | Recall | F1-Score |
|-------|-----------|--------|----------|
| A     | ~76%      | ~77%   | ~76%     |
| B     | ~58%      | ~74%   | ~65%     |
| C     | ~83%      | ~58%   | ~68%     |
| HOLA  | ~88%      | ~98%   | ~93%     |
| NADA  | ~100%     | ~94%   | ~97%     |

**Accuracy Global:** ~83-85%  
**Funciona en tiempo real:** ✅

---

## 🔧 Cambios por Componente

### 1. Preprocesamiento (`preprocessor.py`)

#### Función `is_hand_valid()`
Valida si una mano detectada es confiable:

```python
def is_hand_valid(keypoints, scores, hand_start, hand_end, wrist_idx,
                  max_spread=150, min_confidence=0.8, 
                  high_confidence=0.9, max_wrist_distance=100):
    """
    Checks:
    1. Mínimo 5 puntos válidos
    2. Coherencia espacial (spread < 150px)
    3. Confianza promedio ≥ 0.8
    4. Proximidad a muñeca ≤ 100px
    """
```

#### Función `filter_incoherent_hands()`
Aplica validación a ambas manos:
- Mano izquierda: índices 91-111, muñeca: 9
- Mano derecha: índices 112-132, muñeca: 10

#### Versionado de datos
- Cada `--full` crea `processed_v[TIMESTAMP]/`
- `.latest_processed` apunta a la versión más reciente
- Scripts usan automáticamente la última versión

---

### 2. Entrenamiento (`train.py`)

#### Data Augmentation Mejorada

| Técnica | Probabilidad | Descripción |
|---------|--------------|-------------|
| Zero lower body | 100% | Siempre elimina índices 13-22 |
| Ruido gaussiano | 70% | std variable 0.01-0.05 |
| Dropout frames | 30% | 5-15% de frames |
| Spatial jitter | 40% | Offset X,Y ±0.05 |
| Scale augmentation | 30% | 0.9x - 1.1x |
| Keypoint dropout | 20% | 10-30% de cuerpo/cara |
| ~~Horizontal flip~~ | DESHABILITADO | Causaba confusión A↔B↔C |

#### Hiperparámetros Optimizados

```python
# settings.py
DROPOUT = 0.5           # Balance regularización
LABEL_SMOOTHING = 0.15  # No sobre-regularizar

FEATURE_WEIGHTS = {
    'body': 0.5,        # Reducido
    'feet': 0.0,        # IGNORADO
    'face': 0.1,        # Bajo
    'left_hand': 3.0,   # MÁXIMO
    'right_hand': 3.0   # MÁXIMO
}
```

---

### 3. Inferencia (`ipad_demo.py`, `video_demo.py`)

#### Hand Validation Consistente
Ambos scripts incluyen:
- `is_hand_valid()` - misma lógica que preprocessor
- `filter_incoherent_hands()` - filtra antes de smoothing
- `SKIP_INDICES = {13-22}` - ignora piernas/pies

#### Detección de Reposo Mejorada

```python
def is_pose_active(raw_keypoints, frame_height, debug=False):
    """
    Usa posición RELATIVA al torso:
    - Centro torso = promedio(hombros, caderas)
    - Umbral = centro + 20% altura torso
    - Activo si muñeca ARRIBA del umbral
    """
```

Funciona para cualquier complexión corporal.

---

### 4. Análisis (`confusion_analysis.py`, `inspect_processed.py`)

#### Selección de versión de datos
```bash
# Usa última versión automáticamente
python -m src.analysis.confusion_analysis

# Versión específica
python -m src.analysis.confusion_analysis -d processed_v20251226_130002
```

#### `get_latest_processed_dir()` en settings.py
Busca en orden:
1. `.latest_processed` file
2. Directorio `processed_v*` más reciente
3. `dataset/processed` (fallback)

---

### 5. Debug RTMPose (`debug_rtmpose.py`)

Muestra validación de manos en tiempo real:
- Estado de cada mano: `VÁLIDA` / `INVÁLIDA (razón)`
- Razones: `pocos`, `dispersa`, `conf(X.XX)`, `lejos(Xpx)`, `sin_muñeca`

---

## 📁 Archivos Modificados

```
src/
├── config/
│   └── settings.py          # FEATURE_WEIGHTS, DROPOUT, LABEL_SMOOTHING, get_latest_processed_dir()
├── extraction/
│   ├── preprocessor.py      # is_hand_valid, filter_incoherent_hands, versionado
│   └── debug_rtmpose.py     # Visualización de validación
├── training/
│   └── train.py             # Data augmentation mejorada
├── inference/
│   ├── ipad_demo.py         # Hand validation + is_pose_active mejorado
│   └── video_demo.py        # Hand validation
├── analysis/
│   ├── confusion_analysis.py # Selección de datos -d
│   ├── temporal_analysis.py  # Hand validation
│   └── live_temporal_demo.py # Hand validation
└── data/
    └── inspect_processed.py  # Selección de datos -d

docs/
└── DATA_PIPELINE.md          # Documentación actualizada
```

---

## 🚀 Comandos Principales

### Preprocesar datos
```bash
# Incremental (solo nuevos)
python -m src.extraction.preprocessor

# Completo (nueva versión)
python -m src.extraction.preprocessor --full
```

### Entrenar
```bash
# Usa última versión automáticamente
python -m src.training.train

# Versión específica
python -m src.training.train -d processed_v20251226_130002

# Ver versiones disponibles
python -m src.training.train --list-versions
```

### Evaluar
```bash
python -m src.analysis.confusion_analysis
```

### Inferencia tiempo real
```bash
python -m src.inference.ipad_demo
python -m src.inference.ipad_demo --debug  # Ver debug
```

### Inferencia en videos
```bash
python -m src.inference.video_demo
python -m src.inference.video_demo -v "path/video.mp4"
```

---

## 🎓 Lecciones Aprendidas

1. **Consistencia es clave** - Preprocessing e inferencia DEBEN usar la misma lógica
2. **Overfitting sutil** - Val Acc 100% no significa éxito en producción
3. **Data augmentation** - Necesita balance, demasiada causa underfitting
4. **Horizontal flip** - Malo para señas donde izquierda ≠ derecha (A, B, C)
5. **Detección de reposo** - Debe ser relativa al cuerpo, no absoluta

---

## 🔮 Próximos Pasos Sugeridos

1. **Más datos** - Grabar más señantes con diferentes condiciones
2. **MediaPipe Hands** - Para mejor detección de dedos en señas estáticas
3. **Señas dinámicas** - El modelo funciona mejor con HOLA (movimiento) que A, B, C (estáticas)
4. **Fine-tuning per-user** - Calibración para cada usuario

---

## ✅ Checklist de Verificación

- [ ] Modelo guardado en `experiments/mlruns/best_model.pth`
- [ ] Datos procesados en `dataset/processed_v[TIMESTAMP]/`
- [ ] `.latest_processed` apunta a versión correcta
- [ ] Confusion matrix guardada
- [ ] Probado en tiempo real con éxito
