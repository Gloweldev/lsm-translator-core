# Documentación: Sistema de Entrenamiento LSM-Core

## 📌 Resumen Ejecutivo

El Transformer **NO aprende imágenes ni "posiciones visuales"**. Aprende **secuencias de coordenadas numéricas** que representan movimientos del cuerpo a través del tiempo.

---

## 🎯 ¿Qué Aprende el Modelo?

### Entrada: Vector de 266 números por frame

```
Frame → [x0, y0, x1, y1, x2, y2, ..., x132, y132]
              ← 133 keypoints × 2 coordenadas = 266 ─→
```

### Estructura de los 133 keypoints (RTMPose WholeBody)

| Región | Keypoints | Índices | Dimensiones |
|--------|-----------|---------|-------------|
| Cuerpo | 17 puntos | 0-16 | 0-33 |
| Pies | 6 puntos | 17-22 | 34-45 |
| Cara | 68 puntos | 23-90 | 46-181 |
| Mano Izq | 21 puntos | 91-111 | 182-223 |
| Mano Der | 21 puntos | 112-132 | 224-265 |

### Secuencia temporal

```
Video de 90 frames:
├─ Frame 1:  [266 coordenadas]
├─ Frame 2:  [266 coordenadas]
├─ Frame 3:  [266 coordenadas]
...
└─ Frame 90: [266 coordenadas]

Tensor de entrada: (90, 266) → Secuencia × Features
```

---

## 🧠 Arquitectura del Modelo

### `LSMTransformer` (transformer.py)

```
                    ┌─────────────────┐
Entrada (90×266) →  │ Feature Weights │ → Prioriza manos (×2.5)
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Embedding     │ → Proyecta 266 → 512 dims
                    │   + LayerNorm   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Positional Enc. │ → Codifica ORDEN temporal
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Transformer    │ → 4 capas, 8 heads
                    │    Encoder      │ → Aprende patrones
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Clasificador  │ → 512 → 256 → 5 clases
                    └────────┬────────┘
                             ↓
               Salida: [prob_a, prob_b, prob_c, prob_hola, prob_nada]
```

---

## ⚙️ Configuración (settings.py)

### Dimensiones del modelo
```python
INPUT_DIM = 266       # 133 keypoints × 2 (x,y)
D_MODEL = 512         # Dimensión interna del Transformer
N_HEADS = 8           # Cabezas de atención
N_LAYERS = 4          # Capas del encoder
MAX_SEQ_LEN = 90      # Máximo frames por secuencia
```

### Feature Weights (prioridad por región)
```python
FEATURE_WEIGHTS = {
    'body': 1.0,       # Torso normal
    'feet': 0.3,       # Pies menos relevantes
    'face': 0.1,       # Cara casi ignorada
    'left_hand': 2.5,  # Mano izquierda ×2.5
    'right_hand': 2.5  # Mano derecha ×2.5
}
```

Las manos tienen peso **25× mayor** que la cara. Esto porque las señas dependen de las manos, no de expresiones faciales.

---

## 📊 Flujo de Entrenamiento (train.py)

### 1. Carga de datos
```python
# Lee archivos .npy del preprocessing
sequences, labels = load_dataset(PROCESSED_DATA_DIR)
# Cada .npy contiene: array de shape (num_frames, 266)
```

### 2. Dataset con Augmentación
```python
LSMDataset(sequences, labels, augment=True)
```

**Augmentaciones temporales (críticas):**

| Augmentación | Probabilidad | Propósito |
|--------------|--------------|-----------|
| Random Crop | 50% | Corta el video para evitar "bias de retorno al reposo" |
| **FPS Warping** | **50%** | **Simula 15-60 FPS (0.5x-2.0x)** |
| Start Offset | 30% | Simula inicio tardío del gesto |
| Gaussian Noise | 50% | Robustez a ruido en keypoints |
| Frame Dropout | 20% | Simula frames perdidos |

#### FPS Warping (nuevo en v2.1)
```python
warp_type = choice(['slow', 'normal', 'fast'], p=[0.3, 0.4, 0.3])

slow:   0.5x - 0.7x  # Simula video a 60 FPS
normal: 0.9x - 1.1x  # Velocidad original
fast:   1.5x - 2.0x  # Simula video a 15 FPS (como real-time!)
```

Esto permite que el modelo funcione correctamente en inferencia real-time a 14-15 FPS.

### 3. Balanceo de clases
```python
WeightedRandomSampler(weights, len(labels))
# Muestrea más las clases con menos ejemplos
```

### 4. Training loop
```python
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(...)  # Forward + Backward
    val_loss, val_acc = evaluate(...)         # Solo forward
    
    # Early stopping si no mejora en 10 epochs
    if no_improvement:
        break
```

---

## 🎓 ¿Qué "Entiende" el Transformer?

El modelo aprende **patrones temporales de coordenadas**:

1. **Posición relativa de manos** respecto a caderas
2. **Trayectoria** de movimiento (hacia arriba, circular, etc.)
3. **Velocidad** del gesto
4. **Forma de la mano** (configuración de dedos)
5. **Sincronización** entre mano izquierda y derecha

### Ejemplo conceptual:

```
Seña "HOLA" → El modelo aprende:
├─ Mano cerca de cara al inicio
├─ Movimiento lateral ondulatorio
├─ Dedos extendidos y separados
└─ Secuencia temporal de ~60 frames
```

---

## 📁 Archivos del sistema

| Archivo | Función |
|---------|---------|
| `settings.py` | Configuración global |
| `transformer.py` | Arquitectura del modelo + `forward_with_analysis()` |
| `train.py` | Script de entrenamiento |
| `preprocessor.py` | Extrae keypoints de videos |
| `inspect_processed.py` | Valida dataset + análisis de FPS |
| `ipad_demo.py` | Inferencia real-time con diagnóstico |
| `video_demo.py` | Validación con videos (soporta .mov, .mp4) |

---

## ❓ Preguntas Frecuentes

### ¿El modelo ve la imagen del video?
**NO.** Solo ve coordenadas numéricas (x, y) de cada keypoint.

### ¿Por qué las manos tienen más peso?
Porque las señas en LSM se definen principalmente por las manos. La cara tiene peso 0.1 porque no aporta a la clasificación.

### ¿Qué pasa si el FPS varía?
El modelo fue entrenado con **augmentación de FPS** (0.5x-2.0x) que cubre desde 15 FPS hasta 60 FPS. Esto lo hace robusto a variaciones de velocidad.

### ¿Por qué Random Crop es tan importante?
Sin él, el modelo aprende a detectar señas solo cuando las manos bajan al final del video ("bias de retorno al reposo").

### ¿Cómo puedo ver qué está "pensando" el modelo?
Usa el **Panel de Diagnóstico** en `ipad_demo.py` (tecla [D]) que muestra:
- Probabilidades de todas las clases
- Importancia por región del cuerpo
- Frame más importante de la secuencia

---

## 📈 Métricas Típicas

```
Entrenamiento exitoso:
├─ Train Accuracy: ~98%
├─ Val Accuracy: ~96-97%
├─ Epochs: 50-80
└─ Early stopping por patience
```

