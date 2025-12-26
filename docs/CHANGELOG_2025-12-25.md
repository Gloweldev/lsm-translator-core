# Changelog - 25 de Diciembre 2025

## 🎯 Resumen
Mejoras críticas en inferencia real-time a través de augmentación de FPS en entrenamiento y panel de diagnóstico para interpretabilidad del modelo.

---

## ✨ Nuevas Funcionalidades

### 1. Panel de Diagnóstico en Tiempo Real (`ipad_demo.py`)
**Tecla [D]** para ver:
- Probabilidades de todas las clases (rankeadas)
- Importancia por región del cuerpo (manos, cara, cuerpo)
- Frame más importante de la secuencia
- Tamaño del buffer actual

### 2. Augmentación de FPS (`train.py`)
Nueva augmentación "FPS Warping" que cubre 15-60 FPS:
```python
slow:   0.5x - 0.7x  # Simula 60 FPS
normal: 0.9x - 1.1x  # Original
fast:   1.5x - 2.0x  # Simula 15 FPS (real-time)
```
**Resultado:** El modelo ahora funciona correctamente en inferencia a 14-15 FPS.

### 3. Soporte Multi-formato (`video_demo.py`)
- Soporta: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`
- Auto-rotación de videos verticales (iPhone)
- CLI: `python -m src.inference.video_demo -v "ruta/video.mov"`

### 4. Análisis de FPS (`inspect_processed.py`)
- Escanea FPS de todos los videos en `dataset/raw`
- Genera histograma y recomendaciones
- Exporta `dataset/fps_analysis.png`

---

## 🔧 Mejoras

### Model Interpretability (`transformer.py`)
- Nuevo método `forward_with_analysis()` que retorna:
  - Importancia por frame
  - Importancia por región
  - Feature weights aprendidos

### Inferencia (`ipad_demo.py`)
- Duplicación de frames para compensar FPS bajo
- Diagnóstico inicia activo por defecto
- Nuevo método `predict_with_analysis()`

---

## 📁 Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src/training/train.py` | FPS Warping augmentation (0.5x-2.0x) |
| `src/models/transformer.py` | `forward_with_analysis()` method |
| `src/inference/ipad_demo.py` | Panel diagnóstico, predict_with_analysis |
| `src/inference/video_demo.py` | Multi-formato, CLI args, auto-rotate |
| `src/data/inspect_processed.py` | Análisis FPS de videos raw |
| `docs/TRAINING_EXPLAINED.md` | Documentación actualizada |

---

## 🚀 Commit Sugerido

```
feat(training): FPS augmentation + diagnostic panel

- Add FPS warping (0.5x-2.0x) for 15-60 FPS robustness
- Add real-time diagnostic panel [D] in ipad_demo
- Add forward_with_analysis() for model interpretability  
- Add multi-format support in video_demo (.mov, .mp4, etc)
- Add FPS analysis in inspect_processed
- Fix real-time inference at 14-15 FPS
```
