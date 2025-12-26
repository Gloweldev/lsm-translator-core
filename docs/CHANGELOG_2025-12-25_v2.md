# Changelog - 25 de Diciembre 2025 (Commit 2)

## 🎯 Resumen
Implementación de detector de transición y duration embedding para mejorar detección de señas a diferentes velocidades.

---

## ✨ Nuevas Funcionalidades

### 1. Detector de Transición (`ipad_demo.py`)
Detecta cuando las manos pasan de activo → inactivo y limpia el buffer automáticamente.

```python
if prev_pose_active and not pose_active:
    # Transición: manos bajaron → limpiar buffer
    inference.buffer.clear()
```

**Resultado:** Elimina "inercia" del buffer entre señas.

### 2. Duration Embedding (`transformer.py`)
Nueva clase `DurationEmbedding` que codifica la duración de la secuencia.

```python
class DurationEmbedding(nn.Module):
    # Proyecta duración normalizada a d_model dims
    # Permite al modelo adaptarse a señas rápidas/lentas
```

**Arquitectura actualizada:**
```
Input → Embedding → Positional Enc → Duration Emb (NUEVO) → Transformer → Classifier
```

---

## 🔧 Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src/models/transformer.py` | + `DurationEmbedding` class, modificados `forward()` y `forward_with_analysis()` |
| `src/training/train.py` | Dataset retorna `(seq, label, duration)`, train/evaluate usan duration |
| `src/inference/ipad_demo.py` | + Detector de transición, predict usa duration |

---

## ⚠️ Notas Importantes

1. **Requiere reentrenar** - La arquitectura cambió
2. El modelo anterior NO es compatible
3. El duration embedding es **opcional** (retrocompatible para inferencia)

---

## 🚀 Commit Sugerido

```
feat(model): transition detector + duration embedding

- Add transition detector for buffer cleanup on hand lowering
- Add DurationEmbedding class for speed-adaptive inference
- Modify forward() to accept optional duration parameter
- Update training to pass duration from dataset
- Update inference to calculate and pass duration

BREAKING CHANGE: Model architecture changed, requires retraining
```

---

## 📋 Para entrenar

```bash
python -m src.training.train
```

El modelo ahora aprenderá a manejar señas a diferentes velocidades.
