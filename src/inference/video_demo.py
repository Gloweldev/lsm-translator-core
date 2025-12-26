"""
Demo de Validación con Videos del Dataset.

Toma videos aleatorios de dataset/raw y muestra las predicciones
para validar que el modelo está funcionando correctamente.

Uso:
    # Videos aleatorios del dataset
    python -m src.inference.video_demo
    
    # Video específico
    python -m src.inference.video_demo --video path/to/video.mp4
    python -m src.inference.video_demo -v path/to/video.mp4

Controles:
    [Q] - Salir
    [N] - Siguiente video
    [ESPACIO] - Pausar
"""

import cv2
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    RAW_DATA_DIR,
    MLRUNS_DIR,
    KEYPOINTS_PER_FRAME,
    INPUT_DIM,
    MAX_SEQ_LEN,
    CONFIDENCE_THRESHOLD,
    FILTER_MIN_CUTOFF,
    FILTER_BETA,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    RTMPOSE_MODEL
)
from src.utils.smoothing import OneEuroFilter
from src.models.transformer import LSMTransformer
from mmpose.apis import MMPoseInferencer

# =============================================
# CONFIGURACIÓN
# =============================================
NUM_VIDEOS = 10  # Videos aleatorios a probar
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']
BUFFER_SIZE = MAX_SEQ_LEN  # 90 frames
MIN_FRAMES = 30

# Conexiones del esqueleto
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class VideoValidator:
    """Valida el modelo con videos del dataset."""
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Buffer
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # Índices de caderas para normalización
        self.LEFT_HIP_IDX = 11
        self.RIGHT_HIP_IDX = 12
        
        # Smoothers (OneEuroFilter por keypoint - igual que preprocessing)
        self.smoothers = None
        self.frame_count = 0
        
        # Cargar modelos
        self._load_rtmpose()
        self._load_transformer()
    
    def _load_rtmpose(self):
        print(f"🚀 Cargando RTMPose en {self.device}...")
        self.pose_inferencer = MMPoseInferencer(
            pose2d=RTMPOSE_MODEL,
            device=str(self.device)
        )
        print("✅ RTMPose cargado")
    
    def _load_transformer(self):
        """Carga el Transformer usando config del checkpoint."""
        model_path = MLRUNS_DIR / "best_model.pth"
        if not model_path.exists():
            for p in MLRUNS_DIR.rglob("best_model.pth"):
                model_path = p
                break
        
        if not model_path.exists():
            raise FileNotFoundError("No se encontró best_model.pth")
        
        # Cargar checkpoint para obtener config
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Mostrar info del modelo
        epoch = checkpoint.get('epoch', '?')
        val_acc = checkpoint.get('val_acc', 0)
        run_id = checkpoint.get('run_id', 'N/A')
        
        print(f"🧠 Modelo: {model_path.name}")
        print(f"   Epoch: {epoch}, Val Acc: {val_acc:.4f}")
        if run_id != 'N/A':
            print(f"   Run ID: {run_id}")
        
        self.model = LSMTransformer(
            input_dim=config.get('input_dim', INPUT_DIM),
            num_classes=config.get('num_classes', len(CLASS_NAMES)),
            d_model=config.get('d_model', D_MODEL),
            nhead=config.get('n_heads', N_HEADS),
            num_layers=config.get('n_layers', N_LAYERS),
            dropout=0.0,  # Sin dropout en inferencia
            max_seq_len=config.get('max_seq_len', BUFFER_SIZE)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Transformer cargado")
    
    def reset(self):
        """Reset buffer y smoothers para nuevo video."""
        self.buffer.clear()
        self.smoothers = None
        self.frame_count = 0
    
    def _normalize_to_center(self, keypoints_flat: np.ndarray) -> np.ndarray:
        """
        Normaliza keypoints restando el centro de caderas.
        Igual que en preprocessor.py para que coincida con el entrenamiento.
        """
        # Reconstruir a Nx2
        kp = keypoints_flat.reshape(-1, 2)
        
        left_hip = kp[self.LEFT_HIP_IDX]
        right_hip = kp[self.RIGHT_HIP_IDX]
        
        # Calcular centro
        if np.any(left_hip != 0) and np.any(right_hip != 0):
            center = (left_hip + right_hip) / 2
        elif np.any(left_hip != 0):
            center = left_hip
        elif np.any(right_hip != 0):
            center = right_hip
        else:
            # Usar centroide de puntos válidos
            valid = kp[np.any(kp != 0, axis=1)]
            center = valid.mean(axis=0) if len(valid) > 0 else np.array([0.0, 0.0])
        
        # Restar centro
        normalized = kp - center
        
        return normalized.flatten()
    
    def extract_keypoints(self, frame) -> tuple:
        """
        Extrae keypoints del frame.
        
        Returns:
            (raw_keypoints, normalized_keypoints)
            - raw: para visualización
            - normalized: para predicción
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = list(self.pose_inferencer(rgb, return_vis=False))
        
        raw_keypoints = np.zeros(INPUT_DIM)
        
        if results and results[0].get('predictions'):
            preds = results[0]['predictions']
            if preds and preds[0]:
                kp = np.array(preds[0][0].get('keypoints', []))
                scores = np.array(preds[0][0].get('keypoint_scores', []))
                
                for i in range(min(len(kp), KEYPOINTS_PER_FRAME)):
                    if scores[i] >= CONFIDENCE_THRESHOLD:
                        raw_keypoints[i*2] = kp[i, 0]
                        raw_keypoints[i*2 + 1] = kp[i, 1]
        
        # Aplicar smoothing (OneEuroFilter - igual que preprocessing)
        self.frame_count += 1
        t = self.frame_count / 30.0  # Timestamp en segundos
        
        if self.smoothers is None:
            self.smoothers = []
            for i in range(INPUT_DIM):
                self.smoothers.append(OneEuroFilter(
                    t0=t, x0=raw_keypoints[i],
                    min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA
                ))
            smoothed_keypoints = raw_keypoints.copy()
        else:
            smoothed_keypoints = np.array([self.smoothers[i](t, raw_keypoints[i]) for i in range(INPUT_DIM)])
        
        # Normalizar para predicción
        normalized_keypoints = self._normalize_to_center(smoothed_keypoints.copy())
        
        return raw_keypoints, normalized_keypoints
    
    def predict(self) -> tuple:
        if len(self.buffer) < MIN_FRAMES:
            return None, 0.0
        
        sequence = np.array(list(self.buffer))
        actual_frames = len(sequence)  # Duración real antes del padding
        
        # Padding
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([sequence, padding])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        duration = torch.LongTensor([actual_frames]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor, duration=duration)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        return CLASS_NAMES[predicted.item()], confidence.item()
    
    def draw_skeleton(self, frame, keypoints):
        """Dibuja esqueleto usando keypoints RAW (no normalizados)."""
        h, w = frame.shape[:2]
        points = {}
        
        for i in range(KEYPOINTS_PER_FRAME):
            x = int(keypoints[i*2])
            y = int(keypoints[i*2 + 1])
            if x > 0 or y > 0:
                if 0 <= x < w and 0 <= y < h:
                    points[i] = (x, y)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        for i, j in SKELETON_CONNECTIONS:
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (0, 200, 0), 2)
        
        return frame


def find_videos():
    """Busca videos organizados por clase."""
    videos = []
    for class_dir in RAW_DATA_DIR.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for video in class_dir.glob("*.mp4"):
                videos.append((video, class_name))
    return videos


def run_demo(video_path: str = None):
    """Ejecuta el demo.
    
    Args:
        video_path: Ruta a un video específico (opcional). Si es None, usa videos aleatorios.
    """
    print("=" * 60)
    print("🎬 Demo de Validación con Videos")
    print("=" * 60)
    
    # Formatos soportados
    SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    
    # Determinar videos a procesar
    if video_path:
        # Video específico
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"❌ Video no encontrado: {video_path}")
            return
        
        # Verificar formato
        ext = video_file.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            print(f"⚠️ Formato {ext} no probado. Formatos soportados: {SUPPORTED_FORMATS}")
            print("   Intentando abrir de todas formas...")
        
        # Verificar que OpenCV puede abrir el video
        test_cap = cv2.VideoCapture(str(video_file))
        if not test_cap.isOpened():
            print(f"❌ No se pudo abrir el video: {video_path}")
            print("   Si es .mov de iPhone, puede necesitar codecs adicionales.")
            print("   Intenta convertir con: ffmpeg -i input.mov -c:v libx264 output.mp4")
            test_cap.release()
            return
        
        fps = test_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        test_cap.release()
        
        # Intentar determinar clase del path
        true_class = video_file.parent.name if video_file.parent.name in CLASS_NAMES else "desconocida"
        selected = [(video_file, true_class)]
        print(f"🎬 Video: {video_path}")
        print(f"   Formato: {ext}")
        print(f"   FPS: {fps:.1f}, Frames: {frame_count}, Duración: {duration:.1f}s")
        print(f"   Clase detectada: {true_class.upper()}")
    else:
        # Videos aleatorios del dataset
        all_videos = find_videos()
        if not all_videos:
            print("❌ No hay videos en dataset/raw/")
            return
        
        selected = random.sample(all_videos, min(NUM_VIDEOS, len(all_videos)))
        print(f"📂 Seleccionados {len(selected)} videos aleatorios\n")
    
    # Inicializar
    try:
        validator = VideoValidator()
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎮 Controles: [Q]Salir [N]Siguiente [ESPACIO]Pausar")
    print("=" * 60)
    
    # Estadísticas
    correct = 0
    total = 0
    
    for idx, (video_path, true_class) in enumerate(selected):
        validator.reset()
        
        print(f"\n[{idx+1}/{len(selected)}] {true_class}/{video_path.name}")
        print(f"   Clase verdadera: {true_class.upper()}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        paused = False
        last_prediction = None
        predictions_for_video = []
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                
                # Auto-rotar si viene horizontal pero debería ser vertical (iPhone .mov)
                if w > h:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    h, w = frame.shape[:2]
                
                # Extraer keypoints (raw para visualización, normalized para predicción)
                raw_keypoints, normalized_keypoints = validator.extract_keypoints(frame)
                
                # Acumular normalized para predicción
                validator.buffer.append(normalized_keypoints)
                
                # Predecir
                prediction, confidence = validator.predict()
                
                if prediction:
                    last_prediction = (prediction, confidence)
                    predictions_for_video.append(prediction)
                
                # Dibujar esqueleto con keypoints RAW (posición real)
                frame = validator.draw_skeleton(frame, raw_keypoints)
                
                # UI
                cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
                
                # Info del video
                cv2.putText(frame, f"Video {idx+1}/{len(selected)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"Clase REAL: {true_class.upper()}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Buffer
                buffer_pct = len(validator.buffer) / BUFFER_SIZE * 100
                cv2.putText(frame, f"Buffer: {len(validator.buffer)}/{BUFFER_SIZE}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.rectangle(frame, (10, 85), (210, 100), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 85), (10 + int(buffer_pct * 2), 100), (0, 255, 0), -1)
                
                # Predicción
                if last_prediction:
                    pred, conf = last_prediction
                    is_correct = pred.lower() == true_class.lower()
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    status = "✓" if is_correct else "✗"
                    
                    cv2.putText(frame, f"Prediccion: {pred.upper()} {conf:.0%} {status}", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Lado derecho
                cv2.putText(frame, video_path.name[:30], (w - 250, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            if paused:
                cv2.putText(frame, "PAUSED", (w//2-50, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Video Validation Demo', frame)
            
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                
                # Resumen final
                if total > 0:
                    print(f"\n📊 Accuracy: {correct}/{total} = {correct/total:.1%}")
                return
            elif key == ord('n'):
                break
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        
        # Evaluar video
        if predictions_for_video:
            # Predicción más común
            from collections import Counter
            most_common = Counter(predictions_for_video).most_common(1)[0][0]
            is_correct = most_common.lower() == true_class.lower()
            
            if is_correct:
                correct += 1
                print(f"   ✅ Predicción final: {most_common.upper()}")
            else:
                print(f"   ❌ Predicción final: {most_common.upper()} (debía ser {true_class.upper()})")
            total += 1
        else:
            print(f"   ⚠️ Sin predicción (video muy corto)")
    
    cv2.destroyAllWindows()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"   Videos evaluados: {total}")
    print(f"   Correctos: {correct}")
    print(f"   Accuracy: {correct/total:.1%}" if total > 0 else "   N/A")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo de validación con videos")
    parser.add_argument("-v", "--video", type=str, default=None,
                       help="Ruta a un video específico (opcional)")
    args = parser.parse_args()
    
    run_demo(video_path=args.video)
