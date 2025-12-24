"""
Demo Visual con Análisis Temporal en Tiempo Real.

Muestra el video junto con una gráfica que se construye en tiempo real
mostrando las probabilidades del Transformer frame por frame.

Uso:
    python -m src.analysis.live_temporal_demo

Controles:
    [Q] - Salir
    [SPACE] - Pausar/Reanudar
    [N] - Siguiente video
"""

import cv2
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from collections import deque
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    MLRUNS_DIR,
    RAW_DATA_DIR,
    KEYPOINTS_PER_FRAME,
    INPUT_DIM,
    MAX_SEQ_LEN,
    CONFIDENCE_THRESHOLD,
    FILTER_MIN_CUTOFF,
    FILTER_BETA,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    CLASS_NAMES,
    RTMPOSE_MODEL,
    INFERENCE_CONFIDENCE_THRESHOLD
)
from src.utils.smoothing import OneEuroFilter
from src.models.transformer import LSMTransformer
from mmpose.apis import MMPoseInferencer

# =============================================
# CONFIGURACIÓN
# =============================================
NUM_VIDEOS = 5
BUFFER_SIZE = MAX_SEQ_LEN
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# Colores para gráfica
COLORS = {
    'a': '#2ecc71',      # Verde
    'b': '#3498db',      # Azul
    'c': '#9b59b6',      # Púrpura
    'hola': '#e74c3c',   # Rojo
    'nada': '#95a5a6',   # Gris
}


class LiveTemporalDemo:
    """Demo con video y gráfica en tiempo real."""
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Buffer
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # Smoothers
        self.smoothers = None
        self.frame_count = 0
        
        # Historial de probabilidades
        self.history = {cls: [] for cls in CLASS_NAMES}
        
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
        model_path = MLRUNS_DIR / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        print(f"🧠 Modelo: best_model.pth (Val Acc: {checkpoint.get('val_acc', 0):.4f})")
        
        self.model = LSMTransformer(
            input_dim=config.get('input_dim', INPUT_DIM),
            num_classes=config.get('num_classes', len(CLASS_NAMES)),
            d_model=config.get('d_model', D_MODEL),
            nhead=config.get('n_heads', N_HEADS),
            num_layers=config.get('n_layers', N_LAYERS),
            dropout=0.0,
            max_seq_len=config.get('max_seq_len', BUFFER_SIZE)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✅ Transformer cargado")
    
    def reset(self):
        """Reset para nuevo video."""
        self.buffer.clear()
        self.smoothers = None
        self.frame_count = 0
        self.history = {cls: [] for cls in CLASS_NAMES}
    
    def _normalize_to_center(self, keypoints_flat: np.ndarray) -> np.ndarray:
        kp = keypoints_flat.reshape(-1, 2)
        
        left_hip = kp[LEFT_HIP_IDX]
        right_hip = kp[RIGHT_HIP_IDX]
        
        if np.any(left_hip != 0) and np.any(right_hip != 0):
            center = (left_hip + right_hip) / 2
        elif np.any(left_hip != 0):
            center = left_hip
        elif np.any(right_hip != 0):
            center = right_hip
        else:
            valid = kp[np.any(kp != 0, axis=1)]
            center = valid.mean(axis=0) if len(valid) > 0 else np.array([0.0, 0.0])
        
        return (kp - center).flatten()
    
    def extract_keypoints(self, frame) -> np.ndarray:
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
        
        # Smoothing
        self.frame_count += 1
        t = self.frame_count / 30.0
        
        if self.smoothers is None:
            self.smoothers = [OneEuroFilter(t0=t, x0=raw_keypoints[i],
                              min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
                              for i in range(INPUT_DIM)]
            smoothed = raw_keypoints.copy()
        else:
            smoothed = np.array([self.smoothers[i](t, raw_keypoints[i]) for i in range(INPUT_DIM)])
        
        return self._normalize_to_center(smoothed)
    
    def predict(self) -> tuple:
        if len(self.buffer) == 0:
            return None, np.zeros(len(CLASS_NAMES))
        
        sequence = np.array(list(self.buffer))
        
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([padding, sequence])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
        
        probs_np = probs[0].cpu().numpy()
        predicted_idx = probs_np.argmax()
        
        return CLASS_NAMES[predicted_idx], probs_np
    
    def create_graph_image(self, target_class: str, width: int = 600, height: int = 300) -> np.ndarray:
        """Crea imagen de la gráfica actual."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        num_frames = len(self.history[CLASS_NAMES[0]])
        if num_frames == 0:
            ax.text(0.5, 0.5, 'Esperando datos...', ha='center', va='center', fontsize=14)
        else:
            frames = np.arange(num_frames)
            
            # Dibujar todas las clases
            for cls in CLASS_NAMES:
                if cls == target_class:
                    ax.plot(frames, self.history[cls], linewidth=3, 
                           color=COLORS.get(cls, 'green'), label=f'{cls} (TARGET)')
                elif cls == 'nada':
                    ax.plot(frames, self.history[cls], linewidth=2, 
                           color='gray', linestyle='--', alpha=0.7, label='nada')
                else:
                    ax.plot(frames, self.history[cls], linewidth=1, 
                           alpha=0.4, color=COLORS.get(cls, 'blue'))
            
            # Umbral
            ax.axhline(y=INFERENCE_CONFIDENCE_THRESHOLD, color='red', 
                      linestyle='-', linewidth=1.5, alpha=0.7)
            
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, max(num_frames, 30))
            ax.set_ylabel('Probabilidad')
            ax.set_xlabel('Frame')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        ax.set_title(f'Clase Real: {target_class}', fontsize=12)
        
        # Convertir a imagen
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = canvas.buffer_rgba()
        graph_img = np.asarray(buf)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return graph_img
    
    def run_video(self, video_path: Path, true_class: str):
        """Procesa un video mostrando video y gráfica."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ No se pudo abrir: {video_path}")
            return False
        
        self.reset()
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = int(1000 / fps)
        
        paused = False
        frame_idx = 0
        
        print(f"\n▶️ {video_path.name} (clase: {true_class})")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Video terminado - mostrar resultado final
                    cv2.waitKey(2000)
                    break
                
                # Procesar frame
                features = self.extract_keypoints(frame)
                self.buffer.append(features)
                
                # Predecir
                prediction, probs = self.predict()
                
                # Guardar historial
                for i, cls in enumerate(CLASS_NAMES):
                    self.history[cls].append(probs[i])
                
                frame_idx += 1
            
            # Crear visualización
            h, w = frame.shape[:2]
            
            # Resize video a altura fija
            target_height = 480
            scale = target_height / h
            frame = cv2.resize(frame, (int(w * scale), target_height))
            h, w = frame.shape[:2]
            
            # Crear gráfica (misma altura que video, ancho fijo)
            graph_img = self.create_graph_image(true_class, width=600, height=target_height)
            
            # Agregar info al video
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Real: {true_class}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if prediction:
                color = (0, 255, 0) if prediction == true_class else (0, 0, 255)
                confidence = max(probs)
                cv2.putText(frame, f"Pred: {prediction} ({confidence:.2f})", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Combinar video (izquierda) y gráfica (derecha)
            combined = np.hstack([frame, graph_img])
            
            cv2.imshow('Live Temporal Analysis', combined)
            
            key = cv2.waitKey(frame_delay if not paused else 50)
            
            if key == ord('q'):
                cap.release()
                return True  # Salir completamente
            elif key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                break  # Siguiente video
        
        cap.release()
        return False


def get_random_videos(num_videos: int) -> list:
    all_videos = []
    for cls in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / cls
        if class_dir.exists():
            for v in class_dir.glob("*.mp4"):
                all_videos.append((v, cls))
    
    if len(all_videos) < num_videos:
        return all_videos
    
    return random.sample(all_videos, num_videos)


def main():
    print("=" * 70)
    print("🎬 Live Temporal Analysis Demo")
    print("   Video + Gráfica de Probabilidades en Tiempo Real")
    print("=" * 70)
    print("\nControles:")
    print("   [Q] - Salir")
    print("   [SPACE] - Pausar/Reanudar")
    print("   [N] - Siguiente video")
    print()
    
    videos = get_random_videos(NUM_VIDEOS)
    
    if not videos:
        print(f"❌ No se encontraron videos en {RAW_DATA_DIR}")
        return
    
    print(f"📹 {len(videos)} videos seleccionados\n")
    
    demo = LiveTemporalDemo()
    
    for i, (video_path, true_class) in enumerate(videos):
        print(f"[{i+1}/{len(videos)}]", end=" ")
        
        should_exit = demo.run_video(video_path, true_class)
        
        if should_exit:
            break
    
    cv2.destroyAllWindows()
    print("\n✅ Demo completado")


if __name__ == "__main__":
    main()
