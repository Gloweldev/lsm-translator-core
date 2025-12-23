"""
Demo de Inferencia en Tiempo Real con iPad/DroidCam.

Pipeline completo: RTMPose → Transformer → Predicción
Fuente de video: Cámara IP (DroidCam, EpocCam, etc.)

Uso:
    python -m src.inference.ipad_demo

Controles:
    [Q] - Salir
    [R] - Reset buffer
    [S] - Screenshot
"""

import cv2
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    MLRUNS_DIR,
    KEYPOINTS_PER_FRAME,
    INPUT_DIM,
    MAX_SEQ_LEN,
    CONFIDENCE_THRESHOLD,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    RTMPOSE_MODEL
)
from src.models.transformer import LSMTransformer

# RTMPose
from mmpose.apis import MMPoseInferencer

# =============================================
# CONFIGURACIÓN
# =============================================
# Cámara IP (DroidCam/EpocCam)
CAMERA_IP = "192.168.100.134"
CAMERA_PORT = 4747
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"

# O usar webcam local (cambiar a 0)
USE_WEBCAM = False
WEBCAM_ID = 0

# Clases (deben coincidir con el entrenamiento)
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']

# Inferencia
PREDICTION_THRESHOLD = 0.85  # Confianza mínima para mostrar
STABILITY_FRAMES = 5         # Frames consecutivos para estabilizar
BUFFER_SIZE = MAX_SEQ_LEN    # 90 frames

# Visualización
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class LSMInference:
    """Pipeline de inferencia en tiempo real."""
    
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Buffer de frames
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # Estabilización
        self.last_prediction = None
        self.stable_count = 0
        self.stable_prediction = None
        
        # Cargar modelos
        self._load_rtmpose()
        self._load_transformer(model_path)
    
    def _load_rtmpose(self):
        """Carga RTMPose."""
        print(f"🚀 Cargando RTMPose en {self.device}...")
        self.pose_inferencer = MMPoseInferencer(
            pose2d=RTMPOSE_MODEL,
            device=str(self.device)
        )
        print("✅ RTMPose cargado")
    
    def _load_transformer(self, model_path: str = None):
        """Carga el Transformer entrenado usando config del checkpoint."""
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path is None:
            raise FileNotFoundError("No se encontró modelo entrenado")
        
        print(f"🧠 Cargando Transformer: {model_path}")
        
        # Cargar checkpoint para obtener config
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Crear modelo con config del entrenamiento
        self.model = LSMTransformer(
            input_dim=config.get('input_dim', INPUT_DIM),
            num_classes=config.get('num_classes', len(CLASS_NAMES)),
            d_model=config.get('d_model', D_MODEL),
            nhead=config.get('n_heads', N_HEADS),
            num_layers=config.get('n_layers', N_LAYERS),
            dropout=0.0,  # Sin dropout en inferencia
            max_seq_len=config.get('max_seq_len', BUFFER_SIZE)
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Transformer cargado (val_acc: {checkpoint.get('val_acc', 'N/A'):.4f})")
    
    def _find_best_model(self) -> str:
        """Busca el mejor modelo en mlruns."""
        # Buscar best_model.pth
        best_model = MLRUNS_DIR / "best_model.pth"
        if best_model.exists():
            return str(best_model)
        
        # Buscar en subdirectorios
        for pth_file in MLRUNS_DIR.rglob("best_model.pth"):
            return str(pth_file)
        
        return None
    
    def _normalize_to_center(self, keypoints_flat: np.ndarray) -> np.ndarray:
        """
        Normaliza keypoints restando el centro de caderas.
        Igual que en preprocessor.py para que coincida con el entrenamiento.
        """
        LEFT_HIP_IDX = 11
        RIGHT_HIP_IDX = 12
        
        # Reconstruir a Nx2
        kp = keypoints_flat.reshape(-1, 2)
        
        left_hip = kp[LEFT_HIP_IDX]
        right_hip = kp[RIGHT_HIP_IDX]
        
        # Calcular centro
        if np.any(left_hip != 0) and np.any(right_hip != 0):
            center = (left_hip + right_hip) / 2
        elif np.any(left_hip != 0):
            center = left_hip
        elif np.any(right_hip != 0):
            center = right_hip
        else:
            valid = kp[np.any(kp != 0, axis=1)]
            center = valid.mean(axis=0) if len(valid) > 0 else np.array([0.0, 0.0])
        
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
        
        # Normalizar para predicción
        normalized_keypoints = self._normalize_to_center(raw_keypoints.copy())
        
        return raw_keypoints, normalized_keypoints
    
    def predict(self) -> tuple:
        """
        Predice la seña actual.
        Funciona con buffer parcial usando padding.
        
        Returns:
            (class_name, confidence) o (None, 0) si no hay suficientes frames
        """
        MIN_FRAMES = 30  # Mínimo para predecir
        
        if len(self.buffer) < MIN_FRAMES:
            return None, 0.0
        
        # Convertir buffer a array
        sequence = np.array(list(self.buffer))
        
        # Padding si es necesario (para llegar a BUFFER_SIZE)
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([sequence, padding])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Inferencia
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        class_idx = predicted.item()
        conf = confidence.item()
        
        return CLASS_NAMES[class_idx], conf
    
    def stabilize(self, prediction: str, confidence: float) -> tuple:
        """
        Estabiliza la predicción para evitar parpadeo.
        Solo muestra si la misma predicción se mantiene por N frames con alta confianza.
        """
        if confidence < PREDICTION_THRESHOLD:
            self.stable_count = 0
            return None, 0.0
        
        if prediction == self.last_prediction:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_prediction = prediction
        
        if self.stable_count >= STABILITY_FRAMES:
            self.stable_prediction = prediction
            return prediction, confidence
        
        # Mientras no sea estable, mantener la última predicción estable
        if self.stable_prediction:
            return self.stable_prediction, confidence
        
        return None, confidence
    
    def reset(self):
        """Reinicia el buffer y estabilización."""
        self.buffer.clear()
        self.last_prediction = None
        self.stable_count = 0
        self.stable_prediction = None
    
    def draw_skeleton(self, frame, keypoints) -> np.ndarray:
        """Dibuja el esqueleto sobre el frame usando keypoints RAW."""
        h, w = frame.shape[:2]
        
        # Reconstruir puntos
        points = {}
        for i in range(KEYPOINTS_PER_FRAME):
            x = int(keypoints[i*2])
            y = int(keypoints[i*2 + 1])
            if x > 0 or y > 0:  # Solo si no es (0,0)
                if 0 <= x < w and 0 <= y < h:
                    points[i] = (x, y)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Conexiones del cuerpo
        for i, j in SKELETON_CONNECTIONS:
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (0, 200, 0), 2)
        
        # Conexiones de manos
        for ci, cj in HAND_CONNECTIONS:
            # Mano izquierda
            i, j = 91 + ci, 91 + cj
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (255, 200, 0), 1)
            # Mano derecha
            i, j = 112 + ci, 112 + cj
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (200, 0, 255), 1)
        
        return frame


def run_demo():
    """Ejecuta el demo de inferencia."""
    print("=" * 60)
    print("🤟 LSM-Core: Demo de Inferencia en Tiempo Real")
    print("=" * 60)
    
    # Inicializar inferencia
    try:
        inference = LSMInference()
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        return
    
    # Conectar a cámara
    if USE_WEBCAM:
        source = WEBCAM_ID
        print(f"📷 Usando webcam {WEBCAM_ID}")
    else:
        source = CAMERA_URL
        print(f"📱 Conectando a: {CAMERA_URL}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ No se pudo conectar a la cámara")
        print("   - Verifica que DroidCam esté activo")
        print(f"   - Revisa la IP: {CAMERA_IP}:{CAMERA_PORT}")
        return
    
    print("✅ Cámara conectada")
    print("\n🎮 Controles: [Q]Salir [R]Reset [S]Screenshot")
    print("=" * 60)
    
    fps_counter = deque(maxlen=30)
    frame_count = 0
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Conexión perdida, reconectando...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(source)
            continue
        
        h, w = frame.shape[:2]
        
        # Extraer keypoints (raw para visualización, normalized para predicción)
        raw_keypoints, normalized_keypoints = inference.extract_keypoints(frame)
        
        # Acumular normalized para predicción
        inference.buffer.append(normalized_keypoints)
        
        # Dibujar esqueleto con keypoints RAW
        frame = inference.draw_skeleton(frame, raw_keypoints)
        
        # Predecir
        prediction, confidence = inference.predict()
        stable_pred, stable_conf = inference.stabilize(prediction, confidence)
        
        # UI
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Buffer status
        buffer_pct = len(inference.buffer) / BUFFER_SIZE * 100
        cv2.putText(frame, f"Buffer: {len(inference.buffer)}/{BUFFER_SIZE}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(frame, (10, 35), (210, 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 35), (10 + int(buffer_pct * 2), 50), (0, 255, 0), -1)
        
        # Predicción
        if stable_pred:
            color = (0, 255, 0) if stable_conf > 0.9 else (0, 255, 255)
            cv2.putText(frame, stable_pred.upper(), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            cv2.putText(frame, f"{stable_conf:.1%}", (w - 120, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        elif prediction:
            cv2.putText(frame, f"({prediction} {confidence:.1%})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        # FPS
        fps_counter.append(1.0 / (time.time() - start_time + 1e-6))
        fps = np.mean(fps_counter)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('LSM-Core Demo', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            inference.reset()
            print("🔄 Buffer reiniciado")
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Guardado: {filename}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Demo terminado")


if __name__ == "__main__":
    run_demo()
