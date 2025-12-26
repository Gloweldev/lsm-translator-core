"""
Demo de Inferencia en Tiempo Real con iPad/DroidCam.

Pipeline completo: RTMPose → Transformer → FSM → Predicción + Gráfica Temporal
Fuente de video: Cámara IP (DroidCam, EpocCam, etc.)

Uso:
    python -m src.inference.ipad_demo

Controles:
    [Q] - Salir
    [R] - Reset buffer
    [S] - Screenshot
    [C] - Clear historial
"""

import cv2
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from collections import deque
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
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

# RTMPose
from mmpose.apis import MMPoseInferencer

# =============================================
# CONFIGURACIÓN
# =============================================
# Cámara IP (DroidCam/EpocCam)
CAMERA_IP = "192.168.100.123"
CAMERA_PORT = 4747
CAMERA_URL = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"

# O usar webcam local (cambiar a 0)
USE_WEBCAM = False
WEBCAM_ID = 0

# Clases (deben coincidir con el entrenamiento)
CLASS_NAMES = ['a', 'b', 'c', 'hola', 'nada']

# FSM Thresholds
TRIGGER_THRESHOLD = 0.85
RELEASE_THRESHOLD = 0.50
BUFFER_SIZE = MAX_SEQ_LEN
MIN_FRAMES = 30

# FPS Normalization - Compensar diferencia entre entrenamiento y tiempo real
TRAINING_FPS = 30  # FPS asumido durante entrenamiento
FPS_SMOOTHING = 10  # Frames para calcular FPS promedio

# Physical Veto - Índices de keypoints (COCO-WholeBody)
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
POSE_MARGIN = 0.08  # Margen de seguridad (8% de altura) para señas bajas

# Historial
MAX_HISTORY = 5
MAX_GRAPH_POINTS = 150  # Puntos max en la gráfica

# Colores para gráfica
COLORS = {
    'a': '#2ecc71',
    'b': '#3498db',
    'c': '#9b59b6',
    'hola': '#e74c3c',
    'nada': '#95a5a6',
}

# Visualización esqueleto
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


# =============================================
# VETO FÍSICO - Detección de Posición de Descanso
# =============================================
def is_pose_active(raw_keypoints: np.ndarray, frame_height: int) -> bool:
    """
    Determina si el usuario está en posición activa para señas.
    
    Retorna False (INACTIVO) si AMBAS muñecas están por debajo de las caderas.
    Esto indica que el usuario tiene las manos abajo (posición de descanso).
    
    Args:
        raw_keypoints: Vector de 266 dims (133 keypoints x 2)
        frame_height: Altura del frame para calcular margen
    
    Returns:
        True si está activo (al menos una mano arriba), False si en descanso
    """
    # Extraer coordenadas Y (recordar: Y mayor = más abajo en imagen)
    left_wrist_y = raw_keypoints[LEFT_WRIST_IDX * 2 + 1]
    right_wrist_y = raw_keypoints[RIGHT_WRIST_IDX * 2 + 1]
    left_hip_y = raw_keypoints[LEFT_HIP_IDX * 2 + 1]
    right_hip_y = raw_keypoints[RIGHT_HIP_IDX * 2 + 1]
    
    # Si no hay detección de keypoints, asumir activo
    if left_hip_y == 0 and right_hip_y == 0:
        return True
    
    # Calcular margen de seguridad (permite señas a la altura del ombligo)
    margin = frame_height * POSE_MARGIN
    
    # Usar la cadera más alta detectada como referencia
    hip_y = min(left_hip_y if left_hip_y > 0 else 9999, 
               right_hip_y if right_hip_y > 0 else 9999)
    
    if hip_y == 9999:
        return True  # No hay referencia de cadera
    
    # Umbral: cadera + margen (más abajo que la cadera)
    threshold_y = hip_y + margin
    
    # Verificar si AMBAS muñecas están por debajo del umbral
    # (Si una está a 0, significa no detectada - ignorar)
    left_below = (left_wrist_y > threshold_y) if left_wrist_y > 0 else True
    right_below = (right_wrist_y > threshold_y) if right_wrist_y > 0 else True
    
    # Inactivo solo si AMBAS manos están abajo Y al menos una fue detectada
    both_detected = left_wrist_y > 0 or right_wrist_y > 0
    
    if left_below and right_below and both_detected:
        return False  # Posición de descanso
    
    return True  # Al menos una mano activa


# =============================================
# FSM - MÁQUINA DE ESTADOS FINITA
# =============================================
class State(Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


class WordFSM:
    """Máquina de Estados para detección de palabras."""
    
    def __init__(self):
        self.state = State.IDLE
        self.last_word = None
        self.word_history = deque(maxlen=MAX_HISTORY)
    
    def update(self, prediction: str, confidence: float) -> str:
        if prediction is None:
            return None
        
        new_word = None
        
        if self.state == State.IDLE:
            if confidence > TRIGGER_THRESHOLD and prediction != "nada":
                self.state = State.ACTIVE
                self.last_word = prediction
                self.word_history.append(prediction)
                new_word = prediction
                print(f"🎯 [{self.state.value}] Detectado: {prediction.upper()} ({confidence:.1%})")
        
        elif self.state == State.ACTIVE:
            if confidence < RELEASE_THRESHOLD or prediction == "nada":
                self.state = State.IDLE
                self.last_word = None
                print(f"⏸️ [{self.state.value}] Liberado")
            elif prediction != self.last_word and confidence > TRIGGER_THRESHOLD:
                self.last_word = prediction
                self.word_history.append(prediction)
                new_word = prediction
                print(f"🔄 [{self.state.value}] Hot-Swap: {prediction.upper()} ({confidence:.1%})")
        
        return new_word
    
    def get_current_word(self) -> str:
        if self.state == State.ACTIVE:
            return self.last_word
        return None
    
    def get_history(self) -> list:
        return list(self.word_history)
    
    def clear_history(self):
        self.word_history.clear()
    
    def reset(self):
        self.state = State.IDLE
        self.last_word = None


class LSMInference:
    """Pipeline de inferencia con gráfica temporal en tiempo real."""
    
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Buffer de frames
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # FSM
        self.fsm = WordFSM()
        
        # Historial de probabilidades para gráfica
        self.prob_history = {cls: deque(maxlen=MAX_GRAPH_POINTS) for cls in CLASS_NAMES}
        
        # Smoothers
        self.smoothers = None
        self.frame_count = 0
        
        # Índices para normalización
        self.LEFT_HIP_IDX = 11
        self.RIGHT_HIP_IDX = 12
        
        # Cargar modelos
        self._load_rtmpose()
        self._load_transformer(model_path)
    
    def _load_rtmpose(self):
        print(f"🚀 Cargando RTMPose en {self.device}...")
        self.pose_inferencer = MMPoseInferencer(
            pose2d=RTMPOSE_MODEL,
            device=str(self.device)
        )
        print("✅ RTMPose cargado")
    
    def _load_transformer(self, model_path: str = None):
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path is None:
            raise FileNotFoundError("No se encontró modelo entrenado")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        epoch = checkpoint.get('epoch', '?')
        val_acc = checkpoint.get('val_acc', 0)
        run_id = checkpoint.get('run_id', 'N/A')
        
        print(f"🧠 Modelo: best_model.pth")
        print(f"   Epoch: {epoch}, Val Acc: {val_acc:.4f}")
        if run_id != 'N/A':
            print(f"   Run ID: {run_id}")
        
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
    
    def _find_best_model(self) -> str:
        best_model = MLRUNS_DIR / "best_model.pth"
        if best_model.exists():
            return str(best_model)
        for p in MLRUNS_DIR.rglob("best_model.pth"):
            return str(p)
        return None
    
    def _normalize_to_center(self, keypoints_flat: np.ndarray) -> np.ndarray:
        kp = keypoints_flat.reshape(-1, 2)
        left_hip = kp[self.LEFT_HIP_IDX]
        right_hip = kp[self.RIGHT_HIP_IDX]
        
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
    
    def extract_keypoints(self, frame) -> tuple:
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
        
        self.frame_count += 1
        t = self.frame_count / 30.0
        
        if self.smoothers is None:
            self.smoothers = [OneEuroFilter(t0=t, x0=raw_keypoints[i],
                              min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA)
                              for i in range(INPUT_DIM)]
            smoothed = raw_keypoints.copy()
        else:
            smoothed = np.array([self.smoothers[i](t, raw_keypoints[i]) for i in range(INPUT_DIM)])
        
        return raw_keypoints, self._normalize_to_center(smoothed.copy())
    
    def predict(self) -> tuple:
        if len(self.buffer) < MIN_FRAMES:
            return None, 0.0, None
        
        sequence = np.array(list(self.buffer))
        
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([sequence, padding])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        probs_np = probs[0].cpu().numpy()
        
        # Guardar historial de probabilidades
        for i, cls in enumerate(CLASS_NAMES):
            self.prob_history[cls].append(probs_np[i])
        
        return CLASS_NAMES[predicted.item()], confidence.item(), probs_np
    
    def predict_with_analysis(self) -> dict:
        """
        Predicción con análisis de interpretabilidad.
        
        Returns:
            dict con predicción, confianza, todas las probabilidades,
            importancia por frame, y importancia por región del cuerpo.
        """
        if len(self.buffer) < MIN_FRAMES:
            return None
        
        sequence = np.array(list(self.buffer))
        actual_frames = len(sequence)
        
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([sequence, padding])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            analysis = self.model.forward_with_analysis(tensor)
            logits = analysis['logits']
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        probs_np = probs[0].cpu().numpy()
        
        # Ordenar clases por probabilidad
        sorted_indices = np.argsort(probs_np)[::-1]
        ranked_classes = [(CLASS_NAMES[i], probs_np[i]) for i in sorted_indices]
        
        # Encontrar frames más importantes (últimos 'actual_frames')
        frame_importance = analysis['frame_importance'][:actual_frames]
        top_frame_idx = np.argmax(frame_importance)
        
        # Guardar historial
        for i, cls in enumerate(CLASS_NAMES):
            self.prob_history[cls].append(probs_np[i])
        
        return {
            'prediction': CLASS_NAMES[predicted.item()],
            'confidence': confidence.item(),
            'all_probs': probs_np,
            'ranked_classes': ranked_classes,
            'frame_importance': frame_importance,
            'peak_frame': top_frame_idx,
            'region_importance': analysis['region_importance'],
            'actual_frames': actual_frames
        }
    
    def create_graph_image(self, current_class: str, width: int, height: int) -> np.ndarray:
        """Crea imagen de la gráfica temporal."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        num_points = len(self.prob_history[CLASS_NAMES[0]])
        
        if num_points == 0:
            ax.text(0.5, 0.5, 'Esperando...', ha='center', va='center', 
                   fontsize=14, color='white')
        else:
            frames = np.arange(num_points)
            
            for cls in CLASS_NAMES:
                data = list(self.prob_history[cls])
                if cls == current_class and current_class != "nada":
                    ax.plot(frames, data, linewidth=3, 
                           color=COLORS.get(cls, 'green'), label=cls.upper())
                elif cls == 'nada':
                    ax.plot(frames, data, linewidth=2, 
                           color='gray', linestyle='--', alpha=0.7)
                else:
                    ax.plot(frames, data, linewidth=1, 
                           alpha=0.3, color=COLORS.get(cls, 'blue'))
            
            # Umbral
            ax.axhline(y=TRIGGER_THRESHOLD, color='red', linestyle='-', 
                      linewidth=2, alpha=0.8, label=f'Umbral {TRIGGER_THRESHOLD:.0%}')
            
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, max(num_points, 30))
            ax.set_ylabel('Prob', color='white', fontsize=10)
            ax.set_xlabel('Frame', color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', fontsize=8, facecolor='black', 
                     labelcolor='white', framealpha=0.5)
            ax.grid(True, alpha=0.2)
            
            for spine in ax.spines.values():
                spine.set_color('gray')
        
        plt.tight_layout()
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = canvas.buffer_rgba()
        graph_img = np.asarray(buf)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return graph_img
    
    def reset(self):
        self.buffer.clear()
        self.fsm.reset()
        self.smoothers = None
        self.frame_count = 0
        for cls in CLASS_NAMES:
            self.prob_history[cls].clear()
    
    def draw_skeleton(self, frame, keypoints) -> np.ndarray:
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
        
        for ci, cj in HAND_CONNECTIONS:
            i, j = 91 + ci, 91 + cj
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (255, 200, 0), 1)
            i, j = 112 + ci, 112 + cj
            if i in points and j in points:
                cv2.line(frame, points[i], points[j], (200, 0, 255), 1)
        
        return frame


def draw_diagnostic_panel(frame, analysis: dict, w: int, h: int) -> np.ndarray:
    """
    Dibuja panel de diagnóstico mostrando por qué el modelo tomó su decisión.
    """
    if analysis is None:
        return frame
    
    # Panel semi-transparente a la izquierda
    panel_width = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 100), (panel_width, h - 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 130
    
    # Título
    cv2.putText(frame, "DIAGNOSTICO", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 30
    
    # Todas las probabilidades (rankeadas)
    cv2.putText(frame, "Probabilidades:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 20
    
    for cls_name, prob in analysis['ranked_classes']:
        # Barra de probabilidad
        bar_width = int(prob * 150)
        
        # Color según clase
        if cls_name == analysis['prediction']:
            color = (0, 255, 0)  # Verde para predicción
        elif prob > 0.2:
            color = (0, 200, 255)  # Amarillo para alternativas
        else:
            color = (100, 100, 100)  # Gris para bajas
        
        cv2.rectangle(frame, (10, y_offset - 12), (10 + bar_width, y_offset + 2), color, -1)
        cv2.putText(frame, f"{cls_name}: {prob:.0%}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 22
    
    y_offset += 10
    
    # Información de frames
    cv2.putText(frame, f"Frames: {analysis['actual_frames']}/{BUFFER_SIZE}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 20
    
    cv2.putText(frame, f"Pico: frame {analysis['peak_frame']}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 30
    
    # Importancia por región
    cv2.putText(frame, "Regiones:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 20
    
    region_labels = {
        'left_hand': 'Mano Izq',
        'right_hand': 'Mano Der',
        'body': 'Cuerpo',
        'face': 'Cara',
        'feet': 'Pies'
    }
    
    for region, label in region_labels.items():
        imp = analysis['region_importance'].get(region, 0)
        bar_width = int(min(imp, 3) / 3 * 100)  # Normalizar a max 3
        
        color = (0, 200, 0) if imp > 1.5 else (100, 100, 100)
        cv2.rectangle(frame, (10, y_offset - 10), (10 + bar_width, y_offset), color, -1)
        cv2.putText(frame, f"{label}: {imp:.1f}", (15, y_offset - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        y_offset += 18
    
    return frame


def run_demo():
    """Ejecuta el demo con gráfica temporal en tiempo real."""
    print("=" * 60)
    print("🤟 LSM-Core: Demo con Gráfica Temporal")
    print("=" * 60)
    
    try:
        inference = LSMInference()
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        return
    
    if USE_WEBCAM:
        source = WEBCAM_ID
        print(f"📷 Usando webcam {WEBCAM_ID}")
    else:
        source = CAMERA_URL
        print(f"📱 Conectando a: {CAMERA_URL}")
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ No se pudo conectar a la cámara")
        return
    
    print("✅ Cámara conectada")
    print("\n🎮 Controles: [Q]Salir [R]Reset [S]Screenshot [C]Clear [D]Diagnóstico")
    print("=" * 60)
    
    fps_counter = deque(maxlen=30)
    frame_count = 0
    diagnostic_mode = True  # Inicia con diagnóstico activo (toggle con D)
    
    # Dimensiones objetivo (video vertical)
    TARGET_HEIGHT = 720
    GRAPH_WIDTH = 400
    
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
        
        # Rotar si viene horizontal (ancho > alto)
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]
        
        # Escalar a altura objetivo
        scale = TARGET_HEIGHT / h
        new_w = int(w * scale)
        frame = cv2.resize(frame, (new_w, TARGET_HEIGHT))
        h, w = frame.shape[:2]
        
        # Extraer keypoints
        raw_keypoints, normalized_keypoints = inference.extract_keypoints(frame)
        
        # =============================================
        # VETO FÍSICO - Verificar posición de manos
        # =============================================
        pose_active = is_pose_active(raw_keypoints, h)
        
        # =============================================
        # NORMALIZACIÓN DE FPS - Duplicar frames si FPS < 30
        # Esto compensa que el modelo fue entrenado a ~30 FPS
        # =============================================
        fps_counter.append(1.0 / (time.time() - start_time + 1e-6))
        current_fps = np.mean(list(fps_counter)[-FPS_SMOOTHING:])
        
        # Calcular cuántas veces duplicar el frame
        duplicate_factor = max(1, round(TRAINING_FPS / max(current_fps, 1)))
        
        if pose_active:
            # Posición activa - ejecutar inferencia normal
            # Duplicar frame para compensar FPS bajo
            for _ in range(duplicate_factor):
                inference.buffer.append(normalized_keypoints)
            
            # Dibujar esqueleto (verde)
            frame = inference.draw_skeleton(frame, raw_keypoints)
            
            # Predecir (con o sin análisis)
            if diagnostic_mode:
                analysis = inference.predict_with_analysis()
                if analysis:
                    prediction = analysis['prediction']
                    confidence = analysis['confidence']
                    probs = analysis['all_probs']
                else:
                    prediction, confidence, probs = None, 0.0, None
                    analysis = None
            else:
                prediction, confidence, probs = inference.predict()
                analysis = None
            
            # Actualizar FSM
            inference.fsm.update(prediction, confidence)
            current_word = inference.fsm.get_current_word()
            veto_active = False
        else:
            # Posición de descanso - VETO ACTIVO
            # CRÍTICO: Limpiar buffer para matar memoria de seña anterior
            if len(inference.buffer) > 0:
                inference.buffer.clear()
                inference.fsm.state = State.IDLE
                inference.fsm.last_word = None
                # Limpiar historial de probabilidades
                for cls in CLASS_NAMES:
                    inference.prob_history[cls].clear()
            
            # Dibujar esqueleto en gris (indicar veto)
            frame = inference.draw_skeleton(frame, raw_keypoints)
            
            prediction = "nada"
            confidence = 0.0
            probs = None
            current_word = None
            veto_active = True
            analysis = None  # No hay análisis en veto
        
        # =============================================
        # UI sobre el video
        # =============================================
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Estado FSM + Veto
        if veto_active:
            cv2.putText(frame, "VETO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "(manos abajo)", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            state_color = (0, 255, 0) if inference.fsm.state == State.ACTIVE else (150, 150, 150)
            cv2.putText(frame, f"{inference.fsm.state.value}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        
        # Palabra actual
        if current_word:
            cv2.putText(frame, current_word.upper(), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif prediction and prediction != "nada" and not veto_active:
            cv2.putText(frame, f"({prediction})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
        
        # Confianza
        if prediction and not veto_active:
            cv2.putText(frame, f"{confidence:.0%}", (w - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Historial (parte inferior)
        history = inference.fsm.get_history()
        if history:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
            history_text = " → ".join(history[-3:])
            cv2.putText(frame, history_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS y Factor de duplicación
        fps = current_fps  # Ya calculado arriba
        cv2.putText(frame, f"FPS:{fps:.0f} x{duplicate_factor}", (w - 120, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # =============================================
        # Panel de diagnóstico (si está activo)
        # =============================================
        if diagnostic_mode and analysis:
            frame = draw_diagnostic_panel(frame, analysis, w, h)
        
        # =============================================
        # Crear gráfica temporal
        # =============================================
        target_class = current_word if current_word else (prediction if prediction else "nada")
        graph_img = inference.create_graph_image(target_class, GRAPH_WIDTH, TARGET_HEIGHT)
        
        # =============================================
        # Combinar video (izq) + gráfica (der)
        # =============================================
        combined = np.hstack([frame, graph_img])
        
        cv2.imshow('LSM-Core Demo + Temporal', combined)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            inference.reset()
            print("🔄 Reset completo")
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.png"
            cv2.imwrite(filename, combined)
            print(f"📸 Guardado: {filename}")
        elif key == ord('c'):
            inference.fsm.clear_history()
            print("🗑️ Historial limpio")
        elif key == ord('d'):
            diagnostic_mode = not diagnostic_mode
            status = "✅ ACTIVADO" if diagnostic_mode else "❌ DESACTIVADO"
            print(f"🔍 Modo diagnóstico: {status}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Demo terminado")


if __name__ == "__main__":
    run_demo()
