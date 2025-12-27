"""
Demo de Inferencia en Tiempo Real con iPad/DroidCam.

Pipeline completo: RTMPose → Transformer → FSM → Predicción + Gráfica Temporal
Fuente de video: Cámara IP (DroidCam, EpocCam, etc.) o video local

Uso:
    # Cámara en vivo
    python -m src.inference.ipad_demo
    
    # Video grabado (para evaluación)
    python -m src.inference.ipad_demo -v "ruta/video.mp4"
    python -m src.inference.ipad_demo -v "ruta/video.mp4" --debug

Controles:
    [Q] - Salir
    [R] - Reset buffer
    [S] - Screenshot
    [C] - Clear historial
    [D] - Toggle diagnóstico
"""

import cv2
import sys
import time
import argparse
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
# ACTIVO si muñeca está ARRIBA de (hip - margen)
# Margen negativo = debe estar claramente ARRIBA de la cadera
POSE_MARGIN = -0.03  # Negativo = manos deben estar 3% ARRIBA de cadera para ser activo

# Historial
MAX_HISTORY = 5
MAX_GRAPH_POINTS = 150  # Puntos max en la gráfica

# =============================================
# BUFFER INTELIGENTE POR GESTOS
# =============================================
class GestureState(Enum):
    """Estados del buffer inteligente."""
    IDLE = "idle"           # Esperando que manos suban
    CAPTURING = "capturing" # Capturando gesto (manos activas)
    COOLDOWN = "cooldown"   # Mostrando resultado, esperando

RESULT_DISPLAY_TIME = 1.5  # Segundos que se muestra el resultado
MIN_CAPTURE_FRAMES = 15    # Mínimo de frames para considerar un gesto válido

# Normalización adaptativa de longitud
TARGET_SEQ_LEN = 60        # Longitud objetivo para normalización
MIN_FRAMES_FOR_STRETCH = 30  # Si < 30 frames, estirar a TARGET
MAX_FRAMES_BEFORE_SUBSAMPLE = 120  # Si > 120 frames, submuestrear


def normalize_sequence_length(sequence: np.ndarray, target_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    """
    Normaliza la longitud de la secuencia para manejar señas rápidas y lentas.
    
    - Señas cortas (< 30 frames): Interpola para estirar a target_len
    - Señas normales (30-120 frames): Sin cambio
    - Señas largas (> 120 frames): Submuestrea uniformemente a target_len
    
    Args:
        sequence: Array de shape (N, 266) con los keypoints
        target_len: Longitud objetivo para normalización
        
    Returns:
        Secuencia normalizada
    """
    current_len = len(sequence)
    
    if current_len < MIN_FRAMES_FOR_STRETCH:
        # Señas muy rápidas - ESTIRAR (interpolar)
        # Crear más frames duplicando/interpolando
        indices = np.linspace(0, current_len - 1, target_len)
        indices = np.clip(np.round(indices).astype(int), 0, current_len - 1)
        return sequence[indices]
    
    elif current_len > MAX_FRAMES_BEFORE_SUBSAMPLE:
        # Señas muy largas - COMPRIMIR (submuestrear uniformemente)
        indices = np.linspace(0, current_len - 1, target_len)
        indices = np.clip(np.round(indices).astype(int), 0, current_len - 1)
        return sequence[indices]
    
    else:
        # Longitud normal - sin cambio
        return sequence

# Colores para gráfica
COLORS = {
    'a': '#2ecc71',
    'b': '#3498db',
    'c': '#9b59b6',
    'hola': '#e74c3c',
    'nada': '#95a5a6',
}

# Índices a ignorar (piernas y pies)
SKIP_INDICES = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}


def is_hand_valid(keypoints: np.ndarray, scores: np.ndarray,
                  hand_start: int, hand_end: int, wrist_idx: int,
                  max_spread: float = 150,
                  min_confidence: float = 0.8,
                  high_confidence: float = 0.9,
                  max_wrist_distance: float = 100) -> bool:
    """
    Valida si una mano es confiable (no alucinada).
    
    Checks:
    1. Mínimo 5 puntos válidos
    2. Coherencia espacial
    3. Confianza promedio
    4. Proximidad a la muñeca
    """
    hand_pts = keypoints[hand_start:hand_end]
    hand_scores = scores[hand_start:hand_end]
    
    valid_mask = np.any(hand_pts != 0, axis=1)
    valid_pts = hand_pts[valid_mask]
    valid_scores = hand_scores[valid_mask]
    
    if len(valid_pts) < 5:
        return False
    
    # Coherencia espacial
    spread_x = valid_pts[:, 0].max() - valid_pts[:, 0].min()
    spread_y = valid_pts[:, 1].max() - valid_pts[:, 1].min()
    
    if spread_x > max_spread or spread_y > max_spread:
        return False
    
    # Confianza promedio
    avg_conf = valid_scores.mean()
    if avg_conf < min_confidence:
        return False
    
    # Proximidad a muñeca
    wrist = keypoints[wrist_idx]
    wrist_conf = scores[wrist_idx]
    
    if wrist_conf >= 0.5 and wrist[0] > 0 and wrist[1] > 0:
        hand_center = valid_pts.mean(axis=0)
        distance = np.sqrt(((hand_center - wrist) ** 2).sum())
        if distance > max_wrist_distance:
            return False
    else:
        if avg_conf < high_confidence:
            return False
    
    return True


def filter_incoherent_hands(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Filtra manos incoherentes poniéndolas a cero."""
    filtered = keypoints.copy()
    
    # Mano izquierda: 91-111, muñeca: 9
    if not is_hand_valid(keypoints.reshape(-1, 2), scores, 91, 112, wrist_idx=9):
        filtered[91*2:112*2] = 0.0
    
    # Mano derecha: 112-132, muñeca: 10
    if not is_hand_valid(keypoints.reshape(-1, 2), scores, 112, 133, wrist_idx=10):
        filtered[112*2:133*2] = 0.0
    
    return filtered


# Visualización esqueleto (sin piernas)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
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
def is_pose_active(raw_keypoints: np.ndarray, frame_height: int, debug: bool = False) -> bool:
    """
    Determina si el usuario está en posición activa para señas.
    
    Usa posición RELATIVA al cuerpo en lugar de posición absoluta.
    Compara muñecas con el centro del torso (entre hombros y caderas).
    
    Args:
        raw_keypoints: Vector de 266 dims (133 keypoints x 2)
        frame_height: Altura del frame (para fallback)
        debug: Si True, imprime valores de detección
    
    Returns:
        True si está activo (al menos una mano arriba), False si en descanso
    """
    # Índices para hombros
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    
    # Extraer coordenadas Y
    left_wrist_y = raw_keypoints[LEFT_WRIST_IDX * 2 + 1]
    right_wrist_y = raw_keypoints[RIGHT_WRIST_IDX * 2 + 1]
    left_hip_y = raw_keypoints[LEFT_HIP_IDX * 2 + 1]
    right_hip_y = raw_keypoints[RIGHT_HIP_IDX * 2 + 1]
    left_shoulder_y = raw_keypoints[LEFT_SHOULDER_IDX * 2 + 1]
    right_shoulder_y = raw_keypoints[RIGHT_SHOULDER_IDX * 2 + 1]
    
    # Calcular centro del torso (entre hombros y caderas)
    shoulders = []
    hips = []
    
    if left_shoulder_y > 0:
        shoulders.append(left_shoulder_y)
    if right_shoulder_y > 0:
        shoulders.append(right_shoulder_y)
    if left_hip_y > 0:
        hips.append(left_hip_y)
    if right_hip_y > 0:
        hips.append(right_hip_y)
    
    # Necesitamos al menos un hombro y una cadera
    if not shoulders or not hips:
        if debug:
            print(f"[DEBUG] Sin detección de torso → INACTIVO")
        return False
    
    shoulder_y = sum(shoulders) / len(shoulders)
    hip_y = sum(hips) / len(hips)
    
    # Centro del torso (punto medio entre hombros y caderas)
    torso_center_y = (shoulder_y + hip_y) / 2
    
    # Margen: 20% de la altura del torso
    torso_height = hip_y - shoulder_y
    margin = torso_height * 0.2
    
    # Umbral: un poco abajo del centro del torso
    threshold_y = torso_center_y + margin
    
    # Verificar muñecas
    wrists_detected = []
    wrist_positions = []
    
    if left_wrist_y > 0:
        wrists_detected.append('L')
        wrist_positions.append(left_wrist_y)
    if right_wrist_y > 0:
        wrists_detected.append('R')
        wrist_positions.append(right_wrist_y)
    
    # Sin muñecas detectadas = manos ocultas = inactivo
    if not wrist_positions:
        if debug:
            print(f"[DEBUG] Sin muñecas detectadas → INACTIVO")
        return False
    
    # Calcular cuántas muñecas están ARRIBA del umbral
    above_count = sum(1 for y in wrist_positions if y < threshold_y)
    
    if debug:
        wrists_str = '/'.join([f"{w}:{y:.0f}" for w, y in zip(wrists_detected, wrist_positions)])
        print(f"[DEBUG] muñecas:{wrists_str} | torso_c:{torso_center_y:.0f} thresh:{threshold_y:.0f} | arriba:{above_count}")
    
    # ACTIVO si al menos una muñeca está ARRIBA del umbral
    return above_count > 0


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
        scores = np.zeros(KEYPOINTS_PER_FRAME)
        
        if results and results[0].get('predictions'):
            preds = results[0]['predictions']
            if preds and preds[0]:
                kp = np.array(preds[0][0].get('keypoints', []))
                sc = np.array(preds[0][0].get('keypoint_scores', []))
                
                for i in range(min(len(kp), KEYPOINTS_PER_FRAME)):
                    scores[i] = sc[i]
                    # Ignorar piernas y pies
                    if i in SKIP_INDICES:
                        continue
                    if sc[i] >= CONFIDENCE_THRESHOLD:
                        raw_keypoints[i*2] = kp[i, 0]
                        raw_keypoints[i*2 + 1] = kp[i, 1]
        
        # Filtrar manos incoherentes (alucinaciones)
        raw_keypoints = filter_incoherent_hands(raw_keypoints, scores)
        
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
        actual_frames = len(sequence)  # Duración real antes del padding
        
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([sequence, padding])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        duration = torch.LongTensor([actual_frames]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor, duration=duration)
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
        duration = torch.LongTensor([actual_frames]).to(self.device)
        
        with torch.no_grad():
            analysis = self.model.forward_with_analysis(tensor, duration=duration)
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


def run_demo(video_path: str = None, debug: bool = False):
    """
    Ejecuta el demo con gráfica temporal en tiempo real.
    
    Args:
        video_path: Ruta opcional a video grabado (para evaluación offline)
        debug: Si True, imprime valores de detección de pose
    """
    print("=" * 60)
    print("🤟 LSM-Core: Demo con Gráfica Temporal")
    print("=" * 60)
    
    try:
        inference = LSMInference()
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        return
    
    # Determinar fuente de video
    if video_path:
        source = video_path
        print(f"🎬 Usando video: {video_path}")
    elif USE_WEBCAM:
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
    
    # =============================================
    # BUFFER INTELIGENTE POR GESTOS
    # =============================================
    gesture_state = GestureState.IDLE
    gesture_buffer = []  # Buffer temporal para un gesto completo
    last_prediction = None  # Último resultado
    last_confidence = 0.0
    last_analysis = None
    cooldown_start = 0  # Tiempo cuando empezó el cooldown
    
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
        
        # Verificar si manos están activas
        pose_active = is_pose_active(raw_keypoints, h, debug=debug)
        
        # FPS
        fps_counter.append(1.0 / (time.time() - start_time + 1e-6))
        current_fps = np.mean(list(fps_counter)[-FPS_SMOOTHING:])
        duplicate_factor = max(1, round(TRAINING_FPS / max(current_fps, 1)))
        
        # =============================================
        # MÁQUINA DE ESTADOS DEL BUFFER INTELIGENTE
        # =============================================
        
        if gesture_state == GestureState.IDLE:
            # Esperando que manos suban
            if pose_active:
                # Transición: IDLE → CAPTURING
                gesture_state = GestureState.CAPTURING
                gesture_buffer = []
                # Agregar primer frame
                for _ in range(duplicate_factor):
                    gesture_buffer.append(normalized_keypoints)
                # print("📹 Iniciando captura de gesto")
            
            # Dibujar esqueleto gris (inactivo)
            frame = inference.draw_skeleton(frame, raw_keypoints)
            
        elif gesture_state == GestureState.CAPTURING:
            # Capturando gesto
            if pose_active:
                # Seguir capturando
                for _ in range(duplicate_factor):
                    gesture_buffer.append(normalized_keypoints)
                # Dibujar esqueleto verde (capturando)
                frame = inference.draw_skeleton(frame, raw_keypoints)
            else:
                # Transición: CAPTURING → COOLDOWN (manos bajaron)
                # Hacer predicción con el gesto completo
                if len(gesture_buffer) >= MIN_CAPTURE_FRAMES:
                    # Preparar secuencia con normalización adaptativa
                    raw_sequence = np.array(gesture_buffer)
                    
                    # Normalizar longitud (estirar si muy corto, comprimir si muy largo)
                    sequence = normalize_sequence_length(raw_sequence)
                    actual_frames = len(sequence)
                    
                    # Truncar o hacer padding a BUFFER_SIZE
                    if len(sequence) > BUFFER_SIZE:
                        sequence = sequence[-BUFFER_SIZE:]  # Tomar últimos
                        actual_frames = BUFFER_SIZE
                    elif len(sequence) < BUFFER_SIZE:
                        padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
                        sequence = np.vstack([sequence, padding])
                    
                    tensor = torch.FloatTensor(sequence).unsqueeze(0).to(inference.device)
                    duration = torch.LongTensor([actual_frames]).to(inference.device)
                    
                    with torch.no_grad():
                        if diagnostic_mode:
                            analysis = inference.model.forward_with_analysis(tensor, duration=duration)
                            logits = analysis['logits']
                            probs = F.softmax(logits, dim=1)
                            confidence, predicted = probs.max(1)
                            
                            probs_np = probs[0].cpu().numpy()
                            sorted_indices = np.argsort(probs_np)[::-1]
                            ranked_classes = [(CLASS_NAMES[i], probs_np[i]) for i in sorted_indices]
                            frame_importance = analysis['frame_importance'][:actual_frames]
                            
                            last_analysis = {
                                'prediction': CLASS_NAMES[predicted.item()],
                                'confidence': confidence.item(),
                                'ranked_classes': ranked_classes,
                                'actual_frames': actual_frames,
                                'peak_frame': np.argmax(frame_importance),
                                'region_importance': analysis['region_importance']
                            }
                        else:
                            logits = inference.model(tensor, duration=duration)
                            probs = F.softmax(logits, dim=1)
                            confidence, predicted = probs.max(1)
                            last_analysis = None
                        
                        last_prediction = CLASS_NAMES[predicted.item()]
                        last_confidence = confidence.item()
                    
                    # Actualizar FSM para historial
                    inference.fsm.update(last_prediction, last_confidence)
                    
                    # print(f"✅ Gesto detectado: {last_prediction} ({last_confidence:.0%})")
                else:
                    # Gesto muy corto, ignorar
                    last_prediction = None
                    last_confidence = 0.0
                    last_analysis = None
                
                gesture_state = GestureState.COOLDOWN
                cooldown_start = time.time()
                gesture_buffer = []
                
                # Dibujar esqueleto gris
                frame = inference.draw_skeleton(frame, raw_keypoints)
        
        elif gesture_state == GestureState.COOLDOWN:
            # Mostrando resultado
            time_in_cooldown = time.time() - cooldown_start
            
            if pose_active:
                # Usuario quiere hacer otro gesto, salir de cooldown
                gesture_state = GestureState.CAPTURING
                gesture_buffer = []
                for _ in range(duplicate_factor):
                    gesture_buffer.append(normalized_keypoints)
                frame = inference.draw_skeleton(frame, raw_keypoints)
            elif time_in_cooldown > RESULT_DISPLAY_TIME:
                # Timeout, volver a IDLE
                gesture_state = GestureState.IDLE
                frame = inference.draw_skeleton(frame, raw_keypoints)
            else:
                # Seguir mostrando resultado
                frame = inference.draw_skeleton(frame, raw_keypoints)
        
        # Variables para UI
        current_word = inference.fsm.get_current_word()
        
        # =============================================
        # UI sobre el video
        # =============================================
        
        # Header
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Estado del buffer inteligente
        if gesture_state == GestureState.IDLE:
            cv2.putText(frame, "ESPERANDO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            cv2.putText(frame, "(levanta manos)", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        elif gesture_state == GestureState.CAPTURING:
            cv2.putText(frame, "CAPTURANDO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"frames: {len(gesture_buffer)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        elif gesture_state == GestureState.COOLDOWN:
            cv2.putText(frame, "RESULTADO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Mostrar última predicción
        if last_prediction and last_prediction != "nada":
            cv2.putText(frame, last_prediction.upper(), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"{last_confidence:.0%}", (w - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif current_word:
            cv2.putText(frame, current_word.upper(), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Historial (parte inferior)
        history = inference.fsm.get_history()
        if history:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
            history_text = " → ".join(history[-3:])
            cv2.putText(frame, history_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS y Factor de duplicación
        fps = current_fps
        cv2.putText(frame, f"FPS:{fps:.0f} x{duplicate_factor}", (w - 120, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # =============================================
        # Panel de diagnóstico (si está activo)
        # =============================================
        if diagnostic_mode and last_analysis:
            frame = draw_diagnostic_panel(frame, last_analysis, w, h)
        
        # =============================================
        # Crear gráfica temporal
        # =============================================
        target_class = current_word if current_word else (last_prediction if last_prediction else "nada")
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
    parser = argparse.ArgumentParser(description="LSM-Core: Demo de Inferencia")
    parser.add_argument("-v", "--video", type=str, help="Ruta a video para evaluación offline")
    parser.add_argument("--debug", action="store_true", help="Mostrar valores de detección de pose")
    args = parser.parse_args()
    
    run_demo(video_path=args.video, debug=args.debug)
