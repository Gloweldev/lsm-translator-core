"""
Debug RTMPose con configuración EXACTA del Preprocessor.

Visualiza el pipeline de preprocesamiento en 10 videos aleatorios:
    1. RTMPose extrae 133 keypoints
    2. Filtro de confianza (score < threshold → gris)
    3. Suavizado OneEuroFilter

Esto permite evaluar la calidad ANTES de ejecutar el preprocessor completo.

Controles:
    [Q] - Salir
    [N] - Siguiente video
    [ESPACIO] - Pausar
    [F] - Toggle suavizado
"""

import cv2
import sys
import random
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    RAW_DATA_DIR,
    KEYPOINTS_PER_FRAME,
    RTMPOSE_MODEL,
    CONFIDENCE_THRESHOLD,
    FILTER_MIN_CUTOFF,
    FILTER_BETA,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX
)
from src.utils.smoothing import OneEuroFilter

import torch
from mmpose.apis import MMPoseInferencer

# Colores
COLOR_HIGH_CONF = (0, 255, 0)      # Verde
COLOR_MED_CONF = (0, 255, 255)    # Amarillo
COLOR_FILTERED = (128, 128, 128)  # Gris
COLOR_CENTER = (255, 0, 0)        # Azul

# Conexiones
BODY_CONNECTIONS = [
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
    (5, 9), (9, 13), (13, 17),
]


class KeypointSmoother:
    """Suavizado OneEuroFilter para todos los keypoints."""
    
    def __init__(self, num_keypoints=KEYPOINTS_PER_FRAME, min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA):
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filters = None
        self.initialized = False
    
    def reset(self):
        self.filters = None
        self.initialized = False
    
    def _init(self, t0, keypoints):
        self.filters = []
        for i in range(self.num_keypoints):
            x = keypoints[i, 0] if i < len(keypoints) else 0.0
            y = keypoints[i, 1] if i < len(keypoints) else 0.0
            self.filters.append((
                OneEuroFilter(t0=t0, x0=x, min_cutoff=self.min_cutoff, beta=self.beta),
                OneEuroFilter(t0=t0, x0=y, min_cutoff=self.min_cutoff, beta=self.beta)
            ))
        self.initialized = True
    
    def smooth(self, t, keypoints):
        if keypoints is None:
            return np.zeros((self.num_keypoints, 2))
        if not self.initialized:
            self._init(t, keypoints)
            return keypoints[:, :2].copy()
        
        smoothed = np.zeros((self.num_keypoints, 2))
        for i in range(min(len(keypoints), self.num_keypoints)):
            fx, fy = self.filters[i]
            smoothed[i, 0] = fx(t, keypoints[i, 0])
            smoothed[i, 1] = fy(t, keypoints[i, 1])
        return smoothed


def is_hand_valid(hand_points: dict, hand_scores: dict, 
                   w: int, h: int,
                   all_keypoints: np.ndarray = None,
                   all_scores: np.ndarray = None,
                   wrist_idx: int = None,
                   max_spread: float = 150,
                   min_confidence: float = 0.8,
                   edge_margin: float = 30,
                   high_confidence: float = 0.9,
                   max_wrist_distance: float = 100) -> tuple:
    """
    Valida si una mano es confiable para usar.
    
    Checks:
    1. Coherencia espacial (puntos agrupados)
    2. Confianza mínima
    3. Proximidad a la muñeca (mano debe estar cerca de su muñeca)
    4. Manejo inteligente de bordes
    
    Args:
        hand_points: dict {idx: (x, y)} de puntos de la mano
        hand_scores: dict {idx: score} de confianzas
        w, h: dimensiones del frame
        all_keypoints: array completo de keypoints (para verificar muñeca)
        all_scores: array completo de scores
        wrist_idx: índice de la muñeca correspondiente (9=izq, 10=der)
        max_spread: distancia máxima entre puntos de la mano
        min_confidence: confianza promedio mínima
        edge_margin: margen de borde
        high_confidence: umbral para aceptar en borde
        max_wrist_distance: distancia máxima del centro de mano a la muñeca
        
    Returns:
        (is_valid, reason)
    """
    if len(hand_points) < 5:
        return False, "pocos"
    
    points = list(hand_points.values())
    scores = list(hand_scores.values())
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Check 1: Coherencia espacial
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    
    if spread_x > max_spread or spread_y > max_spread:
        return False, "dispersa"
    
    # Check 2: Confianza promedio
    avg_conf = sum(scores) / len(scores)
    if avg_conf < min_confidence:
        return False, f"conf({avg_conf:.2f})"
    
    # Check 3: Proximidad a la muñeca
    if all_keypoints is not None and all_scores is not None and wrist_idx is not None:
        wrist = all_keypoints[wrist_idx]
        wrist_conf = all_scores[wrist_idx]
        
        # Si la muñeca es visible, la mano debe estar cerca
        if wrist_conf >= 0.5 and wrist[0] > 0 and wrist[1] > 0:
            hand_center_x = sum(xs) / len(xs)
            hand_center_y = sum(ys) / len(ys)
            
            distance = ((hand_center_x - wrist[0])**2 + (hand_center_y - wrist[1])**2)**0.5
            
            if distance > max_wrist_distance:
                return False, f"lejos({distance:.0f}px)"
        else:
            # Muñeca no visible → mano probablemente oculta
            # Solo aceptar si tiene muy alta confianza
            if avg_conf < high_confidence:
                return False, "sin_muñeca"
    
    # Check 4: Manejo de bordes inteligente
    near_edge = 0
    for x, y in points:
        if x < edge_margin or x > (w - edge_margin) or y < edge_margin or y > (h - edge_margin):
            near_edge += 1
    
    edge_ratio = near_edge / len(points)
    
    if edge_ratio > 0.5:
        if avg_conf >= high_confidence:
            return True, "borde_ok"
        else:
            return False, f"borde({avg_conf:.2f})"
    
    return True, "ok"


# Mantener compatibilidad con nombre anterior
def is_hand_coherent(hand_points: dict, max_spread: float = 150) -> bool:
    """Wrapper de compatibilidad para is_hand_valid."""
    if len(hand_points) < 5:
        return False
    
    points = list(hand_points.values())
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    
    return spread_x < max_spread and spread_y < max_spread


def draw_keypoints(frame, keypoints, scores, threshold, h, w):
    """Dibuja keypoints. Ignora piernas/pies, oculta filtrados y valida coherencia de manos."""
    drawn = {}
    
    # Índices a ignorar (piernas y pies)
    SKIP_INDICES = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
    
    # Primero, validar manos con todos los checks
    left_hand_indices = set(range(91, 112))
    right_hand_indices = set(range(112, 133))
    
    # Recopilar puntos y scores de manos para validación
    left_hand_pts = {}
    left_hand_scores = {}
    right_hand_pts = {}
    right_hand_scores = {}
    
    for i in range(91, 112):
        if scores[i] >= threshold and keypoints[i][0] > 0 and keypoints[i][1] > 0:
            px, py = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= px < w and 0 <= py < h:
                left_hand_pts[i] = (px, py)
                left_hand_scores[i] = scores[i]
    
    for i in range(112, 133):
        if scores[i] >= threshold and keypoints[i][0] > 0 and keypoints[i][1] > 0:
            px, py = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= px < w and 0 <= py < h:
                right_hand_pts[i] = (px, py)
                right_hand_scores[i] = scores[i]
    
    # Validar manos con función mejorada (incluye check de muñeca)
    # Muñeca izquierda = índice 9, Muñeca derecha = índice 10
    left_valid, left_reason = is_hand_valid(
        left_hand_pts, left_hand_scores, w, h,
        all_keypoints=keypoints, all_scores=scores, wrist_idx=9
    )
    right_valid, right_reason = is_hand_valid(
        right_hand_pts, right_hand_scores, w, h,
        all_keypoints=keypoints, all_scores=scores, wrist_idx=10
    )
    
    # Indices a ignorar por manos inválidas
    invalid_indices = set()
    if not left_valid:
        invalid_indices.update(left_hand_indices)
    if not right_valid:
        invalid_indices.update(right_hand_indices)
    
    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        # Ignorar piernas, pies y manos incoherentes
        if i in SKIP_INDICES or i in invalid_indices:
            continue
            
        px, py = int(kp[0]), int(kp[1])
        
        if not (0 <= px < w and 0 <= py < h):
            continue
        
        if score < threshold:
            continue
        
        color = COLOR_HIGH_CONF if score > 0.7 else COLOR_MED_CONF
        cv2.circle(frame, (px, py), 4, color, -1)
        drawn[i] = (px, py)
    
    # Conexiones del cuerpo (sin piernas)
    UPPER_BODY_CONNECTIONS = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
    ]
    for i, j in UPPER_BODY_CONNECTIONS:
        if i in drawn and j in drawn:
            cv2.line(frame, drawn[i], drawn[j], (100, 255, 100), 1)
    
    # Conexiones de manos (solo si son coherentes)
    if left_valid:
        for ci, cj in HAND_CONNECTIONS:
            i, j = 91 + ci, 91 + cj
            if i in drawn and j in drawn:
                cv2.line(frame, drawn[i], drawn[j], (255, 200, 0), 1)
    
    if right_valid:
        for ci, cj in HAND_CONNECTIONS:
            i, j = 112 + ci, 112 + cj
            if i in drawn and j in drawn:
                cv2.line(frame, drawn[i], drawn[j], (200, 0, 255), 1)
    
    return frame, (left_valid, left_reason), (right_valid, right_reason)


def nothing(x):
    pass


def run_debug():
    print("=" * 60)
    print("🔍 Debug RTMPose - Configuración del Preprocessor")
    print("=" * 60)
    print(f"📂 Dataset: {RAW_DATA_DIR}")
    print(f"🎯 Model: {RTMPOSE_MODEL}")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"🔧 OneEuro: min_cutoff={FILTER_MIN_CUTOFF}, beta={FILTER_BETA}")
    print("=" * 60)
    
    # Buscar videos
    videos = []
    for class_dir in RAW_DATA_DIR.iterdir():
        if class_dir.is_dir():
            for v in class_dir.glob("*.mp4"):
                videos.append((v, class_dir.name))
    
    if not videos:
        print("❌ No hay videos")
        return
    
    selected = random.sample(videos, min(10, len(videos)))
    print(f"🎲 Seleccionados {len(selected)} videos\n")
    
    # Cargar RTMPose
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Cargando RTMPose en {device}...")
    
    try:
        inferencer = MMPoseInferencer(pose2d=RTMPOSE_MODEL, device=device)
        print("✅ Cargado\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Ventana
    win = 'Preprocessor Debug'
    cv2.namedWindow(win)
    cv2.createTrackbar('Confidence', win, int(CONFIDENCE_THRESHOLD * 100), 100, nothing)
    
    print("🎮 Controles: [Q]Salir [N]Siguiente [ESPACIO]Pausa [F]Suavizado")
    print("🎨 Verde=Alta conf (>0.7), Amarillo=Media, Gris=Filtrado, X Azul=Centro")
    print("=" * 60)
    
    smooth_on = True
    
    for idx, (video_path, class_name) in enumerate(selected):
        print(f"\n[{idx+1}/{len(selected)}] {class_name}/{video_path.name}")
        
        smoother = KeypointSmoother()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        start = time.time()
        paused = False
        cache = None
        
        while cap.isOpened():
            threshold = cv2.getTrackbarPos('Confidence', win) / 100.0
            
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                t = time.time() - start
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = list(inferencer(rgb, return_vis=False))
                
                keypoints = np.zeros((KEYPOINTS_PER_FRAME, 2))
                scores = np.zeros(KEYPOINTS_PER_FRAME)
                
                if results and results[0].get('predictions'):
                    preds = results[0]['predictions']
                    if preds and preds[0]:
                        kp = np.array(preds[0][0].get('keypoints', []))
                        sc = np.array(preds[0][0].get('keypoint_scores', []))
                        n = min(len(kp), KEYPOINTS_PER_FRAME)
                        keypoints[:n] = kp[:n, :2]
                        scores[:n] = sc[:n]
                
                # Suavizado
                if smooth_on:
                    keypoints = smoother.smooth(t, keypoints)
                
                cache = (frame.copy(), keypoints, scores, h, w)
            
            if cache is None:
                continue
            
            frame, keypoints, scores, h, w = cache
            display = frame.copy()
            
            # Dibujar keypoints (ahora retorna estado de manos con razones)
            display, left_result, right_result = draw_keypoints(display, keypoints, scores, threshold, h, w)
            left_ok, left_reason = left_result
            right_ok, right_reason = right_result
            
            # Dibujar centro de caderas (cruz azul)
            left_hip = keypoints[LEFT_HIP_IDX]
            right_hip = keypoints[RIGHT_HIP_IDX]
            if scores[LEFT_HIP_IDX] >= threshold or scores[RIGHT_HIP_IDX] >= threshold:
                cx = int((left_hip[0] + right_hip[0]) / 2)
                cy = int((left_hip[1] + right_hip[1]) / 2)
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.drawMarker(display, (cx, cy), COLOR_CENTER, cv2.MARKER_CROSS, 20, 2)
            
            # UI
            cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Clase: {class_name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Archivo: {video_path.name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"Threshold: {threshold:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Estado
            cv2.putText(display, f"Smooth: {'ON' if smooth_on else 'OFF'}", (w-120, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if smooth_on else (0,0,255), 1)
            
            visible = np.sum(scores >= threshold)
            cv2.putText(display, f"Visible: {visible}/{KEYPOINTS_PER_FRAME}", (w-120, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            
            # Estado de manos con razones
            left_status = "ok" if left_ok else left_reason
            right_status = "ok" if right_ok else right_reason
            left_color = (255, 200, 0) if left_ok else (0, 0, 255)
            right_color = (200, 0, 255) if right_ok else (0, 0, 255)
            cv2.putText(display, f"L:{left_status}", (w-120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, left_color, 1)
            cv2.putText(display, f"R:{right_status}", (w-120, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, right_color, 1)
            
            if paused:
                cv2.putText(display, "PAUSED", (w//2-50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            cv2.imshow(win, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('f'):
                smooth_on = not smooth_on
                smoother.reset()
                print(f"   Smooth: {'ON' if smooth_on else 'OFF'}")
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("\n✅ Debug completado")


if __name__ == "__main__":
    run_debug()
