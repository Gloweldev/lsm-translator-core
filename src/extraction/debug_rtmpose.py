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


def draw_keypoints(frame, keypoints, scores, threshold, h, w):
    """Dibuja keypoints. RTMPose devuelve coordenadas en píxeles."""
    drawn = {}
    
    for i, (kp, score) in enumerate(zip(keypoints, scores)):
        px, py = int(kp[0]), int(kp[1])
        
        if not (0 <= px < w and 0 <= py < h):
            continue
        
        # Punto filtrado por threshold
        if score < threshold:
            cv2.circle(frame, (px, py), 2, COLOR_FILTERED, -1)
            continue
        
        # Color según confianza
        color = COLOR_HIGH_CONF if score > 0.7 else COLOR_MED_CONF
        cv2.circle(frame, (px, py), 4, color, -1)
        drawn[i] = (px, py)
    
    # Conexiones del cuerpo
    for i, j in BODY_CONNECTIONS:
        if i in drawn and j in drawn:
            cv2.line(frame, drawn[i], drawn[j], (100, 255, 100), 1)
    
    # Conexiones de manos
    for ci, cj in HAND_CONNECTIONS:
        # Mano izquierda (91-111)
        i, j = 91 + ci, 91 + cj
        if i in drawn and j in drawn:
            cv2.line(frame, drawn[i], drawn[j], (255, 200, 0), 1)
        # Mano derecha (112-132)
        i, j = 112 + ci, 112 + cj
        if i in drawn and j in drawn:
            cv2.line(frame, drawn[i], drawn[j], (200, 0, 255), 1)
    
    return frame


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
            
            # Dibujar keypoints
            display = draw_keypoints(display, keypoints, scores, threshold, h, w)
            
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
