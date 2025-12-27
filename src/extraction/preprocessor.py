"""
Preprocesador de Videos LSM.

Convierte videos .mp4 a tensores .npy usando RTMPose-WholeBody.
Aplica:
    1. Extracción de 133 keypoints con RTMPose
    2. Filtro de confianza (scores < threshold -> (0,0))
    3. Suavizado OneEuroFilter
    4. Normalización relativa (centrado en caderas)

Uso:
    python -m src.extraction.preprocessor
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RTMPOSE_MODEL,
    KEYPOINTS_PER_FRAME,
    INPUT_DIM,
    CONFIDENCE_THRESHOLD,
    FILTER_MIN_CUTOFF,
    FILTER_BETA,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    ensure_dirs,
    load_classes
)
from src.utils.smoothing import OneEuroFilter

# RTMPose
import torch
from mmpose.apis import MMPoseInferencer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class KeypointSmoother:
    """
    Aplica OneEuroFilter a todos los keypoints.
    Gestiona 133 puntos × 2 coordenadas = 266 filtros.
    """
    
    def __init__(
        self,
        num_keypoints: int = KEYPOINTS_PER_FRAME,
        min_cutoff: float = FILTER_MIN_CUTOFF,
        beta: float = FILTER_BETA
    ):
        self.num_keypoints = num_keypoints
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filters = None
        self.initialized = False
    
    def reset(self):
        """Reinicia los filtros (llamar al iniciar nuevo video)."""
        self.filters = None
        self.initialized = False
    
    def _init_filters(self, t0: float, keypoints: np.ndarray):
        """Inicializa filtros con el primer frame."""
        self.filters = []
        for i in range(self.num_keypoints):
            x = keypoints[i, 0] if i < len(keypoints) else 0.0
            y = keypoints[i, 1] if i < len(keypoints) else 0.0
            
            filter_x = OneEuroFilter(
                t0=t0, x0=x,
                min_cutoff=self.min_cutoff,
                beta=self.beta
            )
            filter_y = OneEuroFilter(
                t0=t0, x0=y,
                min_cutoff=self.min_cutoff,
                beta=self.beta
            )
            self.filters.append((filter_x, filter_y))
        self.initialized = True
    
    def smooth(self, t: float, keypoints: np.ndarray) -> np.ndarray:
        """
        Suaviza los keypoints.
        
        Args:
            t: Timestamp en segundos
            keypoints: Array Nx2 con coordenadas
            
        Returns:
            Array Nx2 suavizado
        """
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((self.num_keypoints, 2))
        
        if not self.initialized:
            self._init_filters(t, keypoints)
            return keypoints[:, :2].copy()
        
        smoothed = np.zeros((self.num_keypoints, 2))
        for i in range(min(len(keypoints), self.num_keypoints)):
            filter_x, filter_y = self.filters[i]
            smoothed[i, 0] = filter_x(t, keypoints[i, 0])
            smoothed[i, 1] = filter_y(t, keypoints[i, 1])
        
        return smoothed


def apply_confidence_filter(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Aplica filtro de confianza: puntos con score < threshold -> (0, 0).
    
    Args:
        keypoints: Nx2 array de coordenadas
        scores: N array de scores
        threshold: Umbral mínimo de confianza
        
    Returns:
        Nx2 array con puntos filtrados
    """
    filtered = keypoints.copy()
    
    for i, score in enumerate(scores):
        if score < threshold:
            filtered[i] = [0.0, 0.0]
    
    return filtered


def is_hand_valid(keypoints: np.ndarray, scores: np.ndarray,
                  hand_start: int, hand_end: int, wrist_idx: int,
                  max_spread: float = 150,
                  min_confidence: float = 0.8,
                  high_confidence: float = 0.9,
                  max_wrist_distance: float = 100) -> bool:
    """
    Valida si una mano es confiable para usar.
    
    Checks:
    1. Mínimo 5 puntos válidos
    2. Coherencia espacial (puntos agrupados)
    3. Confianza promedio mínima
    4. Proximidad a la muñeca
    
    Args:
        keypoints: Nx2 array de coordenadas
        scores: N array de scores de confianza
        hand_start: Índice inicial de la mano
        hand_end: Índice final de la mano (exclusive)
        wrist_idx: Índice de la muñeca correspondiente (9=izq, 10=der)
        max_spread: Distancia máxima permitida entre puntos
        min_confidence: Confianza promedio mínima
        high_confidence: Umbral para aceptar sin muñeca visible
        max_wrist_distance: Distancia máxima del centro de mano a la muñeca
        
    Returns:
        True si la mano es válida
    """
    hand_pts = keypoints[hand_start:hand_end]
    hand_scores = scores[hand_start:hand_end]
    
    # Puntos válidos (no cero)
    valid_mask = np.any(hand_pts != 0, axis=1)
    valid_pts = hand_pts[valid_mask]
    valid_scores = hand_scores[valid_mask]
    
    # Check 1: Mínimo de puntos
    if len(valid_pts) < 5:
        return False
    
    # Check 2: Coherencia espacial
    spread_x = valid_pts[:, 0].max() - valid_pts[:, 0].min()
    spread_y = valid_pts[:, 1].max() - valid_pts[:, 1].min()
    
    if spread_x > max_spread or spread_y > max_spread:
        return False
    
    # Check 3: Confianza promedio
    avg_conf = valid_scores.mean()
    if avg_conf < min_confidence:
        return False
    
    # Check 4: Proximidad a la muñeca
    wrist = keypoints[wrist_idx]
    wrist_conf = scores[wrist_idx]
    
    if wrist_conf >= 0.5 and wrist[0] > 0 and wrist[1] > 0:
        # Muñeca visible: mano debe estar cerca
        hand_center = valid_pts.mean(axis=0)
        distance = np.sqrt(((hand_center - wrist) ** 2).sum())
        
        if distance > max_wrist_distance:
            return False
    else:
        # Muñeca no visible: solo aceptar si muy alta confianza
        if avg_conf < high_confidence:
            return False
    
    return True


def filter_incoherent_hands(keypoints: np.ndarray, scores: np.ndarray = None, 
                            max_spread: float = 150) -> np.ndarray:
    """
    Filtra manos incoherentes (alucinadas) poniéndolas a cero.
    
    Args:
        keypoints: Nx2 array de coordenadas
        scores: N array de scores (opcional, si no se provee usa check simple)
        max_spread: Distancia máxima permitida para una mano coherente
        
    Returns:
        Nx2 array con manos incoherentes en (0, 0)
    """
    filtered = keypoints.copy()
    
    if scores is not None:
        # Validación completa con scores
        # Mano izquierda: 91-111, muñeca izquierda: 9
        if not is_hand_valid(keypoints, scores, 91, 112, wrist_idx=9, max_spread=max_spread):
            filtered[91:112] = [0.0, 0.0]
        
        # Mano derecha: 112-132, muñeca derecha: 10
        if not is_hand_valid(keypoints, scores, 112, 133, wrist_idx=10, max_spread=max_spread):
            filtered[112:133] = [0.0, 0.0]
    else:
        # Validación simple sin scores (compatibilidad)
        hand_pts = keypoints[91:112]
        valid_pts = hand_pts[np.any(hand_pts != 0, axis=1)]
        if len(valid_pts) < 5 or (valid_pts[:, 0].max() - valid_pts[:, 0].min()) > max_spread:
            filtered[91:112] = [0.0, 0.0]
        
        hand_pts = keypoints[112:133]
        valid_pts = hand_pts[np.any(hand_pts != 0, axis=1)]
        if len(valid_pts) < 5 or (valid_pts[:, 0].max() - valid_pts[:, 0].min()) > max_spread:
            filtered[112:133] = [0.0, 0.0]
    
    return filtered


def normalize_to_center(keypoints: np.ndarray, left_hip_idx: int = LEFT_HIP_IDX, right_hip_idx: int = RIGHT_HIP_IDX) -> np.ndarray:
    """
    Normaliza keypoints restando el centro de las caderas.
    Esto hace que las señas sean invariantes a la posición del usuario.
    
    Args:
        keypoints: Nx2 array de coordenadas
        left_hip_idx: Índice de cadera izquierda
        right_hip_idx: Índice de cadera derecha
        
    Returns:
        Nx2 array normalizado
    """
    # Calcular centro de caderas
    left_hip = keypoints[left_hip_idx] if left_hip_idx < len(keypoints) else np.array([0, 0])
    right_hip = keypoints[right_hip_idx] if right_hip_idx < len(keypoints) else np.array([0, 0])
    
    # Si ambas caderas son válidas (no cero)
    if np.any(left_hip != 0) and np.any(right_hip != 0):
        center = (left_hip + right_hip) / 2
    elif np.any(left_hip != 0):
        center = left_hip
    elif np.any(right_hip != 0):
        center = right_hip
    else:
        # Si no hay caderas, usar el centroide de puntos válidos
        valid_mask = np.any(keypoints != 0, axis=1)
        if np.any(valid_mask):
            center = keypoints[valid_mask].mean(axis=0)
        else:
            center = np.array([0.5, 0.5])  # Default al centro
    
    # Restar centro
    normalized = keypoints - center
    
    return normalized


class VideoPreprocessor:
    """
    Procesa videos y genera tensores .npy para el Transformer.
    """
    
    def __init__(self, device: str = None):
        """
        Inicializa el preprocesador.
        
        Args:
            device: 'cuda' o 'cpu' (None = auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.inferencer = None
        self.smoother = KeypointSmoother()
    
    def _init_inferencer(self):
        """Inicializa RTMPose (lazy loading)."""
        if self.inferencer is None:
            logger.info(f"🚀 Cargando RTMPose en {self.device}...")
            self.inferencer = MMPoseInferencer(
                pose2d=RTMPOSE_MODEL,
                device=self.device
            )
            logger.info("✅ RTMPose cargado")
    
    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        apply_smoothing: bool = True,
        apply_normalization: bool = True
    ) -> dict:
        """
        Procesa un video y guarda el tensor .npy.
        
        Args:
            video_path: Ruta al video .mp4
            output_path: Ruta donde guardar el .npy
            apply_smoothing: Aplicar OneEuroFilter
            apply_normalization: Aplicar normalización relativa
            
        Returns:
            Dict con estadísticas del procesamiento
        """
        stats = {
            'status': 'error',
            'frames': 0,
            'valid_frames': 0,
            'error': None
        }
        
        # Skip si ya existe
        if output_path.exists():
            stats['status'] = 'skipped'
            return stats
        
        try:
            self._init_inferencer()
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                stats['error'] = 'Could not open video'
                return stats
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Reset smoother para este video
            self.smoother.reset()
            
            all_keypoints = []
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Convertir a RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inferencia RTMPose
                results = list(self.inferencer(rgb, return_vis=False))
                
                keypoints = np.zeros((KEYPOINTS_PER_FRAME, 2))
                scores = np.zeros(KEYPOINTS_PER_FRAME)
                
                if results and len(results) > 0:
                    preds = results[0].get('predictions', [])
                    if preds and len(preds) > 0 and len(preds[0]) > 0:
                        pred = preds[0][0]
                        raw_kp = np.array(pred.get('keypoints', []))
                        raw_scores = np.array(pred.get('keypoint_scores', []))
                        
                        # Copiar lo que tengamos
                        n = min(len(raw_kp), KEYPOINTS_PER_FRAME)
                        keypoints[:n] = raw_kp[:n, :2]
                        scores[:n] = raw_scores[:n]
                        
                        stats['valid_frames'] += 1
                
                # Paso A: Filtro de confianza
                keypoints = apply_confidence_filter(keypoints, scores, CONFIDENCE_THRESHOLD)
                
                # Paso A.5: Filtrar manos incoherentes (alucinaciones)
                # Pasamos scores para validación completa con muñeca
                keypoints = filter_incoherent_hands(keypoints, scores=scores)
                
                # Paso B: Suavizado
                if apply_smoothing:
                    keypoints = self.smoother.smooth(timestamp, keypoints)
                
                # Paso C: Normalización
                if apply_normalization:
                    keypoints = normalize_to_center(keypoints)
                
                # Aplanar a vector de 266
                flat = keypoints.flatten()
                all_keypoints.append(flat)
                
                frame_idx += 1
            
            cap.release()
            
            stats['frames'] = frame_idx
            
            if len(all_keypoints) == 0:
                stats['error'] = 'No frames extracted'
                return stats
            
            # Guardar como numpy
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tensor = np.array(all_keypoints, dtype=np.float32)
            np.save(str(output_path), tensor)
            
            stats['status'] = 'success'
            stats['shape'] = tensor.shape
            
        except Exception as e:
            stats['error'] = str(e)
            logger.error(f"Error procesando {video_path.name}: {e}")
        
        return stats
    
    def process_all(
        self,
        apply_smoothing: bool = True,
        apply_normalization: bool = True,
        full_reprocess: bool = False
    ):
        """
        Procesa todos los videos en RAW_DATA_DIR.
        
        Args:
            apply_smoothing: Aplicar OneEuroFilter
            apply_normalization: Aplicar normalización relativa
            full_reprocess: Si True, reprocesa todos. Si False, solo nuevos.
        """
        ensure_dirs()
        load_classes()
        
        # Crear directorio versionado con timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if full_reprocess:
            # Nuevo directorio versionado
            versioned_dir = PROCESSED_DATA_DIR.parent / f"processed_v{timestamp}"
            versioned_dir.mkdir(parents=True, exist_ok=True)
            output_base = versioned_dir
            logger.info(f"📁 Creando versión: {versioned_dir.name}")
            
            # Guardar referencia a última versión
            latest_file = PROCESSED_DATA_DIR.parent / ".latest_processed"
            latest_file.write_text(versioned_dir.name)
        else:
            # Modo incremental: usar directorio actual
            output_base = PROCESSED_DATA_DIR
        
        # Recolectar todos los videos
        videos = []
        skipped_existing = 0
        
        for class_dir in RAW_DATA_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            for video_file in class_dir.glob("*.mp4"):
                output_path = output_base / class_name / (video_file.stem + ".npy")
                
                # En modo incremental, saltar si ya existe
                if not full_reprocess and output_path.exists():
                    skipped_existing += 1
                    continue
                
                videos.append((video_file, output_path, class_name))
        
        if skipped_existing > 0:
            logger.info(f"⏭️ Saltados {skipped_existing} videos ya procesados")
        
        if not videos:
            if skipped_existing > 0:
                logger.info("✅ Todos los videos ya están procesados")
            else:
                logger.error(f"❌ No se encontraron videos en {RAW_DATA_DIR}")
            return
        
        logger.info(f"📂 Videos a procesar: {len(videos)}")
        
        # Estadísticas
        stats_total = {
            'success': 0,
            'skipped': 0,
            'error': 0
        }
        errors = []
        
        # Procesar con barra de progreso
        for video_path, output_path, class_name in tqdm(videos, desc="Procesando videos"):
            # En modo full, eliminar existente primero
            if full_reprocess and output_path.exists():
                output_path.unlink()
            
            result = self.process_video(
                video_path,
                output_path,
                apply_smoothing=apply_smoothing,
                apply_normalization=apply_normalization
            )
            
            if result['status'] == 'success':
                stats_total['success'] += 1
            elif result['status'] == 'skipped':
                stats_total['skipped'] += 1
            else:
                stats_total['error'] += 1
                errors.append((video_path.name, result.get('error', 'Unknown')))
        
        # Resumen
        logger.info("=" * 50)
        logger.info("📊 RESUMEN DEL PROCESAMIENTO")
        logger.info("=" * 50)
        logger.info(f"✅ Exitosos:   {stats_total['success']}")
        logger.info(f"⏭️  Saltados:   {stats_total['skipped']}")
        logger.info(f"❌ Errores:    {stats_total['error']}")
        
        if errors:
            logger.warning("\n⚠️ Videos con errores:")
            for name, err in errors[:10]:  # Mostrar máximo 10
                logger.warning(f"   - {name}: {err}")
            if len(errors) > 10:
                logger.warning(f"   ... y {len(errors) - 10} más")
        
        # Verificar output
        logger.info("\n📁 Archivos generados por clase:")
        for class_dir in PROCESSED_DATA_DIR.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.npy")))
                logger.info(f"   {class_dir.name}: {count}")


def main():
    """Punto de entrada."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocesador de videos LSM")
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Reprocesa todos los videos (ignora existentes)"
    )
    args = parser.parse_args()
    
    mode = "COMPLETO" if args.full else "INCREMENTAL"
    
    print("=" * 60)
    print("🔄 LSM-Core Video Preprocessor")
    print("=" * 60)
    print(f"📂 Input:  {RAW_DATA_DIR}")
    print(f"📦 Output: {PROCESSED_DATA_DIR}")
    print(f"🎯 Model:  {RTMPOSE_MODEL}")
    print(f"🔧 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"🔧 Smoothing: min_cutoff={FILTER_MIN_CUTOFF}, beta={FILTER_BETA}")
    print(f"📋 Modo: {mode}")
    print("=" * 60)
    
    preprocessor = VideoPreprocessor()
    preprocessor.process_all(
        apply_smoothing=True,
        apply_normalization=True,
        full_reprocess=args.full
    )
    
    print("\n✅ Procesamiento completado!")


if __name__ == "__main__":
    main()

