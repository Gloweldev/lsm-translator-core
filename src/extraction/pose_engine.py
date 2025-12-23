"""
Motor de detección de pose usando RTMPose-WholeBody.
Extrae 133 keypoints: Cuerpo (17) + Pies (6) + Cara (68) + Manos (42)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List

# MMPose
from mmpose.apis import MMPoseInferencer

from src.config.settings import RTMPOSE_MODEL, KEYPOINTS_PER_FRAME, VALUES_PER_KEYPOINT


class PoseEngine:
    """
    Extractor de landmarks usando RTMPose-WholeBody.
    Detecta 133 keypoints por frame.
    """
    
    def __init__(
        self,
        model: str = RTMPOSE_MODEL,
        device: str = None
    ):
        """
        Inicializa el motor de detección.
        
        Args:
            model: Alias del modelo RTMPose
            device: 'cuda' o 'cpu' (None = auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = model
        
        print(f"🔄 Cargando RTMPose en {device}...")
        self.inferencer = MMPoseInferencer(
            pose2d=model,
            device=device
        )
        print("✅ RTMPose cargado")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Procesa un frame y extrae keypoints.
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            Dict con 'keypoints' (Nx3 array), 'scores', 'visualization'
        """
        # Convertir BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar
        results = list(self.inferencer(rgb, return_vis=True))
        
        output = {
            'keypoints': None,
            'scores': None,
            'visualization': None,
            'raw_results': results
        }
        
        if results and len(results) > 0:
            result = results[0]
            
            # Visualización
            if 'visualization' in result and len(result['visualization']) > 0:
                vis = result['visualization'][0]
                output['visualization'] = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            
            # Keypoints
            if 'predictions' in result and len(result['predictions']) > 0:
                pred = result['predictions'][0]
                if len(pred) > 0 and 'keypoints' in pred[0]:
                    output['keypoints'] = np.array(pred[0]['keypoints'])
                    output['scores'] = np.array(pred[0].get('keypoint_scores', []))
        
        return output
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """
        Extrae keypoints en formato de vector único para el Transformer.
        
        Args:
            frame: Imagen BGR de OpenCV
            
        Returns:
            Vector numpy de (133 * 2) = 266 dimensiones (x, y por cada punto)
        """
        results = self.process_frame(frame)
        
        if results['keypoints'] is not None:
            # keypoints es Nx3 (x, y, score), tomamos solo x, y
            kp = results['keypoints'][:, :VALUES_PER_KEYPOINT]  # Nx2
            return kp.flatten()  # 133*2 = 266
        else:
            return np.zeros(KEYPOINTS_PER_FRAME * VALUES_PER_KEYPOINT)
    
    def draw_landmarks(self, frame: np.ndarray, results: Dict = None) -> np.ndarray:
        """
        Obtiene el frame con landmarks dibujados.
        
        Args:
            frame: Imagen BGR
            results: Resultados de process_frame() o None para procesar
            
        Returns:
            Frame con landmarks dibujados
        """
        if results is None:
            results = self.process_frame(frame)
        
        if results['visualization'] is not None:
            return results['visualization']
        
        return frame.copy()


def create_pose_engine(device: str = None) -> PoseEngine:
    """Factory function para crear el motor de pose."""
    return PoseEngine(device=device)
