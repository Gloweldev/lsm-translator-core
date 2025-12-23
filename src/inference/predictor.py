"""
Predictor: clase para cargar el modelo y hacer predicciones.
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path

from src.config.settings import (
    MLRUNS_DIR, CLASSES, MAX_SEQ_LEN, MIN_SEQ_LEN,
    CONFIDENCE_THRESHOLD, STABILITY_FRAMES
)
from src.models.transformer import LSMTransformer


class LSMPredictor:
    """
    Predictor en tiempo real para LSM.
    Mantiene un buffer de frames y predice la seña.
    """
    
    def __init__(
        self,
        model_path: Path = None,
        device: str = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        stability_frames: int = STABILITY_FRAMES
    ):
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Cargar modelo
        if model_path is None:
            model_path = MLRUNS_DIR / "best_model.pth"
        
        self.model = LSMTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Config
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        
        # Estado
        self.sequence = deque(maxlen=MAX_SEQ_LEN)
        self.predictions = []
        self.last_stable_prediction = ""
    
    def reset(self):
        """Reinicia el estado del predictor."""
        self.sequence.clear()
        self.predictions.clear()
        self.last_stable_prediction = ""
    
    def add_frame(self, keypoints: np.ndarray) -> dict:
        """
        Agrega un frame y retorna la predicción actual.
        
        Args:
            keypoints: Vector de 258 dimensiones con los landmarks
            
        Returns:
            Dict con 'prediction', 'confidence', 'is_stable', 'raw_prediction'
        """
        self.sequence.append(keypoints)
        
        result = {
            'prediction': self.last_stable_prediction,
            'confidence': 0.0,
            'is_stable': False,
            'raw_prediction': None
        }
        
        # No predecir si no hay suficientes frames
        if len(self.sequence) < MIN_SEQ_LEN:
            return result
        
        # Preparar tensor
        data_np = np.array(self.sequence)
        if len(data_np) < MAX_SEQ_LEN:
            pad = np.zeros((MAX_SEQ_LEN - len(data_np), 258))
            data_np = np.vstack([data_np, pad])
        
        data_tensor = torch.FloatTensor(data_np).unsqueeze(0).to(self.device)
        
        # Predicción
        with torch.no_grad():
            output = self.model(data_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction_idx = torch.max(probs, 1)
            
            confidence = confidence.item()
            prediction_idx = prediction_idx.item()
        
        raw_prediction = CLASSES[prediction_idx]
        result['raw_prediction'] = raw_prediction
        result['confidence'] = confidence
        
        # Lógica de estabilización
        if confidence > self.confidence_threshold:
            self.predictions.append(raw_prediction)
            
            if len(self.predictions) > self.stability_frames:
                self.predictions = self.predictions[-self.stability_frames:]
                
                # Si todos son iguales -> predicción estable
                if len(set(self.predictions)) == 1:
                    result['is_stable'] = True
                    if raw_prediction != "Nothing":
                        self.last_stable_prediction = raw_prediction
                    result['prediction'] = self.last_stable_prediction
        else:
            self.predictions.append("Unsure")
        
        return result
