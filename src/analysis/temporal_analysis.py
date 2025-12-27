"""
Análisis Temporal del Transformer - Diagnóstico de Inercia del Buffer.

Visualiza gráficamente cómo cambia la probabilidad de predicción cuadro por cuadro.
Útil para diagnosticar el fenómeno de "lag" en el buffer rodante.

Uso:
    1. Editar VIDEO_PATH y TARGET_CLASS abajo
    2. python -m src.analysis.temporal_analysis
"""

import cv2
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from tqdm import tqdm

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
NUM_VIDEOS = 10  # Videos aleatorios a analizar

# Configuración del buffer (igual que inferencia real)
BUFFER_SIZE = MAX_SEQ_LEN  # 90 frames

# Índices para normalización
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12

# Índices a ignorar (piernas y pies)
SKIP_INDICES = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}


def is_hand_valid(keypoints: np.ndarray, scores: np.ndarray,
                  hand_start: int, hand_end: int, wrist_idx: int,
                  max_spread: float = 150,
                  min_confidence: float = 0.8,
                  high_confidence: float = 0.9,
                  max_wrist_distance: float = 100) -> bool:
    """Valida si una mano es confiable (no alucinada)."""
    hand_pts = keypoints[hand_start:hand_end]
    hand_scores = scores[hand_start:hand_end]
    
    valid_mask = np.any(hand_pts != 0, axis=1)
    valid_pts = hand_pts[valid_mask]
    valid_scores = hand_scores[valid_mask]
    
    if len(valid_pts) < 5:
        return False
    
    spread_x = valid_pts[:, 0].max() - valid_pts[:, 0].min()
    spread_y = valid_pts[:, 1].max() - valid_pts[:, 1].min()
    
    if spread_x > max_spread or spread_y > max_spread:
        return False
    
    avg_conf = valid_scores.mean()
    if avg_conf < min_confidence:
        return False
    
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
    
    if not is_hand_valid(keypoints.reshape(-1, 2), scores, 91, 112, wrist_idx=9):
        filtered[91*2:112*2] = 0.0
    
    if not is_hand_valid(keypoints.reshape(-1, 2), scores, 112, 133, wrist_idx=10):
        filtered[112*2:133*2] = 0.0
    
    return filtered

class TemporalAnalyzer:
    """Analiza probabilidades frame por frame."""
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Buffer rodante
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
        # Smoothers
        self.smoothers = None
        self.frame_count = 0
        
        # Cargar modelos
        self._load_rtmpose()
        self._load_transformer()
    
    def _load_rtmpose(self):
        """Carga RTMPose."""
        print(f"🚀 Cargando RTMPose en {self.device}...")
        self.pose_inferencer = MMPoseInferencer(
            pose2d=RTMPOSE_MODEL,
            device=str(self.device)
        )
        print("✅ RTMPose cargado")
    
    def _load_transformer(self):
        """Carga el Transformer."""
        model_path = MLRUNS_DIR / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró: {model_path}")
        
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
    
    def _normalize_to_center(self, keypoints_flat: np.ndarray) -> np.ndarray:
        """Normaliza restando centro de caderas."""
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
        
        normalized = kp - center
        return normalized.flatten()
    
    def extract_and_preprocess(self, frame) -> np.ndarray:
        """Extrae y preprocesa keypoints (igual que inferencia)."""
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
        
        # Filtrar manos incoherentes
        raw_keypoints = filter_incoherent_hands(raw_keypoints, scores)
        
        # Aplicar smoothing
        self.frame_count += 1
        t = self.frame_count / 30.0
        
        if self.smoothers is None:
            self.smoothers = []
            for i in range(INPUT_DIM):
                self.smoothers.append(OneEuroFilter(
                    t0=t, x0=raw_keypoints[i],
                    min_cutoff=FILTER_MIN_CUTOFF, beta=FILTER_BETA
                ))
            smoothed = raw_keypoints.copy()
        else:
            smoothed = np.array([self.smoothers[i](t, raw_keypoints[i]) for i in range(INPUT_DIM)])
        
        # Normalizar
        normalized = self._normalize_to_center(smoothed)
        return normalized
    
    def predict_with_buffer(self) -> np.ndarray:
        """Predice usando buffer actual con padding si es necesario."""
        sequence = np.array(list(self.buffer))
        actual_frames = len(sequence)  # Duración real antes del padding
        
        # Padding al inicio si buffer incompleto (Cold Start)
        if len(sequence) < BUFFER_SIZE:
            padding = np.zeros((BUFFER_SIZE - len(sequence), INPUT_DIM))
            sequence = np.vstack([padding, sequence])
        
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        duration = torch.LongTensor([actual_frames]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor, duration=duration)
            probs = F.softmax(logits, dim=1)
        
        return probs[0].cpu().numpy()
    
    def analyze_video(self, video_path: str) -> dict:
        """Analiza un video completo frame por frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Almacenar probabilidades por clase
        history = {cls: [] for cls in CLASS_NAMES}
        
        # Reset
        self.buffer.clear()
        self.smoothers = None
        self.frame_count = 0
        
        frame_idx = 0
        pbar = tqdm(desc="Analizando", unit=" frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocesar
            features = self.extract_and_preprocess(frame)
            self.buffer.append(features)
            
            # Predecir
            probs = self.predict_with_buffer()
            
            # Guardar probabilidades
            for i, cls in enumerate(CLASS_NAMES):
                history[cls].append(probs[i])
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        total_frames = len(history[CLASS_NAMES[0]]) if CLASS_NAMES else 0
        
        print(f"   Frames procesados: {total_frames}, FPS: {fps:.1f}")
        
        return {
            'history': history,
            'total_frames': total_frames,
            'fps': fps
        }


def plot_temporal_analysis(result: dict, target_class: str, output_path: str, video_name: str = ""):
    """Genera gráfica de probabilidades temporales."""
    history = result['history']
    total_frames = result['total_frames']
    
    plt.figure(figsize=(16, 8))
    
    frames = np.arange(total_frames)
    
    # Otras clases (transparentes)
    for cls in CLASS_NAMES:
        if cls not in [target_class, 'nada']:
            plt.plot(frames, history[cls], 
                    alpha=0.3, linewidth=1, label=cls, color='lightblue')
    
    # Línea de "nada" (gris punteada)
    plt.plot(frames, history['nada'], 
            linestyle='--', linewidth=2, color='gray', label='nada', alpha=0.8)
    
    # Línea de TARGET_CLASS (verde gruesa)
    plt.plot(frames, history[target_class], 
            linewidth=3, color='green', label=f'{target_class} (TARGET)')
    
    # Umbral de confianza
    plt.axhline(y=INFERENCE_CONFIDENCE_THRESHOLD, 
               color='red', linestyle='-', linewidth=1.5, 
               label=f'Umbral ({INFERENCE_CONFIDENCE_THRESHOLD})')
    
    # Configuración
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    title = f'Análisis Temporal - Clase: {target_class}'
    if video_name:
        title += f' | Video: {video_name}'
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.xlim(0, total_frames)
    
    # Anotaciones
    target_probs = history[target_class]
    if target_probs:
        max_prob = max(target_probs)
        max_frame = target_probs.index(max_prob)
        plt.annotate(f'Max: {max_prob:.2f}', 
                    xy=(max_frame, max_prob),
                    xytext=(max_frame + 10, min(max_prob + 0.05, 1.0)),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"   📊 Gráfica: {output_path}")


def get_random_videos(num_videos: int = 10) -> list:
    """Obtiene videos aleatorios del dataset."""
    import random
    
    all_videos = []
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4"))
            for v in videos:
                all_videos.append((v, class_name))
    
    if len(all_videos) < num_videos:
        return all_videos
    
    return random.sample(all_videos, num_videos)


def main():
    """Punto de entrada."""
    print("=" * 70)
    print("🔬 Análisis Temporal del Transformer")
    print("   Diagnóstico de Inercia del Buffer")
    print("=" * 70)
    
    # Obtener videos aleatorios
    videos = get_random_videos(NUM_VIDEOS)
    
    if not videos:
        print(f"❌ No se encontraron videos en {RAW_DATA_DIR}")
        return
    
    print(f"\n📹 Analizando {len(videos)} videos aleatorios...")
    
    # Crear directorio de salida
    output_dir = Path("experiments/temporal_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analizar cada video
    analyzer = TemporalAnalyzer()
    
    summary = []
    
    for i, (video_path, true_class) in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {video_path.name} (clase: {true_class})")
        
        try:
            result = analyzer.analyze_video(str(video_path))
            
            # Generar gráfica
            output_path = output_dir / f"{i+1:02d}_{true_class}_{video_path.stem}.png"
            plot_temporal_analysis(result, true_class, str(output_path), video_path.name)
            
            # Estadísticas
            target_probs = result['history'][true_class]
            max_prob = max(target_probs)
            avg_prob = np.mean(target_probs)
            above_threshold = sum(1 for p in target_probs if p > INFERENCE_CONFIDENCE_THRESHOLD)
            
            summary.append({
                'video': video_path.name,
                'class': true_class,
                'max_prob': max_prob,
                'avg_prob': avg_prob,
                'frames_above': above_threshold,
                'total_frames': len(target_probs)
            })
            
            print(f"   Max: {max_prob:.3f}, Avg: {avg_prob:.3f}, "
                  f"Frames>{INFERENCE_CONFIDENCE_THRESHOLD}: {above_threshold}/{len(target_probs)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN")
    print("=" * 70)
    
    for s in summary:
        status = "✅" if s['max_prob'] > INFERENCE_CONFIDENCE_THRESHOLD else "⚠️"
        print(f"{status} {s['class']:6} | Max: {s['max_prob']:.3f} | "
              f"Avg: {s['avg_prob']:.3f} | {s['video']}")
    
    # Estadísticas globales
    if summary:
        avg_max = np.mean([s['max_prob'] for s in summary])
        successful = sum(1 for s in summary if s['max_prob'] > INFERENCE_CONFIDENCE_THRESHOLD)
        print(f"\n🎯 Promedio Max Prob: {avg_max:.3f}")
        print(f"✅ Videos que superan umbral: {successful}/{len(summary)}")
    
    print(f"\n📁 Gráficas guardadas en: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

