"""
Demo en tiempo real usando webcam o stream de video.
"""

import cv2
import argparse
import time

from src.extraction.pose_engine import PoseEngine
from src.inference.predictor import LSMPredictor


def main():
    parser = argparse.ArgumentParser(description='🤟 LSM Translator - Demo en Tiempo Real')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Fuente de video: 0=webcam, URL=stream, archivo.mp4')
    parser.add_argument('--width', '-w', type=int, default=640, help='Ancho del video')
    parser.add_argument('--height', '-H', type=int, default=480, help='Alto del video')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🤟 LSM Translator - Demo en Tiempo Real")
    print("=" * 50)
    
    # Inicializar
    print("🔄 Cargando modelo...")
    try:
        predictor = LSMPredictor()
        print("✅ Modelo cargado")
    except FileNotFoundError:
        print("❌ No se encontró el modelo entrenado.")
        print("   Ejecuta primero: python -m src.training.train")
        return
    
    print("🔄 Inicializando pose engine...")
    pose_engine = PoseEngine(model_complexity=2)
    print("✅ Pose engine listo")
    
    # Conectar a fuente de video
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {actual_width}x{actual_height}")
    print("\n🎬 Presiona ESC para salir\n")
    
    # Loop principal
    frame_count = 0
    fps_time = time.time()
    current_fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calcular FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            current_fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Extraer keypoints y dibujar landmarks
        results = pose_engine.process(frame)
        keypoints = pose_engine.extract_keypoints(frame)
        frame = pose_engine.draw_landmarks(frame, results)
        
        # Predicción
        prediction = predictor.add_frame(keypoints)
        
        # UI
        cv2.rectangle(frame, (0, 0), (actual_width, 80), (0, 0, 0), -1)
        
        # Traducción
        color = (0, 255, 100) if prediction['is_stable'] else (255, 255, 255)
        cv2.putText(frame, f"Traduccion: {prediction['prediction']}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Debug info
        if prediction['raw_prediction']:
            debug = f"Raw: {prediction['raw_prediction']} ({prediction['confidence']:.2f})"
            cv2.putText(frame, debug, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {current_fps}", (actual_width - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('LSM Translator', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    pose_engine.close()
    cv2.destroyAllWindows()
    print("\n👋 ¡Hasta luego!")


if __name__ == "__main__":
    main()
