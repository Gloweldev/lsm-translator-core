"""
Script para descargar videos del servidor API.
Guarda los videos en dataset/raw/ organizados por palabra.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.settings import RAW_DATA_DIR, ensure_dirs

# URL de la API
API_URL = "https://lsm-recorder-api.glowel.com.mx/api/videos/export"


def get_video_list():
    """Obtiene la lista de videos desde la API."""
    print("📡 Conectando a la API...")
    try:
        response = requests.get(API_URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Contar total de videos
        total = sum(len(videos) for videos in data.get("palabras", {}).values())
        palabras = list(data.get("palabras", {}).keys())
        
        print(f"✅ Encontradas {len(palabras)} clases con {total} videos")
        return data.get("palabras", {})
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error conectando a la API: {e}")
        return {}


def download_video(palabra: str, video_info: dict) -> str:
    """
    Descarga un video individual.
    
    Args:
        palabra: Nombre de la clase/palabra
        video_info: Dict con 'suggested_filename', 'download_url'
        
    Returns:
        "Success", "Skipped", o mensaje de error
    """
    try:
        download_url = video_info.get('download_url')
        filename = video_info.get('suggested_filename', 'unknown.mp4')
        
        if not download_url:
            return f"Error: No URL for {filename}"
        
        # Crear directorio para la palabra
        output_dir = RAW_DATA_DIR / palabra
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # Saltar si ya existe
        if output_path.exists():
            return "Skipped"
        
        # Descargar
        response = requests.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return "Success"
        
    except Exception as e:
        return f"Error: {str(e)}"


def download_all(max_workers: int = 4):
    """
    Descarga todos los videos del servidor.
    
    Args:
        max_workers: Número de descargas paralelas
    """
    ensure_dirs()
    
    # Obtener lista de videos
    palabras_dict = get_video_list()
    if not palabras_dict:
        return
    
    # Preparar tareas
    tasks = []
    for palabra, videos in palabras_dict.items():
        for video in videos:
            tasks.append((palabra, video))
    
    print(f"\n📥 Descargando videos a: {RAW_DATA_DIR}")
    print(f"   Hilos paralelos: {max_workers}")
    
    success = 0
    skipped = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_video, palabra, video): (palabra, video) 
                   for palabra, video in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
            result = future.result()
            
            if result == "Success":
                success += 1
            elif result == "Skipped":
                skipped += 1
            else:
                errors += 1
                # Mostrar errores
                palabra, video = futures[future]
                tqdm.write(f"⚠️ {palabra}/{video.get('suggested_filename', '?')}: {result}")
    
    print(f"\n✅ Completado:")
    print(f"   - Nuevos: {success}")
    print(f"   - Existentes: {skipped}")
    print(f"   - Errores: {errors}")
    
    # Resumen por clase
    print("\n📊 Videos por clase:")
    for class_dir in sorted(RAW_DATA_DIR.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.mp4")))
            print(f"   {class_dir.name}: {count}")


def main():
    """Punto de entrada."""
    print("=" * 50)
    print("📥 Descargador de Videos LSM")
    print("=" * 50)
    
    download_all(max_workers=4)


if __name__ == "__main__":
    main()
