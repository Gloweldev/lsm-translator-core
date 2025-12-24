"""
Descarga videos del API de grabación LSM.

Soporta:
    - Descarga completa (primera vez)
    - Descarga incremental (solo nuevos desde último sync)

Uso:
    python -m src.utils.download_videos           # Incremental
    python -m src.utils.download_videos --full    # Completo

API:
    GET /api/videos/export         → Todos los videos
    GET /api/videos/export?since=  → Solo videos desde fecha
"""

import os
import sys
import argparse
import requests
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.settings import RAW_DATA_DIR

# =============================================
# CONFIGURACIÓN
# =============================================
API_URL = "https://lsm-recorder-api.glowel.com.mx/api/videos/export"
LAST_SYNC_FILE = RAW_DATA_DIR.parent / ".last_sync"
MAX_WORKERS = 4


def download_video(video_info: dict, folder: Path) -> tuple:
    """
    Descarga un video individual.
    
    Returns:
        (success, filepath, error_message)
    """
    filepath = folder / video_info["suggested_filename"]
    
    # Skip si ya existe
    if filepath.exists():
        return True, str(filepath), "already exists"
    
    try:
        response = requests.get(video_info["download_url"], timeout=60)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        return True, str(filepath), None
    except Exception as e:
        return False, str(filepath), str(e)


def get_last_sync() -> str:
    """Lee el timestamp del último sync."""
    if LAST_SYNC_FILE.exists():
        with open(LAST_SYNC_FILE, "r") as f:
            return f.read().strip()
    return None


def save_last_sync():
    """Guarda el timestamp actual para próximo sync."""
    LAST_SYNC_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_SYNC_FILE, "w") as f:
        f.write(datetime.utcnow().isoformat() + "Z")


def download_all(full_sync: bool = False):
    """
    Descarga videos del API.
    
    Args:
        full_sync: Si True, descarga todo. Si False, solo nuevos.
    """
    print("=" * 60)
    print("📥 Descargador de Videos LSM")
    print("=" * 60)
    
    # Determinar modo
    since = None
    if not full_sync:
        since = get_last_sync()
        if since:
            print(f"📅 Modo incremental (desde: {since})")
        else:
            print("📦 Primera descarga (completa)")
    else:
        print("🔄 Modo: Descarga completa (ignorando último sync)")
    
    # Request al API
    url = f"{API_URL}?since={since}" if since else API_URL
    print(f"🌐 API: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Error conectando al API: {e}")
        return
    
    # Verificar respuesta
    if "palabras" not in data:
        print("❌ Respuesta del API inválida (falta 'palabras')")
        return
    
    total_videos = data.get("total_videos", 0)
    palabras = data["palabras"]
    
    print(f"\n📊 Total: {total_videos} videos en {len(palabras)} clases")
    
    if total_videos == 0:
        print("✅ No hay videos nuevos para descargar")
        return
    
    # Preparar lista de descargas
    downloads = []
    for palabra, videos in palabras.items():
        folder = RAW_DATA_DIR / palabra
        folder.mkdir(parents=True, exist_ok=True)
        
        for video in videos:
            downloads.append((video, folder))
    
    print(f"\n🚀 Iniciando descarga de {len(downloads)} videos...")
    print("-" * 60)
    
    # Descargar en paralelo
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_video, video, folder): (video, folder)
            for video, folder in downloads
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            success, filepath, error = future.result()
            filename = Path(filepath).name
            
            if success:
                if error == "already exists":
                    skip_count += 1
                    status = "⏭️"
                else:
                    success_count += 1
                    status = "✅"
            else:
                error_count += 1
                status = "❌"
            
            # Progress
            pct = i / len(downloads) * 100
            print(f"{status} [{i}/{len(downloads)}] {pct:.0f}% - {filename}")
    
    # Guardar timestamp
    save_last_sync()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"  ✅ Descargados: {success_count}")
    print(f"  ⏭️ Omitidos (ya existían): {skip_count}")
    print(f"  ❌ Errores: {error_count}")
    print(f"\n📁 Guardados en: {RAW_DATA_DIR}")
    print(f"📅 Último sync: {get_last_sync()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Descarga videos del API LSM")
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Descarga completa (ignora último sync)"
    )
    args = parser.parse_args()
    
    download_all(full_sync=args.full)


if __name__ == "__main__":
    main()
