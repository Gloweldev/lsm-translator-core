"""
Utilidades para manejo de archivos.
"""

from pathlib import Path
from typing import List, Dict
from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASSES


def list_files(directory: Path, extension: str = "*") -> List[Path]:
    """Lista archivos en un directorio."""
    if not directory.exists():
        return []
    return list(directory.glob(f"*.{extension}"))


def count_samples(data_dir: Path = None) -> Dict[str, int]:
    """
    Cuenta muestras por clase.
    
    Args:
        data_dir: Directorio de datos (raw o processed)
        
    Returns:
        Dict con nombre de clase -> cantidad de muestras
    """
    if data_dir is None:
        data_dir = PROCESSED_DATA_DIR
    
    counts = {}
    
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if class_dir.exists():
            if data_dir == RAW_DATA_DIR:
                counts[class_name] = len(list(class_dir.glob("*.mp4")))
            else:
                counts[class_name] = len(list(class_dir.glob("*.npy")))
        else:
            counts[class_name] = 0
    
    return counts


def print_dataset_summary():
    """Imprime resumen del dataset."""
    print("=" * 40)
    print("📊 Resumen del Dataset")
    print("=" * 40)
    
    print("\n📂 Videos RAW:")
    raw_counts = count_samples(RAW_DATA_DIR)
    for cls, count in raw_counts.items():
        print(f"   {cls}: {count}")
    print(f"   Total: {sum(raw_counts.values())}")
    
    print("\n📦 Procesados (.npy):")
    proc_counts = count_samples(PROCESSED_DATA_DIR)
    for cls, count in proc_counts.items():
        print(f"   {cls}: {count}")
    print(f"   Total: {sum(proc_counts.values())}")


if __name__ == "__main__":
    print_dataset_summary()
