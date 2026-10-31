"""Načítání a zobrazování snímků – sdílené pro všechny notebooky."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Kořen projektu = složka nad src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_rgb(path) -> np.ndarray:
    """Načte snímek jako RGB uint8 (OpenCV načítá BGR, proto převod)."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Snímek nelze načíst: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_images(images, titles=None, cmap="gray", cols=None, size=5):
    """Zobrazí několik snímků vedle sebe (šedotónové s danou colormapou)."""
    n = len(images)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(size * cols, size * 1.3 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
