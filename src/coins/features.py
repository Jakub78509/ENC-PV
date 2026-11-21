"""Příznaky jednotlivých mincí: průměr v mm a barva uprostřed a na okraji."""

import cv2
import numpy as np
import pandas as pd

from coins import COIN_DIAMETERS_MM

FEATURES = ["prumer_mm", "S_stred", "H_stred", "S_okraj", "H_okraj"]


def calibrate(diameters_px, values):
    """Měřítko px/mm: nejmenší čtverce přes počátek, d_px ≈ k · d_mm (známé průměry podle ČNB)."""
    d_px = np.asarray(diameters_px, dtype=float)
    d_mm = np.array([COIN_DIAMETERS_MM[v] for v in values], dtype=float)
    return float((d_px * d_mm).sum() / (d_mm ** 2).sum())


def _ring_mask(shape, x, y, r, inner, outer):
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), int(outer * r), 1, -1)
    if inner > 0:
        cv2.circle(mask, (int(round(x)), int(round(y))), int(inner * r), 0, -1)
    return mask > 0


def diameter_px(coin):
    """Průměr = menší osa elipsy proložené konturou.

    Je méně citlivá na výběžky (stín, kus sousední mince po watershedu) než opsaná
    kružnice: rozptyl chyby u dotýkajících se mincí 0,28 mm místo 0,89 mm.
    """
    (_, _), (a, b), _ = cv2.fitEllipse(coin["contour"])
    return min(a, b)


def coin_features(img_rgb, coin, px_per_mm):
    """Střed = kruh do 0,5 r, okraj = mezikruží 0,72–0,92 r (u 50 Kč jiný kov než střed)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    x, y, r = coin["x"], coin["y"], coin["r"]
    center = _ring_mask(hsv.shape[:2], x, y, r, 0, 0.5)
    ring = _ring_mask(hsv.shape[:2], x, y, r, 0.72, 0.92)
    return {
        "prumer_mm": diameter_px(coin) / px_per_mm,
        "S_stred": hsv[..., 1][center].mean(),
        "H_stred": hsv[..., 0][center].mean(),
        "S_okraj": hsv[..., 1][ring].mean(),
        "H_okraj": hsv[..., 0][ring].mean(),
    }


def match_to_truth(coins, truth, max_dist=25):
    """Přiřadí detekcím hodnotu z ruční anotace (nejbližší střed do max_dist px).

    truth: seznam (x, y, hodnota). Vrací seznam hodnot (None = falešná detekce)
    a počet anotovaných mincí, které detekce nenašla.
    """
    free = list(range(len(truth)))
    labels = []
    for c in coins:
        best = min(free, key=lambda i: np.hypot(truth[i][0] - c["x"], truth[i][1] - c["y"]), default=None)
        if best is not None and np.hypot(truth[best][0] - c["x"], truth[best][1] - c["y"]) < max_dist:
            labels.append(truth[best][2])
            free.remove(best)
        else:
            labels.append(None)
    return labels, len(free)


def features_table(images, detections, px_per_mm, truth=None):
    """Tabulka příznaků všech detekovaných mincí; s truth i sloupec 'hodnota' (None = falešná)."""
    rows = []
    for name, coins in detections.items():
        labels = match_to_truth(coins, truth[name])[0] if truth is not None else [None] * len(coins)
        for i, (coin, value) in enumerate(zip(coins, labels)):
            row = {"soubor": name, "id": i, "x": coin["x"], "y": coin["y"]}
            row.update(coin_features(images[name], coin, px_per_mm))
            if truth is not None:
                row["hodnota"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def frame_features(img_rgb, coins, px_per_mm):
    """Stejné příznaky jako coin_features, ale rychleji pro živý obraz.

    HSV se převádí jednou za snímek a masky středu a mezikruží se kreslí jen do výřezu kolem mince.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    rows = []
    for coin in coins:
        cx, cy, r = int(round(coin["x"])), int(round(coin["y"])), coin["r"]
        x0, y0 = max(cx - int(r) - 2, 0), max(cy - int(r) - 2, 0)
        x1, y1 = min(cx + int(r) + 3, w), min(cy + int(r) + 3, h)
        roi = hsv[y0:y1, x0:x1]
        center = _ring_mask(roi.shape[:2], cx - x0, cy - y0, r, 0, 0.5)
        ring = _ring_mask(roi.shape[:2], cx - x0, cy - y0, r, 0.72, 0.92)
        rows.append({
            "prumer_mm": diameter_px(coin) / px_per_mm,
            "S_stred": roi[..., 1][center].mean(),
            "H_stred": roi[..., 0][center].mean(),
            "S_okraj": roi[..., 1][ring].mean(),
            "H_okraj": roi[..., 0][ring].mean(),
        })
    return pd.DataFrame(rows, columns=FEATURES)
