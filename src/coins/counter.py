"""Celý řetězec pro jeden snímek: detekce → příznaky → hodnota mince.

Parametry detekce (plochy, jádra) jsou naladěné na dataset z 14. 11.: 1428 px na šířku,
4,876 px/mm. Každý snímek z kamery se proto nejdřív přeškáluje na stejné měřítko px/mm,
aby platily stejné parametry i stejný klasifikátor.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coins import COIN_DIAMETERS_MM
from coins.classify import NearestNeighborScores
from coins.features import diameter_px, frame_features
from coins.segment import detect_coins_bg

DATASET_PX_PER_MM = 4.876
TRAIN_SETS = ["samostatne", "mix", "dotyk"]
FEATURES = ["prumer_mm", "S_stred"]


def train_model(features_csv, k=3):
    T = pd.read_csv(features_csv)
    T = T[T.sada.isin(TRAIN_SETS) & T.hodnota.notna()]
    return NearestNeighborScores(k).fit(T[FEATURES], T.hodnota.astype(int))


def load_calibration(path):
    path = Path(path)
    return json.loads(path.read_text())["px_per_mm"] if path.exists() else None


def save_calibration(path, px_per_mm):
    Path(path).write_text(json.dumps({"px_per_mm": round(px_per_mm, 4)}))


class CoinCounter:
    def __init__(self, features_csv, px_per_mm=None, k=3):
        self.model = train_model(features_csv, k)
        self.px_per_mm = px_per_mm

    def scale_for(self, frame):
        """Faktor, kterým se snímek zmenší/zvětší na měřítko datasetu."""
        if self.px_per_mm:
            return DATASET_PX_PER_MM / self.px_per_mm
        return 1428 / frame.shape[1]

    def calibrate(self, frame_rgb, value):
        """Kalibrace z mincí jedné známé hodnoty: px/mm = medián průměru v px / průměr podle ČNB."""
        s = self.scale_for(frame_rgb)
        coins = detect_coins_bg(cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        if not coins:
            return None
        d_px = np.median([diameter_px(c) for c in coins]) / s
        self.px_per_mm = d_px / COIN_DIAMETERS_MM[value]
        return self.px_per_mm

    def process(self, frame_rgb):
        """Vrátí seznam mincí v souřadnicích původního snímku, každou s hodnotou a skóre."""
        s = self.scale_for(frame_rgb)
        small = cv2.resize(frame_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else frame_rgb
        coins = detect_coins_bg(small)
        if not coins:
            return []
        X = frame_features(small, coins, DATASET_PX_PER_MM)
        values = self.model.predict(X[FEATURES])
        scores = self.model.decision_function(X[FEATURES])
        out = []
        for coin, value, score, d in zip(coins, values, scores, X.prumer_mm):
            out.append({"x": coin["x"] / s, "y": coin["y"] / s, "r": coin["r"] / s,
                        "value": int(value), "score": float(score.max()), "diameter_mm": float(d)})
        return out
