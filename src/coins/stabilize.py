"""Stabilní živý součet: sledování mincí, hlasování o hodnotě, hystereze, medián v čase, detekce pohybu."""

from collections import Counter, deque

import cv2
import numpy as np


class Track:
    def __init__(self, track_id, coin, classify=None):
        self.id = track_id
        self.classify = classify
        self.features = deque([coin["features"]], maxlen=15) if "features" in coin else None
        self.x, self.y, self.r = coin["x"], coin["y"], coin["r"]
        self.d = coin.get("diameter_mm", 0.0)
        self.votes = deque([coin["value"]], maxlen=9)
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    @property
    def value(self):
        # s klasifikátorem: hodnota z příznaků zprůměrovaných přes snímky (šum se vyruší),
        # jinak hlasování z hodnot jednotlivých snímků
        if self.classify is not None and self.features:
            return self.classify(np.mean(self.features, axis=0))
        return Counter(self.votes).most_common(1)[0][0]

    def update(self, coin):
        self.x, self.y, self.r = coin["x"], coin["y"], coin["r"]
        self.d = 0.7 * self.d + 0.3 * coin.get("diameter_mm", self.d)
        self.votes.append(coin["value"])
        if self.features is not None and "features" in coin:
            self.features.append(coin["features"])
        self.hits += 1
        self.misses = 0


class Stabilizer:
    """
    min_hits:   mince se započítá až po tolika snímcích v řadě (hystereze nahoru)
    max_misses: mince se odebere až po tolika snímcích bez detekce (hystereze dolů)
    window:     součet = medián z posledních window snímků
    motion_thr: kolik % pixelů se mezi snímky výrazně změní, aby se součet zmrazil (ruka v záběru)
    settle:     kolik snímků po skončení pohybu ještě počkat
    max_dd:     mince se spáruje se stopou jen při podobném průměru (hodnoty se liší aspoň o 1,5 mm),
                takže se po přeskládání rukou nepřevezme hodnota jiné mince
    classify:   funkce příznaky → hodnota; když je zadaná, hodnota stopy se počítá ze zprůměrovaných příznaků
    """

    def __init__(self, min_hits=3, max_misses=3, window=5, motion_thr=2.0, settle=2, max_dd=0.8, classify=None):
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.sums = deque(maxlen=window)
        self.motion_thr = motion_thr
        self.settle = settle
        self.max_dd = max_dd
        self.classify = classify
        self.calm = 0
        self.tracks = []
        self.next_id = 0
        self.prev_gray = None
        self.moving = False
        self.stable_sum = 0

    def motion(self, frame_rgb):
        """% pixelů, které se od minulého snímku výrazně změnily (po odečtení celkové změny jasu)."""
        small = cv2.resize(frame_rgb, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff = gray - self.prev_gray
        self.prev_gray = gray
        return float((np.abs(diff - np.median(diff)) > 25).mean() * 100)

    def _match(self, coins):
        pairs, used = [], set()
        for ci, c in enumerate(coins):
            best, best_d = None, None
            for t in self.tracks:
                if t.id in used:
                    continue
                d = np.hypot(t.x - c["x"], t.y - c["y"])
                same_size = abs(t.d - c.get("diameter_mm", t.d)) < self.max_dd
                if d < 0.5 * max(t.r, c["r"]) and same_size and (best_d is None or d < best_d):
                    best, best_d = t, d
            if best is not None:
                used.add(best.id)
                pairs.append((best, c))
            else:
                pairs.append((None, c))
        return pairs, used

    def update(self, frame_rgb, coins):
        self.calm = 0 if self.motion(frame_rgb) > self.motion_thr else self.calm + 1
        self.moving = self.calm <= self.settle
        if self.moving:
            return self.stable_sum

        pairs, used = self._match(coins)
        for t in self.tracks:
            if t.id not in used:
                t.misses += 1
        for track, coin in pairs:
            if track is None:
                self.tracks.append(Track(self.next_id, coin, self.classify))
                self.next_id += 1
            else:
                track.update(coin)
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
        # nepotvrzená mince musí být vidět v každém snímku, potvrzená smí chybět max_misses snímků
        self.tracks = [t for t in self.tracks
                       if t.misses <= (self.max_misses if t.confirmed else 0)]

        self.sums.append(sum(t.value for t in self.confirmed_tracks()))
        self.stable_sum = int(sorted(self.sums)[len(self.sums) // 2])
        return self.stable_sum

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.confirmed]
