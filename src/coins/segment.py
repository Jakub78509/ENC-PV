import cv2
import numpy as np

from coins.preprocess import preprocess


def threshold_hysteresis(v):
    _, high = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, low = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    _, labels = cv2.connectedComponents(low)
    keep = np.unique(labels[high > 0])
    keep = keep[keep > 0]
    return np.isin(labels, keep).astype(np.uint8) * 255


def fill_small_holes(mask, max_area=800):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=4)
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < max_area]
    out = mask.copy()
    out[np.isin(labels, small)] = 255
    return out


def clean_mask(mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return fill_small_holes(mask)


def segment(v):
    return clean_mask(threshold_hysteresis(v))


def split_touching(img_rgb, mask, peak_size=21, min_peak=0.4):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    peaks = (dist == cv2.dilate(dist, kernel)) & (dist > min_peak * dist.max())
    peaks = cv2.dilate(peaks.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    _, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[(mask > 0) & (peaks == 0)] = 0
    return cv2.watershed(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), markers)


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0


def _coin_from_contour(c, min_area, min_circularity):
    area = cv2.contourArea(c)
    circ = circularity(c)
    if area < min_area or circ < min_circularity:
        return None
    (x, y), r = cv2.minEnclosingCircle(c)
    return {"x": x, "y": y, "r": r, "area": area, "circularity": circ, "contour": c}


def find_coins(mask, labels=None, min_area=2000, min_circularity=0.7):
    if labels is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours = []
        for label in range(2, labels.max() + 1):
            region = ((labels == label) & (mask > 0)).astype(np.uint8) * 255
            found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if found:
                contours.append(max(found, key=cv2.contourArea))
    coins = [_coin_from_contour(c, min_area, min_circularity) for c in contours]
    coins = [c for c in coins if c is not None]
    coins.sort(key=lambda coin: (coin["y"], coin["x"]))
    return coins


def detect_coins(img_rgb):
    mask = segment(preprocess(img_rgb))
    return find_coins(mask, split_touching(img_rgb, mask))


def draw_coins(img_rgb, coins, color=(0, 255, 0)):
    vis = img_rgb.copy()
    for i, coin in enumerate(coins):
        center = (int(coin["x"]), int(coin["y"]))
        cv2.circle(vis, center, int(coin["r"]), color, 3)
        cv2.putText(vis, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis
