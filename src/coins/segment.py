import cv2
import numpy as np


def threshold_hysteresis(v):
    _, high = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, low = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    _, labels = cv2.connectedComponents(low)
    keep = np.unique(labels[high > 0])
    keep = keep[keep > 0]
    return np.isin(labels, keep).astype(np.uint8) * 255


def clean_mask(mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment(v):
    return clean_mask(threshold_hysteresis(v))


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0


def find_coins(mask, min_area=2000, min_circularity=0.7):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coins = []
    for c in contours:
        area = cv2.contourArea(c)
        circ = circularity(c)
        if area < min_area or circ < min_circularity:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        coins.append({"x": x, "y": y, "r": r, "area": area, "circularity": circ, "contour": c})
    coins.sort(key=lambda coin: (coin["y"], coin["x"]))
    return coins


def draw_coins(img_rgb, coins, color=(0, 255, 0)):
    vis = img_rgb.copy()
    for i, coin in enumerate(coins):
        center = (int(coin["x"]), int(coin["y"]))
        cv2.circle(vis, center, int(coin["r"]), color, 3)
        cv2.putText(vis, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis
