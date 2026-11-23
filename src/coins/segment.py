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


def split_touching(img_rgb, mask, peak_size=21, min_peak=0.4, min_peak_px=None):
    # min_peak_px: pevný práh na vzdálenost od okraje; jinak min_peak × maximum ve snímku
    # (velký objekt, např. ruka, by relativní práh vytáhl nad poloměr všech mincí)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    threshold = min_peak_px if min_peak_px is not None else min_peak * dist.max()
    peaks = (dist == cv2.dilate(dist, kernel)) & (dist > threshold)
    peaks = cv2.dilate(peaks.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    _, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[(mask > 0) & (peaks == 0)] = 0
    return cv2.watershed(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), markers)


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0


def _coin_from_contour(c, min_area, min_circularity, hull=False):
    # hull=True: plochu a kruhovitost měříme na konvexním obalu, takže „ukousnutá“
    # mince (tmavý lesklý kus splyne s podložkou) projde filtrem
    if hull:
        c = cv2.convexHull(c)
    area = cv2.contourArea(c)
    circ = circularity(c)
    if area < min_area or circ < min_circularity:
        return None
    (x, y), r = cv2.minEnclosingCircle(c)
    return {"x": x, "y": y, "r": r, "area": area, "circularity": circ, "contour": c}


def find_coins(mask, labels=None, min_area=2000, min_circularity=0.7, hull=False):
    if labels is None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours = []
        for label in range(2, labels.max() + 1):
            region = ((labels == label) & (mask > 0)).astype(np.uint8) * 255
            found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if found:
                contours.append(max(found, key=cv2.contourArea))
    coins = [_coin_from_contour(c, min_area, min_circularity, hull) for c in contours]
    coins = [c for c in coins if c is not None]
    coins.sort(key=lambda coin: (coin["y"], coin["x"]))
    return coins


def detect_coins(img_rgb):
    mask = segment(preprocess(img_rgb))
    return find_coins(mask, split_touching(img_rgb, mask))


# --- Dataset z 14. 11.: šedá podložka, pravítko nahoře, snímky 1428×1071 px ---

def ruler_bottom(img_rgb, min_fraction=0.3, margin=10):
    """Spodní řádek červeného pravítka (vše nad ním se ignoruje), 0 pokud pravítko chybí."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 60)
    rows = np.where(red.mean(axis=1) > min_fraction)[0]
    rows = rows[rows < img_rgb.shape[0] // 2]
    return int(rows.max()) + margin if len(rows) else 0


def background_diff(v, scale=4, ksize=75):
    """|V − pozadí|: pozadí = mediánový filtr s jádrem větším než mince (na zmenšeném snímku).

    Odstraní pozvolné změny jasu podložky (vinětace, boční světlo), takže práh je pro
    celou fotku stejný a najde i tmavé lesklé mince, které jsou tmavší než podložka.
    """
    small = cv2.resize(v, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, ksize), (v.shape[1], v.shape[0]))
    diff = np.abs(v.astype(np.float32) - bg.astype(np.float32))
    return np.clip(diff * 2, 0, 255).astype(np.uint8)


def segment_bg(img_rgb, thr=50, ksize=7):
    mask = cv2.threshold(background_diff(preprocess(img_rgb)), thr, 255, cv2.THRESH_BINARY)[1]
    mask[:ruler_bottom(img_rgb)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = fill_small_holes(mask, max_area=20000)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def remove_border_blobs(mask):
    """Vyhodí objekty, které se dotýkají okraje snímku (ruka, mince napůl mimo záběr)."""
    n, labels = cv2.connectedComponents(mask)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    out = mask.copy()
    out[np.isin(labels, border[border > 0])] = 0
    return out


def detect_coins_bg(img_rgb, thr=50, min_peak_px=20):
    # min_peak_px = 20: zhruba 0,4 × poloměr 1 Kč (10 mm × 4,876 px/mm ≈ 49 px)
    mask = remove_border_blobs(segment_bg(img_rgb, thr))
    labels = split_touching(img_rgb, mask, min_peak_px=min_peak_px)
    return find_coins(mask, labels, min_area=5000, min_circularity=0.75, hull=True)


def draw_coins(img_rgb, coins, color=(0, 255, 0)):
    vis = img_rgb.copy()
    for i, coin in enumerate(coins):
        center = (int(coin["x"]), int(coin["y"]))
        cv2.circle(vis, center, int(coin["r"]), color, 3)
        cv2.putText(vis, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return vis
