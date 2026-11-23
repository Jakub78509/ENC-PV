"""Živé počítání mincí z kamery.

    python live.py                      # kamera 0
    python live.py --source 1           # jiná kamera (iPhone přes Continuity Camera bývá 0 nebo 1)
    python live.py --list               # vypíše dostupné kamery
    python live.py --source video.mov   # místo kamery video

Klávesy: q konec, k kalibrace (na podložce jen mince hodnoty --calib-value), s uloží snímek.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from coins.counter import CoinCounter, load_calibration, save_calibration  # noqa: E402
from coins.stabilize import Stabilizer  # noqa: E402

GREEN, YELLOW, WHITE, BLACK = (0, 200, 0), (0, 200, 255), (255, 255, 255), (0, 0, 0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="číslo kamery nebo cesta k videu")
    p.add_argument("--list", action="store_true", help="vypsat dostupné kamery a skončit")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--calibration", default=str(ROOT / "kalibrace.json"))
    p.add_argument("--calib-value", type=int, default=10, help="hodnota mincí pro kalibraci klávesou k")
    p.add_argument("--record", help="uložit video s popisky (.mp4)")
    p.add_argument("--log", help="uložit součty po snímcích (.csv)")
    p.add_argument("--no-window", action="store_true", help="bez okna (zpracování videa)")
    return p.parse_args()


def list_cameras(n=5):
    for i in range(n):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        if ok:
            print(f"kamera {i}: {frame.shape[1]}×{frame.shape[0]}")
        cap.release()


def open_source(args):
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Zdroj {args.source} nejde otevřít (zkus --list).")
    return cap


def draw(frame_bgr, coins, stab, fps, px_per_mm):
    for c in coins:
        cv2.circle(frame_bgr, (int(c["x"]), int(c["y"])), int(c["r"]), (128, 128, 128), 1)
    for t in stab.tracks:
        color = GREEN if t.confirmed else YELLOW
        center = (int(t.x), int(t.y))
        cv2.circle(frame_bgr, center, int(t.r), color, 3)
        label = f"{t.value}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        org = (center[0] - size[0] // 2, center[1] + size[1] // 2)
        cv2.putText(frame_bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK, 6)
        cv2.putText(frame_bgr, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    text = "merim..." if stab.moving else f"Soucet: {stab.stable_sum} Kc"
    cv2.rectangle(frame_bgr, (10, 10), (430, 75), BLACK, -1)
    cv2.putText(frame_bgr, text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 3)
    info = f"{fps:4.1f} fps | " + (f"{px_per_mm:.2f} px/mm" if px_per_mm else "bez kalibrace (k)")
    cv2.putText(frame_bgr, info, (20, frame_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)


def main():
    args = parse_args()
    if args.list:
        list_cameras()
        return

    counter = CoinCounter(ROOT / "results" / "05_priznaky.csv", load_calibration(args.calibration))
    stab = Stabilizer(classify=counter.classify)
    cap = open_source(args)
    writer, log, fps, frame_no = None, [], 0.0, 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        coins = counter.process(frame_rgb)
        stable = stab.update(frame_rgb, coins)
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-3) if fps else 1 / max(time.perf_counter() - t0, 1e-3)

        log.append({"snimek": frame_no, "cas_s": round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 3),
                    "surovy": sum(c["value"] for c in coins), "stabilni": stable,
                    "minci": len(coins), "potvrzenych": len(stab.confirmed_tracks()), "pohyb": stab.moving})
        frame_no += 1

        draw(frame_bgr, coins, stab, fps, counter.px_per_mm)
        if args.record:
            if writer is None:
                h, w = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            writer.write(frame_bgr)
        if args.no_window:
            continue

        cv2.imshow("Pocitani minci", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("k"):
            px = counter.calibrate(frame_rgb, args.calib_value)
            if px:
                save_calibration(args.calibration, px)
                print(f"kalibrace: {px:.3f} px/mm uložena do {args.calibration}")
            else:
                print("kalibrace: na podložce nejsou žádné mince")
        if key == ord("s"):
            path = ROOT / "results" / f"07_snimek_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame_bgr)
            print("uloženo", path)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if args.log:
        pd.DataFrame(log).to_csv(args.log, index=False)


if __name__ == "__main__":
    main()
