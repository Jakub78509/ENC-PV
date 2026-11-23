#!/bin/bash
# Dvojklik spustí živé počítání mincí (u Jakuba je iPhone kamera 0, MacBook kamera 1).
# Jinou kameru: ./spustit_live.command 1
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif ! python3 -c "import cv2, sklearn, pandas" 2>/dev/null; then
  echo "Vytvářím prostředí .venv a instaluji knihovny (jen poprvé)..."
  python3 -m venv .venv && source .venv/bin/activate && pip install -q -r requirements.txt
fi

{
  echo "=== $(date) ==="
  echo "Kamery podle macOS:"
  system_profiler SPCameraDataType 2>/dev/null
  echo "Kamery, které vidí Python (OpenCV):"
  python3 live.py --list 2>&1
  python3 -c "import cv2, platform; print('OpenCV', cv2.__version__, '| macOS', platform.mac_ver()[0], '| Python', platform.python_version())" 2>&1
} | tee results/07_kamery.txt
CAM="${1:-0}"
if ! python3 live.py --list 2>/dev/null | grep -q "kamera $CAM:"; then CAM=0; fi
echo "Spouštím kameru $CAM (q = konec, k = kalibrace, s = snímek)"
python3 live.py --source "$CAM" --log results/07_kamera.csv --record results/07_kamera.mp4
echo "Hotovo. Záznam: results/07_kamera.csv a results/07_kamera.mp4"
read -n 1 -s -r -p "Stiskni libovolnou klávesu pro zavření..."
