import cv2

# Převedeme barevný obrázek go HSV
def to_v(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]

# Získáme jasový kanal
def preprocess(img_rgb, ksize=5):
    return cv2.medianBlur(to_v(img_rgb), ksize)
