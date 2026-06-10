"""
config.py — shared constants for IntelliTrack.
All colour and threshold values that need to stay in sync across
the Python backend and the Jinja2 template live here.
"""

import colorsys

# ── COCO-80 class names ────────────────────────────────────────────
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
    76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

# ── Visually distinct colour per class (golden-ratio HSV spacing) ──
def _gen_colors(n: int):
    bgr_list, hex_list = [], []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.92)
        ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
        bgr_list.append((bi, gi, ri))          # OpenCV uses BGR
        hex_list.append(f"#{ri:02x}{gi:02x}{bi:02x}")
    return bgr_list, hex_list

CLASS_COLORS_BGR, CLASS_COLORS_HEX = _gen_colors(80)

# Convenience: class-name → hex (used in Jinja template)
CLASS_COLOR_MAP: dict[str, str] = {
    name: CLASS_COLORS_HEX[idx] for idx, name in COCO_CLASSES.items()
}

# ── Detection defaults ─────────────────────────────────────────────
DEFAULT_CONF    = 0.40   # confidence threshold
DEFAULT_IOU     = 0.45   # NMS IoU threshold
TRAIL_LENGTH    = 35     # positions kept per object trail
MODEL_PATH      = "yolov8n.pt"

# ── Capture defaults ───────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 80
TARGET_FPS   = 30
