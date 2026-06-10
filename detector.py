"""
detector.py — YOLOv8-nano inference wrapper with ByteTrack tracking.

Each call to process_frame():
  1. Runs model.track() with persist=True so ByteTrack keeps object IDs
     stable across frames.
  2. Draws bounding boxes, ID labels, and fading motion trails.
  3. Returns (annotated_frame, detections_list).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config


class Detector:
    def __init__(self, model_path: str = config.MODEL_PATH) -> None:
        print(f"[Detector] Loading {model_path} …")
        self.model = YOLO(model_path)
        print("[Detector] Model ready.")

        # Runtime-tunable settings (written by the Flask /api/config route)
        self.conf: float = config.DEFAULT_CONF
        self.iou: float  = config.DEFAULT_IOU
        self.active_classes: Optional[list[int]] = None  # None = all 80 classes

        # Per-object motion trails  {id: deque[(cx, cy)]}
        self._trails: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.TRAIL_LENGTH)
        )
        # How many consecutive frames an object has been absent
        self._missing: dict[int, int] = {}

        # FPS bookkeeping
        self._fps: int  = 0
        self._fps_t: float = time.time()
        self._fps_n: int   = 0

    # ── Public API ────────────────────────────────────────────────

    @property
    def fps(self) -> int:
        return self._fps

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[dict]]:
        """Run ByteTrack-enabled YOLOv8 on *frame*.

        Returns
        -------
        annotated : np.ndarray  — copy of frame with boxes / trails drawn
        detections : list[dict] — one dict per detected object
        """
        annotated   = frame.copy()
        detections: list[dict] = []

        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.conf,
                iou=self.iou,
                classes=self.active_classes,
                verbose=False,
            )
        except Exception as exc:
            print(f"[Detector] inference error: {exc}")
            self._tick_fps()
            return annotated, detections

        boxes_obj = results[0].boxes
        if boxes_obj is not None and len(boxes_obj) > 0:
            h, w = frame.shape[:2]

            xyxy    = boxes_obj.xyxy.cpu().numpy()
            cls_ids = boxes_obj.cls.cpu().numpy().astype(int)
            confs   = boxes_obj.conf.cpu().numpy()
            ids     = (
                boxes_obj.id.cpu().numpy().astype(int)
                if boxes_obj.id is not None
                else np.full(len(xyxy), -1, dtype=int)
            )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cls_id  = int(cls_ids[i])
                conf    = float(confs[i])
                obj_id  = int(ids[i])
                cls_name = config.COCO_CLASSES.get(cls_id, "unknown")
                color    = config.CLASS_COLORS_BGR[cls_id]
                cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2

                # --- trail ---
                if obj_id >= 0:
                    self._trails[obj_id].append((cx, cy))
                    self._draw_trail(annotated, obj_id, color)

                # --- box + label ---
                self._draw_box(annotated, x1, y1, x2, y2,
                               obj_id, cls_name, conf, color)

                detections.append({
                    "id":       obj_id,
                    "class_id": cls_id,
                    "class":    cls_name,
                    "conf":     round(conf, 3),
                    "box":      [int(x1), int(y1), int(x2), int(y2)],
                    "cx":       int(cx),
                    "cy":       int(cy),
                    "cx_norm":  float(cx) / w,
                    "cy_norm":  float(cy) / h,
                })

        # Age-out stale trails (>90 absent frames ≈ 3 s at 30 fps)
        active = {d["id"] for d in detections if d["id"] >= 0}
        for oid in list(self._trails):
            if oid in active:
                self._missing.pop(oid, None)
            else:
                self._missing[oid] = self._missing.get(oid, 0) + 1
                if self._missing[oid] > 90:
                    del self._trails[oid]
                    del self._missing[oid]

        self._tick_fps()
        self._draw_hud(annotated)
        return annotated, detections

    # ── Drawing helpers ───────────────────────────────────────────

    def _draw_trail(
        self,
        frame: np.ndarray,
        obj_id: int,
        color: tuple[int, int, int],
    ) -> None:
        trail = list(self._trails[obj_id])
        for j in range(1, len(trail)):
            alpha = j / len(trail)
            c = tuple(int(v * alpha * 0.9) for v in color)
            thick = max(1, int(3 * alpha))
            cv2.line(frame, trail[j - 1], trail[j], c, thick, cv2.LINE_AA)

    def _draw_box(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        obj_id: int,
        cls_name: str,
        conf: float,
        color: tuple[int, int, int],
    ) -> None:
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Corner accents (top-left & bottom-right)
        corner = 10
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 3, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 3, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 3, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 3, cv2.LINE_AA)

        # Label
        id_part = f"#{obj_id} " if obj_id >= 0 else ""
        label   = f"{id_part}{cls_name} {conf:.0%}"
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        (tw, th), baseline = cv2.getTextSize(label, font, fscale, fthick)
        pad = 4
        lx1 = x1
        ly1 = max(0, y1 - th - baseline - pad * 2)
        lx2 = x1 + tw + pad * 2
        ly2 = y1

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(
            frame, label,
            (lx1 + pad, ly2 - baseline - pad // 2),
            font, fscale, (10, 10, 10), fthick, cv2.LINE_AA,
        )

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Bottom-right FPS chip."""
        h, w = frame.shape[:2]
        label = f" FPS {self._fps:3d} "
        font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        x, y = w - tw - 8, h - 10
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4),
                      (20, 20, 20), -1)
        cv2.putText(frame, label, (x, y), font, fscale,
                    (0, 230, 118), fthick, cv2.LINE_AA)

    def _tick_fps(self) -> None:
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps   = self._fps_n
            self._fps_n = 0
            self._fps_t = now
