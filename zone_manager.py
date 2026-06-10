"""
zone_manager.py — Virtual-zone and tripwire analytics.

Zones  : convex or concave polygons defined in normalised [0,1] coords.
         Emits zone_entry / zone_exit events when a tracked object moves
         across the polygon boundary.

Tripwires: directed line segments.  Emits wire_in / wire_out events when an
           object's centroid crosses the line (direction determined by the
           sign of the cross-product with the line vector).

All coordinates stored as normalised floats so they stay valid regardless
of the actual frame resolution used at draw time.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


class ZoneManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        # --- Zone storage ---
        # zone_name → np.ndarray shape (N,2) normalised [0,1]
        self._zones: dict[str, np.ndarray] = {}

        # zone_name → set of object IDs currently inside
        self._zone_occupants: dict[str, set[int]] = {}
        # zone_name → cumulative entry count
        self._zone_total: dict[str, int] = {}

        # --- Tripwire storage ---
        # wire_name → (p1_norm, p2_norm)  each shape (2,)
        self._wires: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._wire_in:  dict[str, int] = {}
        self._wire_out: dict[str, int] = {}

        # Previous centroid per object (for tripwire crossing)
        self._prev_pos: dict[int, tuple[float, float]] = {}

        # Recent events (ring buffer, last 200)
        self._events: list[dict] = []

    # ── Zone CRUD ─────────────────────────────────────────────────

    def add_zone(self, name: str, points_norm: list[list[float]]) -> None:
        with self._lock:
            self._zones[name] = np.array(points_norm, dtype=np.float32)
            self._zone_occupants[name] = set()
            self._zone_total[name] = 0

    def remove_zone(self, name: str) -> None:
        with self._lock:
            for d in [self._zones, self._zone_occupants, self._zone_total]:
                d.pop(name, None)

    def list_zones(self) -> list[str]:
        with self._lock:
            return list(self._zones.keys())

    # ── Tripwire CRUD ─────────────────────────────────────────────

    def add_tripwire(
        self,
        name: str,
        p1_norm: list[float],
        p2_norm: list[float],
    ) -> None:
        with self._lock:
            self._wires[name] = (
                np.array(p1_norm, dtype=np.float32),
                np.array(p2_norm, dtype=np.float32),
            )
            self._wire_in[name]  = 0
            self._wire_out[name] = 0

    def remove_tripwire(self, name: str) -> None:
        with self._lock:
            for d in [self._wires, self._wire_in, self._wire_out]:
                d.pop(name, None)

    def list_tripwires(self) -> list[str]:
        with self._lock:
            return list(self._wires.keys())

    # ── Per-frame update ─────────────────────────────────────────

    def update(self, detections: list[dict]) -> None:
        """Call once per processed frame with the detector's detections list."""
        now_str = datetime.now().strftime("%H:%M:%S")

        with self._lock:
            current_ids = {d["id"] for d in detections if d["id"] >= 0}

            for det in detections:
                oid = det["id"]
                if oid < 0:
                    continue

                cx_n, cy_n = det["cx_norm"], det["cy_norm"]
                cls = det["class"]

                # ---- Zone entry / exit --------------------------------
                for z_name, pts in self._zones.items():
                    # pointPolygonTest works purely in normalised coords
                    inside = (
                        cv2.pointPolygonTest(
                            pts.reshape(-1, 1, 2),
                            (float(cx_n), float(cy_n)),
                            False,
                        ) >= 0
                    )
                    was_inside = oid in self._zone_occupants.get(z_name, set())

                    if inside and not was_inside:
                        self._zone_occupants[z_name].add(oid)
                        self._zone_total[z_name] = (
                            self._zone_total.get(z_name, 0) + 1
                        )
                        self._push_event({
                            "time": now_str,
                            "type": "zone_entry",
                            "zone": z_name,
                            "class": cls,
                            "id": oid,
                        })

                    elif not inside and was_inside:
                        self._zone_occupants[z_name].discard(oid)
                        self._push_event({
                            "time": now_str,
                            "type": "zone_exit",
                            "zone": z_name,
                            "class": cls,
                            "id": oid,
                        })

                # ---- Tripwire crossing --------------------------------
                if oid in self._prev_pos:
                    px, py = self._prev_pos[oid]
                    for w_name, (p1, p2) in self._wires.items():
                        prev_side = _cross(px, py, p1, p2)
                        curr_side = _cross(cx_n, cy_n, p1, p2)
                        if prev_side * curr_side < 0:          # crossed
                            direction = "wire_in" if curr_side > 0 else "wire_out"
                            if direction == "wire_in":
                                self._wire_in[w_name] = (
                                    self._wire_in.get(w_name, 0) + 1
                                )
                            else:
                                self._wire_out[w_name] = (
                                    self._wire_out.get(w_name, 0) + 1
                                )
                            self._push_event({
                                "time": now_str,
                                "type": direction,
                                "zone": w_name,
                                "class": cls,
                                "id": oid,
                            })

                self._prev_pos[oid] = (cx_n, cy_n)

            # Clean up objects that have left the scene
            gone = set(self._prev_pos) - current_ids
            for oid in gone:
                del self._prev_pos[oid]
                for occ in self._zone_occupants.values():
                    occ.discard(oid)

    # ── Draw overlays on frame ────────────────────────────────────

    def draw_on_frame(self, frame: np.ndarray) -> None:
        """Mutate *frame* in-place: draw zones (filled) and tripwires."""
        h, w = frame.shape[:2]

        with self._lock:
            for z_name, pts_n in self._zones.items():
                pts_px = (pts_n * np.array([w, h])).astype(np.int32)

                # Semi-transparent fill
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_px], (0, 230, 118))
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
                cv2.polylines(frame, [pts_px], True, (0, 230, 118), 2, cv2.LINE_AA)

                # Label at centroid
                cx_px = int(pts_px[:, 0].mean())
                cy_px = int(pts_px[:, 1].mean())
                now_c  = len(self._zone_occupants.get(z_name, set()))
                total  = self._zone_total.get(z_name, 0)
                _put_chip(frame, f"{z_name}  {now_c} | {total}",
                          cx_px - 50, cy_px, (0, 230, 118))

            for w_name, (p1_n, p2_n) in self._wires.items():
                p1 = (int(p1_n[0] * w), int(p1_n[1] * h))
                p2 = (int(p2_n[0] * w), int(p2_n[1] * h))
                cv2.line(frame, p1, p2, (60, 60, 255), 3, cv2.LINE_AA)

                cnt_in  = self._wire_in.get(w_name, 0)
                cnt_out = self._wire_out.get(w_name, 0)
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                _put_chip(frame, f"{w_name}  ↑{cnt_in} ↓{cnt_out}",
                          mx + 6, my - 6, (100, 100, 255))

    # ── Stats snapshot ────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            zones = {
                name: {
                    "current": len(self._zone_occupants.get(name, set())),
                    "total":   self._zone_total.get(name, 0),
                    "points":  pts.tolist(),
                }
                for name, pts in self._zones.items()
            }
            wires = {
                name: {
                    "in":  self._wire_in.get(name, 0),
                    "out": self._wire_out.get(name, 0),
                    "points": [p.tolist() for p in self._wires[name]],
                }
                for name in self._wires
            }
            return {
                "zones":     zones,
                "tripwires": wires,
                "events":    list(self._events[-20:]),
            }

    def reset_tracking(self) -> None:
        """Clear per-object state (call when the video source changes)."""
        with self._lock:
            self._prev_pos.clear()
            for occ in self._zone_occupants.values():
                occ.clear()

    # ── Internal helpers ─────────────────────────────────────────

    def _push_event(self, ev: dict) -> None:
        self._events.append(ev)
        if len(self._events) > 200:
            self._events = self._events[-100:]


# ── Module-level helper (no lock needed) ─────────────────────────

def _cross(
    px: float, py: float,
    a: np.ndarray, b: np.ndarray,
) -> float:
    """Signed cross-product: sign tells which side of line AB point P is on."""
    return float((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]))


def _put_chip(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int],
) -> None:
    font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    (tw, th), bl = cv2.getTextSize(text, font, fscale, fthick)
    pad = 4
    cv2.rectangle(frame, (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad), (18, 18, 18), -1)
    cv2.putText(frame, text, (x, y), font, fscale, color, fthick, cv2.LINE_AA)
