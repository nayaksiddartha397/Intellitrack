"""
app.py — IntelliTrack Flask server.

Threading model
───────────────
  FrameReader thread : cap.read() in a tight loop → stores latest raw frame
  InferenceWorker    : pops raw frame, runs YOLOv8+ByteTrack, stores annotated
                       frame + stats dict
  generate_frames()  : MJPEG generator — reads annotated frame, encodes JPEG
  emit_stats_loop()  : daemon thread — broadcasts stats via SocketIO every 250 ms

async_mode='threading' — works on Mac + Windows without eventlet/gevent.
NOTE: werkzeug threading does NOT support WebSocket; the JS client is forced
      to use HTTP long-polling (set in socket.io transports option).
"""

from __future__ import annotations

import io
import os
import platform
import threading
import time
import urllib.request
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO

import config
from detector import Detector
from event_logger import EventLogger
from zone_manager import ZoneManager

# ── Flask / SocketIO ──────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "intellitrack-2024"
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
)


# ── VideoProcessor ────────────────────────────────────────────────

class VideoProcessor:
    """Two daemon threads: FrameReader and InferenceWorker."""

    def __init__(self) -> None:
        self.detector: Detector | None = None
        self.zone_mgr = ZoneManager()
        self.logger   = EventLogger()

        self._cap:          cv2.VideoCapture | None = None
        self._raw_frame:    np.ndarray | None       = None
        self._output_frame: np.ndarray | None       = None
        self._stats:        dict                    = {}

        self._raw_lock  = threading.Lock()
        self._out_lock  = threading.Lock()
        self._stat_lock = threading.Lock()

        self.running      = False
        self._frame_count = 0
        self._video_done  = False   # True when a finite video file reaches its last frame

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self, source: int | str = 0) -> None:
        self.running  = True
        self._cap     = _open_capture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.detector = Detector()

        threading.Thread(target=self._reader,    daemon=True, name="FrameReader").start()
        threading.Thread(target=self._inference, daemon=True, name="InferenceWorker").start()

    def restart(self, source: int | str) -> None:
        self.running      = False
        self._video_done  = False   # reset so new source starts fresh
        time.sleep(0.35)
        if self._cap:
            self._cap.release()
        self.zone_mgr.reset_tracking()
        with self._raw_lock:  self._raw_frame    = None
        with self._out_lock:  self._output_frame = None
        self.start(source)

    # ── Reader thread ──────────────────────────────────────────────

    def _reader(self) -> None:
        while self.running:
            if not (self._cap and self._cap.isOpened()):
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()

            if not ret:
                total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    # Finite video file — mark done, hold position, do NOT loop
                    self._video_done = True
                    while self.running and self._video_done:
                        time.sleep(0.1)
                else:
                    # Webcam transient read error — just retry
                    time.sleep(0.01)
                continue

            with self._raw_lock:
                self._raw_frame = frame

    # ── Inference thread ───────────────────────────────────────────

    def _inference(self) -> None:
        log_every = 15
        while self.running:
            with self._raw_lock:
                frame = self._raw_frame

            if frame is None:
                time.sleep(0.01)
                continue

            frame = frame.copy()
            self.zone_mgr.draw_on_frame(frame)

            annotated, detections = self.detector.process_frame(frame)

            self.zone_mgr.update(detections)

            self._frame_count += 1
            if self._frame_count % log_every == 0 and detections:
                self.logger.log_detection_batch(detections)

            with self._out_lock:
                self._output_frame = annotated

            zone_snap    = self.zone_mgr.get_stats()
            class_counts = dict(Counter(d["class"] for d in detections))

            # Video-file progress (total_frames == 0 for webcam)
            cap_pos   = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self._cap else 0
            cap_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else -1
            cap_vfps  = float(self._cap.get(cv2.CAP_PROP_FPS))       if self._cap else 30.0

            with self._stat_lock:
                self._stats = {
                    "fps":          self.detector.fps,
                    "total":        len(detections),
                    "class_counts": class_counts,
                    "zones":        zone_snap["zones"],
                    "tripwires":    zone_snap["tripwires"],
                    "events":       zone_snap["events"],
                    "active_ids":   [d["id"] for d in detections if d["id"] >= 0],
                    "video": {
                        "current_frame": cap_pos,
                        "total_frames":  max(cap_total, 0),
                        "video_fps":     cap_vfps if cap_vfps > 0 else 30.0,
                        "is_file":       cap_total > 0,
                        "done":          self._video_done,
                    },
                }

    # ── Public API ─────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray | None:
        with self._out_lock:
            return None if self._output_frame is None else self._output_frame.copy()

    def get_stats(self) -> dict:
        with self._stat_lock:
            return dict(self._stats)

    def get_dims(self) -> tuple[int, int]:
        if self._cap is None:
            return config.FRAME_WIDTH, config.FRAME_HEIGHT
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or config.FRAME_WIDTH,
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT,
        )


# ── Global processor ──────────────────────────────────────────────
processor = VideoProcessor()


# ── MJPEG generator ───────────────────────────────────────────────

def generate_frames():
    while True:
        frame = processor.get_frame()
        if frame is None:
            time.sleep(0.033)
            continue
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
        )
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / config.TARGET_FPS)


# ── Background stats emitter ──────────────────────────────────────

def emit_stats_loop() -> None:
    """Push stats to all connected socket.io clients every 250 ms."""
    while True:
        try:
            stats = processor.get_stats()
            if stats:                        # skip empty dict on startup
                socketio.emit("stats", stats, namespace="/")
        except Exception as exc:
            print(f"[StatsEmitter] emit error: {exc}")
        time.sleep(0.25)


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        class_color_map=config.CLASS_COLOR_MAP,
        coco_classes=list(config.COCO_CLASSES.values()),
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST stats endpoint (polling fallback for socket.io) ──────────

@app.route("/api/stats")
def api_stats():
    """Lightweight REST endpoint — used when socket.io polling is slow to connect."""
    return jsonify(processor.get_stats())


@app.route("/api/info")
def api_info():
    w, h = processor.get_dims()
    return jsonify({"width": w, "height": h, "model": config.MODEL_PATH})


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True) or {}
    det  = processor.detector
    if det is None:
        return jsonify({"error": "detector not ready"}), 503

    if "conf" in data:
        det.conf = max(0.05, min(0.99, float(data["conf"])))
    if "iou" in data:
        det.iou  = max(0.10, min(0.90, float(data["iou"])))
    if "classes" in data:
        raw = data["classes"]
        det.active_classes = None if (raw is None or raw == []) else [int(c) for c in raw]

    return jsonify({"conf": det.conf, "iou": det.iou, "classes": det.active_classes})


@app.route("/api/source", methods=["POST"])
def api_source():
    data = request.get_json(force=True) or {}
    src  = data.get("source", "0")
    try:    src = int(src)
    except (ValueError, TypeError): pass
    threading.Thread(target=processor.restart, args=(src,), daemon=True).start()
    return jsonify({"status": "restarting", "source": str(src)})


# ── Video file upload ─────────────────────────────────────────────

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB limit


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a video file, save it to uploads/, switch the processor to it."""
    if "file" not in request.files:
        return jsonify({"error": "no file field"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    # Sanitise filename without werkzeug.utils dependency issues
    safe = "".join(
        c if c.isalnum() or c in "._- " else "_"
        for c in os.path.basename(f.filename)
    ).strip() or "upload.mp4"

    dest = os.path.join(UPLOAD_FOLDER, safe)
    f.save(dest)
    print(f"[Upload] saved → {dest}")

    threading.Thread(target=processor.restart, args=(dest,), daemon=True).start()
    return jsonify({"status": "ok", "path": dest, "filename": safe})


# ── Zone endpoints ────────────────────────────────────────────────

@app.route("/api/zones", methods=["GET", "POST", "DELETE"])
def api_zones():
    if request.method == "GET":
        return jsonify(processor.zone_mgr.get_stats()["zones"])

    data = request.get_json(force=True) or {}

    if request.method == "POST":
        name, points = data.get("name", "").strip(), data.get("points", [])
        if not name or len(points) < 3:
            return jsonify({"error": "need name + ≥3 points"}), 400
        processor.zone_mgr.add_zone(name, points)
        return jsonify({"status": "ok", "name": name})

    if request.method == "DELETE":
        processor.zone_mgr.remove_zone(data.get("name", ""))
        return jsonify({"status": "ok"})


# ── Tripwire endpoints ────────────────────────────────────────────

@app.route("/api/tripwires", methods=["GET", "POST", "DELETE"])
def api_tripwires():
    if request.method == "GET":
        return jsonify(processor.zone_mgr.get_stats()["tripwires"])

    data = request.get_json(force=True) or {}

    if request.method == "POST":
        name, points = data.get("name", "").strip(), data.get("points", [])
        if not name or len(points) < 2:
            return jsonify({"error": "need name + 2 points"}), 400
        processor.zone_mgr.add_tripwire(name, points[0], points[1])
        return jsonify({"status": "ok", "name": name})

    if request.method == "DELETE":
        processor.zone_mgr.remove_tripwire(data.get("name", ""))
        return jsonify({"status": "ok"})


# ── Log endpoints ─────────────────────────────────────────────────

@app.route("/api/export")
def api_export():
    csv_str = processor.logger.export_csv()
    buf     = io.BytesIO(csv_str.encode("utf-8"))
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="intellitrack_export.csv")


@app.route("/api/clear_log", methods=["POST"])
def api_clear_log():
    processor.logger.clear_all()
    return jsonify({"status": "cleared"})


@app.route("/api/summary")
def api_summary():
    return jsonify(processor.logger.event_summary())


# ── SocketIO events ───────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    print(f"[SocketIO] client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"[SocketIO] client disconnected: {request.sid}")


# ── Helpers ───────────────────────────────────────────────────────

def _open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, str):
        return cv2.VideoCapture(source)
    os_name = platform.system()
    if os_name == "Windows":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if os_name == "Darwin":
        return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(source)


def _ensure_socketio_client() -> None:
    dest = os.path.join(app.static_folder, "socket.io.min.js")
    if os.path.isfile(dest):
        return
    url = "https://cdn.socket.io/4.6.0/socket.io.min.js"
    print("  [setup] Downloading socket.io client → static/ …")
    try:
        urllib.request.urlretrieve(url, dest)
        print("  [setup] socket.io client ready.")
    except Exception as exc:
        print(f"  [setup] WARNING: socket.io download failed ({exc})")
        print("          The CDN fallback in index.html will be used instead.")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  IntelliTrack — Real-Time Object Detection")
    print("=" * 55)
    print(f"  OS       : {platform.system()} {platform.release()}")
    print(f"  Model    : {config.MODEL_PATH}")
    print(f"  Conf     : {config.DEFAULT_CONF}")

    _ensure_socketio_client()

    print("  Starting video processor …")
    processor.start(source=0)

    threading.Thread(target=emit_stats_loop, daemon=True, name="StatsEmitter").start()

    print("  Server   : http://127.0.0.1:5000")
    print("  Open the URL above in your browser.")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
    )
