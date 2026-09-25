"""
Smart Video Analytics — Flask backend (pretrained YOLO26, no training)

Three analytics modes on any uploaded video:
  - count : live people count + tracking IDs
  - line  : cumulative in/out line-crossing counter
  - zone  : restricted-zone intrusion detection

Run:
    pip install -r requirements.txt
    python app.py
Open http://127.0.0.1:5000
"""

import os
import uuid
import subprocess
import threading

import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS    = os.path.join(BASE_DIR, "weights", "yolo26n.pt")  # auto-downloads if absent
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIDEO_EXT  = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CONF       = 0.3

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Load pretrained models ONCE at startup. Bigger model = better detections =
# fewer track breaks = fewer ID switches. We keep two and pick per-request.
print("Loading pretrained YOLO26 models...")
MODELS = {}
def get_model(quality):
    """quality: 'fast' -> yolo26n, 'accurate' -> yolo26s. Cached after first load."""
    name = "yolo26s.pt" if quality == "accurate" else "yolo26n.pt"
    if name not in MODELS:
        local = os.path.join(BASE_DIR, "weights", name)
        MODELS[name] = YOLO(local if os.path.exists(local) else name)
    return MODELS[name]

model = get_model("accurate")   # warm the default (yolo26s) at startup
print("Ready. Classes:", len(model.names))

PERSON = 0
JOBS = {}
MIN_TRACK_FRAMES = 5   # an ID must persist this many frames to count as a real person


def allowed(fn):
    return os.path.splitext(fn)[1].lower() in VIDEO_EXT


def encode_h264(src, dst):
    """Re-encode to browser-friendly H.264; fall back to copy if ffmpeg missing."""
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        import shutil
        shutil.copy(src, dst)


def video_props(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return w, h, fps


def run_analytics(job_id, in_path, out_name, mode):
    try:
        JOBS[job_id]["status"] = "processing"
        w, h, fps = video_props(in_path)
        raw = os.path.join(OUTPUT_DIR, f"{job_id}_raw.mp4")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        unique_ids = set()
        confirmed  = set()          # ids seen >= MIN_TRACK_FRAMES (real people)
        seen_count = {}             # id -> how many frames it has appeared
        peak = 0
        line_in = line_out = 0
        max_intruders = 0

        # ---------------- geometry ----------------
        # LINE-CROSSING (region gate): instead of a fragile single line, we use a
        # counting REGION. A person is tallied once when they ENTER the region and
        # once when they EXIT it — direction-agnostic and robust to people milling
        # around or moving diagonally. This is how real footfall systems work.
        gx1, gy1 = int(w * 0.30), int(h * 0.30)
        gx2, gy2 = int(w * 0.70), int(h * 0.80)
        gate_pts = np.array([(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2)], dtype=np.int32)

        inside_state = {}   # track id -> was inside the gate last frame?
        gate_seen    = set()  # confirmed ids that have been inside at least once
        zone_seen    = set()  # confirmed ids that have breached the restricted zone

        # ZONE (intrusion): a proportional, centered rectangle. Scales to any size.
        zx1, zy1 = int(w * 0.20), int(h * 0.18)
        zx2, zy2 = int(w * 0.80), int(h * 0.88)
        zone_pts = np.array([(zx1, zy1), (zx2, zy1), (zx2, zy2), (zx1, zy2)], dtype=np.int32)

        def in_gate(cx, cy):
            return cv2.pointPolygonTest(gate_pts, (float(cx), float(cy)), False) >= 0

        # tracker config path (bundled BoT-SORT with ReID + long buffer)
        tracker_cfg = os.path.join(BASE_DIR, "botsort_robust.yaml")
        if not os.path.exists(tracker_cfg):
            tracker_cfg = "botsort.yaml"

        # pick model by requested quality (accurate=yolo26s default, fast=yolo26n)
        quality = JOBS[job_id].get("quality", "accurate")
        mdl = get_model(quality)

        prev_metric = None   # for the live count-change HUD

        for r in mdl.track(in_path, stream=True, persist=True,
                           tracker=tracker_cfg, classes=[PERSON],
                           conf=CONF, verbose=False):
            frame = r.plot(line_width=2)

            ids = [] if r.boxes.id is None else r.boxes.id.int().cpu().tolist()
            boxes = [] if r.boxes.id is None else r.boxes.xyxy.cpu().numpy()

            # ---- settling filter: an ID must persist a few frames to be "real" ----
            # This removes the flicker IDs that appear for 1-3 frames during
            # occlusions and would otherwise inflate the unique count.
            for tid in ids:
                seen_count[tid] = seen_count.get(tid, 0) + 1
                if seen_count[tid] >= MIN_TRACK_FRAMES:
                    confirmed.add(tid)
            # only confirmed ids count toward "in frame" and totals
            conf_ids   = [t for t in ids if t in confirmed]
            conf_boxes = [b for t, b in zip(ids, boxes) if t in confirmed]
            n = len(conf_ids)
            peak = max(peak, n)
            unique_ids = confirmed          # unique = confirmed people only

            intruders = 0
            gate_now = 0
            for tid, box in zip(conf_ids, conf_boxes):
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2

                if mode == "line":
                    now_in = in_gate(cx, cy)
                    if now_in:
                        gate_now += 1
                    was_in = inside_state.get(tid, False)
                    if now_in and not was_in:
                        line_in += 1        # entered the region
                        gate_seen.add(tid)
                    elif was_in and not now_in:
                        line_out += 1       # exited the region
                    inside_state[tid] = now_in

                elif mode == "zone":
                    if cv2.pointPolygonTest(zone_pts, (float(cx), float(cy)), False) >= 0:
                        intruders += 1
                        zone_seen.add(tid)   # unique people who ever breached

            # ---------------- overlays ----------------
            # every mode reports a CURRENT value (right now) and a TOTAL (cumulative)
            if mode == "count":
                cur, cur_lbl = n, "IN FRAME"
                tot, tot_lbl = len(unique_ids), "TOTAL SEEN"
                _banner(frame, f"In frame: {n}    Unique: {len(unique_ids)}",
                        (0, 255, 120))

            elif mode == "line":
                _draw_gate(frame, gate_pts, gate_now)
                cur, cur_lbl = gate_now, "INSIDE NOW"
                tot, tot_lbl = len(gate_seen), "PASSED THRU"
                _banner(frame, f"ENTERED {line_in}    EXITED {line_out}    INSIDE {gate_now}",
                        (255, 170, 40))

            elif mode == "zone":
                max_intruders = max(max_intruders, intruders)
                breached = intruders > 0
                _draw_zone(frame, zone_pts, breached, intruders)
                cur, cur_lbl = intruders, "INTRUDERS NOW"
                tot, tot_lbl = len(zone_seen), "TOTAL BREACHED"

            # live HUD (top-left): current count with up/down change + total below
            _draw_hud(frame, cur_lbl, cur, prev_metric, tot_lbl, tot)
            prev_metric = cur

            writer.write(frame)

        writer.release()
        final = os.path.join(OUTPUT_DIR, out_name)
        encode_h264(raw, final)
        try: os.remove(raw)
        except OSError: pass

        stats = {"unique_people": len(unique_ids), "peak_in_frame": peak}
        if mode == "line":
            stats.update(entered=line_in, exited=line_out,
                         passed_through=len(gate_seen))
        if mode == "zone":
            stats.update(total_breached=len(zone_seen), max_intruders=max_intruders)

        JOBS[job_id].update(status="done", output=out_name, mode=mode, stats=stats)
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


def _banner(frame, text, color):
    """Info strip along the BOTTOM of the frame (top-left is used by the HUD)."""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 46), (min(len(text)*15+30, frame.shape[1]), h), (10, 12, 16), -1)
    cv2.putText(frame, text, (16, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _draw_hud(frame, cur_label, cur_value, prev, tot_label, tot_value):
    """Top-left HUD: big CURRENT count (with up/down change arrow) + TOTAL below."""
    pw, ph = 270, 118
    x, y = 16, 16
    panel = frame.copy()
    cv2.rectangle(panel, (x, y), (x + pw, y + ph), (12, 14, 20), -1)
    cv2.addWeighted(panel, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), (60, 74, 100), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + 4, y + ph), (255, 170, 40), -1)   # accent bar

    # --- current (big) ---
    cv2.putText(frame, cur_label, (x + 18, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 170, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, str(cur_value), (x + 16, y + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 245, 255), 3, cv2.LINE_AA)
    if prev is not None and cur_value != prev:
        up = cur_value > prev
        col = (90, 220, 110) if up else (80, 90, 240)     # green up / red down
        cv2.putText(frame, ("^" if up else "v") + f" {abs(cur_value - prev)}",
                    (x + 150, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

    # --- total (smaller, below, divider line) ---
    cv2.line(frame, (x + 14, y + 82), (x + pw - 14, y + 82), (50, 62, 88), 1, cv2.LINE_AA)
    cv2.putText(frame, tot_label, (x + 18, y + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 170, 200), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(str(tot_value), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, str(tot_value), (x + pw - tw - 16, y + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 120), 2, cv2.LINE_AA)


def _draw_gate(frame, pts, now_inside):
    """A professional counting gate (region): translucent fill, border,
    corner brackets, and a header showing current occupancy."""
    amber = (40, 170, 255)     # BGR
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], amber)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [pts], True, amber, 2, cv2.LINE_AA)

    x1, y1 = pts[0]; x2, y2 = pts[2]
    L = max(20, int((x2 - x1) * 0.07))
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx+dx*L, cy), amber, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy+dy*L), amber, 4, cv2.LINE_AA)

    label = f"COUNTING GATE  -  {now_inside} INSIDE"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 20, y1), amber, -1)
    cv2.putText(frame, label, (x1 + 10, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (12, 14, 18), 2, cv2.LINE_AA)


def _draw_zone(frame, pts, breached, count):
    """A professional restricted zone: translucent fill + border + labeled header."""
    green = (120, 210, 90)      # clear  (BGR)
    red   = (60, 60, 235)       # breach (BGR)
    color = red if breached else green

    # translucent fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

    # border
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

    # corner accents (thicker) for a "targeted area" look
    x1, y1 = pts[0]; x2, y2 = pts[2]
    L = max(18, int((x2 - x1) * 0.06))
    for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx+dx*L, cy), color, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy+dy*L), color, 4, cv2.LINE_AA)

    # header label pinned to the zone's top-left
    label = f"RESTRICTED  -  {count} INSIDE" if breached else "RESTRICTED  -  CLEAR"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 20, y1), color, -1)
    cv2.putText(frame, label, (x1 + 10, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (12, 14, 18), 2, cv2.LINE_AA)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("media")
    mode = request.form.get("mode", "count")
    if file is None or file.filename == "":
        return jsonify(error="No video selected"), 400
    if not allowed(file.filename):
        return jsonify(error="Please upload a video file"), 400
    if mode not in {"count", "line", "zone"}:
        mode = "count"

    job_id = uuid.uuid4().hex[:12]
    safe = secure_filename(file.filename)
    in_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe}")
    file.save(in_path)

    out_name = f"{job_id}_result.mp4"
    quality = request.form.get("quality", "accurate")
    if quality not in {"fast", "accurate"}:
        quality = "accurate"
    JOBS[job_id] = {"status": "queued", "mode": mode, "quality": quality}
    threading.Thread(target=run_analytics,
                     args=(job_id, in_path, out_name, mode)).start()
    return jsonify(job_id=job_id, mode=mode)


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    resp = dict(job)
    if job.get("status") == "done":
        resp["video_url"] = url_for("result_file", filename=job["output"])
    return jsonify(resp)


@app.route("/outputs/<path:filename>")
def result_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)