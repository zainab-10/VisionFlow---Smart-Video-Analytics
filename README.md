<div align="center">

# 👁️ VisionFlow — Smart Video Analytics

**People counting, line-crossing & restricted-zone intrusion detection — powered by a pretrained YOLO26 + BoT-SORT tracker. Zero training required.**

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.0-black)](https://flask.palletsprojects.com/)
[![Ultralytics](https://img.shields.io/badge/ultralytics-YOLO26-purple)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

[Features](#-features) • [Demo](#-demo) • [Quickstart](#-quickstart) • [How it works](#-how-it-works) • [Project structure](#-project-structure) • [Reuse it](#-reuse-it-for-other-use-cases)

</div>

---

## 🚀 Overview

VisionFlow turns any uploaded video into live analytics using a **pretrained** YOLO26 model — no dataset collection, no labeling, no training loop. YOLO26 already recognizes `person` and 79 other COCO classes out of the box. VisionFlow adds **multi-object tracking** (persistent IDs across frames) on top and layers three real-world analytics products over it, all through a simple web dashboard.

## ✨ Features

| Mode | What it does | Real-world use case |
|---|---|---|
| 🧍 **People Counting** | Live per-frame count + confirmed unique visitors across the whole clip | Occupancy limits, crowd monitoring |
| 🚦 **Line / Gate Crossing** | Cumulative in/out tally through a counting region, direction-agnostic | Retail footfall, capacity tracking |
| 🚫 **Zone Intrusion** | Flags anyone whose position enters a restricted polygon | Perimeter security, no-go areas |

Other things baked in:
- **Persistent ID tracking** via BoT-SORT with ReID (`botsort_robust.yaml`), tuned to minimize ID switches in crowded, occluded scenes.
- **ID "settling" filter** — a track must persist several frames before it counts as a real person, filtering out flicker IDs from partial detections.
- **Fast / Accurate model toggle** — swap between `yolo26n` (speed) and `yolo26s` (accuracy) per job.
- **Browser-friendly output** — results are re-encoded to H.264 automatically (falls back gracefully if `ffmpeg` isn't installed).
- **Async job processing** — uploads are processed in a background thread with live status polling, so the UI never blocks.

## 🎬 Demo

https://drive.google.com/file/d/1NlbpwX-0IqkJm4c6gGmbiixmtMXfTZl0/view?usp=sharing

> Upload `VisionFlow_Smart_Video_Analytics_demo.mp4` directly into a GitHub issue or the README editor on github.com — GitHub will host it and hand you an embeddable `user-attachments` link to drop in above. (Direct `.mp4` links don't autoplay/embed in Markdown, so this is the standard way to show a demo video in a README.)

## 🧩 Tech Stack

- **Detection & Tracking:** [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) + BoT-SORT (ReID)
- **Backend:** Flask
- **Geometry:** Shapely / OpenCV (`pointPolygonTest`)
- **Frontend:** HTML/CSS/JS dashboard (`templates/index.html`)
- **Video I/O:** OpenCV + ffmpeg

## ⚡ Quickstart

### 1. Clone & install
```bash
git clone https://github.com/<your-username>/visionflow-smart-video-analytics.git
cd visionflow-smart-video-analytics
pip install -r requirements.txt
```
> Use **Python 3.11 or 3.12** — CUDA wheels aren't published for 3.14 yet.

### 2. (Optional) GPU acceleration
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ffmpeg (for reliable in-browser playback)
| OS | Command |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

### 4. Run
```bash
python app.py
```
Open **http://127.0.0.1:5000**, pick a mode, drop in a video with people, and hit **Run analysis**.

Need a quick test clip? Grab a free one from [Pexels — people videos](https://www.pexels.com/search/videos/people/).

### 5. (Optional) Explore it in a notebook first
`Smart_Video_Analytics_YOLO26.ipynb` walks through the same pipeline step-by-step — detection → tracking → unique-ID counting → line crossing → zone intrusion — in Colab-friendly cells. Good for understanding the core loop before touching the Flask app.

## 🛠 How It Works

1. A pretrained model (`yolo26n.pt` fast / `yolo26s.pt` accurate) loads once at startup and downloads automatically on first run.
2. `model.track(..., persist=True, tracker="botsort_robust.yaml")` assigns every person a **stable ID** that survives across frames, occlusions, and brief exits.
3. **Counting unique confirmed IDs** = accurate totals — far more reliable than raw per-frame detection counts.
4. **Line/gate mode** watches each tracked center relative to a counting region and tallies entries/exits — robust to diagonal movement, unlike a single trip-wire line.
5. **Zone mode** checks whether each tracked box center falls inside a restricted polygon (`shapely` / `cv2.pointPolygonTest`) and flags intrusions live.
6. The output is rendered frame-by-frame with overlays, then re-encoded to H.264 for the browser.

### Tracker tuning (`botsort_robust.yaml`)
The bundled BoT-SORT config is tuned specifically to **minimize ID switches in crowded scenes**:
- `new_track_thresh: 0.70` — only spawns a new ID for confident, unmatched detections (prevents flicker IDs).
- `track_buffer: 150` (~5s @ 30fps) — lets a person stay "remembered" through long occlusions and keep their original ID.
- `with_reid: True` + low `appearance_thresh` — leans on appearance re-identification to reconnect a track after someone is hidden and reappears.

## 📁 Project Structure
```
visionflow-smart-video-analytics/
├── app.py                       # Flask backend — tracking + 3 analytics modes
├── botsort_robust.yaml          # Tuned BoT-SORT tracker config (ReID, long buffer)
├── requirements.txt
├── Smart_Video_Analytics_YOLO26.ipynb   # Standalone notebook walkthrough
├── templates/
│   └── index.html               # VisionFlow dashboard frontend
├── weights/                     # yolo26n.pt / yolo26s.pt auto-download here
├── uploads/                     # incoming videos (auto-created)
└── outputs/                     # annotated result videos (auto-created)
```

## 🔁 Reuse It for Other Use Cases

No training required to repurpose this for a different object type — just change the class filter in `app.py` (`classes=[0]` is `person`):

| Classes | Use case |
|---|---|
| `[2]` | Traffic / car counting |
| `[2, 3, 5, 7]` | Mixed vehicle flow (car, motorcycle, bus, truck) |
| any COCO class ID | Whatever that class is |

Same pipeline, same UI, new application.

## 🚢 Production Notes
The bundled Flask dev server and in-memory `JOBS` dict are meant for local/demo use. For a real deployment:
- Serve with **gunicorn + nginx** instead of `flask run` / debug mode.
- Move `JOBS` state to **Redis** (or a DB) so jobs survive restarts and scale across workers.
- Put uploads/outputs on object storage (e.g. S3) instead of local disk for anything beyond a single instance.

## 🗺 Roadmap
- [ ] Configurable line/zone geometry from the UI (currently proportional defaults)
- [ ] Multi-camera support
- [ ] Export analytics as CSV/JSON
- [ ] Dockerfile for one-command deployment

## 📄 License
This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO26 and the tracking/solutions API
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) for the ReID-based tracker

---

<div align="center">
Made with ❤️ using pretrained models — no dataset, no training, just clever layering.
</div>
