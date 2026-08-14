# TundraNVR

Local camera detection MVP: **camera or video file → motion filter → object detection → save events/clips → simple web UI**.

One camera or one sample video is enough.

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 |
| API / UI | FastAPI + static HTML |
| Ingest | OpenCV `VideoCapture` |
| Motion | OpenCV frame differencing |
| Detection | Ultralytics YOLO nano on CPU |
| Storage | SQLite + filesystem clips/thumbs |
| Live view | MJPEG / JPEG over HTTP |
| Config | `config.yaml` |

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch` installs a CUDA wheel on a CPU-only machine, install the CPU builds first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`torchvision` must come from the same PyTorch CPU index; a PyPI wheel can fail with `operator torchvision::nms does not exist`.

Download a short sample clip (people walking) so you can run without a camera:

```bash
python scripts/download_sample.py
```

Or point `camera.source` in `config.yaml` at an RTSP URL, HTTP stream, local file, or camera index (`0`). The live page can also change source and model without editing the file.

## Run

```bash
python -m app.main
```

Then open http://127.0.0.1:8000 — live view — and http://127.0.0.1:8000/events for history.

| Route | Role |
| --- | --- |
| `GET /health` | ingest status, fps, last motion/detections |
| `GET /api/frame.jpg` | latest JPEG (with overlay) |
| `GET /api/stream.mjpg` | live MJPEG |
| `GET /api/events` | recent events JSON |
| `GET /api/settings` | current video source and YOLO model |
| `PUT /api/settings` | set source/model, persist, restart pipeline |
| `GET /media/...` | thumbs and clips |

YOLO downloads `yolov8n.pt` on first detection run.

## Stay local — do not re-clone

The UI and the model already run on **the same machine** that started `python -m app.main`. Changing source or model in the browser does not hit GitHub and does not re-download the repo. Weights (`yolov8n.pt`, …) sit in the project folder after the first fetch.

Clone **once**. After that:

| You want to… | Do this |
| --- | --- |
| Point at another file, webcam (`0`), or RTSP | Live page → Apply (writes `data/settings.json`) |
| Try a different YOLO size | Live page → pick `yolov8s.pt` / `yolo11n.pt` → Apply (downloads that `.pt` once) |
| Pick up code from GitHub | `./scripts/update.sh` (or `git pull`), then restart the app |
| Reinstall Python packages | Only if `requirements.txt` changed |

Do **not** clone into a new folder each time. That re-downloads torch (~hundreds of MB) and throws away your venv, events, and weights. `git pull` updates a few source files; `.venv/`, `*.pt`, and `data/` stay put.

A split “UI in the cloud, YOLO on the Mac” setup is more moving parts than this MVP needs. Run the app on the MacBook; use the browser on that same machine (http://127.0.0.1:8000). If you edit in Cursor Cloud, pull those commits locally instead of starting from a fresh clone.

## Config keys

See [`config.yaml`](config.yaml). The live page **Apply** button writes `data/settings.json`, which overrides `camera.source` and `detection.model` until you delete that file.

- `camera.source` — file, URL, or index; files loop when `loop_file` is true. Also editable on the live page.
- `pipeline.detect_fps` — detection rate; extra frames are dropped
- `motion.min_area` — ignore small pixel changes
- `detection.model` — Ultralytics weights (`yolov8n.pt` by default). Also editable on the live page.
- `detection.classes` — allowlist (default `person`, `car`, `dog`, `cat`)
- `events.pre_seconds` / `post_seconds` / `cooldown_seconds` — clip window and anti-flood
- `events.retention_days` — delete old events, thumbs, and clips
- `server.host` / `server.port`

## Pipeline

Motion without a matching class does **not** create events. A matching detection writes an event, a JPEG thumb, and a short MP4 clip (H.264 via ffmpeg when available). If the stream drops, ingest reopens the capture after a short backoff.

## Non-goals

Multi-camera scaling, home-automation integrations, WebRTC, continuous 24/7 recording, zone/mask editors, face/LPR/semantic search.
