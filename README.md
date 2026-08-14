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

Or point `camera.source` in `config.yaml` at an RTSP URL, HTTP stream, local file, or camera index (`0`).

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
| `GET /media/...` | thumbs and clips |

YOLO downloads `yolov8n.pt` on first detection run.

## How to test

There are no automated tests. The check is: start the app on the sample clip, then confirm a detection event (database row + JPEG thumb + MP4 clip) shows up.

### 1. One-time setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_sample.py
```

Confirm `ffmpeg` is on `PATH` (`ffmpeg -version`) so event clips encode as H.264. The default `camera.source` in `config.yaml` is `data/sample.mp4` (people walking, loops).

### 2. Start the app

```bash
source .venv/bin/activate
python -m app.main
```

The API listens on http://127.0.0.1:8000. First run downloads `yolov8n.pt` into the repo root (needs internet) and can take a minute before detections start.

### 3. Confirm ingest is alive

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expect `"status": "ok"` and `"opened": true`. `fps` is ingest rate (the looping sample is paced near the file's native fps). `last_error` should be `null`.

Save a still (useful if the live page is blank for a moment):

```bash
curl -s -o /tmp/frame.jpg http://127.0.0.1:8000/api/frame.jpg
```

`503` means no frame yet; wait a couple of seconds and retry.

### 4. Watch the UI

| Page | What to look for |
| --- | --- |
| http://127.0.0.1:8000 | Live MJPEG. Status should go from `waiting` to `live`. Motion text and class labels (`person …`) appear only for a short TTL after a detection, so one glance can miss them. |
| http://127.0.0.1:8000/events | History table. Click a row to play the clip. |

Detection runs only when motion is present, and only allow-listed classes (`person`, `car`, `dog`, `cat` by default) create events. The sample clip has people, so you should get `person` events within about 10–20 seconds after the model is loaded.

### 5. Confirm an event was saved

Poll until the list is non-empty (cooldown between events is 8 seconds):

```bash
curl -s 'http://127.0.0.1:8000/api/events?limit=5' | python -m json.tool
```

A good event looks like:

- `classes` includes `person`
- `thumb_url` like `/media/thumbs/….jpg`
- `clip_url` like `/media/clips/….mp4`

On disk that is a row in `data/events.db` plus files under `data/thumbs/` and `data/clips/`. Open the clip URL in the browser (or `ffplay` / any player) to check the MP4 plays.

That trio — SQLite row, JPEG thumb, MP4 clip — is the end-to-end signal that the pipeline works.

### 6. Optional: camera or RTSP

Point `camera.source` in `config.yaml` at `0` (webcam), an RTSP/HTTP URL, or another local file, restart, and repeat steps 3–5. Walk in front of the camera (or play a clip with people/cars/pets). Motion without a matching class does **not** create events.

### Gotchas

- `RuntimeError: operator torchvision::nms does not exist` — `torch` and `torchvision` are ABI-mismatched. Reinstall CPU wheels: `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.
- Live boxes / `last_detections` are transient. Prefer `/api/events` and a saved clip over a single screenshot of `/`.
- Empty events after a minute: check `/health` `last_error`, that `data/sample.mp4` exists, and that `detection.classes` still includes `person`.

## Config keys

See [`config.yaml`](config.yaml):

- `camera.source` — file, URL, or index; files loop when `loop_file` is true
- `pipeline.detect_fps` — detection rate; extra frames are dropped
- `motion.min_area` — ignore small pixel changes
- `detection.classes` — allowlist (default `person`, `car`, `dog`, `cat`)
- `events.pre_seconds` / `post_seconds` / `cooldown_seconds` — clip window and anti-flood
- `events.retention_days` — delete old events, thumbs, and clips
- `server.host` / `server.port`

## Pipeline

Motion without a matching class does **not** create events. A matching detection writes an event, a JPEG thumb, and a short MP4 clip (H.264 via ffmpeg when available). If the stream drops, ingest reopens the capture after a short backoff.

## Non-goals

Multi-camera scaling, home-automation integrations, WebRTC, continuous 24/7 recording, zone/mask editors, face/LPR/semantic search.
