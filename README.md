# TundraNVR

Local building CCTV monitor: **fixed camera or video file → motion filter → object detection (people/vehicles + drones) → log what happened → save clips → web UI**.

One camera or one sample clip is enough. Samples are **static** security-camera views (entrance, corridor, lobby, parking), not handheld or moving shots.

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 |
| API / UI | FastAPI + static HTML |
| Ingest | OpenCV `VideoCapture` |
| Motion | OpenCV frame differencing |
| Detection | Ultralytics YOLO nano on CPU, plus a UAV/drone model |
| Scene notes | Ollama, OpenAI vision, or YOLO fallback |
| Alerts | Rule-based anomaly flags (drone, aircraft, unattended bag, unexpected class) |
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

Download static building-CCTV clips and the drone weights:

```bash
python scripts/download_sample.py
```

Default source is `data/samples/entrance.mp4` (CAVIAR building door). The live page lists each clip as a button. Or point `camera.source` at an RTSP URL, HTTP stream, JPEG still URL, local file, or camera index (`0`).

## Run

```bash
python -m app.main
```

Then open http://127.0.0.1:8000 — Monitor — and http://127.0.0.1:8000/events for Activity. Detector choices are explained on the Monitor page.

| Route | Role |
| --- | --- |
| `GET /health` | ingest status, last detections, last scene note, anomaly flag, version |
| `GET /api/frame.jpg` | latest JPEG (with overlay) |
| `GET /api/stream.mjpg` | live MJPEG |
| `GET /api/events` | recent events JSON (`?alerts=true` for flagged only) |
| `GET /api/settings` | current video source and YOLO model |
| `PUT /api/settings` | set source/model, persist, restart pipeline |
| `GET /media/...` | thumbs and clips |

YOLO downloads `yolov8n.pt` on first detection run. `scripts/download_sample.py` also fetches `drone-yolo.pt`.

## Config keys

See [`config.yaml`](config.yaml). The live page **Apply** button writes `data/settings.json`, which overrides `camera.source` and `detection.model` until you delete that file.

- `camera.source` — file, URL, or index; files loop when `loop_file` is true. Also editable on the live page.
- `pipeline.detect_fps` — detection rate; extra frames are dropped
- `motion.min_area` — ignore small pixel changes
- `detection.model` — Ultralytics weights (`yolov8n.pt` by default). Also editable on the live page.
- `detection.drone_model` — second YOLO trained on quadcopters and fixed-wing UAVs (`drone-yolo.pt`)
- `detection.classes` — building allowlist (people, vehicles, bags, pets, drone, airplane)
- `monitoring.expected_classes` — ordinary for a building camera
- `monitoring.alert_classes` — always flagged (`drone`, `airplane`)
- `monitoring.unattended_bags` — flag backpack/handbag/suitcase with no person
- `events.pre_seconds` / `post_seconds` / `cooldown_seconds` — clip window and anti-flood
- `events.retention_days` — delete old events, thumbs, and clips
- `server.host` / `server.port`
- `vision.enabled` / `vision.provider` — scene notes after each event (`auto`, `ollama`, `openai`, or `off`)
- `vision.ollama_url` / `vision.ollama_model` — local vision model (default `moondream` at `http://127.0.0.1:11434`)
- `vision.openai_model` — used when `OPENAI_API_KEY` is set (default `gpt-4o-mini`)

## Scene log and alerts

After an event is saved, a short caption is written to the event’s `summary` and shown on `/events`. The live page also shows the latest `last_scene` line from `/health`.

With `vision.provider: auto` the app tries, in order:

1. A local Ollama vision model (`ollama pull moondream`)
2. OpenAI if `OPENAI_API_KEY` is in the environment
3. A YOLO-based sentence from the detected classes (always available)

Alerts are a first-cut rule check, not a trained anomaly model:

- `drone` or `airplane` → always an alert
- bag class with no person → unattended bag
- any other class not in `monitoring.expected_classes` → unexpected

A learned “this is not how this camera usually looks” model is the next step; the `anomaly` / `anomaly_reason` fields on each event are the place that will plug in.

## Public webcams

The live page still lists official [NYC DOT](https://webcams.nyctmc.org/) JPEG stills as optional live sources. Those are street cameras, not building CCTV. JPEG still URLs are re-fetched about every 1.5 seconds.

Do not point the source at random unsecured IP cameras. Stick to feeds the operator publishes.

## Pipeline

Motion without a matching class does **not** create events. A matching detection writes an event, a JPEG thumb, a short MP4 clip (H.264 via ffmpeg when available), a scene note, and an optional anomaly flag. If the stream drops, ingest reopens the capture after a short backoff. JPEG still URLs are re-fetched on an interval instead of treated as end-of-file.

Entrance/corridor/lobby clips are from the EC CAVIAR project (IST 2001 37540).

## Non-goals

Multi-camera scaling, home-automation integrations, WebRTC, continuous 24/7 recording, zone/mask editors, face/LPR/semantic search.
