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
| Scene notes | Ollama, OpenAI vision, or YOLO fallback |
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

Download sample clips (city street, parking lot, cars, wildlife, livestock, aircraft, drone, indoor people) so you can run without a camera:

```bash
python scripts/download_sample.py
```

Default source is `data/samples/city.mp4`. The live page lists each clip as a button, plus official NYC DOT traffic stills for a live outdoor test. Or point `camera.source` at an RTSP URL, HTTP stream, JPEG still URL, local file, or camera index (`0`).

## Run

```bash
python -m app.main
```

Then open http://127.0.0.1:8000 — live view — and http://127.0.0.1:8000/events for history. The header shows the app version.

| Route | Role |
| --- | --- |
| `GET /health` | ingest status, fps, last motion/detections, app version |
| `GET /api/frame.jpg` | latest JPEG (with overlay) |
| `GET /api/stream.mjpg` | live MJPEG |
| `GET /api/events` | recent events JSON |
| `GET /api/settings` | current video source and YOLO model |
| `PUT /api/settings` | set source/model, persist, restart pipeline |
| `GET /media/...` | thumbs and clips |

YOLO downloads `yolov8n.pt` on first detection run.

## Config keys

See [`config.yaml`](config.yaml). The live page **Apply** button writes `data/settings.json`, which overrides `camera.source` and `detection.model` until you delete that file.

- `camera.source` — file, URL, or index; files loop when `loop_file` is true. Also editable on the live page.
- `pipeline.detect_fps` — detection rate; extra frames are dropped
- `motion.min_area` — ignore small pixel changes
- `detection.model` — Ultralytics weights (`yolov8n.pt` by default). Also editable on the live page.
- `detection.classes` — allowlist (people, vehicles, aircraft, common animals). Stock YOLO has no drone class.
- `events.pre_seconds` / `post_seconds` / `cooldown_seconds` — clip window and anti-flood
- `events.retention_days` — delete old events, thumbs, and clips
- `server.host` / `server.port`
- `vision.enabled` / `vision.provider` — scene notes after each event (`auto`, `ollama`, `openai`, or `off`)
- `vision.ollama_url` / `vision.ollama_model` — local vision model (default `moondream` at `http://127.0.0.1:11434`)
- `vision.openai_model` — used when `OPENAI_API_KEY` is set (default `gpt-4o-mini`)

## Scene notes

After an event is saved, a short caption is written to the event’s `summary` and shown on `/events`. With `vision.provider: auto` the app tries, in order:

1. A local Ollama vision model (`ollama pull moondream`)
2. OpenAI if `OPENAI_API_KEY` is in the environment
3. A YOLO-based sentence from the detected classes (always available; no extra install)

This is a still-image caption, not a full video narrative. Set `vision.provider: off` to skip the LLM and keep only the YOLO note.

## Public webcams

The live page lists official [NYC DOT](https://webcams.nyctmc.org/) JPEG stills. Those URLs are single frames, not video streams; ingest re-fetches about every 1.5 seconds. They are public traffic cameras, so expect road scenes, not a private property feed.

Do not point the source at random unsecured IP cameras. Stick to feeds the operator publishes.

## Pipeline

Motion without a matching class does **not** create events. A matching detection writes an event, a JPEG thumb, a short MP4 clip (H.264 via ffmpeg when available), and a scene note. If the stream drops, ingest reopens the capture after a short backoff. JPEG still URLs are re-fetched on an interval instead of treated as end-of-file.

## Non-goals

Multi-camera scaling, home-automation integrations, WebRTC, continuous 24/7 recording, zone/mask editors, face/LPR/semantic search.
