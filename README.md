# TundraNVR

Local building CCTV monitor: **fixed camera or video file → Edge (Pattern of Life) → Node (what is it) → Hub (what is it doing) → operator**. One process, one GUI; the Monitor page swaps seats so you can see how the ladder feeds forward.

One camera or one sample clip is enough. Samples are **static** security-camera views (entrance, corridor, lobby, parking), not handheld or moving shots.

## Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.12 |
| API / UI | FastAPI + static HTML |
| Ingest | OpenCV `VideoCapture` |
| Motion | OpenCV frame differencing + occupancy grid |
| Detection | Ultralytics YOLO nano on CPU, plus a UAV/drone model |
| Pattern of Life | Per-camera visual baseline and usual-motion grid (Edge) |
| Scene notes | Ollama, OpenAI vision, or YOLO fallback — Hub only |
| Alerts | Edge unusualness, class priors (drone/aircraft), unattended bag; operator confirm/dismiss |
| Storage | SQLite + filesystem clips/thumbs + per-camera PoL JSON |
| Live view | MJPEG / JPEG over HTTP (`?seat=edge|node|hub`)
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

Then open http://127.0.0.1:8000 — Monitor — and http://127.0.0.1:8000/events for Activity. On Monitor, switch **Raspberry / Node / Hub** to sit at each stage of the same pipeline.

| Route | Role |
| --- | --- |
| `GET /health` | ingest status, last detections, handoff strip, PoL state, version |
| `GET /api/frame.jpg?seat=edge\|node\|hub` | latest JPEG for that seat |
| `GET /api/stream.mjpg?seat=edge\|node\|hub` | live MJPEG for that seat |
| `GET /api/events` | recent events JSON (`?alerts=true` for operator-paged only) |
| `POST /api/events/{id}/review` | `{ "action": "confirm" \| "dismiss" }` — dismiss trains PoL |
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

## Escalation (proof of concept)

The live page is one GUI with three seats on the same process:

1. **Raspberry (Edge)** — motion + Pattern of Life. Usual frames would not leave the Pi. Overlay is an occupancy grid, not class names.
2. **Node** — names objects on Edge trips (or on a policy skip: drone, aircraft, unattended bag). Detector cards live here.
3. **Hub** — captions *what it is doing* only when Node cannot close the packet.
4. **Operator** — Activity: confirm (real incident) or dismiss (fold into this camera’s baseline).

A compact handoff strip shows the climb. Click a stage to jump to that seat.

Each camera source has its own profile under `data/pol/`. The first ~80 detect ticks are **learning** (Edge is jumpy and uploads). After that, doorway traffic that matches the grid is kept locally. Dismissed events update that profile; confirmed ones do not.

## Scene log and alerts

After a Hub escalation, a short caption is written to the event’s `summary` and shown on `/events`. Node-closed packets get a short “named and closed” note without calling a VLM.

With `vision.provider: auto` the app tries, in order:

1. A local Ollama vision model (`ollama pull moondream`)
2. OpenAI if `OPENAI_API_KEY` is in the environment
3. A YOLO-based sentence from the detected classes (always available)

Class priors still force a climb even if Edge looks usual:

- `drone` or `airplane` → Node + Hub + operator
- bag class with no person → unattended bag
- any other class not in `monitoring.expected_classes` → unexpected

Pattern of Life is the main gate, not a pile of geometry rules.

## Public webcams

The live page still lists official [NYC DOT](https://webcams.nyctmc.org/) JPEG stills as optional live sources. Those are street cameras, not building CCTV. JPEG still URLs are re-fetched about every 1.5 seconds.

Do not point the source at random unsecured IP cameras. Stick to feeds the operator publishes.

## Pipeline

Motion without a matching class does **not** create events. Edge-usual frames with no policy skip do not create events either (they would stay on the Pi). A Node-received packet writes an event, a JPEG thumb, a short MP4 clip (H.264 via ffmpeg when available), a scene note if Hub ran, and an optional operator flag. If the stream drops, ingest reopens the capture after a short backoff. JPEG still URLs are re-fetched on an interval instead of treated as end-of-file.

Entrance/corridor/lobby clips are from the EC CAVIAR project (IST 2001 37540).

## Non-goals

Multi-camera scaling, home-automation integrations, WebRTC, continuous 24/7 recording, zone/mask editors, face/LPR/semantic search.
