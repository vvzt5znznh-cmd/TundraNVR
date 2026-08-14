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

## Do I need a VM?

**No.** Install Git, Python 3.12, and ffmpeg on the machine itself. This is a single Python process plus a browser; a VM only slows CPU detection and makes a webcam harder to use later.

Skip Docker / WSL too for the first test. A laptop with ~8 GB RAM and internet is enough — detection runs on CPU (`yolov8n`). You do **not** need a GPU or a camera; the sample clip is the default source.

## Fresh machine

You need three tools, then the repo.

| Tool | Why |
| --- | --- |
| Git | clone this repo |
| Python 3.12 | run the app (3.11+ is usually fine; 3.12 is what we use) |
| ffmpeg | encode H.264 event clips |

Pick your OS. After that, the Python steps are the same.

### Windows

In **PowerShell as Administrator** (or a normal prompt if `winget` already works):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
```

Close and reopen the terminal so `git`, `python`, and `ffmpeg` are on `PATH`. Confirm:

```powershell
git --version
python --version
ffmpeg -version
```

Python should print 3.12.x. If `python` is missing, try `py -3.12`. If PowerShell blocks venv activation later, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### macOS

Install [Homebrew](https://brew.sh) if you do not have it, then:

```bash
brew install git python@3.12 ffmpeg
python3.12 --version
ffmpeg -version
```

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv python3.12-dev ffmpeg
python3.12 --version
ffmpeg -version
```

On older Ubuntu where `python3.12` is not in the default repos, use the distro `python3` if it is 3.11+, or install 3.12 from [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).

### Clone

```bash
git clone https://github.com/vvzt5znznh-cmd/TundraNVR.git
cd TundraNVR
```

### Python env and sample clip

Always install the **CPU** PyTorch wheels first. A generic `torchvision` from PyPI can fail at detection with `operator torchvision::nms does not exist`.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_sample.py
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_sample.py
```

That last command writes `data/sample.mp4` (people walking). First-time pip will download a few hundred MB (torch). Stay on this machine's disk; you do not need a camera yet.

To use a webcam or RTSP later, set `camera.source` in `config.yaml` to `0`, a URL, or another file.

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

### 1. Start the app

If you just finished **Fresh machine**, skip ahead — the venv, torch CPU wheels, `ffmpeg`, and `data/sample.mp4` are already in place. Otherwise do that section first.

```bash
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m app.main
```

The API listens on http://127.0.0.1:8000. First run downloads `yolov8n.pt` into the repo root (needs internet) and can take a minute before detections start.

### 2. Confirm ingest is alive

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Expect `"status": "ok"` and `"opened": true`. `fps` is ingest rate (the looping sample is paced near the file's native fps). `last_error` should be `null`.

Save a still (useful if the live page is blank for a moment):

```bash
curl -s -o frame.jpg http://127.0.0.1:8000/api/frame.jpg
```

`503` means no frame yet; wait a couple of seconds and retry.

### 3. Watch the UI

| Page | What to look for |
| --- | --- |
| http://127.0.0.1:8000 | Live MJPEG. Status should go from `waiting` to `live`. Motion text and class labels (`person …`) appear only for a short TTL after a detection, so one glance can miss them. |
| http://127.0.0.1:8000/events | History table. Click a row to play the clip. |

Detection runs only when motion is present, and only allow-listed classes (`person`, `car`, `dog`, `cat` by default) create events. The sample clip has people, so you should get `person` events within about 10–20 seconds after the model is loaded.

### 4. Confirm an event was saved

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

### 5. Optional: camera or RTSP

Point `camera.source` in `config.yaml` at `0` (webcam), an RTSP/HTTP URL, or another local file, restart, and repeat steps 2–4. Walk in front of the camera (or play a clip with people/cars/pets). Motion without a matching class does **not** create events.

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
