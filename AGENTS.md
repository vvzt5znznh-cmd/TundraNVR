# TundraNVR

Local camera-detection MVP: camera/video file → motion filter → YOLO object detection → save events (SQLite + JPEG thumb + MP4 clip) → FastAPI web UI. See [`README.md`](README.md) for the full overview, routes, and config keys.

## Cursor Cloud specific instructions

Single Python service. There are no tests or linters configured in this repo.

### Environment
- Python 3.12 in a `.venv` at the repo root. Dependencies (including the correct CPU builds of `torch`/`torchvision`) are installed by the startup update script, so you normally do not reinstall them.
- Activate the venv before running anything: `. .venv/bin/activate`.
- `ffmpeg` (system) is used to encode H.264 event clips; it is preinstalled.

### torch / torchvision gotcha (important)
- `torch` and `torchvision` MUST both be the CPU builds from `https://download.pytorch.org/whl/cpu` and must match. Installing `requirements.txt` on its own pulls a generic `torchvision` wheel from PyPI that is ABI-incompatible with the CPU `torch`, producing `RuntimeError: operator torchvision::nms does not exist` at first detection. The update script fixes this by force-reinstalling the `+cpu` `torchvision` wheel last. If you ever see that NMS error, run `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.

### Sample video + model (one-time, persisted in the snapshot; not in the update script)
- `data/sample.mp4` is the default `camera.source` and is git-ignored. If missing, fetch it with `python scripts/download_sample.py` (needs internet).
- YOLO downloads `yolov8n.pt` into the repo root on the first detection run (needs internet).

### Run
- Start the app (dev): `python -m app.main` — serves on `http://0.0.0.0:8000` (`config.yaml` → `server`). There is no separate dev/prod command; this is the dev run.
- Web UI: `/` (live MJPEG view) and `/events` (history). JSON/health: `/health`, `/api/events`, latest frame at `/api/frame.jpg`, event media under `/media/...`.

### Testing / behavior notes
- The bundled `data/sample.mp4` is a short clip that loops, so `/health` reports a very high ingest `fps` (hundreds) — that is expected for a looping file source, not a bug.
- Live-view bounding boxes and the `last_detections`/motion status are transient: overlays are only drawn for a short TTL after a detection, so a single screenshot of `/` may show none. Prefer `/api/events` and `/health` (polled) as reliable evidence that detection is working. Detection only runs when motion is present, and only allow-listed classes (default `person`, `car`, `dog`, `cat`) create events.
- Events (row in `data/events.db` + thumb in `data/thumbs/` + clip in `data/clips/`) are the core end-to-end signal that the pipeline works.
