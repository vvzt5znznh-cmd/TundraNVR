# TundraNVR

Local camera-detection MVP: camera → Edge (OpenCV motion + Pattern of Life, no neural net) → Node (YOLO on trips only) → Hub (caption once) → operator. See [`README.md`](README.md) for the full overview, routes, and config keys.

## Cursor Cloud specific instructions

Single Python service. There are no tests or linters configured in this repo.

### Environment
- Python 3.12 in a `.venv` at the repo root. Dependencies (including the correct CPU builds of `torch`/`torchvision`) are installed by the startup update script, so you normally do not reinstall them.
- Activate the venv before running anything: `. .venv/bin/activate`.
- `ffmpeg` (system) is used to encode H.264 event clips; it is preinstalled.

### torch / torchvision gotcha (important)
- `torch` and `torchvision` MUST both be the CPU builds from `https://download.pytorch.org/whl/cpu` and must match. Installing `requirements.txt` on its own pulls a generic `torchvision` wheel from PyPI that is ABI-incompatible with the CPU `torch`, producing `RuntimeError: operator torchvision::nms does not exist` at first detection. The update script fixes this by force-reinstalling the `+cpu` `torchvision` wheel last. If you ever see that NMS error, run `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.

### Sample video + model (one-time, persisted in the snapshot; not in the update script)
- Default `camera.source` is index `0` (or an RTSP URL set in Behind the scenes). A looping sample file is not the product story.
- YOLO downloads `yolov8n.pt` into the repo root on the first **Edge trip** (unusual or still-learning motion), not on every motion frame. Needs internet.

### Run
- Start the app (dev): `python -m app.main` — serves on `http://0.0.0.0:8000` (`config.yaml` → `server`). There is no separate dev/prod command; this is the dev run.
- Web UI: `/` (live MJPEG view) and `/events` (history). JSON/health: `/health`, `/api/events`, latest frame at `/api/frame.jpg`, event media under `/media/...`.

### Testing / behavior notes
- Motion is OpenCV frame difference. Pattern of Life is a learned occupancy footprint. Neither is a neural net.
- YOLO (`handoff.node.ran` / `yolo_ran`) runs only when Edge uploads a trip. Usual doorway traffic should show `node.received: false` and `yolo_ran: false`.
- Live-view bounding boxes are transient. Prefer `/api/events` and `/health` (polled) as evidence. Hub captions once per escalated event, not per frame.
- Events (row in `data/events.db` + thumb in `data/thumbs/` + clip in `data/clips/`) are the core end-to-end signal that an unusual trip climbed the ladder.
