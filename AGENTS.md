# TundraNVR

Single Python service: camera → Edge (Pattern of Life) → Node (YOLO) → Hub (caption) → operator. `python -m app.main` on port 8000.

- Python 3.12 `.venv` at repo root. Activate: `. .venv/bin/activate`.
- `torch` / `torchvision` must be CPU builds from `https://download.pytorch.org/whl/cpu`. If you see `torchvision::nms does not exist`: `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.
- `ffmpeg` is used for event clips.
- Point `camera.source` at a file, RTSP URL, or index `0`. YOLO fetches `yolov8n.pt` on first detect. Drone weights: `python scripts/download_sample.py`.
- Live `/`, Activity `/events`. Evidence: `/health` (handoff, PoL progress, model names) and `/api/events`.
