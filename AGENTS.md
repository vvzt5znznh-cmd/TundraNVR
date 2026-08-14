# TundraNVR

Single Python service: camera → Edge (Pattern of Life) → Node (YOLO) → Hub (caption) → operator. `python -m app.main` on port 8000.

- Python 3.12 `.venv` at repo root. Activate: `. .venv/bin/activate`.
- `torch` / `torchvision` must be CPU builds from `https://download.pytorch.org/whl/cpu`. If you see `torchvision::nms does not exist`: `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.
- `ffmpeg` is used for event clips.
- Samples + drone weights: `python scripts/download_sample.py` (needs internet). Default source `data/samples/entrance.mp4`. YOLO fetches `yolov8n.pt` on first detect.
- Live `/`, Activity `/events`. Evidence: `/health` (handoff + PoL) and `/api/events`. Looping files report high ingest fps — expected.
