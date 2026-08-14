# TundraNVR

Camera → **Raspberry** (unusual?) → **Node** (detect + track) → **Hub** (VLM **verdict**) → **operator**.

Motion is pixel change. Pattern of Life is a learned occupancy footprint. YOLO names Edge trips only. Node assigns stable track IDs (one track ≤ one event). Hub adjudicates with a Set-of-Mark prompt and structured JSON. Captions are a search byproduct, never the decision.

Default vision is **local-only**. Cloud OpenAI requires `vision.allow_cloud: true`. See [`LICENSING.md`](LICENSING.md).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` to index `0` or an RTSP URL. Activity is confirm / dismiss — dismiss trains Pattern of Life.

Offline ablation (fixtures only — not headline numbers):

```bash
python scripts/eval.py --smoke
```

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip.
