# TundraNVR

Camera → **Raspberry** (unusual?) → **Node** (what is it?) → **Hub** (what is it doing?) → **operator**.

Motion is pixel change (OpenCV frame difference), not a model. Pattern of Life is a learned occupancy footprint, also not a model. YOLO names a trip only after Edge says unusual (or is still learning). A caption runs once on Hub, only when Node cannot close.

One process. Switch seats on the live page. Open **Behind the scenes** for learning progress, scores, and which stage actually ran.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Default `camera.source` is camera index `0`. Point it at an RTSP/HTTP URL, a file, or another index in `config.yaml` or Behind the scenes. Let a static doorway sit for a few minutes so Pattern of Life can learn. Activity is confirm / dismiss — dismiss trains that footprint.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip, not on every motion frame.
