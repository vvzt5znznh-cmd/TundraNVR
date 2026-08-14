# TundraNVR

Camera → **Raspberry** (unusual?) → **Node** (what is it?) → **Hub** (what is it doing?) → **operator**.

One process. Switch seats on the live page. Open **Behind the scenes** for learning progress, scores, and which models are running.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` in `config.yaml` (or Behind the scenes) to a file, RTSP/HTTP URL, or camera index `0`. Activity is confirm / dismiss — dismiss trains Pattern of Life.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. Drone weights: `python scripts/download_sample.py`.
