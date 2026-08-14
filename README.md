# TundraNVR

Camera → **Raspberry** (unusual?) → **Node** (what is it?) → **Hub** (what is it doing?) → **operator**.

One process. Switch seats on the live page to see the handoff.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_sample.py
python -m app.main
```

Open http://127.0.0.1:8000. Clips: Entrance, Drone, Bag left. Activity is confirm / dismiss (dismiss trains Pattern of Life).

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`.
