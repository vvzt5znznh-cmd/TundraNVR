# TundraNVR

Camera → **Raspberry** (unusual?) → **Node** (detect + track) → **Hub** (VLM **verdict**) → **operator**.

Motion is pixel change. Pattern of Life is a learned occupancy footprint. YOLO names Edge trips only. Node assigns stable track IDs (one track ≤ one event). Hub adjudicates with a Set-of-Mark prompt and structured JSON. Captions are a search byproduct, never the decision.

**Escalation is recall-oriented** (`escalation.mode: recall`): Node hands Hub any plausible track. Do not gate on YOLO or VLM confidence. Hub suppresses. `pol_score` restores the old PoL ≥ 0.7 gate.

Default vision is **local-only**. Cloud OpenAI requires `vision.allow_cloud: true`. See [`LICENSING.md`](LICENSING.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

This PoC is **one building camera**. Critical-infrastructure buyers will still scrutinise it as Annex III-adjacent; that is documentation and later work, not this binary. Face recognition, LPR, emotion recognition, and audio are **out**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` to index `0` or an RTSP URL (Details on Live). Events is Incident / Normal — Normal trains Raspberry’s Pattern of Life. Dark ops UI: Live video + verdict rail; Events inbox.

Optional mutating-API token: `server.api_token` or `TUNDRANVR_API_TOKEN` (Bearer or `X-API-Token`) on `PUT /api/settings` and event review. Live MJPEG stays open.

Offline ablation (fixtures only — **not** headline NAR/Pd/FAR):

```bash
python scripts/eval.py --smoke
```

`/health` reports `escalation` counts (Raspberry trips → Node proposals → Hub alerts → operator confirms).

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip (AGPL — do not fine-tune until the detector licence is decided).
