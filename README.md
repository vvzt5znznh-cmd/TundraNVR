# TundraNVR

Camera → **Edge** (unusual?) → **Detect** (YOLO + track) → **Verify** (VLM **verdict**) → **Review**.

Motion is pixel change. Pattern of Life is a learned occupancy footprint. YOLO names Edge trips only. Detect assigns stable track IDs (one track ≤ one event). Verify adjudicates with a Set-of-Mark prompt and structured JSON. Captions are a search byproduct, never the decision.

**Escalation default is `auto`** (`escalation.mode`): recall while Verify is healthy, else `pol_score`. Explicit `recall` / `pol_score` remain for eval. Detect is a namer in recall (it does not suppress). Verify suppresses. When Verify is down, live unusual traffic goes to an Unverified shelf — not mixed with pending Review. Sample fallback keeps Detect running but does **not** page Review.

Default vision is **local-only**. Cloud OpenAI requires `vision.allow_cloud: true`. See [`LICENSING.md`](LICENSING.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

This PoC is **one building camera**. Critical-infrastructure buyers will still scrutinise it as Annex III-adjacent; that is documentation and later work, not this binary. Face recognition, LPR, emotion recognition, and audio are **out**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` to index `0` or an RTSP URL (Details on Live). Events is Incident / Normal — Normal trains Edge’s Pattern of Life (two Normals before a large absorb; sample/fixture never absorb). Live: video + verdict rail. Events: marked still of what was spotted, then the clip. Review leads with **why it was paged** (`paged_because`). A few percent of Verify-suppressed trips are still shown as **Audit** so false negatives are measurable (`/health` `audit_shown` / `audit_confirmed`).

Tracks age in **wall-clock seconds** (`tracking.max_age_s`). Detect still runs every `pipeline.idle_detect_seconds` when the scene is still, so unattended-bag / loiter can fire in a quiet corridor. Learning ready is **cell coverage**, not a tick count; Review is not paged while learning.

Optional API token: `server.api_token` or `TUNDRANVR_API_TOKEN` (Bearer or `X-API-Token`) on `GET /api/events`, `GET /media`, `PUT /api/settings`, and event review. Live MJPEG stays open. Empty token = no auth; Live shows a **NO AUTH** pill. Event `source` in SQLite is redacted (no RTSP userinfo).

Offline ablation (fixtures only — **not** headline NAR/Pd/FAR unless provenance is `live`):

```bash
python scripts/eval.py --smoke
```

`--smoke` implies `--allow-fixture` so the table still has numbers, stamped `fixture`. Without that flag, eval refuses headline NAR/Pd/FAR on sample/fixture provenance. Every row stamps `mode` and `mode_effective`.

`/health` reports `escalation` counts (Edge trips → Detect proposals → Verify alerts → Review confirms). Internal keys remain `raspberry_trips` / `node_proposals` / `hub_alerts`.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip (AGPL — do not fine-tune until the detector licence is decided).
