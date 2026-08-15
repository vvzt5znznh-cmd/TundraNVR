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

Set `camera.source` to index `0` or an RTSP URL (Details on Live). Details also has **Street / Indoor / Left bag** presets for the bundled demo files. Events is Incident / Normal — Normal trains Edge’s occupancy map (two Normals before a large absorb; sample/fixture never absorb). Live: clean video (no chrome on the JPEG) plus a verdict rail. Detect/Verify show **situation lines** templated from tracks (dwell, zone, person near a vehicle, unattended bag) — not an LLM narration, and not identity. Events: marked still of what was spotted, then the clip. Review leads with **why it was paged** (`paged_because`). A few percent of Verify-suppressed trips are still shown as **Audit** so false negatives are measurable (`/health` `audit_shown` / `audit_confirmed`).

Tracks age in **wall-clock seconds** (`tracking.max_age_s`). Detect still runs every `pipeline.idle_detect_seconds` when the scene is still, so unattended-bag / loiter can fire in a quiet corridor. The 16-cell fill is a **motion sketch**, not a seasonal Pattern of Life. Review is not paged while that sketch is filling, and never paged on sample fallback.

## Demo clips (no camera)

This host often has no USB camera. The process then loops a file under `data/samples/` (gitignored). Fetch street + indoor:

```bash
python scripts/download_sample.py
```

| Clip | What it is for | How to use |
| --- | --- | --- |
| `street.mp4` | Intel parking lot — people, bicycles, cars | **Default fallback.** Mixed-object Detect namer. |
| `indoor.mp4` | Intel indoor pedestrians | Details → Indoor. Tracks and dwell. |
| `package.mp4` | CAVIAR LeftBag — unattended bag | Details → Left bag. Bag rule: backpack/handbag/suitcase, dwell ≥8s, no person within 120px in the last 2s. |
| `entrance.mp4` | CAVIAR 2004 mall / Promod | **Not a preset.** Grain, glass, and mannequins produce false person boxes. |

Situation lines on Detect/Verify are templates from those tracks (`#4 car · #2 person nearby`). They never say someone entered a vehicle or returned N times — track ids die after `max_age_s` and are not a person. Face / ReID / LPR stay out.

A 15–50s loop **cannot** stand in for months of Pattern of Life. Sixteen of 64 cells with two motion hits fill in seconds on a loop; that only answers “which cells have moved in this session.” Sample provenance does not page Review and does not absorb into the occupancy map. A real PoL needs **this** live camera over days.

Optional API token: `server.api_token` or `TUNDRANVR_API_TOKEN` (Bearer or `X-API-Token`) on `GET /api/events`, `GET /media`, `PUT /api/settings`, and event review. Live MJPEG stays open. Empty token = no auth; Live shows a **NO AUTH** pill. Event `source` in SQLite is redacted (no RTSP userinfo).

Offline ablation (fixtures only — **not** headline NAR/Pd/FAR unless provenance is `live`):

```bash
python scripts/eval.py --smoke
```

`--smoke` implies `--allow-fixture` so the table still has numbers, stamped `fixture`. Without that flag, eval refuses headline NAR/Pd/FAR on sample/fixture provenance. Every row stamps `mode` and `mode_effective`.

`/health` reports `escalation` counts (Edge trips → Detect proposals → Verify alerts → Review confirms). Internal keys remain `raspberry_trips` / `node_proposals` / `hub_alerts`.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip (AGPL — do not fine-tune until the detector licence is decided).
