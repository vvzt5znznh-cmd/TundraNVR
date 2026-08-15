# TundraNVR

Camera → **Edge** (unusual?) → **Detect** (YOLO + track) → **Verify** (VLM **verdict**) → **Review**.

That is the whole product. Side channels (badge fusion, thumb-novelty kNN, MQTT, a second class-allowlist “anomaly” checker, a caption-only VLM pass) are **off** and not part of the decision.

**What runs**

1. **Edge** — OpenCV frame difference plus a learned occupancy footprint. No neural net. Unusual, still-learning, or idle-sweep frames go to Detect.
2. **Detect** — YOLO names those trips; ByteTrack holds an id for wall-clock dwell (one track ≤ one event). Detect does **not** suppress in recall. The only Detect rule is unattended bag (backpack/handbag/suitcase, dwell, no person nearby).
3. **Verify** — local VLM, Set-of-Mark JPEG, JSON `alert` / `category` / `reason`. Captions are a search byproduct, never the decision. Fail-open: if Verify is down, the rule alert sits on an **Unverified** shelf.
4. **Review** — Incident or Normal. Normal absorbs into Edge’s occupancy map (two Normals before a large absorb). Sample/fixture never absorb and never page.

**Escalation default is `auto`**: recall while Verify is healthy, else `pol_score`. Detect is a namer. Verify suppresses. A few percent of Verify-suppressed trips still page as **Audit**.

Default vision is **local-only**. Cloud OpenAI requires `vision.allow_cloud: true`. See [`LICENSING.md`](LICENSING.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

This PoC is **one building camera**. Face recognition, LPR, emotion recognition, audio, badge fusion, and 24/7 NVR recording are **out** until they have a real feed and a seat in this cascade.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` to index `0` or an RTSP URL (Details on Live). Details also has **Street / Indoor / Left bag** presets for the bundled demo files. Live: clean video (no chrome on the JPEG) plus a verdict rail. Detect/Verify show **situation lines** templated from tracks (dwell, zone, person near a vehicle, unattended bag) — not an LLM narration, and not identity. Events: marked still of what was spotted, then the clip. Review leads with **why it was paged** (`paged_because`).

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

`--smoke` implies `--allow-fixture` so the table still has numbers, stamped `fixture`. Without that flag, eval refuses headline NAR/Pd/FAR on sample/fixture provenance. Stages are `motion` / `detect` / `track` / `verifier`. Every row stamps `mode` and `mode_effective`.

`/health` reports `escalation` counts (`edge_trips` → `node_proposals` → `hub_alerts`) plus `paged_because`, audit, and latency.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip (AGPL — do not fine-tune until the detector licence is decided).
