# TundraNVR

Camera → **Edge** (unusual?) → **Detect** (YOLO + track) → **Verify** (VLM **verdict**) → **Review**.

That is the whole product. Side channels (badge fusion, thumb-novelty kNN, MQTT, a second class-allowlist “anomaly” checker, a caption-only VLM pass) are **off** and not part of the decision.

**What runs**

1. **Edge** — OpenCV frame difference plus a learned occupancy footprint. No neural net. Unusual, still-learning, or idle-sweep frames go to Detect.
2. **Detect** — YOLO names those trips; ByteTrack holds an id for wall-clock dwell (one track ≤ one event). Detect does **not** suppress in recall. The only Detect rule is unattended bag (backpack/handbag/suitcase, dwell, no person nearby).
3. **Verify** — local VLM, Set-of-Mark JPEG, JSON `alert` / `category` / `reason`. Captions are a search byproduct, never the decision. Fail-open: if Verify is down, the rule alert sits on an **Unverified** shelf.
4. **Review** — Incident or Normal. Normal absorbs into Edge’s occupancy map (two Normals before a large absorb). Sample/fixture never absorb and never page (unless `demo.page_review`).

**Escalation default is `auto`** (`escalation.mode`): recall while Verify is healthy, else `pol_score`. Explicit `recall` / `pol_score` remain for eval. Detect is a namer. Verify suppresses. A few percent of Verify-suppressed trips still page as **Audit**. Sample/fixture provenance never absorbs into Pattern of Life. By default samples do **not** page Review; set `demo.page_review: true` to allow allowlisted showcase clips (see below).

Default vision is **local-only**. Cloud OpenAI requires `vision.allow_cloud: true`. See [`LICENSING.md`](LICENSING.md) and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

Optional **Jev page/suppress gate** (spike, off by default): TypeSafe’s typed noul over **structured trip state only** (dwell, zone, pol_score, classes, bag/situation templates, Verify health) — never frames. Enable with `jev.enabled: true`, remote calls also need `jev.allow_cloud: true` plus `OPENROUTER_API_KEY` (or `TYPESAFE_API_KEY`). Fail-open when disabled, denied, or unreachable (same spirit as Verify). Offline smoke: `python scripts/jev_smoke.py`. `/health` → `jev` (last action, counts, latency).

This PoC is **one building camera**. Face recognition, LPR, emotion recognition, audio, badge fusion, and 24/7 NVR recording are **out** until they have a real feed and a seat in this cascade.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m app.main
```

Set `camera.source` to index `0` or an RTSP URL (Details on Live). Details also has demo presets for the bundled sample files. Events is Incident / Normal — Normal trains Edge’s occupancy map (two Normals before a large absorb; sample/fixture never absorb). Live: clean video (no chrome on the JPEG) plus a verdict rail. Detect/Verify show **situation lines** templated from tracks (dwell, zone, person near a vehicle, unattended bag) — not an LLM narration, and not identity. Events: marked still of what was spotted, then the clip. Review leads with **why it was paged** (`paged_because`). A few percent of Verify-suppressed trips are still shown as **Audit** so false negatives are measurable (`/health` `audit_shown` / `audit_confirmed`).

Tracks age in **wall-clock seconds** (`tracking.max_age_s`). Detect still runs every `pipeline.idle_detect_seconds` when the scene is still, so unattended-bag / loiter can fire in a quiet corridor. The 16-cell fill is a **motion sketch**, not a seasonal Pattern of Life. Review is not paged while that sketch is filling. Sample paths under `data/samples/` keep provenance `sample` even when selected in Details.

## Demo clips (no camera)

This host often has no USB camera. The process then loops a file under `data/samples/` (gitignored). Fetch the catalog (plus optional `drone-yolo.pt`):

```bash
python scripts/download_sample.py
```

| Clip | What it showcases | Review paging |
| --- | --- | --- |
| `street.mp4` | Intel parking lot — people, bicycles, cars | Default fallback. Not allowlisted. |
| `indoor.mp4` | Intel indoor pedestrians — tracks / dwell | Not allowlisted. |
| `parking.mp4` | Intel car lot — vehicle namer | Not allowlisted. |
| `courtyard.mp4` | Pexels outdoor street (trimmed ≤3 min) | Showcase allowlist. |
| `lobby.mp4` | CAVIAR INRIA lobby Browse_WhileWaiting2 — longer indoor dwell | Showcase allowlist. |
| `package.mp4` | CAVIAR LeftBag — unattended bag rule | Showcase allowlist. |
| `drones.mp4` | Wikimedia ground/static camera filming a quadcopter (UAV **in** frame) | Showcase allowlist. |
| `entrance.mp4` | CAVIAR 2004 mall / Promod | **Not downloaded / not a preset.** Grain, glass, and mannequins produce false person boxes. |

**Showcase Review paging:** set `demo.page_review: true` in `config.yaml`, restart, then pick an allowlisted preset (Courtyard / Lobby / Left bag / Drones). Provenance stays `sample` — Normal dismissals do **not** absorb into PoL. Street / Indoor / Parking stay Detect-only demos. Catalog + licenses live in `app/samples.py` and the download script header.

**Drones / aircraft:** stock YOLOv8n has `airplane` but **no** `drone` class. `detection.drone_model: drone-yolo.pt` (fetched by the download script) runs a secondary UAV namer. Prefer `drones.mp4` (camera looking at airspace/yard with a UAV flying in scene) — not drone-POV or gimbal-moving footage.

Situation lines on Detect/Verify are templates from those tracks (`#4 car · #2 person nearby`). They never say someone entered a vehicle or returned N times — track ids die after `max_age_s` and are not a person. Face / ReID / LPR stay out.

A multi-minute loop **cannot** stand in for months of Pattern of Life. Sixteen of 64 cells with two motion hits fill in seconds on a loop; that only answers “which cells have moved in this session.” A real PoL needs **this** live camera over days.

Optional API token: `server.api_token` or `TUNDRANVR_API_TOKEN` (Bearer or `X-API-Token`) on `GET /api/events`, `GET /media`, `PUT /api/settings`, and event review. Live MJPEG stays open. Empty token = no auth; Live shows a **NO AUTH** pill. Event `source` in SQLite is redacted (no RTSP userinfo).

Offline ablation (fixtures only — **not** headline NAR/Pd/FAR unless provenance is `live`):

```bash
python scripts/eval.py --smoke
```

`--smoke` implies `--allow-fixture` so the table still has numbers, stamped `fixture`. Without that flag, eval refuses headline NAR/Pd/FAR on sample/fixture provenance. Stages are `motion` / `detect` / `track` / `verifier`. Every row stamps `mode` and `mode_effective`.

`/health` reports `escalation` counts (`edge_trips` → `node_proposals` → `hub_alerts`) plus `paged_because`, audit, and latency.

`torch` and `torchvision` must both be the CPU wheels from that index, or detection fails with `torchvision::nms does not exist`. YOLO fetches `yolov8n.pt` on the first Edge trip (AGPL — do not fine-tune until the detector licence is decided).
