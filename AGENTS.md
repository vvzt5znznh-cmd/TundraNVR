# TundraNVR

Local camera-detection MVP: camera → Edge (OpenCV motion + Pattern of Life) → Detect (YOLO on trips + ByteTrack) → Verify (VLM verifier JSON, fail-open) → Review. See [`README.md`](README.md), [`LICENSING.md`](LICENSING.md), and [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Cursor Cloud specific instructions

Single Python service. There are no tests or linters configured in this repo besides `python scripts/eval.py --smoke`.

### Environment
- Python 3.12 in a `.venv` at the repo root. Dependencies (including the correct CPU builds of `torch`/`torchvision`) are installed by the startup update script, so you normally do not reinstall them.
- Activate the venv before running anything: `. .venv/bin/activate`.
- `ffmpeg` (system) is used to encode H.264 event clips; it is preinstalled.

### torch / torchvision gotcha (important)
- `torch` and `torchvision` MUST both be the CPU builds from `https://download.pytorch.org/whl/cpu` and must match. Installing `requirements.txt` on its own pulls a generic `torchvision` wheel from PyPI that is ABI-incompatible with the CPU `torch`, producing `RuntimeError: operator torchvision::nms does not exist` at first detection. The update script fixes this by force-reinstalling the `+cpu` `torchvision` wheel last. If you ever see that NMS error, run `pip install --force-reinstall --no-deps torchvision --index-url https://download.pytorch.org/whl/cpu`.

### Sample video + model
- Default `camera.source` is index `0` (or an RTSP URL set in Details).
- YOLO downloads `yolov8n.pt` on the first **Edge trip**. Needs internet. AGPL — see LICENSING.md.

### Run
- Start the app (dev): `python -m app.main` — `http://0.0.0.0:8000`.
- Web UI: `/` (Live) and `/events` (Review). JSON: `/health`, `/api/events`. Dark ops chrome; Incident/Normal on Events. Seats: Edge / Detect / Verify / Review.

### Testing / behavior notes
- Motion is OpenCV frame difference. YOLO only on Edge trips. Tracks: one track ≤ one event. Dwell is on `/api/events` (`dwell_s`, `track_id`).
- Hub verifier is local-only unless `vision.allow_cloud: true`. Fail-open: `verifier_status=unavailable` keeps the rule alert.
- Escalation default is `recall` (any plausible Node trip → Hub). Do not gate on YOLO/VLM confidence. `/health` → `escalation` and `models` (want vs running per seat).
- `python scripts/eval.py --smoke` writes a fixture ablation table (NAR/Pd/FAR language); never treat it as site headline numbers.
- Do not add audio, face recognition, LPR, or emotion recognition.
