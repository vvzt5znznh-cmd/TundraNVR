# Threat model (PoC)

TundraNVR is a monitoring stack. For a building — and especially for a site adjacent to critical infrastructure — a compromised NVR is a reconnaissance asset and a pivot. This note maps the Frigate/go2rtc CVE class (command injection via unsanitised config, incomplete nested-YAML patches, RTSP credential leak to a viewer role, missing WebSocket auth, `--privileged` container escape) onto this Python service.

This is a checklist for the PoC, not a full IEC 62443 zone/conduit design.

## What we do now

- **No privileged containers.** Do not run this process or any sidecar with `--privileged` or host network unless a later deployment note says otherwise. GPU access, if added, should use device mounts, not a privileged flag.
- **Typed config, not free-form exec.** `config.yaml` is loaded with `yaml.safe_load` into dataclasses (`app/config.py`). Unknown nested keys are ignored. Live settings accept a single `source` string (`PUT /api/settings`), not an arbitrary YAML document. Source URLs are restricted to `rtsp` / `rtsps` / `http` / `https` / `file` (or a camera index / existing file path). There is no `exec:` camera scheme.
- **Do not log RTSP secrets.** Logs and `/health` redact URL userinfo. `GET /api/settings` returns the same redacted source. Paste a full URL to change the camera; a redacted `***@` value is rejected.
- **Optional token on mutating APIs.** Set `server.api_token` or `TUNDRANVR_API_TOKEN`. Then `PUT /api/settings` and `POST /api/events/{id}/review` require `Authorization: Bearer <token>` or `X-API-Token`. Live MJPEG (`/api/stream.mjpg`) stays open for the PoC. Empty token = no auth (default).
- **No audio, no face/LPR, no emotion recognition.** Out of scope on both privacy and AI Act grounds. Do not copy Frigate audio events.

## Still open (later slices)

- Authn/z on every GET API and any future WebSocket
- Camera credential vault (today the source may sit in `config.yaml` or `data/settings.json`)
- Signed, tamper-evident event logs and Ed25519 evidence bundles
- Network segmentation (camera VLAN / analytics / operator UI) mapped to IEC 62443 zones
- Nested-YAML review if a future feature accepts operator-uploaded YAML

## Operator notes

- Do not put RTSP passwords in tickets, screenshots of `/health`, or CI logs.
- If `api_token` is set, the Behind-the-scenes camera form and Activity confirm/dismiss buttons need that token; the unauthenticated UI will get 401.
