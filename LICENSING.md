# Licensing

TundraNVR application code in this repository is intended for a **local building-camera** proof of concept. Critical-infrastructure (energy, water, telecom) deployments sit close to EU AI Act Annex III and, if the site is designated *skjermingsverdig*, Norway's Sikkerhetsloven. That adjacency is documented, not implemented: this binary is still one building camera.

## Detector weights and AGPL — still an open decision

The default namer, Ultralytics **YOLOv8n** (`yolov8n.pt` via the `ultralytics` package), is licensed **AGPL-3.0**. That is acceptable for this PoC while `detection.model` stays a swap point.

Do **not** fine-tune, redistribute, or commercially package a YOLO8/YOLO11/YOLO26 checkpoint from this repo until a licence decision is recorded here.

**WS2 (not started):** evaluate **RF-DETR** (Apache 2.0) on our own building footage as the commercial-path detector. YOLO26-N remains a CPU fallback only after the AGPL question is closed.

Until that decision: keep `detection.model` config-swappable; write no fine-tuning scripts.

## Biometrics — out

Face recognition, licence-plate recognition, biometric categorisation, and emotion recognition are **not** in this product. That is an explicit decision (EU AI Act prohibitions / high-risk concentration, Norwegian camera law, reputation), not a missing feature. Do not add them.

## Vision / cloud

Frames of identifiable people must not leave the machine unless `vision.allow_cloud: true` is set explicitly. The default provider is local (Ollama) or off. See `config.yaml`.

## Audio

This project must not record or analyse audio. Norwegian law generally forbids recording others' conversations; do not add Frigate-style audio events.
