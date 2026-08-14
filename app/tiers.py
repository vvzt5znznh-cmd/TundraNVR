"""Wanted vs running models at each cascade seat."""

from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.vision import effective_provider

# Operator-facing seat names. Internal keys stay edge / node / hub / operator.
SEAT_LABELS = {
    "edge": "Edge",
    "node": "Detect",
    "hub": "Verify",
    "operator": "Review",
}


def _norm(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _match(want: str, running: str) -> bool:
    w, r = _norm(want), _norm(running)
    if not w or not r:
        return False
    return w in r or r in w


def _hub_running(cfg: AppConfig, provider: str) -> str:
    if not cfg.vision.enabled or provider in {"off", "none"}:
        return "off"
    if provider == "denied-cloud":
        return "cloud blocked"
    if provider in {"local", "ollama"}:
        return f"{cfg.vision.ollama_model} via Ollama"
    if provider in {"openai", "cloud"}:
        return f"{cfg.vision.openai_model} via OpenAI"
    return provider


def models_payload(
    cfg: AppConfig,
    *,
    provider: str,
    yolo_ran: bool,
    hub_ran: bool,
    edge_active: bool,
) -> dict:
    node_name = Path(cfg.detection.model).name or cfg.detection.model
    hub_name = _hub_running(cfg, provider)
    edge_running = "OpenCV + Pattern of Life"
    return {
        "edge": {
            "label": SEAT_LABELS["edge"],
            "want": cfg.targets.edge,
            "running": edge_running,
            "match": _match(cfg.targets.edge, edge_running) or "no neural" in cfg.targets.edge.lower() or "no nn" in cfg.targets.edge.lower(),
            "active": bool(edge_active),
            "note": "No neural net on this seat.",
        },
        "node": {
            "label": SEAT_LABELS["node"],
            "want": cfg.targets.node,
            "running": f"{node_name} + ByteTrack",
            "match": _match(cfg.targets.node, node_name),
            "active": bool(yolo_ran),
            "note": "Detector only on Edge trips.",
        },
        "hub": {
            "label": SEAT_LABELS["hub"],
            "want": cfg.targets.hub,
            "running": hub_name,
            "match": _match(cfg.targets.hub, hub_name),
            "active": bool(hub_ran),
            "note": "Verifier; captions are a byproduct.",
        },
    }
