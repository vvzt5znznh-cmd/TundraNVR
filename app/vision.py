from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from app.config import VisionConfig

log = logging.getLogger(__name__)


def fallback_summary(
    classes: list[str],
    score: float,
    anomaly_reason: str = "",
) -> str:
    labels = ", ".join(classes) if classes else "motion"
    if not classes:
        return "Motion was detected, but no allow-listed object was named."
    if anomaly_reason:
        return f"Detected {labels} (score {score:.2f}). {anomaly_reason}."
    if "drone" in classes:
        return f"Detected {labels} (score {score:.2f}). A drone near a building is worth an alert."
    if {"person"} & set(classes):
        kind = "ordinary foot traffic around the building"
    else:
        kind = "activity in the scene"
    return f"Detected {labels} (score {score:.2f}). This looks like {kind}."


def effective_provider(cfg: VisionConfig) -> str:
    if not cfg.enabled:
        return "off"
    provider = (cfg.provider or "local").strip().lower()
    if provider in {"auto", "none", "false"}:
        provider = "local"
    if provider in {"openai", "cloud"} and not cfg.allow_cloud:
        return "denied-cloud"
    return provider


def complete_vision(cfg: VisionConfig, jpeg: bytes, prompt: str) -> tuple[str | None, str, str]:
    """Return (text, provider, status). Never silently sends frames to the cloud."""
    provider = effective_provider(cfg)
    if provider in {"off", "none", "false"}:
        return None, "off", "off"
    if provider == "denied-cloud":
        log.warning("Cloud vision blocked: vision.allow_cloud is false")
        return None, "denied-cloud", "unavailable"
    if not jpeg:
        return None, provider, "unavailable"
    if provider in {"local", "ollama"}:
        text = _ollama(cfg, jpeg, prompt)
        if text:
            return text, "ollama", "ok"
        return None, "ollama", "unavailable"
    if provider in {"openai", "cloud"}:
        if not cfg.allow_cloud:
            return None, "denied-cloud", "unavailable"
        text = _openai(cfg, jpeg, prompt)
        if text:
            return text, "openai", "ok"
        return None, "openai", "unavailable"
    return None, provider, "unavailable"


def describe_event(
    cfg: VisionConfig,
    thumb: Path,
    classes: list[str],
    score: float,
    anomaly_reason: str = "",
) -> tuple[str, str]:
    """Search-facing caption only. Never the alert decision."""
    fallback = fallback_summary(classes, score, anomaly_reason)
    if not thumb.is_file():
        return fallback, "fallback"
    jpeg = thumb.read_bytes()
    prompt = (
        "You are looking at a still from a fixed building CCTV camera. "
        f"Detected objects: {', '.join(classes) or 'none'}. "
        f"Anomaly flag: {anomaly_reason or 'none'}. "
        "In 2-3 short sentences say what is in the frame and what it is doing. "
        "Do not mention models."
    )
    text, provider, status = complete_vision(cfg, jpeg, prompt)
    if text and status == "ok":
        return text, provider
    return fallback, status if status != "ok" else "fallback"


def _ollama(cfg: VisionConfig, jpeg: bytes, prompt: str) -> str | None:
    url = cfg.ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "images": [base64.b64encode(jpeg).decode("ascii")],
        "stream": False,
    }
    try:
        data = _post_json(url, payload, cfg.timeout_seconds, headers={})
    except Exception as exc:
        log.debug("Ollama vision skipped: %s", exc)
        return None
    text = str(data.get("response") or "").strip()
    return text or None


def _openai(cfg: VisionConfig, jpeg: bytes, prompt: str) -> str | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    b64 = base64.b64encode(jpeg).decode("ascii")
    payload = {
        "model": cfg.openai_model,
        "max_tokens": 400,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {key}"}
    try:
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            cfg.timeout_seconds,
            headers=headers,
        )
    except Exception as exc:
        log.debug("OpenAI vision skipped: %s", exc)
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    text = str((choices[0].get("message") or {}).get("content") or "").strip()
    return text or None


def _post_json(url: str, payload: dict, timeout: float, headers: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc
    return json.loads(raw) if raw else {}
