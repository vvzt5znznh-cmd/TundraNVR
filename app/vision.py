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

PROMPT = (
    "You are looking at a security-camera still. Detected objects: {objects}. "
    "In 2-3 short sentences describe what is in the frame, what is likely happening, "
    "and whether it looks like ordinary activity around roads or buildings or something unusual. "
    "Do not mention models or that this is a test."
)


def fallback_summary(classes: list[str], score: float) -> str:
    labels = ", ".join(classes) if classes else "motion"
    if not classes:
        return "Motion was detected, but no allow-listed object was named."
    unusual = {"airplane", "bear", "elephant", "zebra", "giraffe"}
    if unusual.intersection(classes):
        kind = "unusual for a typical street — worth a look"
    elif {"person", "car", "truck", "bus", "bicycle", "motorcycle"} & set(classes):
        kind = "ordinary traffic or foot traffic around infrastructure"
    elif {"dog", "cat", "horse", "sheep", "cow", "bird"} & set(classes):
        kind = "an animal near a road or building"
    else:
        kind = "activity in the scene"
    return (
        f"Detected {labels} (score {score:.2f}). This looks like {kind}."
    )


def describe_event(cfg: VisionConfig, thumb: Path, classes: list[str], score: float) -> tuple[str, str]:
    """Return (summary, provider_used). Never raises."""
    objects = ", ".join(classes) if classes else "none"
    prompt = PROMPT.format(objects=objects)
    provider = (cfg.provider or "auto").lower()
    if provider in {"off", "none", "false"}:
        return fallback_summary(classes, score), "off"
    if not cfg.enabled:
        return fallback_summary(classes, score), "off"
    if not thumb.is_file():
        return fallback_summary(classes, score), "fallback"
    jpeg = thumb.read_bytes()
    if provider in {"auto", "ollama"}:
        text = _ollama(cfg, jpeg, prompt)
        if text:
            return text, "ollama"
        if provider == "ollama":
            return fallback_summary(classes, score), "fallback"
    if provider in {"auto", "openai"}:
        text = _openai(cfg, jpeg, prompt)
        if text:
            return text, "openai"
    return fallback_summary(classes, score), "fallback"


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
        "max_tokens": 220,
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
