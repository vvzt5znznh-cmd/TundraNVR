"""Optional TypeSafe Jev page/suppress gate (spike).

Sends **structured trip state only** (dwell, zone, pol_score, classes, bag /
situation signals, Verify health). Never frames, faces, audio, or identity.

Gating mirrors Verify cloud policy:
- `jev.enabled` master switch
- `jev.allow_cloud` must be true for remote OpenRouter / TypeSafe calls
- missing key, errors, or cloud denied → fail-open (`fallback` → current Verify path)

Actions from calibrated noul P(should_page):
- noul >= page_threshold → page
- noul < suppress_threshold → suppress
- else → verify (send to Verify as today)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from app.config import JevConfig

log = logging.getLogger(__name__)

ACTIONS = ("page", "suppress", "verify", "fallback")
STATUSES = (
    "ok",
    "disabled",
    "skipped",
    "denied-cloud",
    "unavailable",
    "mock",
    "fallback",
    "dry_run",
)

DEFAULT_NOUL_INSTRUCTIONS = (
    "Should this fixed building-camera trip page a human operator for review?"
)
DEFAULT_NOUL_CRITERIA = {
    "true": (
        "Unattended bag, after-hours person without badge, intrusion, "
        "drone/airplane near the building, or clearly unusual activity "
        "worth interrupting an operator."
    ),
    "false": (
        "Ordinary doorway traffic, expected vehicles or pedestrians for "
        "this camera, learning/sketch noise, or activity that should stay suppressed."
    ),
}


@dataclass(frozen=True)
class JevGateResult:
    action: str
    status: str
    noul: float | None = None
    reason: str = ""
    provider: str = "off"
    latency_ms: float = 0.0
    applied: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "status": self.status,
            "noul": None if self.noul is None else round(float(self.noul), 4),
            "reason": self.reason,
            "provider": self.provider,
            "latency_ms": round(float(self.latency_ms), 1),
            "applied": bool(self.applied),
        }


@dataclass
class JevStats:
    """Counters exposed on /health."""

    last: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "calls": 0,
            "page": 0,
            "suppress": 0,
            "verify": 0,
            "fallback": 0,
            "denied_cloud": 0,
            "unavailable": 0,
            "dry_run": 0,
        }
    )
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, result: JevGateResult) -> None:
        self.last = result.as_dict()
        self.counts["calls"] = int(self.counts.get("calls", 0)) + 1
        key = result.action if result.action in {"page", "suppress", "verify", "fallback"} else "fallback"
        self.counts[key] = int(self.counts.get(key, 0)) + 1
        if result.status == "denied-cloud":
            self.counts["denied_cloud"] = int(self.counts.get("denied_cloud", 0)) + 1
        if result.status == "unavailable":
            self.counts["unavailable"] = int(self.counts.get("unavailable", 0)) + 1
        if result.status == "dry_run":
            self.counts["dry_run"] = int(self.counts.get("dry_run", 0)) + 1
        if result.latency_ms > 0:
            self.latencies_ms.append(float(result.latency_ms))
            if len(self.latencies_ms) > 64:
                self.latencies_ms = self.latencies_ms[-64:]

    def health(self, cfg: JevConfig) -> dict[str, Any]:
        lat = list(self.latencies_ms)
        return {
            "enabled": bool(cfg.enabled),
            "allow_cloud": bool(cfg.allow_cloud),
            "provider": (cfg.provider or "openrouter").strip().lower(),
            "model": cfg.model,
            "dry_run": bool(cfg.dry_run),
            "page_threshold": float(cfg.page_threshold),
            "suppress_threshold": float(cfg.suppress_threshold),
            "has_api_key": bool(resolve_api_key(cfg)),
            "last": dict(self.last),
            "counts": dict(self.counts),
            "latency_ms": {
                "count": len(lat),
                "p50": round(_percentile(lat, 50), 1) if lat else None,
                "p95": round(_percentile(lat, 95), 1) if lat else None,
            },
        }


def resolve_api_key(cfg: JevConfig) -> str:
    """Prefer OpenRouter, then TypeSafe, then config (env wins)."""
    for name in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return str(cfg.api_key or "").strip()


def build_trip_state(
    *,
    classes: list[str],
    score: float,
    pol_score: float,
    dwell_s: float | None = None,
    zone: str = "",
    bag: bool = False,
    named: bool = False,
    unusual: bool = False,
    learning: bool = False,
    no_badge: bool = False,
    verify_healthy: bool = True,
    anomaly_reason: str = "",
    situation: list[str] | None = None,
    provenance: str = "live",
    mode_effective: str = "",
) -> dict[str, Any]:
    """Structured state only — never include images or raw frames."""
    lines = [str(s) for s in (situation or []) if str(s).strip()][:8]
    return {
        "camera_context": "fixed building CCTV trip",
        "classes": list(classes or []),
        "detector_score": round(float(score), 3),
        "pol_score": round(float(pol_score), 3),
        "dwell_s": None if dwell_s is None else round(float(dwell_s), 1),
        "zone": str(zone or ""),
        "bag_unattended": bool(bag),
        "named": bool(named),
        "unusual": bool(unusual),
        "learning": bool(learning),
        "no_badge_within_window": bool(no_badge),
        "verify_healthy": bool(verify_healthy),
        "anomaly_reason": str(anomaly_reason or ""),
        "situation_lines": lines,
        "provenance": str(provenance or "live"),
        "escalation_mode_effective": str(mode_effective or ""),
    }


def action_from_noul(
    noul: float,
    *,
    page_threshold: float,
    suppress_threshold: float,
) -> str:
    page_t = max(0.0, min(1.0, float(page_threshold)))
    suppress_t = max(0.0, min(1.0, float(suppress_threshold)))
    if suppress_t > page_t:
        suppress_t, page_t = page_t, suppress_t
    if noul >= page_t:
        return "page"
    if noul < suppress_t:
        return "suppress"
    return "verify"


def rule_fallback(*, reason: str = "jev unavailable — keep Verify path") -> JevGateResult:
    return JevGateResult(
        action="fallback",
        status="fallback",
        noul=None,
        reason=reason,
        provider="rules",
        applied=False,
    )


def decide_page_gate(
    cfg: JevConfig,
    state: dict[str, Any],
    *,
    post_json: Callable[..., dict] | None = None,
) -> JevGateResult:
    """Return page / suppress / verify / fallback. Never raises into the pipeline."""
    t0 = time.monotonic()
    if not cfg.enabled:
        return JevGateResult(
            action="fallback",
            status="disabled",
            reason="jev.enabled is false",
            provider="off",
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    provider = (cfg.provider or "openrouter").strip().lower()
    if provider in {"mock", "dry", "fixture"}:
        result = _mock_decide(cfg, state, t0=t0)
        return _maybe_dry_run(cfg, result)

    if not cfg.allow_cloud:
        return JevGateResult(
            action="fallback",
            status="denied-cloud",
            reason="jev.allow_cloud is false — structured state not sent",
            provider="denied-cloud",
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    key = resolve_api_key(cfg)
    if not key:
        return JevGateResult(
            action="fallback",
            status="unavailable",
            reason="no OPENROUTER_API_KEY or TYPESAFE_API_KEY",
            provider=provider,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    try:
        noul = _remote_noul(cfg, state, api_key=key, post_json=post_json)
    except Exception as exc:
        log.warning("Jev gate failed open: %s", exc)
        return JevGateResult(
            action="fallback",
            status="unavailable",
            reason=f"jev error: {exc}",
            provider=provider,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    if noul is None:
        return JevGateResult(
            action="fallback",
            status="unavailable",
            reason="jev response missing noul",
            provider=provider,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    action = action_from_noul(
        noul,
        page_threshold=cfg.page_threshold,
        suppress_threshold=cfg.suppress_threshold,
    )
    result = JevGateResult(
        action=action,
        status="ok",
        noul=float(noul),
        reason=f"noul={noul:.3f} → {action}",
        provider=provider,
        latency_ms=(time.monotonic() - t0) * 1000.0,
        applied=action in {"page", "suppress"},
    )
    return _maybe_dry_run(cfg, result)


def _maybe_dry_run(cfg: JevConfig, result: JevGateResult) -> JevGateResult:
    if not cfg.dry_run:
        return result
    if result.action in {"page", "suppress"} and result.status in {"ok", "mock"}:
        return JevGateResult(
            action="fallback",
            status="dry_run",
            noul=result.noul,
            reason=f"dry_run: would {result.action} ({result.reason})",
            provider=result.provider,
            latency_ms=result.latency_ms,
            applied=False,
        )
    return result


def _mock_decide(cfg: JevConfig, state: dict[str, Any], *, t0: float) -> JevGateResult:
    """Deterministic offline path for smoke tests (no network, no key)."""
    if state.get("bag_unattended"):
        noul = 0.92
    elif state.get("unusual") and state.get("no_badge_within_window"):
        noul = 0.8
    elif state.get("learning") or state.get("provenance") in {"sample", "fixture"}:
        noul = 0.12
    elif state.get("unusual"):
        noul = 0.55
    else:
        noul = 0.28
    action = action_from_noul(
        noul,
        page_threshold=cfg.page_threshold,
        suppress_threshold=cfg.suppress_threshold,
    )
    return JevGateResult(
        action=action,
        status="mock",
        noul=noul,
        reason=f"mock noul={noul:.3f} → {action}",
        provider="mock",
        latency_ms=(time.monotonic() - t0) * 1000.0,
        applied=action in {"page", "suppress"},
    )


def _remote_noul(
    cfg: JevConfig,
    state: dict[str, Any],
    *,
    api_key: str,
    post_json: Callable[..., dict] | None = None,
) -> float | None:
    url = (cfg.base_url or "https://openrouter.ai/api/alpha/decisions").strip()
    payload = {
        "model": cfg.model or "typesafe/jev-1.13",
        "state": state,
        "questions": {
            "should_page": {
                "type": "noul",
                "instructions": (cfg.instructions or DEFAULT_NOUL_INSTRUCTIONS).strip(),
                "criteria": dict(cfg.criteria or DEFAULT_NOUL_CRITERIA),
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/vvzt5znznh-cmd/tundranvr",
        "X-Title": "TundraNVR Jev gate spike",
    }
    poster = post_json or _post_json
    data = poster(url, payload, float(cfg.timeout_seconds), headers)
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        return None
    should = answers.get("should_page") or {}
    if not isinstance(should, dict):
        return None
    raw = should.get("noul")
    if raw is None:
        return None
    return max(0.0, min(1.0, float(raw)))


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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)
