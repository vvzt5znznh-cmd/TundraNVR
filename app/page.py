"""Why an event was (or was not) paged to Review."""

from __future__ import annotations

REASONS = (
    "learning",
    "sample",
    "unusual",
    "named_object",
    "rule",
    "verify_unavailable",
    "verified",
    "audit",
    "jev_page",
    "jev_suppress",
)

LABELS = {
    "learning": "Motion map still filling",
    "sample": "Sample clip — not a live alert",
    "unusual": "Unusual for this camera",
    "named_object": "Named object",
    "rule": "Rule (unattended bag)",
    "verify_unavailable": "Unverified — Verify offline",
    "verified": "Verify alert",
    "audit": "Audit sample",
    "jev_page": "Jev gated page",
    "jev_suppress": "Jev suppressed",
}


def choose_paged_because(
    *,
    provenance: str,
    learning: bool,
    bag: bool,
    unusual: bool,
    named: bool,
    verify_status: str = "",
    alert: bool = False,
    audit: bool = False,
    demo_paging: bool = False,
    jev_action: str = "",
) -> str:
    # Default: looping sample/fixture never looks like a live site page.
    # Showcase demos opt in via demo.page_review + allowlisted clips.
    if provenance in {"sample", "fixture"} and not demo_paging:
        return "sample"
    if learning:
        return "learning"
    if jev_action == "suppress":
        return "jev_suppress"
    if jev_action == "page":
        return "jev_page"
    if audit:
        return "audit"
    if verify_status in {"unavailable", "malformed", "error"}:
        return "verify_unavailable"
    if alert and verify_status == "ok":
        return "verified"
    if bag:
        return "rule"
    if unusual:
        return "unusual"
    if named:
        return "named_object"
    return "named_object"


def label(reason: str) -> str:
    return LABELS.get(reason, reason or "—")
