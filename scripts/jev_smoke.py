#!/usr/bin/env python3
"""Smoke-exercise the optional Jev page/suppress gate without a live API key.

Examples:
  python scripts/jev_smoke.py
  python scripts/jev_smoke.py --provider mock --enabled
  python scripts/jev_smoke.py --show-denied
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import JevConfig
from app.jev_gate import (
    action_from_noul,
    build_trip_state,
    decide_page_gate,
)


def _print(title: str, result) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(result.as_dict(), indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enabled", action="store_true", help="set jev.enabled")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=("mock", "openrouter"),
        help="mock needs no key; openrouter needs env key + allow_cloud",
    )
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-denied", action="store_true", help="also print denied-cloud path")
    args = parser.parse_args()

    bag = build_trip_state(
        classes=["suitcase", "person"],
        score=0.81,
        pol_score=0.88,
        dwell_s=14.0,
        zone="doorway",
        bag=True,
        named=True,
        unusual=True,
        learning=False,
        no_badge=True,
        verify_healthy=True,
        anomaly_reason="unattended bag (track)",
        situation=["#3 suitcase · dwell 14s · no person nearby"],
        provenance="live",
        mode_effective="recall",
    )
    quiet = build_trip_state(
        classes=["person"],
        score=0.62,
        pol_score=0.21,
        dwell_s=2.0,
        zone="doorway",
        bag=False,
        named=True,
        unusual=False,
        learning=False,
        verify_healthy=True,
        anomaly_reason="",
        situation=["#1 person · doorway"],
        provenance="live",
        mode_effective="recall",
    )

    cfg = JevConfig(
        enabled=bool(args.enabled) or args.provider == "mock",
        allow_cloud=bool(args.allow_cloud),
        provider=args.provider,
        dry_run=bool(args.dry_run),
        page_threshold=0.75,
        suppress_threshold=0.35,
    )

    # Threshold unit check (no network).
    assert action_from_noul(0.9, page_threshold=0.75, suppress_threshold=0.35) == "page"
    assert action_from_noul(0.5, page_threshold=0.75, suppress_threshold=0.35) == "verify"
    assert action_from_noul(0.1, page_threshold=0.75, suppress_threshold=0.35) == "suppress"

    _print("bag trip", decide_page_gate(cfg, bag))
    _print("quiet trip", decide_page_gate(cfg, quiet))

    off = JevConfig(enabled=False, provider="mock")
    _print("disabled → fallback", decide_page_gate(off, bag))

    if args.show_denied or args.provider == "openrouter":
        denied = JevConfig(enabled=True, allow_cloud=False, provider="openrouter")
        _print("cloud denied → fallback", decide_page_gate(denied, bag))

    print("\nok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
