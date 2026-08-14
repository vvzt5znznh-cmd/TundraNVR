"""Thin NVR hardening: redact camera secrets, optional mutating-API token."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

ALLOWED_SOURCE_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https", "file"})


def redact_source(source: str | int | None) -> str:
    """Strip userinfo and query from URLs so RTSP passwords never hit logs."""
    if source is None:
        return ""
    text = str(source).strip()
    if not text:
        return ""
    if "://" not in text:
        return text
    parts = urlsplit(text)
    netloc = parts.netloc
    if "@" in netloc:
        host = netloc.rsplit("@", 1)[-1]
        netloc = f"***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def source_scheme(source: str | int) -> str | None:
    text = str(source).strip()
    if "://" not in text:
        return None
    return text.split("://", 1)[0].strip().lower()
