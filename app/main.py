from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    SettingsError,
    load_config,
    parse_settings_update,
    public_settings,
    save_runtime_settings,
)
from app.pipeline import Pipeline
from app.security import redact_source
from app.version import VERSION, version_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tundranvr")

cfg = load_config()
pipeline = Pipeline(cfg)
_settings_lock = threading.Lock()


class SettingsUpdate(BaseModel):
    source: str = Field(min_length=1, max_length=2000)


class EventReview(BaseModel):
    action: str = Field(min_length=1, max_length=20)


def _require_mutating_auth(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
) -> None:
    token = (cfg.server.api_token or "").strip()
    if not token:
        return
    presented = (x_api_token or "").strip()
    if not presented and authorization:
        scheme, _, rest = authorization.partition(" ")
        if scheme.lower() == "bearer":
            presented = rest.strip()
    if presented != token:
        raise HTTPException(status_code=401, detail="unauthorized")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Starting TundraNVR")
    pipeline.start()
    try:
        yield
    finally:
        log.info("Shutting down TundraNVR")
        pipeline.stop()


app = FastAPI(title="TundraNVR", version=VERSION, lifespan=lifespan)
web_dir = cfg.web_dir
app.mount("/media", StaticFiles(directory=str(cfg.data_dir)), name="media")


@app.get("/health")
def health() -> dict:
    payload = pipeline.health()
    payload.update(version_payload())
    return payload


@app.get("/api/frame.jpg")
def latest_frame(seat: str = "node") -> Response:
    jpeg = pipeline.latest_jpeg(seat)
    if not jpeg:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/stream.mjpg")
async def mjpeg_stream(seat: str = "node") -> StreamingResponse:
    boundary = b"frame"
    interval = 1.0 / max(cfg.pipeline.live_fps, 1.0)

    async def generate():
        while True:
            jpeg = pipeline.latest_jpeg(seat)
            if jpeg:
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary.decode()}",
    )


@app.get("/api/events")
def list_events(limit: int = 50, alerts: bool = False) -> JSONResponse:
    limit = max(1, min(limit, 200))
    events = []
    for row in pipeline.store.list_events(limit, alerts_only=alerts):
        events.append(_event_json(row))
    return JSONResponse(events)


@app.get("/api/events/{event_id}")
def get_event(event_id: int) -> JSONResponse:
    row = pipeline.store.get(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="event not found")
    return JSONResponse(_event_json(row))


@app.post("/api/events/{event_id}/review")
def review_event(
    event_id: int,
    body: EventReview,
    _: None = Depends(_require_mutating_auth),
) -> JSONResponse:
    action = (body.action or "").strip().lower()
    if action not in {"confirm", "dismiss"}:
        raise HTTPException(status_code=400, detail="action must be confirm or dismiss")
    try:
        row = pipeline.review_event(event_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not row:
        raise HTTPException(status_code=404, detail="event not found")
    return JSONResponse(_event_json(row))


@app.get("/api/settings")
def get_settings() -> dict:
    return public_settings(cfg)


@app.put("/api/settings")
def update_settings(
    body: SettingsUpdate,
    _: None = Depends(_require_mutating_auth),
) -> dict:
    global cfg, pipeline
    try:
        source = parse_settings_update(body.source)
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _settings_lock:
        save_runtime_settings(source=source, model=cfg.detection.model)
        log.info("Applying camera source %s", redact_source(source))
        old = pipeline
        old.stop()
        cfg = load_config()
        pipeline = Pipeline(cfg)
        pipeline.start()
    return public_settings(cfg)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/events")
def events_page() -> FileResponse:
    return FileResponse(web_dir / "events.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(web_dir / "ui.js", media_type="text/javascript")


def _event_json(row: dict) -> dict:
    thumb = row.get("thumb_path")
    clip = row.get("clip_path")
    return {
        "id": row["id"],
        "ts_start": row["ts_start"],
        "ts_end": row["ts_end"],
        "classes": row.get("classes") or [],
        "score": row.get("score"),
        "thumb_url": f"/media/{thumb}" if thumb else None,
        "clip_url": f"/media/{clip}" if clip else None,
        "summary": row.get("summary") or "",
        "anomaly": bool(row.get("anomaly")),
        "anomaly_reason": row.get("anomaly_reason") or "",
        "source": row.get("source") or "",
        "pol_score": row.get("pol_score"),
        "stopped_at": row.get("stopped_at") or "",
        "handoff": row.get("handoff") or {},
        "operator_status": row.get("operator_status") or "",
        "track_id": row.get("track_id"),
        "dwell_s": (row.get("features") or {}).get("dwell_s"),
        "verifier_provider": row.get("verifier_provider") or "",
        "verifier_status": row.get("verifier_status") or "",
        "novelty_score": row.get("novelty_score"),
        "features": row.get("features") or {},
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
