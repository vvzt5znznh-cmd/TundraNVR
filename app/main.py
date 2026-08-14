from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
    model: str = Field(min_length=1, max_length=500)


class EventReview(BaseModel):
    action: str = Field(min_length=1, max_length=20)


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
    payload["vision"] = cfg.vision.provider if cfg.vision.enabled else "off"
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
def review_event(event_id: int, body: EventReview) -> JSONResponse:
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
def update_settings(body: SettingsUpdate) -> dict:
    global cfg, pipeline
    try:
        source, model = parse_settings_update(body.source, body.model)
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _settings_lock:
        save_runtime_settings(source=source, model=model)
        log.info("Applying settings source=%s model=%s", source, model)
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


@app.get("/app.css")
def app_css() -> FileResponse:
    return FileResponse(web_dir / "app.css", media_type="text/css")


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
