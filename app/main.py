from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_config
from app.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tundranvr")

cfg = load_config()
pipeline = Pipeline(cfg)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Starting TundraNVR")
    pipeline.start()
    try:
        yield
    finally:
        log.info("Shutting down TundraNVR")
        pipeline.stop()


app = FastAPI(title="TundraNVR", lifespan=lifespan)
web_dir = cfg.web_dir
app.mount("/media", StaticFiles(directory=str(cfg.data_dir)), name="media")


@app.get("/health")
def health() -> dict:
    return pipeline.health()


@app.get("/api/frame.jpg")
def latest_frame() -> Response:
    jpeg = pipeline.latest_jpeg()
    if not jpeg:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/stream.mjpg")
async def mjpeg_stream() -> StreamingResponse:
    boundary = b"frame"
    interval = 1.0 / max(cfg.pipeline.live_fps, 1.0)

    async def generate():
        while True:
            jpeg = pipeline.latest_jpeg()
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
def list_events(limit: int = 50) -> JSONResponse:
    limit = max(1, min(limit, 200))
    events = []
    for row in pipeline.store.list_events(limit):
        events.append(_event_json(row))
    return JSONResponse(events)


@app.get("/api/events/{event_id}")
def get_event(event_id: int) -> JSONResponse:
    row = pipeline.store.get(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="event not found")
    return JSONResponse(_event_json(row))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/events")
def events_page() -> FileResponse:
    return FileResponse(web_dir / "events.html")


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
