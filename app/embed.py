from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DIM = 128


def thumb_hist(path: Path, dim: int = DIM) -> np.ndarray:
    """Cheap per-thumb embedding (spatial colour histogram). Swap later for CLIP/Jina."""
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    if img is None:
        return np.zeros(dim, dtype=np.float32)
    small = cv2.resize(img, (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vec = small.reshape(-1)
    if vec.size < dim:
        out = np.zeros(dim, dtype=np.float32)
        out[: vec.size] = vec
        return out
    return vec[:dim].astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


@dataclass
class Novelty:
    score: float
    neighbors: int
    gated: bool = False

    def as_dict(self) -> dict:
        return {"score": round(self.score, 3), "neighbors": self.neighbors, "gated": self.gated}


class EmbeddingIndex:
    """Per-camera kNN over event thumbs. sqlite-vec if present, else numpy."""

    def __init__(self, store) -> None:
        self.store = store
        self._vec = False
        try:
            self._vec = bool(self.store.enable_sqlite_vec())
        except Exception:
            self._vec = False
        if self._vec:
            log.info("sqlite-vec loaded for embeddings")

    def add(self, event_id: int, source: str, vector: np.ndarray, hour: int, cls: str) -> None:
        self.store.insert_embedding(event_id, source, vector.astype(np.float32).tobytes(), hour, cls)

    def novelty(self, source: str, vector: np.ndarray, k: int = 5) -> Novelty:
        rows = self.store.embeddings_for(source)
        if not rows:
            return Novelty(score=1.0, neighbors=0)
        dists = []
        for row in rows:
            other = np.frombuffer(row["vector"], dtype=np.float32)
            dists.append(cosine_distance(vector, other))
        dists.sort()
        take = dists[: max(1, min(k, len(dists)))]
        score = float(sum(take) / len(take))
        return Novelty(score=score, neighbors=len(take))
