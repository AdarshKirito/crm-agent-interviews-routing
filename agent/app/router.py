"""Task-type router: nearest-neighbour vote over labelled example requests.

The examples (data/router_examples.jsonl) are requests + task context from tasks
outside the test split. At run time the router embeds the conversation's user
turns plus the session's task context -- request text only, never a dataset
label -- and takes a similarity-weighted vote of the k nearest examples.
The routing table (data/routing.yaml) then maps the predicted type to a model
tier; it is fitted on dev results by scripts/fit_routing.py.
"""

import hashlib
import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import DATA_DIR

EMBED_MODEL = os.getenv("CRMROUTE_ROUTER_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
K_NEIGHBOURS = int(os.getenv("CRMROUTE_ROUTER_K", "15"))


@dataclass
class Prediction:
    task_type: str
    confidence: float
    top: list[tuple[str, float]]


class TaskRouter:
    def __init__(self, examples_path: Path = DATA_DIR / "router_examples.jsonl", cache_dir: Path | None = None):
        self.examples_path = examples_path
        self.cache_dir = cache_dir or Path(os.getenv("CRMROUTE_CACHE_DIR", DATA_DIR / ".cache"))
        self._lock = threading.Lock()
        self._model = None
        self._matrix: np.ndarray | None = None
        self._labels: list[str] = []

    def _embedder(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(EMBED_MODEL, cache_dir=os.getenv("FASTEMBED_CACHE_PATH"))
        return self._model

    def _load(self) -> None:
        with self._lock:
            if self._matrix is not None:
                return
            rows = [json.loads(line) for line in self.examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._labels = [r["task"] for r in rows]
            digest = hashlib.sha256((EMBED_MODEL + self.examples_path.read_text(encoding="utf-8")).encode()).hexdigest()[:16]
            cache = self.cache_dir / f"router_{digest}.npy"
            if cache.exists():
                self._matrix = np.load(cache)
                return
            vectors = np.array(list(self._embedder().passage_embed([r["text"] for r in rows])), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, vectors)
            self._matrix = vectors

    def warm_up(self) -> None:
        self._load()
        self._embedder()

    def predict(self, text: str) -> Prediction:
        self._load()
        query = np.array(next(iter(self._embedder().query_embed([text]))), dtype=np.float32)
        query /= np.linalg.norm(query)
        sims = self._matrix @ query
        nearest = np.argsort(-sims)[:K_NEIGHBOURS]
        votes: dict[str, float] = defaultdict(float)
        for i in nearest:
            votes[self._labels[i]] += float(max(sims[i], 0.0))
        total = sum(votes.values()) or 1.0
        ranked = sorted(((t, v / total) for t, v in votes.items()), key=lambda x: x[1], reverse=True)
        return Prediction(ranked[0][0], ranked[0][1], ranked[:3])


def load_routing_table(path: Path = DATA_DIR / "routing.yaml") -> dict:
    if not path.exists():
        return {"default": "big", "tiers": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"default": "big", "tiers": {}}


_router: TaskRouter | None = None


def get_router() -> TaskRouter:
    global _router
    if _router is None:
        _router = TaskRouter()
    return _router
