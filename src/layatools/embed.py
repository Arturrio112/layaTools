"""Text embeddings for semantic code search. Standard BERT-family encoder, no remote code."""

from __future__ import annotations

import base64
import os
import threading
from typing import Any

# bge-small: 33M params, 384 dims, 512 tokens; strong for its size and a plain BertModel.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 512


class Embedder:
    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self.model_name = model or os.environ.get("LAYATOOLS_EMBED_MODEL", DEFAULT_MODEL)
        self._device = device
        self._tok: Any = None
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        self._device = device

    def embed(self, texts: list[str], *, query: bool = False, batch_size: int = 32) -> list[list[float]]:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            if query:
                texts = [QUERY_PREFIX + t for t in texts]
            out: list[list[float]] = []
            # Sort by length so batches pad little, then restore the caller's order.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            vecs: dict[int, list[float]] = {}
            for s in range(0, len(order), batch_size):
                idx = order[s : s + batch_size]
                enc = self._tok(
                    [texts[i] for i in idx], padding=True, truncation=True,
                    max_length=MAX_TOKENS, return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    cls = self._model(**enc).last_hidden_state[:, 0]  # bge pools the CLS token
                cls = torch.nn.functional.normalize(cls, dim=-1).float().cpu()
                for i, v in zip(idx, cls.tolist()):
                    vecs[i] = v
            out = [vecs[i] for i in range(len(texts))]
            return out


def pack(vectors: list[list[float]]) -> dict[str, Any]:
    """float32 little-endian, base64: ~4x smaller than JSON floats and lossless."""
    import struct

    dim = len(vectors[0]) if vectors else 0
    raw = b"".join(struct.pack(f"<{dim}f", *v) for v in vectors)
    return {"dim": dim, "count": len(vectors), "data": base64.b64encode(raw).decode()}
