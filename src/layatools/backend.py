"""The seam between the gateway and Laya, so tests can swap in a fake."""

from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]: ...


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class LayaBackend:
    """Lazily loads Laya's Router on first use (model load is slow and heavy)."""

    def __init__(self, preload: bool = False) -> None:
        self._preload = preload
        self._router: Any = None

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        if self._router is None:
            import os

            from laya import Router

            # LAYATOOLS_DEVICE=cpu|cuda|mps; default: CUDA when available (needs gcc + python3-dev for Triton), else CPU.
            self._router = Router(preload=self._preload, device=os.environ.get("LAYATOOLS_DEVICE") or _default_device())
        return self._router.predict(state, questions)
