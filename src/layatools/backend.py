"""The seam between the gateway and Laya, so tests can swap in a fake."""

from __future__ import annotations

import os
from typing import Any, Protocol


class Backend(Protocol):
    def predict(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]: ...


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


LAYA_MODEL_ENV = "LAYATOOLS_LAYA_MODEL"


def configured_model() -> str | None:
    """Laya checkpoint forced for every Router call: `LAYATOOLS_LAYA_MODEL` = english | multilingual | typed-decisions.

    Unset (or `auto`) keeps Laya's own routing: English text -> english, other languages -> multilingual.
    """
    value = (os.environ.get(LAYA_MODEL_ENV) or "").strip().lower()
    return None if value in {"", "auto"} else value


class LayaBackend:
    """Lazily loads Laya's Router on first use (model load is slow and heavy)."""

    def __init__(self, preload: bool = False) -> None:
        self._preload = preload
        self._router: Any = None
        self._tuned: dict[str, Any] = {}
        self.last_routing: dict[str, Any] | None = None  # which checkpoint answered the last Router call

    def _device(self) -> str:
        return os.environ.get("LAYATOOLS_DEVICE") or _default_device()

    def predict(
        self, state: dict[str, Any], questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        """`model` names a fine-tuned checkpoint in MODELS_DIR; if it is not installed, use the Router."""
        if model:
            from .profiles import MODELS_DIR

            path = MODELS_DIR / model
            if path.is_dir():
                if model not in self._tuned:
                    import laya

                    self._tuned[model] = laya.load(str(path), device=self._device())
                return self._tuned[model].predict(state, questions)
        if self._router is None:
            from laya import Router

            # LAYATOOLS_DEVICE=cpu|cuda|mps; default: CUDA when available (needs gcc + python3-dev for Triton), else CPU.
            self._router = Router(preload=self._preload, device=os.environ.get("LAYATOOLS_DEVICE") or _default_device())
        result = self._router.predict(state, questions, model=configured_model())
        self.last_routing = result.get("routing")
        return result
