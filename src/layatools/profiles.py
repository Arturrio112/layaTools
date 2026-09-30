"""Decision profiles: named, reusable sets of Laya questions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_MIN_CONFIDENCE = 0.7
MODELS_DIR = Path.home() / ".local" / "share" / "layatools" / "models"


class Profile(BaseModel):
    """A named decision an LLM can invoke by name instead of restating the questions."""

    name: str
    description: str = ""
    questions: dict[str, dict[str, Any]]
    min_confidence: float = Field(DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0)
    state_field: str | None = None
    # Name of a fine-tuned checkpoint under MODELS_DIR; falls back to the default Router when absent.
    model: str | None = None

    def summary(self) -> dict[str, Any]:
        """Token-lean description for `list_decisions`."""
        return {
            "name": self.name,
            "description": self.description,
            "answers": {
                qid: _describe(spec) for qid, spec in self.questions.items()
            },
        }


def _describe(spec: dict[str, Any]) -> str:
    kind = spec.get("type")
    criteria = spec.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        return "choice: " + "|".join(criteria)
    if kind == "score" and isinstance(criteria, list):
        return f"score 0-{len(criteria) - 1}"
    return "yes/no" if kind == "noul" else str(kind)


def load_profile_file(path: Path) -> Profile:
    data = yaml.safe_load(path.read_text())
    data.setdefault("name", path.stem)
    return Profile.model_validate(data)


SHIPPED_DIR = Path(__file__).parent / "builtin_profiles"
USER_DIR = Path.home() / ".config" / "layatools" / "profiles"


def load_profiles(*directories: Path | None, builtin: bool = True) -> dict[str, Profile]:
    """Load Laya's presets, then the profiles shipped with layatools, then each directory in order.

    Later sources win on a name clash, so project profiles override user ones override shipped ones.
    """
    profiles: dict[str, Profile] = {}
    sources = [SHIPPED_DIR, *directories] if builtin else list(directories)
    if builtin:
        profiles.update(_builtin_profiles())
    for directory in sources:
        if directory is not None and directory.is_dir():
            for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
                profile = load_profile_file(path)
                profiles[profile.name] = profile
    return profiles


def _builtin_profiles() -> dict[str, Profile]:
    from laya.mcp.tools import PRESETS
    import laya
    from laya.presets import state_field

    out: dict[str, Profile] = {}
    for name, builder in PRESETS.items():
        questions = getattr(laya, builder)()
        out[name] = Profile(
            name=name,
            description=f"Laya built-in '{name}' preset",
            questions=questions,
            state_field=state_field(questions),
        )
    return out


PROJECT_SUBDIR = Path(".layatools") / "profiles"


def env_dirs() -> list[Path]:
    """Extra profile directories from `LAYATOOLS_PROFILES` (separated like PATH)."""
    return [Path(d).expanduser() for d in os.environ.get("LAYATOOLS_PROFILES", "").split(os.pathsep) if d]


def profiles_for(project: Path | None) -> dict[str, Profile]:
    """Laya presets + shipped + user-level + `LAYATOOLS_PROFILES` + the project's `.layatools/profiles/`.

    Later sources win, so the project's own profiles have the highest priority.
    """
    return load_profiles(USER_DIR, *env_dirs(), (project / PROJECT_SUBDIR) if project else None)
