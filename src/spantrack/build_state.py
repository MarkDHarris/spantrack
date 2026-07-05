"""Build manifest (`build.json`): the run's parameter fingerprint + per-doc checkpoints.

Resume is emergent from two independent stores: the sentence cache skips already-
synthesized sentences (content-addressed by voice/speed/model/text), and this file skips
already-encoded chapters. `build.json` records the parameters that determine the output;
when the current run's fingerprint differs from the stored one, the chapter checkpoints
are stale (a new voice re-encodes every chapter) and are discarded on load. Audio
invalidation needs no active step here -- a changed parameter simply misses the cache.

The file is a resume optimization, not a deliverable, so a missing or corrupt one is a
clean start rather than an error (unlike `chapter_sidecar`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from spantrack import atomic_write

__all__ = ["BuildParameters", "BuildState", "load", "save"]

# Bumped if the on-disk shape changes; an unrecognized version reads as a clean start.
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BuildParameters:
    """The run parameters that determine output; any change invalidates checkpoints."""

    voice: str
    speed: float
    granularity: str
    audio_format: str
    bitrate: int
    selection: tuple[str, ...]


@dataclass(slots=True)
class BuildState:
    """The resume state for one build: its parameter fingerprint and completed docs."""

    parameters: BuildParameters
    completed: set[str]

    def is_done(self, doc_href: str) -> bool:
        """Return whether `doc_href` has already been fully narrated and encoded."""
        return doc_href in self.completed

    def mark_done(self, doc_href: str) -> None:
        """Record `doc_href` as complete (persisted by the next `save`)."""
        self.completed.add(doc_href)


def load(path: Path, parameters: BuildParameters) -> BuildState:
    """Load the build state for `parameters`, or a fresh one on miss/mismatch.

    A missing, malformed, or wrong-version `build.json`, or one whose stored parameters
    differ from `parameters`, yields a fresh state with no checkpoints. Only a file that
    matches the current parameters restores its checkpoints.
    """
    stored = _read(path)
    if stored is None or stored.parameters != parameters:
        return BuildState(parameters=parameters, completed=set())

    return stored


def save(path: Path, state: BuildState) -> None:
    """Write `state` to `path` as deterministic JSON via an atomic temp + replace."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "parameters": {
            "voice": state.parameters.voice,
            "speed": state.parameters.speed,
            "granularity": state.parameters.granularity,
            "audio_format": state.parameters.audio_format,
            "bitrate": state.parameters.bitrate,
            "selection": list(state.parameters.selection),
        },
        "completed": sorted(state.completed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    atomic_write.replace(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _read(path: Path) -> BuildState | None:
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    document = cast("dict[str, Any]", data)
    if document.get("schema_version") != _SCHEMA_VERSION:
        return None

    try:
        params = document["parameters"]
        return BuildState(
            parameters=BuildParameters(
                voice=str(params["voice"]),
                speed=float(params["speed"]),
                granularity=str(params["granularity"]),
                audio_format=str(params["audio_format"]),
                bitrate=int(params["bitrate"]),
                selection=tuple(str(href) for href in params["selection"]),
            ),
            completed={str(href) for href in document["completed"]},
        )
    except (KeyError, TypeError, ValueError):
        return None
