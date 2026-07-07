"""Versioned per-chapter token-timestamp sidecar: the record word-mode SMIL reads.

One sidecar is written per narrated chapter and always persisted. It captures every
sentence's chapter-absolute clip interval, its display/normalized text, its cache key,
and the per-token timeline, so word-level SMIL can be (re)generated later without
re-synthesizing any audio.

Each sentence also records its tokens' `source_ranges`: the char offsets back into
`display_text` of the source word each token voices, from `word_alignment`. Storing them
lets word-level SMIL regenerate from the sidecar with no re-synthesis or re-alignment.
`source_ranges` is `null` in sentence mode; in word mode every entry is a range, or all
are `null` together for a sentence with tokens but no source words (punctuation only).

Writes go through a temp file + `Path.replace`, and the JSON is built in a fixed key
order, so an identical sidecar always serializes to identical bytes.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from spantrack import atomic_write
from spantrack.tts.backend import TokenTimestamp, token_from_dict, token_to_dict

__all__ = [
    "BackendIdentity",
    "ChapterAudio",
    "ChapterSidecar",
    "SidecarError",
    "SidecarSentence",
    "read",
    "write",
]

# Bumped whenever the on-disk shape changes incompatibly. `read` refuses any other
# version rather than mis-parse it.
_SCHEMA_VERSION = 2


class SidecarError(Exception):
    """Raised when a chapter sidecar is unreadable or of an unsupported schema."""


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    """The backend parameters behind a chapter's audio; mirror the cache key."""

    name: str
    model_version: str
    voice: str
    speed: float


@dataclass(frozen=True, slots=True)
class ChapterAudio:
    """The assembled chapter's audio file and its measured timing."""

    file: str
    sample_rate: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class SidecarSentence:
    """One sentence's entry: its chapter-absolute clip, text, cache key, and tokens.

    `tokens` is None when the backend emitted no timestamps. The token times are
    chapter-absolute (already shifted by `clip_begin_s`), not sentence-relative.

    `source_ranges`, when present, is parallel to `tokens`: entry `i` is the `[start,
    end)` char range in `display_text` of the source word token `i` voices (word mode).
    It is None in sentence mode; in word mode every entry is a range, or all are None
    together for a sentence with tokens but no source words.
    """

    span_id: str
    clip_begin_s: float
    clip_end_s: float
    display_text: str
    normalized_text: str
    cache_key: str
    tokens: tuple[TokenTimestamp, ...] | None
    source_ranges: tuple[tuple[int, int] | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class ChapterSidecar:
    """A whole chapter's token-timestamp record (one per narrated chapter)."""

    spantrack_version: str
    backend: BackendIdentity
    doc_href: str
    audio: ChapterAudio
    sentences: tuple[SidecarSentence, ...]


def write(path: Path, sidecar: ChapterSidecar) -> None:
    """Write `sidecar` to `path` as deterministic JSON via an atomic temp + replace."""
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "spantrack_version": sidecar.spantrack_version,
        "backend": {
            "name": sidecar.backend.name,
            "model_version": sidecar.backend.model_version,
            "voice": sidecar.backend.voice,
            "speed": sidecar.backend.speed,
        },
        "doc_href": sidecar.doc_href,
        "audio": {
            "file": sidecar.audio.file,
            "sample_rate": sidecar.audio.sample_rate,
            "duration_s": sidecar.audio.duration_s,
        },
        "sentences": [_sentence_to_dict(sentence) for sentence in sidecar.sentences],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    atomic_write.replace(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def read(path: Path) -> ChapterSidecar:
    """Read and validate a chapter sidecar written by `write`.

    Raises:
        SidecarError: If the file is missing, not valid JSON, of an unsupported
            `schema_version`, or missing a required field.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"unreadable chapter sidecar at {path}"
        raise SidecarError(msg) from error

    if not isinstance(raw, dict):
        msg = f"chapter sidecar at {path} is not a JSON object"
        raise SidecarError(msg)

    data = cast("dict[str, Any]", raw)

    version = data.get("schema_version")
    if version != _SCHEMA_VERSION:
        msg = (
            f"unsupported sidecar schema_version {version!r}; "
            f"expected {_SCHEMA_VERSION}"
        )
        raise SidecarError(msg)

    try:
        backend = data["backend"]
        audio = data["audio"]
        return ChapterSidecar(
            spantrack_version=str(data["spantrack_version"]),
            backend=BackendIdentity(
                name=str(backend["name"]),
                model_version=str(backend["model_version"]),
                voice=str(backend["voice"]),
                speed=float(backend["speed"]),
            ),
            doc_href=str(data["doc_href"]),
            audio=ChapterAudio(
                file=str(audio["file"]),
                sample_rate=int(audio["sample_rate"]),
                duration_s=float(audio["duration_s"]),
            ),
            sentences=tuple(_sentence_from_dict(s) for s in data["sentences"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = f"malformed chapter sidecar at {path}"
        raise SidecarError(msg) from error


def _sentence_to_dict(sentence: SidecarSentence) -> dict[str, object]:
    return {
        "span_id": sentence.span_id,
        "clip_begin_s": sentence.clip_begin_s,
        "clip_end_s": sentence.clip_end_s,
        "display_text": sentence.display_text,
        "normalized_text": sentence.normalized_text,
        "cache_key": sentence.cache_key,
        "tokens": (
            None
            if sentence.tokens is None
            else [token_to_dict(token) for token in sentence.tokens]
        ),
        "source_ranges": (
            None
            if sentence.source_ranges is None
            else [None if r is None else [r[0], r[1]] for r in sentence.source_ranges]
        ),
    }


def _sentence_from_dict(row: Mapping[str, Any]) -> SidecarSentence:
    tokens = row["tokens"]
    raw_ranges = row.get("source_ranges")
    source_ranges = (
        None
        if raw_ranges is None
        else tuple(None if r is None else (int(r[0]), int(r[1])) for r in raw_ranges)
    )
    return SidecarSentence(
        span_id=str(row["span_id"]),
        clip_begin_s=float(row["clip_begin_s"]),
        clip_end_s=float(row["clip_end_s"]),
        display_text=str(row["display_text"]),
        normalized_text=str(row["normalized_text"]),
        cache_key=str(row["cache_key"]),
        tokens=None if tokens is None else tuple(token_from_dict(t) for t in tokens),
        source_ranges=source_ranges,
    )
