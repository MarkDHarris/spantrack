"""Content-addressed store of synthesized sentences.

The cache is the artifact the determinism and crash-safety guarantees hang on. A
sentence's key is `sha256` over `(model_version, voice, speed, normalized_text)`, so
identical inputs always resolve to the same entry and any change to voice/speed/model
busts it. Each entry is a `<key>.wav` (24 kHz mono float32) plus a `<key>.tokens.json`
sidecar, sharded under the first two hex characters of the key.

Writes are atomic (temp file + `os.replace`) and the `.wav` is renamed last, so it acts
as the entry's completion marker: a crash mid-write leaves at most an orphan sidecar,
which `get` treats as a miss and the next run overwrites. Resume is therefore emergent
-- a cache hit skips synthesis, with no bespoke resume bookkeeping.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import soundfile  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from spantrack.tts.backend import (
    SAMPLE_RATE,
    Pcm,
    TokenTimestamp,
    token_from_dict,
    token_to_dict,
)

__all__ = ["SentenceCache"]

# NUL joins the key fields because it cannot occur in a voice/model id or in normalized
# narration text, so no character can migrate across a boundary and collide two distinct
# inputs -- unlike the illustrative `|` separator (a pipe is ordinary prose).
_KEY_SEPARATOR = "\x00"


class _SoundFile(Protocol):
    """The slice of the untyped `soundfile` module this store uses."""

    def read(self, file: str, *, dtype: str) -> tuple[NDArray[np.float32], int]: ...

    def write(
        self,
        file: str,
        data: NDArray[np.float32],
        samplerate: int,
        *,
        subtype: str,
        format: str,  # noqa: A002  matches soundfile's real keyword name
    ) -> None: ...


# `soundfile` is untyped; the import above suppresses the missing-stub warning and this
# cast keeps every call site fully typed against the Protocol.
_sf = cast("_SoundFile", soundfile)


@dataclass(frozen=True, slots=True)
class SentenceCache:
    """A sharded, content-addressed store of synthesized sentences under `root`."""

    root: Path

    @staticmethod
    def key(model_version: str, voice: str, speed: float, normalized_text: str) -> str:
        """Derive the content-addressed cache key for one sentence.

        `speed` uses its shortest round-tripping `repr`, so equal speeds always hash
        identically while any change produces a distinct key.
        """
        material = _KEY_SEPARATOR.join(
            [model_version, voice, repr(speed), normalized_text],
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[Pcm, tuple[TokenTimestamp, ...] | None] | None:
        """Return the cached `(pcm, tokens)` for `key`, or None on a miss.

        `tokens` is None when the backend that wrote the entry emitted no timestamps.
        """
        wav = self._wav_path(key)
        if not wav.is_file():
            return None

        pcm, _ = _sf.read(str(wav), dtype="float32")
        return pcm, self._read_tokens(self._tokens_path(key))

    def put(
        self,
        key: str,
        pcm: Pcm,
        tokens: tuple[TokenTimestamp, ...] | None,
    ) -> None:
        """Store one synthesized sentence, overwriting any existing entry for `key`.

        The sidecar is written before the audio and the `.wav` is renamed last, so the
        entry only becomes visible to `get` once both parts are on disk.
        """
        self._shard(key).mkdir(parents=True, exist_ok=True)
        self._write_tokens(self._tokens_path(key), tokens)
        self._write_wav(self._wav_path(key), pcm)

    def _shard(self, key: str) -> Path:
        return self.root / key[:2]

    def _wav_path(self, key: str) -> Path:
        return self._shard(key) / f"{key}.wav"

    def _tokens_path(self, key: str) -> Path:
        return self._shard(key) / f"{key}.tokens.json"

    def _write_wav(self, path: Path, pcm: Pcm) -> None:
        tmp = path.with_name(f"{path.name}.tmp")
        # The temp name ends in `.tmp`, so soundfile cannot infer the container from the
        # extension -- name it explicitly.
        _sf.write(str(tmp), pcm, SAMPLE_RATE, subtype="FLOAT", format="WAV")
        tmp.replace(path)

    def _write_tokens(
        self, path: Path, tokens: tuple[TokenTimestamp, ...] | None
    ) -> None:
        rows = None if tokens is None else [token_to_dict(token) for token in tokens]
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _read_tokens(self, path: Path) -> tuple[TokenTimestamp, ...] | None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw is None:
            return None

        return tuple(token_from_dict(row) for row in raw)
