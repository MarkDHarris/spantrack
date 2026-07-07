"""Tests for the TTSBackend protocol contract, exercised via a fake backend."""

import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from spantrack.lexicon import Lexicon, load
from spantrack.tts.backend import TokenTimestamp, TTSBackend

# A fake backend's phoneme "inventory": lowercase letters and space. The real Kokoro
# inventory is the misaki set.
_INVENTORY = frozenset("abcdefghijklmnopqrstuvwxyz ")


class _FakeBackend:
    """A minimal, self-contained TTSBackend used to exercise the protocol contract."""

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        for word, phonemes in lexicon.entries.items():
            if unknown := set(phonemes) - _INVENTORY:
                msg = f"{word}: phonemes outside inventory: {sorted(unknown)}"
                raise ValueError(msg)

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        def override(match: re.Match[str]) -> str:
            phonemes = lexicon.get(match.group(0))
            return f"[{match.group(0)}](/{phonemes}/)" if phonemes else match.group(0)

        return re.sub(r"[A-Za-z]+", override, text)

    def synthesize(self, text: str):
        # One flat second of silence per token, tokens timed back-to-back.
        words = text.split()
        pcm = np.zeros(24_000 * max(len(words), 1), dtype=np.float32)
        tokens = tuple(
            TokenTimestamp(text=w, start_s=float(i), end_s=float(i + 1), whitespace=" ")
            for i, w in enumerate(words)
        )
        return pcm, tokens


# Static conformance: pyright strict rejects a signature drift in `_FakeBackend` here.
# The runtime `isinstance` checks below only see method presence, not signatures.
_STATIC_CONFORMANCE: TTSBackend = _FakeBackend()


class _NoSynthesize:
    """A backend with the lexicon hooks but missing `synthesize`."""

    def validate_lexicon(self, lexicon: Lexicon) -> None: ...

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str: ...


class _OnlyValidate:
    """A partial implementation missing `apply_lexicon` and `synthesize`."""

    def validate_lexicon(self, lexicon: Lexicon) -> None: ...


def _lex(tmp_path: Path, content: str) -> Lexicon:
    path = tmp_path / "lexicon.toml"
    path.write_text(content, encoding="utf-8")

    return load(path)


def test_conforming_fake_satisfies_protocol():
    assert isinstance(_FakeBackend(), TTSBackend)


def test_plain_object_is_not_a_backend():
    assert not isinstance(object(), TTSBackend)


def test_partial_implementation_is_not_a_backend():
    assert not isinstance(_OnlyValidate(), TTSBackend)


def test_backend_without_synthesize_is_not_a_backend():
    assert not isinstance(_NoSynthesize(), TTSBackend)


def test_synthesize_returns_pcm_and_timestamps():
    pcm, tokens = _FakeBackend().synthesize("one two")
    assert pcm.dtype == np.float32
    assert tokens == (
        TokenTimestamp(text="one", start_s=0.0, end_s=1.0, whitespace=" "),
        TokenTimestamp(text="two", start_s=1.0, end_s=2.0, whitespace=" "),
    )


def test_token_timestamp_is_frozen():
    token = TokenTimestamp(text="x", start_s=0.0, end_s=1.0, whitespace="")
    with pytest.raises(FrozenInstanceError):
        token.text = "y"  # type: ignore[misc]


def test_token_timestamp_equality():
    a = TokenTimestamp(text="x", start_s=0.0, end_s=1.0, whitespace=" ")
    b = TokenTimestamp(text="x", start_s=0.0, end_s=1.0, whitespace=" ")
    c = TokenTimestamp(text="x", start_s=0.0, end_s=1.5, whitespace=" ")
    assert a == b
    assert a != c


def test_validate_lexicon_rejects_out_of_inventory(tmp_path: Path):
    lexicon = _lex(tmp_path, '[words]\ngif = "d3if"\n')
    with pytest.raises(ValueError, match="inventory"):
        _FakeBackend().validate_lexicon(lexicon)


def test_validate_lexicon_accepts_in_inventory(tmp_path: Path):
    _FakeBackend().validate_lexicon(_lex(tmp_path, '[words]\ngif = "jif"\n'))


def test_apply_lexicon_rewrites_matches_case_insensitively(tmp_path: Path):
    lexicon = _lex(tmp_path, '[words]\ngif = "jif"\n')
    assert _FakeBackend().apply_lexicon("a GIF here", lexicon) == "a [GIF](/jif/) here"


def test_apply_lexicon_leaves_unmatched_text(tmp_path: Path):
    lexicon = _lex(tmp_path, '[words]\ngif = "jif"\n')
    assert _FakeBackend().apply_lexicon("nothing to see", lexicon) == "nothing to see"
