"""Tests for the TTSBackend protocol contract, exercised via a fake backend."""

import re
from pathlib import Path

import pytest

from spantrack.lexicon import Lexicon, load
from spantrack.tts.backend import TTSBackend

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


class _OnlyValidate:
    """A partial implementation missing `apply_lexicon`."""

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


def test_fake_type_checks_against_protocol():
    # A static conformance check: pyright strict verifies the method signatures.
    backend: TTSBackend = _FakeBackend()
    assert isinstance(backend, TTSBackend)


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
