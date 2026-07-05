"""Tests for the out-of-vocabulary report collector."""

from collections.abc import Sequence

from spantrack.oov_report import build_g2p, find_oov


class _FakeToken:
    def __init__(self, text: str, phonemes: str | None) -> None:
        self.text = text
        self.phonemes = phonemes


class _FakeG2P:
    """Phonemizes known words; leaves the rest `None`, like fallback-disabled misaki."""

    def __init__(self, known: Sequence[str]) -> None:
        self._known = {word.lower() for word in known}

    def __call__(self, text: str) -> tuple[str, Sequence[_FakeToken]]:
        tokens = [
            _FakeToken(word, "X" if word.lower() in self._known else None)
            for word in text.split()
        ]
        return "", tokens


def test_find_oov_reports_unknown_words():
    assert find_oov(["the cat sat"], _FakeG2P(["the", "cat"])) == ["sat"]


def test_find_oov_dedups_case_insensitively_keeping_first():
    assert find_oov(["Foo foo FOO"], _FakeG2P([])) == ["Foo"]


def test_find_oov_excludes_non_alpha_tokens():
    assert find_oov(["hello 123 ..."], _FakeG2P([])) == ["hello"]


def test_find_oov_spans_multiple_texts():
    g2p = _FakeG2P(["known"])
    assert find_oov(["known alpha", "beta known"], g2p) == ["alpha", "beta"]


def test_find_oov_empty_input():
    assert find_oov([], _FakeG2P([])) == []


def test_real_g2p_detects_gibberish_not_real_words():
    words = find_oov(["The quixotic zzzblornak ran."], build_g2p(british=False))

    assert "zzzblornak" in words
    assert "quixotic" not in words
