"""Tests for the content-addressed sentence cache."""

from pathlib import Path

import numpy as np

from spantrack.sentence_cache import SentenceCache
from spantrack.tts.backend import TokenTimestamp

_PCM = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
_TOKENS = (
    TokenTimestamp(text="hello", start_s=0.0, end_s=0.5, whitespace=" "),
    TokenTimestamp(text="world", start_s=0.5, end_s=1.0, whitespace=""),
)


def _wav_path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.wav"


def test_key_is_stable():
    assert SentenceCache.key("m", "v", 1.0, "t") == SentenceCache.key(
        "m", "v", 1.0, "t"
    )


def test_key_busts_on_each_field():
    base = SentenceCache.key("m", "v", 1.0, "t")
    assert SentenceCache.key("m2", "v", 1.0, "t") != base
    assert SentenceCache.key("m", "v2", 1.0, "t") != base
    assert SentenceCache.key("m", "v", 1.1, "t") != base
    assert SentenceCache.key("m", "v", 1.0, "t2") != base


def test_key_avoids_field_boundary_collision():
    # Shifting a character across a field boundary must change the key; a naive
    # separator-less concatenation would collide these.
    assert SentenceCache.key("ab", "c", 1.0, "t") != SentenceCache.key(
        "a", "bc", 1.0, "t"
    )


def test_miss_returns_none(tmp_path: Path):
    cache = SentenceCache(root=tmp_path)
    assert cache.get(SentenceCache.key("m", "v", 1.0, "absent")) is None


def test_round_trip_pcm_and_tokens(tmp_path: Path):
    cache = SentenceCache(root=tmp_path)
    key = SentenceCache.key("m", "v", 1.0, "hello world")
    cache.put(key, _PCM, _TOKENS)

    hit = cache.get(key)
    assert hit is not None
    pcm, tokens = hit
    assert np.array_equal(pcm, _PCM)
    assert pcm.dtype == np.float32
    assert tokens == _TOKENS


def test_round_trip_none_tokens(tmp_path: Path):
    cache = SentenceCache(root=tmp_path)
    key = SentenceCache.key("m", "v", 1.0, "no timestamps")
    cache.put(key, _PCM, None)

    hit = cache.get(key)
    assert hit is not None
    pcm, tokens = hit
    assert np.array_equal(pcm, _PCM)
    assert tokens is None


def test_stored_wav_is_byte_identical_on_re_put(tmp_path: Path):
    cache = SentenceCache(root=tmp_path)
    key = SentenceCache.key("m", "v", 1.0, "determinism")

    cache.put(key, _PCM, _TOKENS)
    first = _wav_path(tmp_path, key).read_bytes()
    cache.put(key, _PCM, _TOKENS)
    second = _wav_path(tmp_path, key).read_bytes()

    assert first == second


def test_entry_lives_under_two_char_shard(tmp_path: Path):
    cache = SentenceCache(root=tmp_path)
    key = SentenceCache.key("m", "v", 1.0, "sharded")
    cache.put(key, _PCM, _TOKENS)

    assert _wav_path(tmp_path, key).is_file()
    assert _wav_path(tmp_path, key).parent.name == key[:2]


def test_orphan_tokens_without_wav_is_a_miss(tmp_path: Path):
    # Simulate a crash after the tokens sidecar but before the audio: the .wav is the
    # completion marker, so a half-written entry re-synthesizes rather than corrupts.
    cache = SentenceCache(root=tmp_path)
    key = SentenceCache.key("m", "v", 1.0, "crashed")
    shard = tmp_path / key[:2]
    shard.mkdir()
    (shard / f"{key}.tokens.json").write_text("null", encoding="utf-8")

    assert cache.get(key) is None
