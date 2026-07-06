"""Tests for run configuration helpers."""

import pytest

from spantrack.run_config import AudioFormat, parse_bitrate


@pytest.mark.parametrize(
    ("bitrate", "expected"),
    [
        ("64k", 64_000),
        ("128000", 128_000),
        ("1.5M", 1_500_000),
        ("96K", 96_000),
        (" 64k ", 64_000),
    ],
)
def test_parse_bitrate_valid(bitrate: str, expected: int):
    assert parse_bitrate(bitrate) == expected


@pytest.mark.parametrize("bitrate", ["fast", "", "k64", "6-4k", "64kb"])
def test_parse_bitrate_rejects_invalid(bitrate: str):
    with pytest.raises(ValueError, match="bitrate"):
        parse_bitrate(bitrate)


def test_audio_format_extension_matches_container():
    assert AudioFormat.MP3.extension == "mp3"
    assert AudioFormat.OPUS.extension == "opus"


def test_audio_format_media_type_is_the_epub_manifest_type():
    assert AudioFormat.MP3.media_type == "audio/mpeg"
    assert AudioFormat.OPUS.media_type == "audio/ogg; codecs=opus"


def test_every_audio_format_resolves_both_facts():
    for audio_format in AudioFormat:
        assert audio_format.extension
        assert audio_format.media_type
