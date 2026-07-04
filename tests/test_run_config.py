"""Tests for run configuration helpers."""

import pytest

from spantrack.run_config import parse_bitrate


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
