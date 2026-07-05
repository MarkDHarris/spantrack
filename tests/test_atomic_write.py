from pathlib import Path

import pytest

from spantrack import atomic_write


def test_replace_writes_produced_content(tmp_path: Path):
    target = tmp_path / "out.json"

    atomic_write.replace(target, lambda tmp: tmp.write_text("payload"))

    assert target.read_text() == "payload"


def test_replace_overwrites_an_existing_file(tmp_path: Path):
    target = tmp_path / "out.json"
    target.write_text("stale")

    atomic_write.replace(target, lambda tmp: tmp.write_text("fresh"))

    assert target.read_text() == "fresh"


def test_replace_leaves_no_temp_file_behind(tmp_path: Path):
    target = tmp_path / "out.json"

    atomic_write.replace(target, lambda tmp: tmp.write_text("payload"))

    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_producer_writes_a_sibling_temp_not_the_target(tmp_path: Path):
    target = tmp_path / "out.json"
    seen: list[Path] = []

    def record(tmp: Path) -> None:
        seen.append(tmp)
        tmp.write_text("x")

    atomic_write.replace(target, record)

    (produced,) = seen
    assert produced != target
    assert produced.parent == target.parent  # same dir -> replace is atomic


def test_a_failing_producer_leaves_the_original_intact(tmp_path: Path):
    target = tmp_path / "out.json"
    target.write_text("original")

    def boom(_tmp: Path) -> None:
        msg = "producer failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="producer failed"):
        atomic_write.replace(target, boom)

    assert target.read_text() == "original"
