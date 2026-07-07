"""Tests for build.json run-parameter fingerprint + per-doc checkpoints."""

import json
from pathlib import Path

from spantrack import build_state
from spantrack.build_state import BuildParameters


def _parameters(
    *,
    voice: str = "af_heart",
    selection: tuple[str, ...] = ("ch1.xhtml", "ch2.xhtml"),
) -> BuildParameters:
    return BuildParameters(
        voice=voice,
        speed=1.0,
        granularity="sentence",
        audio_format="mp3",
        bitrate=64_000,
        selection=selection,
    )


def _stored(
    *,
    schema_version: int = 1,
    parameters: dict[str, object] | None = None,
) -> str:
    # A payload that would restore `completed` if it were well-formed and current.
    payload = {
        "schema_version": schema_version,
        "parameters": parameters
        if parameters is not None
        else {
            "voice": "af_heart",
            "speed": 1.0,
            "granularity": "sentence",
            "audio_format": "mp3",
            "bitrate": 64_000,
            "selection": ["ch1.xhtml", "ch2.xhtml"],
        },
        "completed": ["ch1.xhtml"],
    }
    return json.dumps(payload)


def test_load_missing_returns_a_fresh_state(tmp_path: Path):
    state = build_state.load(tmp_path / "build.json", _parameters())

    assert state.parameters == _parameters()
    assert not state.is_done("ch1.xhtml")


def test_mark_done_tracks_completed_docs(tmp_path: Path):
    state = build_state.load(tmp_path / "build.json", _parameters())

    state.mark_done("ch1.xhtml")

    assert state.is_done("ch1.xhtml")
    assert not state.is_done("ch2.xhtml")


def test_save_then_load_restores_checkpoints(tmp_path: Path):
    path = tmp_path / "build.json"
    state = build_state.load(path, _parameters())
    state.mark_done("ch1.xhtml")
    build_state.save(path, state)

    resumed = build_state.load(path, _parameters())

    assert resumed.is_done("ch1.xhtml")
    assert not resumed.is_done("ch2.xhtml")


def test_parameter_mismatch_discards_checkpoints(tmp_path: Path):
    path = tmp_path / "build.json"
    state = build_state.load(path, _parameters(voice="af_heart"))
    state.mark_done("ch1.xhtml")
    build_state.save(path, state)

    # A different voice means the cached audio (and thus every chapter) is stale.
    resumed = build_state.load(path, _parameters(voice="am_adam"))

    assert not resumed.is_done("ch1.xhtml")
    assert resumed.parameters.voice == "am_adam"


def test_malformed_build_json_is_a_clean_start(tmp_path: Path):
    path = tmp_path / "build.json"
    path.write_text("{ not json", encoding="utf-8")

    state = build_state.load(path, _parameters())

    assert not state.is_done("ch1.xhtml")


def test_unknown_schema_version_is_a_clean_start(tmp_path: Path):
    path = tmp_path / "build.json"
    path.write_text(_stored(schema_version=999), encoding="utf-8")

    assert not build_state.load(path, _parameters()).is_done("ch1.xhtml")


def test_non_object_json_is_a_clean_start(tmp_path: Path):
    path = tmp_path / "build.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert not build_state.load(path, _parameters()).is_done("ch1.xhtml")


def test_wrong_shaped_parameters_is_a_clean_start(tmp_path: Path):
    path = tmp_path / "build.json"
    path.write_text(_stored(parameters={"voice": "af_heart"}), encoding="utf-8")

    assert not build_state.load(path, _parameters()).is_done("ch1.xhtml")


def test_uncoercible_parameter_value_is_a_clean_start(tmp_path: Path):
    # A well-shaped payload whose values are present but uncoercible (speed "fast") must
    # restart cleanly via the ValueError/TypeError arm of _read's except, not just the
    # KeyError arm a missing key hits.
    path = tmp_path / "build.json"
    path.write_text(
        _stored(
            parameters={
                "voice": "af_heart",
                "speed": "fast",  # float("fast") raises ValueError inside _read
                "granularity": "sentence",
                "audio_format": "mp3",
                "bitrate": 64_000,
                "selection": ["ch1.xhtml", "ch2.xhtml"],
            },
        ),
        encoding="utf-8",
    )

    assert not build_state.load(path, _parameters()).is_done("ch1.xhtml")


def test_save_is_byte_deterministic(tmp_path: Path):
    path = tmp_path / "build.json"
    state = build_state.load(path, _parameters())
    state.mark_done("ch2.xhtml")
    state.mark_done("ch1.xhtml")

    build_state.save(path, state)
    first = path.read_bytes()
    build_state.save(path, state)

    assert path.read_bytes() == first
