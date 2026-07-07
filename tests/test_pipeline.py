import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import HAS_FFMPEG
from numpy.typing import NDArray

from spantrack import (
    cli,
    epub_container,
    narration_projection,
    narration_set,
    pipeline,
    smil,
)
from spantrack.content_namespace import ContentNamespaceError
from spantrack.epubcheck import EpubcheckResult
from spantrack.lexicon import Lexicon
from spantrack.progress import ProgressReporter
from spantrack.run_config import RunConfig
from spantrack.tts import kokoro_backend
from spantrack.tts.backend import TokenTimestamp


class _RecordingReporter(ProgressReporter):
    def __init__(self) -> None:
        self.selections: list[narration_set.Selection] = []
        self.projections: list[narration_projection.SelectionProjection] = []

    def info(self, message: str) -> None:
        pass

    def selection(self, selection: narration_set.Selection) -> None:
        self.selections.append(selection)

    def projection(self, projection: narration_projection.SelectionProjection) -> None:
        self.projections.append(projection)


def _config(epub: Path, tmp_path: Path, *extra: str) -> RunConfig:
    return cli.parse_args(
        [
            str(epub),
            "-o",
            str(tmp_path / "out.epub"),
            "--work-dir",
            str(tmp_path / "work"),
            *extra,
        ],
    )


class _StubBackend:
    """A weightless backend: a 0.2s tone + one evenly-timed token per source word."""

    name = "stub"
    model_version = "stub@1"
    voice = "af_heart"
    speed = 1.0

    def validate_lexicon(self, lexicon: Lexicon) -> None:
        _ = lexicon

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str:
        _ = lexicon
        return text

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        pcm = np.full(4_800, 0.3, dtype=np.float32)  # 0.2s at 24 kHz, all voiced
        words = text.split()
        if not words:
            return pcm, (TokenTimestamp(text, 0.0, 0.2, ""),)

        step = 0.2 / len(words)
        tokens = tuple(
            TokenTimestamp(word, index * step, (index + 1) * step, " ")
            for index, word in enumerate(words)
        )

        return pcm, tokens


class _UntimedStub(_StubBackend):
    """A backend that emits audio but no per-token timestamps."""

    def synthesize(
        self,
        text: str,
    ) -> tuple[NDArray[np.float32], tuple[TokenTimestamp, ...] | None]:
        _ = text
        return np.full(4_800, 0.3, dtype=np.float32), None


@pytest.fixture
def stub_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real Kokoro build with a weightless stub; skip without ffmpeg."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    monkeypatch.setattr(kokoro_backend, "build", _stub_build)


@pytest.mark.usefixtures("stub_synthesis")
def test_run_produces_narrated_output(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    config = _config(minimal_epub_path, tmp_path)
    code = pipeline.run(config)

    assert code == 0
    assert config.output.exists()

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)
    # The narrated output carries a chapter mp3, a SMIL overlay, the highlight CSS, and
    # the media-overlay wiring in the OPF.
    assert list(result.rglob("*.mp3"))
    assert list(result.rglob("*.smil"))
    assert list(result.rglob("highlight.css"))

    opf_files = list(result.rglob("*.opf"))
    assert opf_files
    assert "media-overlay" in opf_files[0].read_text(encoding="utf-8")


@pytest.mark.usefixtures("stub_synthesis")
def test_opus_run_writes_opus_audio_referenced_in_smil(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    config = _config(minimal_epub_path, tmp_path, "--audio-format", "opus")

    assert pipeline.run(config) == 0

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)

    assert list(result.rglob("*.opus"))  # the chapter audio is Opus
    assert not list(result.rglob("*.mp3"))  # and no MP3 was written

    smil_text = next(result.rglob("*.smil")).read_text(encoding="utf-8")
    assert ".opus" in smil_text  # the <audio> src points at the .opus file


@pytest.mark.usefixtures("stub_synthesis")
def test_progress_bar_run_narrates_end_to_end(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    # The --progress path drives the tqdm bar through the real narration loop.
    config = _config(minimal_epub_path, tmp_path, "--progress")

    assert pipeline.run(config) == 0
    assert config.output.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("stub_synthesis")
def test_word_granularity_wraps_words_and_references_them_in_smil(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    config = _config(minimal_epub_path, tmp_path, "--granularity", "word")

    assert pipeline.run(config) == 0

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)

    docs = [path.read_text(encoding="utf-8") for path in result.rglob("*.xhtml")]
    assert any('-w0"' in doc for doc in docs)  # word spans were injected

    smil_text = next(result.rglob("*.smil")).read_text(encoding="utf-8")
    assert "-w0" in smil_text  # a <par> points at a word span, not a sentence span
    assert "media-overlay" in next(result.rglob("*.opf")).read_text(encoding="utf-8")


def _stub_build(_config: RunConfig) -> _StubBackend:
    return _StubBackend()


def _untimed_stub(_config: RunConfig) -> _UntimedStub:
    return _UntimedStub()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_word_mode_without_timestamps_fails_loud_by_default(
    minimal_epub_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setattr(kokoro_backend, "build", _untimed_stub)
    config = _config(minimal_epub_path, tmp_path, "--granularity", "word")

    with pytest.raises(pipeline.UntimedNarrationError):
        pipeline.run(config)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_word_mode_without_timestamps_falls_back_with_flag(
    minimal_epub_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setattr(kokoro_backend, "build", _untimed_stub)
    config = _config(
        minimal_epub_path,
        tmp_path,
        "--granularity",
        "word",
        "--allow-untimed-fallback",
    )

    assert pipeline.run(config) == 0

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)
    smil_text = next(result.rglob("*.smil")).read_text(encoding="utf-8")
    # The fallback still emits a <par>, pointed at the sentence's first word span.
    assert "-w0" in smil_text


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.usefixtures("stub_synthesis")
def test_word_mode_resume_reuses_audio_and_smil(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    config = _config(minimal_epub_path, tmp_path, "--granularity", "word")
    assert pipeline.run(config) == 0

    book = config.work_dir / "book"
    book_mp3 = next(book.rglob("*.mp3"))
    encoded_at = book_mp3.stat().st_mtime_ns
    smil_first = next(book.rglob("*.smil")).read_bytes()

    assert pipeline.run(config) == 0  # resume

    assert book_mp3.stat().st_mtime_ns == encoded_at  # not re-encoded
    assert next(book.rglob("*.smil")).read_bytes() == smil_first  # SMIL unchanged


def test_dry_run_writes_no_output(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--dry-run")

    assert pipeline.run(config) == 0
    assert not config.output.exists()


def test_dry_run_reports_selection_and_projection(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    reporter = _RecordingReporter()
    config = _config(minimal_epub2_path, tmp_path, "--dry-run")

    assert pipeline.run(config, reporter=reporter) == 0
    assert not config.output.exists()

    selection = reporter.selections[0]
    assert [doc.href for doc in selection.documents] == ["OEBPS/chapter1.xhtml"]
    assert reporter.projections[0].total_characters > 0


def test_pick_override_limits_narration_set(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    reporter = _RecordingReporter()
    config = _config(
        minimal_epub2_path,
        tmp_path,
        "--dry-run",
        "--pick",
        "cover.xhtml",
    )

    pipeline.run(config, reporter=reporter)

    assert [doc.href for doc in reporter.selections[0].documents] == [
        "OEBPS/cover.xhtml"
    ]


@pytest.mark.usefixtures("stub_synthesis")
def test_epub2_output_is_upgraded_to_epub3(minimal_epub2_path: Path, tmp_path: Path):
    config = _config(minimal_epub2_path, tmp_path)
    assert pipeline.run(config) == 0

    result = tmp_path / "result"
    epub_container.unzip(config.output, result)
    assert 'version="3.0"' in (result / "OEBPS" / "content.opf").read_text()
    assert (result / "OEBPS" / "nav.xhtml").is_file()


def test_dry_run_prints_selection_to_stream(
    minimal_epub2_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    config = _config(minimal_epub2_path, tmp_path, "--dry-run")
    pipeline.run(config)

    err = capsys.readouterr().err
    assert "OEBPS/chapter1.xhtml" in err
    assert "projected" in err


def test_oov_report_lists_missing_words_and_writes_no_output(
    minimal_epub2_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    # End-to-end smoke: the real misaki G2P runs over the fixture's normalized text,
    # the review list is emitted, and no output EPUB is written (emit-then-exit).
    config = _config(minimal_epub2_path, tmp_path, "--oov-report")

    assert pipeline.run(config) == 0
    assert not config.output.exists()
    assert "vocabulary" in capsys.readouterr().err


@pytest.mark.usefixtures("stub_synthesis")
def test_clean_rebuilds_work_dir(minimal_epub_path: Path, tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()

    stale = work / "stale.txt"
    stale.write_text("old")

    config = _config(minimal_epub_path, tmp_path, "--clean")

    assert pipeline.run(config) == 0
    assert not stale.exists()
    assert config.output.exists()


@pytest.mark.usefixtures("stub_synthesis")
def test_validate_pass_returns_zero(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--validate")
    code = pipeline.run(
        config,
        validator=lambda _: EpubcheckResult(ok=True, returncode=0, messages=()),
    )

    assert code == 0


@pytest.mark.usefixtures("stub_synthesis")
def test_validate_failure_returns_nonzero(minimal_epub_path: Path, tmp_path: Path):
    config = _config(minimal_epub_path, tmp_path, "--validate")
    code = pipeline.run(
        config,
        validator=lambda _: EpubcheckResult(
            ok=False,
            returncode=1,
            messages=("ERROR(RSC-005): broken",),
        ),
    )

    assert code == 1


@pytest.mark.usefixtures("stub_synthesis")
def test_validate_invoked_with_output_path(minimal_epub_path: Path, tmp_path: Path):
    seen: dict[str, Path] = {}

    def validator(path: Path) -> EpubcheckResult:
        seen["path"] = path
        return EpubcheckResult(ok=True, returncode=0, messages=())

    config = _config(minimal_epub_path, tmp_path, "--validate")
    pipeline.run(config, validator=validator)

    assert seen["path"] == config.output


@pytest.mark.usefixtures("stub_synthesis")
def test_no_validation_by_default(minimal_epub_path: Path, tmp_path: Path):
    def validator(path: Path) -> EpubcheckResult:  # noqa: ARG001
        raise AssertionError("validator must not run without --validate")

    config = _config(minimal_epub_path, tmp_path)

    assert pipeline.run(config, validator=validator) == 0


def test_non_xhtml_content_document_fails_loud(
    minimal_epub2_src: Path,
    tmp_path: Path,
    zip_epub_tree: Callable[[Path, Path], Path],
):
    # A chapter whose <html> omits the XHTML namespace would be silently narrated as
    # empty by every downstream stage; the ingest guard refuses the build instead. The
    # --dry-run path must fail too, since projection also counts such a document zero.
    src = tmp_path / "book_src"
    shutil.copytree(minimal_epub2_src, src)
    (src / "OEBPS" / "chapter1.xhtml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        "<html><head><title>Chapter 1</title></head>"
        "<body><h1>One</h1><p>Bare namespace.</p></body></html>",
        encoding="utf-8",
    )
    epub = zip_epub_tree(src, tmp_path / "bad.epub")
    config = _config(epub, tmp_path, "--dry-run")

    with pytest.raises(ContentNamespaceError, match=r"chapter1\.xhtml"):
        pipeline.run(config)


@pytest.mark.usefixtures("stub_synthesis")
def test_main_runs_end_to_end(minimal_epub_path: Path, tmp_path: Path):
    out = tmp_path / "out.epub"
    code = cli.main(
        [
            str(minimal_epub_path),
            "-o",
            str(out),
            "--work-dir",
            str(tmp_path / "w"),
        ],
    )

    assert code == 0
    assert out.exists()


_EPUBCHECK_JAR = Path(".epubcheck") / "epubcheck.jar"
_JDK_JAVA = Path(r"C:\Program Files\Java\jdk-26.0.1\bin\java.exe")


def _java_executable() -> str | None:
    found = shutil.which("java")
    if found is not None:
        return found
    return str(_JDK_JAVA) if _JDK_JAVA.is_file() else None


def _first_mp3_bytes(epub: Path, dest: Path) -> bytes:
    epub_container.unzip(epub, dest)
    return next(dest.rglob("*.mp3")).read_bytes()


@pytest.mark.usefixtures("stub_synthesis")
def test_resume_reuses_encoded_audio_without_re_encoding(
    minimal_epub_path: Path,
    tmp_path: Path,
):
    config = _config(minimal_epub_path, tmp_path)

    assert pipeline.run(config) == 0
    book_mp3 = next((config.work_dir / "book").rglob("*.mp3"))
    encoded_at = book_mp3.stat().st_mtime_ns
    first_output = _first_mp3_bytes(config.output, tmp_path / "r1")

    build_json = (config.work_dir / "build.json").read_text(encoding="utf-8")
    assert "chapter1.xhtml" in build_json  # the chapter is checkpointed done

    assert pipeline.run(config) == 0  # resume over the same work dir

    assert book_mp3.stat().st_mtime_ns == encoded_at  # not re-encoded (checkpoint skip)
    assert _first_mp3_bytes(config.output, tmp_path / "r2") == first_output


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.skipif(
    not HAS_FFMPEG or not _EPUBCHECK_JAR.is_file() or _java_executable() is None,
    reason="acceptance needs ffmpeg + epubcheck jar + java",
)
@pytest.mark.parametrize("granularity", ["sentence", "word"])
@pytest.mark.parametrize("fixture_name", ["minimal_epub_path", "minimal_epub2_path"])
def test_narrated_output_is_epubcheck_clean(
    fixture_name: str,
    granularity: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kokoro_backend, "build", _stub_build)
    epub: Path = request.getfixturevalue(fixture_name)
    extra = () if granularity == "sentence" else ("--granularity", "word")
    config = _config(epub, tmp_path, *extra)

    assert pipeline.run(config) == 0

    java = _java_executable()
    assert java is not None
    completed = subprocess.run(
        [java, "-jar", str(_EPUBCHECK_JAR), str(config.output)],
        capture_output=True,
        text=True,
        check=False,
    )

    # epubcheck exits 0 only when there are no errors or fatals (warnings are allowed).
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    not HAS_FFMPEG or not _EPUBCHECK_JAR.is_file() or _java_executable() is None,
    reason="acceptance needs ffmpeg + epubcheck jar + java",
)
@pytest.mark.parametrize("fixture_name", ["minimal_epub_path", "minimal_epub2_path"])
def test_opus_output_is_epubcheck_clean(
    fixture_name: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(kokoro_backend, "build", _stub_build)
    epub: Path = request.getfixturevalue(fixture_name)
    config = _config(epub, tmp_path, "--audio-format", "opus")

    assert pipeline.run(config) == 0

    result = tmp_path / "unzipped"
    epub_container.unzip(config.output, result)
    assert list(result.rglob("*.opus"))  # the chapter audio is Opus

    opf_text = next(result.rglob("*.opf")).read_text(encoding="utf-8")
    assert "audio/ogg; codecs=opus" in opf_text  # declared as the Opus core media type

    java = _java_executable()
    assert java is not None

    completed = subprocess.run(
        [java, "-jar", str(_EPUBCHECK_JAR), str(config.output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_millisecond_clock_collapses_sub_millisecond_clips():
    # The word-par filter drops a clip with no duration on the millisecond clock, since
    # epubcheck (MED-009) rejects clipBegin == clipEnd. Sub-ms bounds collapse to one
    # clock string; a >= 1 ms difference does not.
    assert smil.clock_value(0.5) == smil.clock_value(0.5004)
    assert smil.clock_value(0.5) != smil.clock_value(0.501)
