"""Tests for human-readable progress reporting to a text stream."""

from collections.abc import Callable
from io import StringIO

import pytest
from narration_documents import make_document

from spantrack.narration_projection import SelectionProjection
from spantrack.narration_set import Excluded, Selection
from spantrack.progress import (
    ProgressBarUnavailableError,
    ProgressReporter,
    ensure_tqdm_available,
)


def _render(action: Callable[[ProgressReporter], None]) -> str:
    stream = StringIO()
    action(ProgressReporter(stream))
    return stream.getvalue()


def test_selection_lists_documents_skips_and_unmatched():
    selection = Selection(
        documents=(make_document("ch1.xhtml", 0), make_document("ch2.xhtml", 1)),
        excluded=(
            Excluded(item_id="nav", href="nav.xhtml", reason="nav", spine_index=2),
        ),
        unmatched=("ghost*.xhtml",),
    )

    out = _render(lambda r: r.selection(selection))

    assert "narration set: 2 document(s)" in out
    assert "  narrate  ch1.xhtml" in out
    assert "  narrate  ch2.xhtml" in out
    assert "  skip     nav.xhtml (nav)" in out
    assert "  warning  override matched no document: ghost*.xhtml" in out


def test_excluded_without_href_falls_back_to_item_id():
    selection = Selection(
        documents=(),
        excluded=(
            Excluded(item_id="x1", href="", reason="unresolved-item", spine_index=0),
        ),
        unmatched=(),
    )

    out = _render(lambda r: r.selection(selection))

    assert "  skip     x1 (unresolved-item)" in out


@pytest.mark.parametrize(
    ("duration_seconds", "size_bytes", "expected_duration", "expected_size"),
    [
        (45.0, 900, "45s", "900 B"),
        (90.0, 2048, "1m 30s", "2.0 KB"),
        (3661.0, 5_000_000, "1h 1m 1s", "4.8 MB"),
        (7200.0, 2_000_000_000, "2h 0m 0s", "1.9 GB"),
    ],
)
def test_projection_formats_duration_and_size(
    duration_seconds: float,
    size_bytes: int,
    expected_duration: str,
    expected_size: str,
):
    projection = SelectionProjection(
        documents=(),
        total_characters=1234,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
    )

    out = _render(lambda r: r.projection(projection))

    assert "1,234 characters" in out
    assert f"~{expected_duration} audio" in out
    assert f"~{expected_size}" in out


def test_oov_empty_reports_none_found():
    out = _render(lambda r: r.oov([]))

    assert out.strip() == "no out-of-vocabulary words found"


def test_oov_lists_words_as_lexicon_skeleton():
    out = _render(lambda r: r.oov(["Gandalf", "Frodo"]))

    assert "out-of-vocabulary words: 2" in out
    assert "[words]" in out
    assert '# Gandalf = ""' in out
    assert '# Frodo = ""' in out


def test_narrating_without_a_bar_prints_a_line_per_chapter():
    stream = StringIO()

    with ProgressReporter(stream).narrating(2) as chapters:
        chapters.narrated("OEBPS/chapter1.xhtml")
        chapters.narrated("OEBPS/chapter2.xhtml")

    out = stream.getvalue()
    assert "narrated OEBPS/chapter1.xhtml" in out
    assert "narrated OEBPS/chapter2.xhtml" in out


def test_narrating_with_a_bar_renders_a_progress_bar():
    stream = StringIO()

    with ProgressReporter(stream, progress_bar=True).narrating(2) as chapters:
        chapters.narrated("a.xhtml")
        chapters.narrated("b.xhtml")

    out = stream.getvalue()
    assert "2/2" in out  # tqdm renders the completed/total count
    assert "narrated a.xhtml" not in out  # the bar replaces the per-chapter lines


def test_ensure_tqdm_available_raises_when_absent(monkeypatch: pytest.MonkeyPatch):
    def no_spec(_name: str) -> None:
        return None

    monkeypatch.setattr("importlib.util.find_spec", no_spec)

    with pytest.raises(ProgressBarUnavailableError, match="tqdm"):
        ensure_tqdm_available()
