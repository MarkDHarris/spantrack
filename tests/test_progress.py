"""Tests for human-readable progress reporting to a text stream."""

from collections.abc import Callable
from io import StringIO

import pytest
from narration_documents import make_document

from spantrack.narration_projection import SelectionProjection
from spantrack.narration_set import Excluded, Selection
from spantrack.progress import ProgressReporter


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
