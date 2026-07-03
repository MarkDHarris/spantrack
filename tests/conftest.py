import zipfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_EPUB_SRC = FIXTURES / "minimal_epub"


def _source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


@pytest.fixture
def minimal_epub_src() -> Path:
    return MINIMAL_EPUB_SRC


@pytest.fixture
def real_epub_path() -> Path:
    """A real-world EPUB from .epubs/ for integration checks; skipped when absent.

    The books there are copyrighted and gitignored, so this never runs in CI.
    """
    root = Path(__file__).resolve().parents[1]

    matches = sorted((root / ".epubs").glob("*.epub"))
    if not matches:
        pytest.skip("no real EPUB present in .epubs/")

    return matches[0]


@pytest.fixture
def minimal_epub_path(tmp_path: Path, minimal_epub_src: Path) -> Path:
    """Zip the readable minimal-EPUB source tree into a valid .epub.

    Built with raw zipfile (mimetype first, STORED) so the unzip test does not depend on
    the production repack under test in the next task.
    """
    epub_path = tmp_path / "minimal.epub"
    mimetype = minimal_epub_src / "mimetype"
    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
        for path in _source_files(minimal_epub_src):
            if path == mimetype:
                continue

            zf.write(path, path.relative_to(minimal_epub_src).as_posix())

    return epub_path
