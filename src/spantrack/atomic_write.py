"""Crash-safe file replacement: write a sibling temp file, then `os.replace` over it.

Several stores (`build_state`, `chapter_sidecar`, `sentence_cache`) must never leave a
half-written file that a resumed run would read as complete. They all need the same
primitive -- populate a temp file next to the target, then atomically rename it into
place -- so it lives here once. Callers keep their own serialization (JSON, WAV); this
module owns only the atomicity, and any future durability hardening (fsync of the temp
file and its directory) belongs here where every writer inherits it.
"""

from collections.abc import Callable
from pathlib import Path

__all__ = ["replace"]


def replace(path: Path, write: Callable[[Path], object]) -> None:
    """Write `path` atomically: `write` fills a sibling temp, then it is renamed over.

    Args:
        path: The destination file, overwritten if it already exists.
        write: Fills the passed temp path with the file's content; any return value is
            ignored. It receives a sibling path in the same directory (so the final
            rename stays on one filesystem and is atomic); a store that cannot infer its
            format from the `.tmp` suffix must name it explicitly.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    write(tmp)
    tmp.replace(path)
