"""Read and write the EPUB OCF ZIP container."""

import posixpath
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

MIMETYPE_NAME = "mimetype"


def unzip(epub_path: Path, dest_dir: Path) -> list[str]:
    """Extract the EPUB container into a directory.

    Args:
        epub_path: The .epub (OCF ZIP) file to read.
        dest_dir: Directory to extract into; created if missing.

    Returns:
        Archive member names in archive order (POSIX-separated, directory entries
        excluded), so a later repack can preserve entry order.

    Raises:
        ValueError: If a member path escapes dest_dir (zip-slip).
    """
    names: list[str] = []
    with zipfile.ZipFile(epub_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            target = _safe_destination(dest_dir, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)

            names.append(info.filename)

    return names


def repack(
    source_dir: Path,
    epub_path: Path,
    order: Sequence[str] | None = None,
) -> None:
    """Zip a working directory back into a valid EPUB container.

    The OCF rule is honored: `mimetype` is written first, uncompressed (STORED), with no
    extra field; every other entry is deflated. The round trip preserves file contents
    byte-for-byte, not the archive bytes themselves (deflate output is not guaranteed
    identical).

    Args:
        source_dir: Directory holding the unpacked container.
        epub_path: Destination .epub path; overwritten if it exists.
        order: Member names to emit first, in this order (typically the list returned by
            unzip). Any files not listed follow in sorted order.

    Raises:
        ValueError: If source_dir has no `mimetype` file.
    """
    mimetype = source_dir / MIMETYPE_NAME
    if not mimetype.is_file():
        err_msg = f"missing {MIMETYPE_NAME} in {source_dir}"
        raise ValueError(err_msg)

    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as archive:
        mimetype_info = zipfile.ZipInfo(MIMETYPE_NAME)
        mimetype_info.compress_type = zipfile.ZIP_STORED

        archive.writestr(mimetype_info, mimetype.read_bytes())

        for name in _ordered_members(source_dir, order):
            archive.write(source_dir.joinpath(*name.split("/")), name)


def _ordered_members(source_dir: Path, order: Sequence[str] | None) -> list[str]:
    remaining = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    remaining.discard(MIMETYPE_NAME)

    ordered: list[str] = []
    for name in order or ():
        if name in remaining:
            ordered.append(name)
            remaining.discard(name)

    ordered.extend(sorted(remaining))

    return ordered


def _safe_destination(dest_dir: Path, member_name: str) -> Path:
    normalized = posixpath.normpath(member_name)
    if normalized.startswith(("/", "../")) or normalized == "..":
        err_msg = f"unsafe archive member path: {member_name!r}"
        raise ValueError(err_msg)

    return dest_dir.joinpath(*normalized.split("/"))
