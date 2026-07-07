"""Select the spine-ordered set of content documents to narrate.

Selection is explicit and inspectable: every document the spine offers is either
included or excluded with a recorded reason, so the `--dry-run` printout can justify
each decision with no silent heuristics. Skipping is driven by `epub:type` landmarks
(cover/toc and the `bodymatter` start-of-reading marker), the nav document, spine
`linear="no"`, and media type.
"""

import enum
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from spantrack.lxml_reading import (
    EPUB_NS,
    XHTML_NS,
    XmlElement,
    parse_xml_file,
    qname,
    split_fragment,
)
from spantrack.opf import (
    XHTML_MEDIA_TYPE,
    ManifestItem,
    Package,
    SpineItem,
    resolve_within_container,
)

__all__ = [
    "Document",
    "Excluded",
    "Reason",
    "Selection",
    "apply_overrides",
    "read_landmarks",
    "select",
]

# epub:type landmark values whose target document is navigational or non-prose and must
# never be narrated.
_SKIP_LANDMARKS = frozenset({"cover", "toc", "loi", "lot", "landmarks"})
_BODYMATTER = "bodymatter"


class Reason(enum.StrEnum):
    """A recorded reason a spine document was left out of the narration set.

    A landmark exclusion is emitted as a `_LANDMARK_PREFIX`-tagged string rather than a
    member, because its `epub:type` suffix is data, not a fixed value.
    """

    UNRESOLVED_ITEM = "unresolved-item"
    NAV = "nav"
    NON_LINEAR = "non-linear"
    NON_CONTENT = "non-content"
    FRONT_MATTER = "front-matter"
    NOT_PICKED = "not-picked"
    EXCLUDED = "excluded"


_LANDMARK_PREFIX = "landmark:"

# Soft default exclusions that `--include` may override; `landmark:*` reasons are soft
# too (resurrected by `_is_soft`). Structural reasons never resurrect: those documents
# carry no prose, so nav/non-content/unresolved-item stay excluded.
_SOFT_REASONS = frozenset({Reason.NON_LINEAR, Reason.FRONT_MATTER})


@dataclass(frozen=True, slots=True)
class Document:
    """A content document chosen for narration."""

    item_id: str
    href: str
    spine_index: int


@dataclass(frozen=True, slots=True)
class Excluded:
    """A spine document left out of the narration set, with the reason why."""

    item_id: str
    href: str
    reason: str
    spine_index: int


@dataclass(frozen=True, slots=True)
class Selection:
    """The narration set with its exclusions and any dud override selectors.

    `unmatched` holds override selectors that matched no manifest document, so the
    caller can report a typo rather than silently narrating the wrong set.
    """

    documents: tuple[Document, ...]
    excluded: tuple[Excluded, ...]
    unmatched: tuple[str, ...] = ()


def select(package: Package, *, landmarks: Mapping[str, str]) -> Selection:
    """Choose the spine-ordered content documents to narrate.

    Args:
        package: The parsed (EPUB 3) package.
        landmarks: Map of container-relative document href to its `epub:type` landmark,
            as produced by `read_landmarks`.

    Returns:
        The selection, preserving spine order in both included and excluded lists.
    """
    nav = package.nav_item()
    nav_href = nav.href if nav is not None else None
    body_start = _body_start_index(package, landmarks)

    documents: list[Document] = []
    excluded: list[Excluded] = []
    for index, spine_item in enumerate(package.spine):
        item = package.item_by_id(spine_item.idref)
        if item is None:
            excluded.append(
                Excluded(
                    spine_item.idref,
                    "",
                    Reason.UNRESOLVED_ITEM,
                    index,
                )
            )
            continue

        reason = _exclusion_reason(
            item,
            spine_item,
            nav_href=nav_href,
            landmark=landmarks.get(_strip_fragment(item.href)),
            is_front_matter=index < body_start,
        )
        if reason is None:
            documents.append(Document(item.item_id, item.href, index))
        else:
            excluded.append(Excluded(item.item_id, item.href, reason, index))

    return Selection(tuple(documents), tuple(excluded))


def _exclusion_reason(
    item: ManifestItem,
    spine_item: SpineItem,
    *,
    nav_href: str | None,
    landmark: str | None,
    is_front_matter: bool,
) -> str | None:
    if item.href == nav_href:
        return Reason.NAV

    if not spine_item.linear:
        return Reason.NON_LINEAR

    if item.media_type != XHTML_MEDIA_TYPE:
        return Reason.NON_CONTENT

    if landmark in _SKIP_LANDMARKS:
        return f"{_LANDMARK_PREFIX}{landmark}"

    if is_front_matter:
        return Reason.FRONT_MATTER

    return None


def _body_start_index(package: Package, landmarks: Mapping[str, str]) -> int:
    target = next(
        (href for href, kind in landmarks.items() if kind == _BODYMATTER),
        None,
    )
    if target is None:
        return 0

    for index, spine_item in enumerate(package.spine):
        item = package.item_by_id(spine_item.idref)
        if item is not None and _strip_fragment(item.href) == target:
            return index

    # The bodymatter landmark points outside the spine (a dangling marker): fall back to
    # index 0 so nothing is treated as front matter. Widening narration is safer than
    # letting a bad marker classify the whole book as skippable front matter.
    return 0


def apply_overrides(
    selection: Selection,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    pick: Sequence[str] = (),
) -> Selection:
    """Layer the CLI narration-set overrides onto a default selection.

    `pick` replaces the set entirely with the named documents (in spine order),
    ignoring `include`/`exclude`. Otherwise `include` resurrects documents the default
    dropped for a soft reason and `exclude` removes documents, with `exclude` winning
    ties. Selectors are case-sensitive globs matched against each document's
    container-relative href and its basename; a selector matching no document is
    reported in `unmatched`.

    Args:
        selection: The default selection from `select`.
        include: Globs of documents to add back to the set.
        exclude: Globs of documents to drop from the set.
        pick: Globs naming the exact documents to narrate.

    Returns:
        The overridden selection, with unmatched selectors recorded.
    """
    entries = _ordered_entries(selection)
    if pick:
        return _apply_pick(entries, pick)

    return _apply_include_exclude(entries, include, exclude)


@dataclass(frozen=True, slots=True)
class _Entry:
    """A spine document merged from the default selection for override processing.

    `reason` is the discriminator: None marks a currently-included document, a string
    the default exclusion reason. Keyed by `spine_index` so a document appearing at two
    spine positions keeps its distinct decisions.
    """

    spine_index: int
    item_id: str
    href: str
    reason: str | None


def _ordered_entries(selection: Selection) -> list[_Entry]:
    # Rebuild the per-spine-position view, keyed by spine index rather than href so a
    # document that appears at two positions keeps its distinct default decisions.
    entries = [
        _Entry(doc.spine_index, doc.item_id, doc.href, None)
        for doc in selection.documents
    ]
    entries += [
        _Entry(ex.spine_index, ex.item_id, ex.href, ex.reason)
        for ex in selection.excluded
    ]
    entries.sort(key=lambda entry: entry.spine_index)

    return entries


def _apply_pick(entries: list[_Entry], pick: Sequence[str]) -> Selection:
    documents: list[Document] = []
    excluded: list[Excluded] = []
    for entry in entries:
        if entry.href and _matches_any(entry.href, pick):
            documents.append(Document(entry.item_id, entry.href, entry.spine_index))
        else:
            reason = (
                Reason.NOT_PICKED if entry.href else entry.reason or Reason.EXCLUDED
            )
            excluded.append(
                Excluded(entry.item_id, entry.href, reason, entry.spine_index),
            )

    return Selection(tuple(documents), tuple(excluded), _unmatched(entries, pick))


def _apply_include_exclude(
    entries: list[_Entry],
    include: Sequence[str],
    exclude: Sequence[str],
) -> Selection:
    documents: list[Document] = []
    excluded: list[Excluded] = []
    for entry in entries:
        if not entry.href:
            excluded.append(
                Excluded(
                    entry.item_id,
                    entry.href,
                    entry.reason or Reason.EXCLUDED,
                    entry.spine_index,
                ),
            )
            continue

        included, reason = _override_decision(
            entry.reason,
            has_include_match=_matches_any(entry.href, include),
            has_exclude_match=_matches_any(entry.href, exclude),
        )
        if included:
            documents.append(Document(entry.item_id, entry.href, entry.spine_index))
        else:
            excluded.append(
                Excluded(
                    entry.item_id,
                    entry.href,
                    reason or Reason.EXCLUDED,
                    entry.spine_index,
                ),
            )

    return Selection(
        tuple(documents),
        tuple(excluded),
        _unmatched(entries, [*include, *exclude]),
    )


def _override_decision(
    default_reason: str | None,
    *,
    has_include_match: bool,
    has_exclude_match: bool,
) -> tuple[bool, str | None]:
    if default_reason is None:
        if has_exclude_match:
            return False, Reason.EXCLUDED

        return True, None

    if has_include_match and not has_exclude_match and _is_soft(default_reason):
        return True, None

    return False, default_reason


def _is_soft(reason: str) -> bool:
    return reason in _SOFT_REASONS or reason.startswith(_LANDMARK_PREFIX)


def _matches(selector: str, href: str, basename: str) -> bool:
    return fnmatchcase(href, selector) or fnmatchcase(basename, selector)


def _matches_any(href: str, selectors: Sequence[str]) -> bool:
    basename = posixpath.basename(href)
    return any(_matches(selector, href, basename) for selector in selectors)


def _unmatched(entries: list[_Entry], selectors: Sequence[str]) -> tuple[str, ...]:
    # Pair each href with its basename once, rather than recomputing the basename for
    # every (selector, href) combination in the loop below.
    candidates = [
        (entry.href, posixpath.basename(entry.href)) for entry in entries if entry.href
    ]
    seen: set[str] = set()
    unmatched: list[str] = []
    for selector in selectors:
        if selector in seen:
            continue

        seen.add(selector)
        if not any(_matches(selector, href, base) for href, base in candidates):
            unmatched.append(selector)

    return tuple(unmatched)


def read_landmarks(container_root: Path, package: Package) -> dict[str, str]:
    """Read the landmarks nav into a container-relative href to `epub:type` map.

    Args:
        container_root: The container root directory.
        package: The parsed package (used to locate the nav document).

    Returns:
        A mapping of document href (fragment stripped) to its first landmark
        `epub:type`; empty when there is no nav or no landmarks section.
    """
    nav = package.nav_item()
    if nav is None:
        return {}

    nav_path = container_root / nav.href
    if not nav_path.is_file():
        return {}

    landmarks_nav = _landmarks_nav(parse_xml_file(nav_path))
    if landmarks_nav is None:
        return {}

    nav_dir = posixpath.dirname(nav.href)
    landmarks: dict[str, str] = {}
    for anchor in landmarks_nav.iter(qname(XHTML_NS, "a")):
        kind = anchor.get(qname(EPUB_NS, "type"))
        href = anchor.get("href")
        if not kind or not href:
            continue

        target = _resolve(nav_dir, href)
        if target:
            landmarks.setdefault(target, kind)

    return landmarks


def _landmarks_nav(root: XmlElement) -> XmlElement | None:
    for nav in root.iter(qname(XHTML_NS, "nav")):
        if nav.get(qname(EPUB_NS, "type")) == "landmarks":
            return nav

    return None


def _resolve(base_dir: str, href: str) -> str:
    # A landmark href that is absolute or escapes the container is dropped (empty)
    # rather than seeding a bogus, unmatchable key.
    return resolve_within_container(base_dir, href) or ""


def _strip_fragment(href: str) -> str:
    """Drop the fragment from an href so it keys the fragment-less landmark map."""
    return split_fragment(href)[0]
