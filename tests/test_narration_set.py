from pathlib import Path

from spantrack import epub2_upgrade, epub_container, narration_set, opf


def _item(
    item_id: str,
    href: str,
    *,
    media_type: str = "application/xhtml+xml",
    properties: frozenset[str] = frozenset(),
) -> opf.ManifestItem:
    return opf.ManifestItem(item_id, href, media_type, properties)


def _spine(idref: str, *, linear: bool = True) -> opf.SpineItem:
    return opf.SpineItem(idref, linear)


def _package(
    manifest: list[opf.ManifestItem],
    spine: list[opf.SpineItem],
) -> opf.Package:
    return opf.Package(
        opf_path="OEBPS/content.opf",
        version="3.0",
        unique_identifier=None,
        manifest=tuple(manifest),
        spine=tuple(spine),
        spine_toc=None,
        guide=(),
    )


def test_default_selection_is_spine_order():
    package = _package(
        [_item("c1", "OEBPS/c1.xhtml"), _item("c2", "OEBPS/c2.xhtml")],
        [_spine("c2"), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})

    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c2.xhtml",
        "OEBPS/c1.xhtml",
    ]


def test_excludes_nav_document_in_spine():
    package = _package(
        [
            _item("nav", "OEBPS/nav.xhtml", properties=frozenset({"nav"})),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("nav"), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})
    assert [doc.href for doc in selection.documents] == ["OEBPS/c1.xhtml"]

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/nav.xhtml"] == "nav"


def test_excludes_non_linear_spine_item():
    package = _package(
        [_item("cover", "OEBPS/cover.xhtml"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("cover", linear=False), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/cover.xhtml"] == "non-linear"
    assert [doc.href for doc in selection.documents] == ["OEBPS/c1.xhtml"]


def test_excludes_landmarked_cover_and_toc():
    package = _package(
        [
            _item("cover", "OEBPS/cover.xhtml"),
            _item("toc", "OEBPS/toc.xhtml"),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("cover"), _spine("toc"), _spine("c1")],
    )
    landmarks = {"OEBPS/cover.xhtml": "cover", "OEBPS/toc.xhtml": "toc"}

    selection = narration_set.select(package, landmarks=landmarks)

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/cover.xhtml"] == "landmark:cover"
    assert reasons["OEBPS/toc.xhtml"] == "landmark:toc"
    assert [doc.href for doc in selection.documents] == ["OEBPS/c1.xhtml"]


def test_excludes_frontmatter_before_bodymatter():
    package = _package(
        [
            _item("fm", "OEBPS/preface.xhtml"),
            _item("c1", "OEBPS/c1.xhtml"),
            _item("c2", "OEBPS/c2.xhtml"),
        ],
        [_spine("fm"), _spine("c1"), _spine("c2")],
    )
    landmarks = {"OEBPS/c1.xhtml": "bodymatter"}

    selection = narration_set.select(package, landmarks=landmarks)

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/preface.xhtml"] == "front-matter"
    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c1.xhtml",
        "OEBPS/c2.xhtml",
    ]


def test_excludes_non_content_media_type():
    package = _package(
        [
            _item("img", "OEBPS/cover.svg", media_type="image/svg+xml"),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("img"), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/cover.svg"] == "non-content"


def test_excludes_unresolved_spine_item():
    package = _package(
        [_item("c1", "OEBPS/c1.xhtml")],
        [_spine("ghost"), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})

    assert any(ex.reason == "unresolved-item" for ex in selection.excluded)
    assert [doc.href for doc in selection.documents] == ["OEBPS/c1.xhtml"]


def test_every_exclusion_has_a_reason():
    package = _package(
        [
            _item("nav", "OEBPS/nav.xhtml", properties=frozenset({"nav"})),
            _item("cover", "OEBPS/cover.xhtml"),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("nav"), _spine("cover", linear=False), _spine("c1")],
    )

    selection = narration_set.select(package, landmarks={})

    assert all(ex.reason for ex in selection.excluded)


def test_read_landmarks_from_upgraded_nav(minimal_epub2_path: Path, tmp_path: Path):
    book = tmp_path / "book"
    epub_container.unzip(minimal_epub2_path, book)
    package = epub2_upgrade.upgrade(book, modified="2026-07-03T00:00:00Z")

    landmarks = narration_set.read_landmarks(book, package)

    assert landmarks["OEBPS/cover.xhtml"] == "cover"
    assert landmarks["OEBPS/chapter1.xhtml"] == "bodymatter"


def test_end_to_end_selection_over_upgraded_book(
    minimal_epub2_path: Path,
    tmp_path: Path,
):
    book = tmp_path / "book"
    epub_container.unzip(minimal_epub2_path, book)
    package = epub2_upgrade.upgrade(book, modified="2026-07-03T00:00:00Z")

    landmarks = narration_set.read_landmarks(book, package)
    selection = narration_set.select(package, landmarks=landmarks)

    assert [doc.href for doc in selection.documents] == ["OEBPS/chapter1.xhtml"]


def test_landmark_lookup_ignores_manifest_fragment():
    package = _package(
        [_item("cover", "OEBPS/cover.xhtml#c"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("cover"), _spine("c1")],
    )
    landmarks = {"OEBPS/cover.xhtml": "cover"}

    selection = narration_set.select(package, landmarks=landmarks)

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/cover.xhtml#c"] == "landmark:cover"


def test_bodymatter_matches_despite_manifest_fragment():
    package = _package(
        [_item("fm", "OEBPS/fm.xhtml"), _item("c1", "OEBPS/c1.xhtml#start")],
        [_spine("fm"), _spine("c1")],
    )
    landmarks = {"OEBPS/c1.xhtml": "bodymatter"}

    selection = narration_set.select(package, landmarks=landmarks)

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/fm.xhtml"] == "front-matter"


def test_read_landmarks_drops_container_escaping_href(tmp_path: Path):
    oebps = tmp_path / "book" / "OEBPS"
    oebps.mkdir(parents=True)
    nav = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html>'
        '<html xmlns="http://www.w3.org/1999/xhtml"'
        ' xmlns:epub="http://www.idpf.org/2007/ops">'
        "<head><title>x</title></head><body>"
        '<nav epub:type="landmarks"><ol>'
        '<li><a epub:type="cover" href="../../evil.xhtml">Bad</a></li>'
        '<li><a epub:type="bodymatter" href="c1.xhtml">Body</a></li>'
        "</ol></nav></body></html>"
    )
    (oebps / "nav.xhtml").write_text(nav, encoding="utf-8")
    package = _package(
        [_item("nav", "OEBPS/nav.xhtml", properties=frozenset({"nav"}))],
        [],
    )

    landmarks = narration_set.read_landmarks(tmp_path / "book", package)

    assert all(not key.startswith("..") for key in landmarks)
    assert landmarks == {"OEBPS/c1.xhtml": "bodymatter"}


def _overridden(
    package: opf.Package,
    *,
    landmarks: dict[str, str] | None = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    pick: tuple[str, ...] = (),
) -> narration_set.Selection:
    selection = narration_set.select(package, landmarks=landmarks or {})

    return narration_set.apply_overrides(
        selection,
        include=include,
        exclude=exclude,
        pick=pick,
    )


def _three_chapters() -> opf.Package:
    return _package(
        [
            _item("c1", "OEBPS/c1.xhtml"),
            _item("c2", "OEBPS/c2.xhtml"),
            _item("c3", "OEBPS/c3.xhtml"),
        ],
        [_spine("c1"), _spine("c2"), _spine("c3")],
    )


def test_pick_selects_named_docs_in_spine_order():
    selection = _overridden(_three_chapters(), pick=("c3.xhtml", "c1.xhtml"))

    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c1.xhtml",
        "OEBPS/c3.xhtml",
    ]
    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/c2.xhtml"] == "not-picked"


def test_pick_overrides_default_landmark_exclusion():
    package = _package(
        [_item("cover", "OEBPS/cover.xhtml"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("cover"), _spine("c1")],
    )

    selection = _overridden(
        package,
        landmarks={"OEBPS/cover.xhtml": "cover"},
        pick=("cover.xhtml",),
    )

    assert [doc.href for doc in selection.documents] == ["OEBPS/cover.xhtml"]


def test_include_readds_frontmatter():
    package = _package(
        [_item("fm", "OEBPS/preface.xhtml"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("fm"), _spine("c1")],
    )

    selection = _overridden(
        package,
        landmarks={"OEBPS/c1.xhtml": "bodymatter"},
        include=("preface.xhtml",),
    )

    assert [doc.href for doc in selection.documents] == [
        "OEBPS/preface.xhtml",
        "OEBPS/c1.xhtml",
    ]


def test_exclude_drops_matching_chapter():
    selection = _overridden(_three_chapters(), exclude=("c2.xhtml",))
    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c1.xhtml",
        "OEBPS/c3.xhtml",
    ]

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/c2.xhtml"] == "excluded"


def test_exclude_beats_include():
    package = _package(
        [_item("fm", "OEBPS/preface.xhtml"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("fm"), _spine("c1")],
    )

    selection = _overridden(
        package,
        landmarks={"OEBPS/c1.xhtml": "bodymatter"},
        include=("preface.xhtml",),
        exclude=("preface.xhtml",),
    )
    assert "OEBPS/preface.xhtml" not in {doc.href for doc in selection.documents}


def test_include_does_not_resurrect_nav():
    package = _package(
        [
            _item("nav", "OEBPS/nav.xhtml", properties=frozenset({"nav"})),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("nav"), _spine("c1")],
    )

    selection = _overridden(package, include=("nav.xhtml",))

    reasons = {ex.href: ex.reason for ex in selection.excluded}
    assert reasons["OEBPS/nav.xhtml"] == "nav"


def test_glob_include_matches_multiple():
    package = _package(
        [
            _item("fm", "OEBPS/front.xhtml"),
            _item("c1", "OEBPS/c1.xhtml"),
        ],
        [_spine("fm", linear=False), _spine("c1")],
    )

    selection = _overridden(package, include=("*.xhtml",))
    assert [doc.href for doc in selection.documents] == [
        "OEBPS/front.xhtml",
        "OEBPS/c1.xhtml",
    ]


def test_unmatched_pick_selector_is_reported():
    selection = _overridden(_three_chapters(), pick=("ghost.xhtml",))

    assert selection.documents == ()
    assert "ghost.xhtml" in selection.unmatched


def test_unmatched_exclude_selector_is_reported():
    selection = _overridden(_three_chapters(), exclude=("nope.xhtml",))

    assert "nope.xhtml" in selection.unmatched


def test_redundant_pick_selectors_not_reported_unmatched():
    selection = _overridden(_three_chapters(), pick=("*.xhtml", "c1.xhtml"))

    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c1.xhtml",
        "OEBPS/c2.xhtml",
        "OEBPS/c3.xhtml",
    ]
    assert selection.unmatched == ()


def test_redundant_include_selectors_not_reported_unmatched():
    package = _package(
        [_item("fm", "OEBPS/preface.xhtml"), _item("c1", "OEBPS/c1.xhtml")],
        [_spine("fm"), _spine("c1")],
    )

    selection = _overridden(
        package,
        landmarks={"OEBPS/c1.xhtml": "bodymatter"},
        include=("*.xhtml", "preface.xhtml"),
    )
    assert selection.unmatched == ()


def test_duplicate_spine_item_keeps_body_occurrence():
    package = _package(
        [_item("c1", "OEBPS/c1.xhtml"), _item("c2", "OEBPS/c2.xhtml")],
        [_spine("c1"), _spine("c2"), _spine("c1")],
    )

    selection = _overridden(package, landmarks={"OEBPS/c2.xhtml": "bodymatter"})
    assert [doc.href for doc in selection.documents] == [
        "OEBPS/c2.xhtml",
        "OEBPS/c1.xhtml",
    ]
