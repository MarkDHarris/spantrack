"""Sequence the conversion stages and hand artifacts between them.

`pipeline` is the sole orchestrator: it drives ingest -> upgrade -> selection ->
per-chapter narration -> OPF/CSS patching -> repack. Sibling modules share small types
and helpers with one another, but none of them sequences the stages.
"""

import importlib.metadata
import posixpath
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from lxml import etree

from spantrack import (
    audio_assembly,
    build_state,
    chapter_sidecar,
    content_namespace,
    epub2_upgrade,
    epub_container,
    epubcheck,
    highlight_css,
    lexicon,
    lxml_reading,
    narration_projection,
    narration_set,
    oov_report,
    opf,
    sentence_spans,
    smil,
    synthesis,
    text_normalization,
    word_alignment,
)
from spantrack.lexicon import Lexicon
from spantrack.lxml_reading import XmlElement
from spantrack.progress import ProgressReporter
from spantrack.run_config import (
    AudioFormat,
    Granularity,
    Language,
    RunConfig,
    parse_bitrate,
)
from spantrack.sentence_cache import SentenceCache
from spantrack.tts import kokoro_backend
from spantrack.tts.backend import (
    SAMPLE_RATE,
    Pcm,
    Synthesis,
    TokenTimestamp,
    shift_tokens,
)

__all__ = [
    "DocumentTargets",
    "NarrationContext",
    "NarrationResult",
    "UntimedNarrationError",
    "narrate_document",
    "run",
]

BOOK_SUBDIR = "book"
_CSS_MEDIA_TYPE = "text/css"
_CSS_ITEM_ID = "spantrack-css"

type Validator = Callable[[Path], epubcheck.EpubcheckResult]


class _NarrationBackend(Protocol):
    """The backend surface the narration wiring drives: synthesis, lexicon, identity."""

    @property
    def name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def voice(self) -> str: ...

    @property
    def speed(self) -> float: ...

    def validate_lexicon(self, lexicon: Lexicon) -> None: ...

    def apply_lexicon(self, text: str, lexicon: Lexicon) -> str: ...

    def synthesize(self, text: str) -> Synthesis: ...


type BackendFactory = Callable[[RunConfig], _NarrationBackend]


@dataclass(frozen=True, slots=True)
class NarrationContext:
    """Cross-document synthesis inputs shared by every narrated chapter."""

    backend: _NarrationBackend
    cache: SentenceCache
    lexicon: Lexicon | None
    spantrack_version: str
    bitrate: int
    audio_format: AudioFormat
    granularity: Granularity = Granularity.SENTENCE
    allow_untimed_fallback: bool = False


@dataclass(frozen=True, slots=True)
class DocumentTargets:
    """One document's cross-reference hrefs and output file paths.

    `doc_href`/`audio_href` are relative to the SMIL file (the `<text>`/`<audio>` `src`
    bases); the three paths are where the mp3, SMIL, and sidecar are written.
    """

    doc_href: str
    audio_href: str
    audio_path: Path
    smil_path: Path
    sidecar_path: Path


@dataclass(frozen=True, slots=True)
class NarrationResult:
    """A narrated chapter's contribution: audio length and sentence count."""

    duration_s: float
    sentence_count: int


@dataclass(frozen=True, slots=True)
class _BuildLayout:
    """Where generated artifacts live for one build, relative to the container."""

    container_root: Path
    opf_dir: str
    css_opf_href: str
    sidecars_dir: Path


@dataclass(frozen=True, slots=True)
class _ChapterPaths:
    targets: DocumentTargets
    css_href_rel_doc: str
    audio_opf_href: str
    smil_opf_href: str


def run(
    config: RunConfig,
    *,
    reporter: ProgressReporter | None = None,
    validator: Validator = epubcheck.validate,
    backend_factory: BackendFactory | None = None,
) -> int:
    """Convert one EPUB end to end and return the process exit code.

    Unpacks the container, upgrades EPUB 2 inputs to EPUB 3, resolves the narration set,
    narrates each chapter (synthesis -> assembly -> SMIL + sidecar), wires the OPF and
    highlight CSS, and repacks the output EPUB. `--dry-run` returns after the projection
    and `--oov-report` after listing missed words, both before synthesis.

    Args:
        config: The resolved run configuration.
        reporter: Progress sink; defaults to stderr reporting.
        validator: EPUB validator invoked only when `config.validate` is set.
        backend_factory: Builds the TTS backend from the config; injected so tests drive
            a fake backend without downloading weights.

    Returns:
        0 on success, nonzero if validation fails.
    """
    reporter = reporter or ProgressReporter(progress_bar=config.progress)

    if config.clean and config.work_dir.exists():
        shutil.rmtree(config.work_dir)

    # Unpack fresh over any previous copy: unzip overwrites the original members, while
    # generated audio/SMIL/CSS persist so a resumed run reuses checkpointed chapters.
    # --clean discards the whole work dir first.
    container_root = config.work_dir / BOOK_SUBDIR
    archive_order = epub_container.unzip(config.input, container_root)
    package = epub2_upgrade.upgrade(
        container_root, modified=_source_modified(config.input)
    )

    landmarks = narration_set.read_landmarks(container_root, package)
    selection = narration_set.apply_overrides(
        narration_set.select(package, landmarks=landmarks),
        include=config.include,
        exclude=config.exclude,
        pick=config.pick,
    )

    # Refuse a malformed content document whose root is not XHTML-namespaced before it
    # is silently narrated as empty; runs before projection so --dry-run fails too.
    content_namespace.require_xhtml_namespace(container_root, selection.documents)

    if config.oov_report:
        reporter.selection(selection)
        words = oov_report.find_oov(
            _normalized_sentences(container_root, selection.documents),
            oov_report.build_g2p(british=config.lang is Language.EN_GB),
        )
        reporter.oov(words)
        return 0

    projection = narration_projection.project(
        container_root,
        selection.documents,
        bitrate=config.bitrate,
        speed=config.speed,
    )
    reporter.selection(selection)
    reporter.projection(projection)

    if config.dry_run:
        return 0

    factory = backend_factory if backend_factory is not None else kokoro_backend.build
    _narrate_and_package(config, reporter, package, selection, factory)

    epub_container.repack(container_root, config.output, order=archive_order)
    reporter.info(f"wrote {config.output}")

    if config.validate:
        return _report_validation(validator(config.output), reporter)

    return 0


def _narrate_and_package(
    config: RunConfig,
    reporter: ProgressReporter,
    package: opf.Package,
    selection: narration_set.Selection,
    backend_factory: BackendFactory,
) -> None:
    container_root = config.work_dir / BOOK_SUBDIR
    backend = backend_factory(config)
    opf_dir = posixpath.dirname(package.opf_path)
    css_opf_href = highlight_css.write_stylesheet(
        container_root / opf_dir if opf_dir else container_root
    )
    layout = _BuildLayout(
        container_root=container_root,
        opf_dir=opf_dir,
        css_opf_href=css_opf_href,
        sidecars_dir=config.work_dir / "sidecars",
    )

    parameters = _build_parameters(config, selection)
    build_json = config.work_dir / "build.json"
    state = build_state.load(build_json, parameters)
    context = NarrationContext(
        backend=backend,
        cache=SentenceCache(root=config.work_dir / "cache"),
        lexicon=_load_lexicon(config, backend),
        spantrack_version=_spantrack_version(),
        bitrate=parse_bitrate(config.bitrate),
        audio_format=config.audio_format,
        granularity=config.granularity,
        allow_untimed_fallback=config.allow_untimed_fallback,
    )

    overlays: list[opf.ChapterOverlay] = []
    with reporter.narrating(len(selection.documents)) as chapters:
        for document in selection.documents:
            overlay = _narrate_chapter(context, state, layout, document)
            if overlay is not None:
                overlays.append(overlay)
            build_state.save(build_json, state)
            chapters.narrated(document.href)

    opf.patch_opf_media_overlays(
        container_root / package.opf_path,
        overlays,
        stylesheet=opf.ManifestEntry(_CSS_ITEM_ID, css_opf_href, _CSS_MEDIA_TYPE),
        narrator=config.voice,
        active_class=highlight_css.ACTIVE_CLASS,
    )


def _narrate_chapter(
    context: NarrationContext,
    state: build_state.BuildState,
    layout: _BuildLayout,
    document: narration_set.Document,
) -> opf.ChapterOverlay | None:
    paths = _chapter_paths(layout, document, context.audio_format)
    root = lxml_reading.parse_xml_file(layout.container_root / document.href)

    duration_s = _reuse_duration(state, paths, document)
    if duration_s is None:
        result = narrate_document(context, root, paths.targets)
        if result.sentence_count == 0:
            return None

        duration_s = result.duration_s
        state.mark_done(document.href)
    else:
        # Resume: the mp3/SMIL persist; re-inject spans (their deterministic ids match
        # the persisted SMIL) into the freshly-unpacked content doc.
        _inject_spans(context.granularity, root)

    highlight_css.link_into_head(root, paths.css_href_rel_doc)
    _write_content(root, layout.container_root / document.href)

    return opf.ChapterOverlay(
        content_item_id=document.item_id,
        audio_id=f"{document.item_id}-audio",
        audio_href=paths.audio_opf_href,
        audio_media_type=context.audio_format.media_type,
        smil_id=f"{document.item_id}-smil",
        smil_href=paths.smil_opf_href,
        duration_s=duration_s,
    )


def _chapter_paths(
    layout: _BuildLayout,
    document: narration_set.Document,
    audio_format: AudioFormat,
) -> _ChapterPaths:
    doc_dir = posixpath.dirname(document.href)
    stem = posixpath.splitext(posixpath.basename(document.href))[0]

    smil_container = posixpath.join(doc_dir, f"{stem}.smil")
    audio_container = posixpath.join(
        layout.opf_dir,
        "audio",
        f"{stem}.{audio_format.extension}",
    )
    css_container = posixpath.join(layout.opf_dir, layout.css_opf_href)

    doc_base = doc_dir or "."
    opf_base = layout.opf_dir or "."

    return _ChapterPaths(
        targets=DocumentTargets(
            doc_href=posixpath.basename(document.href),
            audio_href=posixpath.relpath(audio_container, doc_base),
            audio_path=layout.container_root / audio_container,
            smil_path=layout.container_root / smil_container,
            sidecar_path=layout.sidecars_dir / f"{stem}.tokens.json",
        ),
        css_href_rel_doc=posixpath.relpath(css_container, doc_base),
        audio_opf_href=posixpath.relpath(audio_container, opf_base),
        smil_opf_href=posixpath.relpath(smil_container, opf_base),
    )


def _write_content(root: XmlElement, path: Path) -> None:
    path.write_bytes(etree.tostring(root, encoding="utf-8", xml_declaration=True))


def narrate_document(
    context: NarrationContext,
    root: XmlElement,
    targets: DocumentTargets,
) -> NarrationResult:
    """Narrate one content document into an mp3, a SMIL overlay, and a sidecar.

    Injects sentence spans into `root` (mutating it in place), synthesizes each sentence
    through the cache, assembles and encodes the chapter mp3, then writes the SMIL
    overlay and the token-timestamp sidecar. A document with no narratable sentence
    writes nothing and reports a zero-length result. In word granularity each source
    word is wrapped and highlighted, while the sentence stays the synthesis unit.
    """
    if context.granularity is Granularity.WORD:
        return _narrate_words(context, root, targets)

    spans = sentence_spans.inject(root)
    if not spans:
        return NarrationResult(duration_s=0.0, sentence_count=0)

    synthesized, chapter = _synthesize_and_assemble(context, spans)

    sidecar_sentences: list[chapter_sidecar.SidecarSentence] = []
    smil_units: list[smil.SyncUnit] = []
    for span, synth, clip in zip(spans, synthesized, chapter.clips, strict=True):
        sidecar_sentences.append(
            chapter_sidecar.SidecarSentence(
                span_id=span.span_id,
                clip_begin_s=clip.clip_begin_s,
                clip_end_s=clip.clip_end_s,
                display_text=span.display_text,
                normalized_text=synth.normalized_text,
                cache_key=synth.cache_key,
                tokens=shift_tokens(synth.tokens, clip.clip_begin_s),
            ),
        )
        # Drop a zero-duration clip: a sentence trimmed to silence (typically the final
        # one, which absorbs no inter-sentence gap) would emit a clipBegin == clipEnd
        # <par> that epubcheck (MED-009) rejects. Word mode filters the same way.
        if _has_duration(clip.clip_begin_s, clip.clip_end_s):
            smil_units.append(
                smil.SyncUnit(span.span_id, clip.clip_begin_s, clip.clip_end_s),
            )

    return _write_chapter_outputs(
        context, targets, chapter, sidecar_sentences, smil_units
    )


class UntimedNarrationError(RuntimeError):
    """Raised when word mode meets a sentence the backend gave no timestamps for."""


@dataclass(frozen=True, slots=True)
class _Synthesis:
    normalized_text: str
    cache_key: str
    pcm: Pcm
    tokens: tuple[TokenTimestamp, ...] | None


def _synthesize_text(context: NarrationContext, display_text: str) -> _Synthesis:
    normalized = text_normalization.normalize(display_text)
    synth_text = (
        context.backend.apply_lexicon(normalized, context.lexicon)
        if context.lexicon is not None
        else normalized
    )
    cache_key, pcm, tokens = synthesis.synthesize_sentence(
        context.backend,
        context.cache,
        synth_text,
    )

    return _Synthesis(
        normalized_text=normalized,
        cache_key=cache_key,
        pcm=pcm,
        tokens=tokens,
    )


class _Injected(Protocol):
    """A synthesizable unit: its text and the break that sets the gap after it."""

    @property
    def display_text(self) -> str: ...

    @property
    def break_after(self) -> audio_assembly.BreakAfter: ...


def _synthesize_and_assemble(
    context: NarrationContext,
    injected: Sequence[_Injected],
) -> tuple[list[_Synthesis], audio_assembly.AssembledChapter]:
    """Synthesize each unit through the cache and assemble the chapter waveform.

    Shared by both granularities: the sentence stays the synthesis and assembly unit in
    either mode, so the `_Synthesis` list and the gapless `AssembledChapter` (clip
    offsets included) are derived identically before each mode builds its own rows.
    """
    synthesized = [_synthesize_text(context, unit.display_text) for unit in injected]
    chapter = audio_assembly.assemble(
        [
            (synth.pcm, unit.break_after)
            for unit, synth in zip(injected, synthesized, strict=True)
        ],
    )
    return synthesized, chapter


def _write_chapter_outputs(
    context: NarrationContext,
    targets: DocumentTargets,
    chapter: audio_assembly.AssembledChapter,
    sidecar_sentences: list[chapter_sidecar.SidecarSentence],
    smil_units: list[smil.SyncUnit],
) -> NarrationResult:
    """Encode the chapter audio, write the sidecar + SMIL, and report the result.

    A chapter with no highlightable `<par>` (every clip trimmed to silence) is skipped
    before anything is written, so the audio is never left unreferenced by the overlay.
    Both granularities funnel their rows here, so this skip guard cannot diverge.
    """
    if not smil_units:
        return NarrationResult(duration_s=0.0, sentence_count=0)

    for output in (targets.audio_path, targets.smil_path, targets.sidecar_path):
        output.parent.mkdir(parents=True, exist_ok=True)

    audio_assembly.encode_chapter(
        chapter.pcm,
        targets.audio_path,
        settings=audio_assembly.EncodeSettings(context.audio_format, context.bitrate),
    )
    _write_sidecar(context, targets, chapter.duration_s, sidecar_sentences)
    targets.smil_path.write_bytes(
        smil.build_smil(
            targets.doc_href,
            targets.audio_href,
            smil_units,
        ),
    )

    return NarrationResult(
        duration_s=chapter.duration_s,
        sentence_count=len(sidecar_sentences),
    )


def _narrate_words(
    context: NarrationContext,
    root: XmlElement,
    targets: DocumentTargets,
) -> NarrationResult:
    sentences = sentence_spans.inject_words(root)
    if not sentences or not any(sentence.words for sentence in sentences):
        # A chapter with no source words (e.g. a "* * *" divider page) is not narrated.
        return NarrationResult(duration_s=0.0, sentence_count=0)

    synthesized, chapter = _synthesize_and_assemble(context, sentences)

    sidecar_sentences: list[chapter_sidecar.SidecarSentence] = []
    smil_units: list[smil.SyncUnit] = []
    for sentence, synth, clip in zip(
        sentences, synthesized, chapter.clips, strict=True
    ):
        tokens = shift_tokens(synth.tokens, clip.clip_begin_s)
        pars, source_ranges = _word_pars(context, sentence, tokens, clip)
        smil_units.extend(pars)
        sidecar_sentences.append(
            chapter_sidecar.SidecarSentence(
                span_id=sentence.span_id,
                clip_begin_s=clip.clip_begin_s,
                clip_end_s=clip.clip_end_s,
                display_text=sentence.display_text,
                normalized_text=synth.normalized_text,
                cache_key=synth.cache_key,
                tokens=tokens,
                source_ranges=source_ranges,
            ),
        )

    return _write_chapter_outputs(
        context,
        targets,
        chapter,
        sidecar_sentences,
        smil_units,
    )


def _has_duration(clip_begin_s: float, clip_end_s: float) -> bool:
    # Non-empty only once serialized to the millisecond clock -- a sub-millisecond clip
    # rounds to clipBegin == clipEnd, which epubcheck (MED-009) rejects.
    return smil.clock_value(clip_begin_s) != smil.clock_value(clip_end_s)


def _word_pars(
    context: NarrationContext,
    sentence: sentence_spans.WordSentence,
    tokens: tuple[TokenTimestamp, ...] | None,
    clip: audio_assembly.ClipInterval,
) -> tuple[list[smil.SyncUnit], tuple[tuple[int, int] | None, ...] | None]:
    if tokens is None:
        # The backend emits no timestamps at all: fail loud, or fall back if allowed.
        return _untimed_pars(context, sentence, clip), None

    if not tokens:
        # This sentence trimmed to silence (an empty token tuple): nothing to highlight.
        return [], None

    word_spans = [
        word_alignment.WordSpan(word.display_text, word.start, word.end)
        for word in sentence.words
    ]
    aligned = word_alignment.align(word_spans, tokens)
    # Drop a clip with no serialized duration: a word the backend never voiced has no
    # audio to highlight, and the span simply stays unreferenced by any <par>.
    pars = [
        smil.SyncUnit(word.span_id, wordclip.clip_begin_s, wordclip.clip_end_s)
        for word, wordclip in zip(sentence.words, aligned.words, strict=True)
        if _has_duration(wordclip.clip_begin_s, wordclip.clip_end_s)
    ]

    return pars, aligned.source_ranges


def _untimed_pars(
    context: NarrationContext,
    sentence: sentence_spans.WordSentence,
    clip: audio_assembly.ClipInterval,
) -> list[smil.SyncUnit]:
    if not context.allow_untimed_fallback:
        msg = (
            "word mode needs per-token timestamps, but the backend produced none for "
            f"{sentence.span_id!r}; pass --allow-untimed-fallback to highlight it whole"
        )
        raise UntimedNarrationError(msg)

    if not sentence.words or not _has_duration(clip.clip_begin_s, clip.clip_end_s):
        return []

    # Fall back to a single sentence-length highlight on the sentence's first word span.
    first = sentence.words[0].span_id
    return [smil.SyncUnit(first, clip.clip_begin_s, clip.clip_end_s)]


def _reuse_duration(
    state: build_state.BuildState,
    paths: _ChapterPaths,
    document: narration_set.Document,
) -> float | None:
    # Reusable only when checkpointed, all three artifacts persist, and the sidecar
    # still parses -- a stale/incompatible sidecar re-narrates rather than crashing.
    persisted = (
        state.is_done(document.href)
        and paths.targets.audio_path.is_file()
        and paths.targets.smil_path.is_file()
        and paths.targets.sidecar_path.is_file()
    )
    if not persisted:
        return None

    try:
        return chapter_sidecar.read(paths.targets.sidecar_path).audio.duration_s
    except chapter_sidecar.SidecarError:
        return None


def _inject_spans(granularity: Granularity, root: XmlElement) -> None:
    if granularity is Granularity.WORD:
        sentence_spans.inject_words(root)
    else:
        sentence_spans.inject(root)


def _write_sidecar(
    context: NarrationContext,
    targets: DocumentTargets,
    duration_s: float,
    sentences: list[chapter_sidecar.SidecarSentence],
) -> None:
    sidecar = chapter_sidecar.ChapterSidecar(
        spantrack_version=context.spantrack_version,
        backend=chapter_sidecar.BackendIdentity(
            name=context.backend.name,
            model_version=context.backend.model_version,
            voice=context.backend.voice,
            speed=context.backend.speed,
        ),
        doc_href=targets.doc_href,
        audio=chapter_sidecar.ChapterAudio(
            file=targets.audio_href,
            sample_rate=SAMPLE_RATE,
            duration_s=duration_s,
        ),
        sentences=tuple(sentences),
    )
    chapter_sidecar.write(targets.sidecar_path, sidecar)


def _load_lexicon(config: RunConfig, backend: _NarrationBackend) -> Lexicon | None:
    if config.lexicon is None:
        return None

    loaded = lexicon.load(config.lexicon)
    backend.validate_lexicon(loaded)

    return loaded


def _build_parameters(
    config: RunConfig,
    selection: narration_set.Selection,
) -> build_state.BuildParameters:
    return build_state.BuildParameters(
        voice=config.voice,
        speed=config.speed,
        granularity=config.granularity.value,
        audio_format=config.audio_format.value,
        bitrate=parse_bitrate(config.bitrate),
        selection=tuple(document.href for document in selection.documents),
    )


def _spantrack_version() -> str:
    try:
        return importlib.metadata.version("spantrack")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _normalized_sentences(
    container_root: Path,
    documents: Sequence[narration_set.Document],
) -> Iterator[str]:
    # Preview the text synthesis will see, reusing the segment + normalize stages.
    # The tree is injected in memory and discarded; the container is untouched.
    for document in documents:
        root = lxml_reading.parse_xml_file_or_none(container_root / document.href)
        if root is None:
            continue

        for span in sentence_spans.inject(root):
            yield text_normalization.normalize(span.display_text)


def _source_modified(path: Path) -> str:
    # Derive dcterms:modified from the input's mtime so an EPUB 2 upgrade is
    # reproducible across runs of the same file — never datetime.now(), which would
    # break the determinism guarantee.
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)

    return modified.strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_validation(
    result: epubcheck.EpubcheckResult,
    reporter: ProgressReporter,
) -> int:
    if result.ok:
        reporter.info("epubcheck: no errors")
        return 0

    for message in result.messages:
        reporter.info(message)

    reporter.info(f"epubcheck failed with {len(result.messages)} message(s)")
    return 1
