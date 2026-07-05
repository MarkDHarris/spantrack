# Example: The Lighthouse Keeper (EPUB 2)

A small, hand-written EPUB 2 for manually testing spantrack end to end with real Kokoro synthesis.

- `the-lighthouse-keeper.epub` — the book to run spantrack on.
- `the-lighthouse-keeper/` — the same files unzipped, for inspection or tweaking.

## What it exercises

- **EPUB 2 to 3 upgrade:** version 2.0 package + NCX; spantrack synthesizes `nav.xhtml`.
- **Narration-set selection:** the title page is front matter (before the `text` guide reference) and is **skipped**; only the three chapters narrate.
- **Block variety for highlighting:** headings, multi-sentence paragraphs, a `<blockquote>`, a `<ul>` list, and inline `<em>` / `<a>` spans.
- **TTS-input normalization:** a year (`1892`), an em-dash pause, and an ellipsis.

## Run it

First run downloads the Kokoro weights (one-time). `ffmpeg` must be on PATH; add `--validate` to run epubcheck (needs a JRE).

```bash
spantrack examples/the-lighthouse-keeper.epub --validate
```

Useful flags: `--voice af_heart`, `--speed 1.0`, `--dry-run` (print the narration set and projected duration, no synthesis), `--oov-report` (list words the lexicon misses).

Output defaults to `examples/the-lighthouse-keeper.narrated.epub`; the work directory (`<input>.spantrack/`) holds the cache, sidecars, and per-chapter checkpoints, so a killed run resumes without re-synthesizing.

## What to check in a reader (Thorium desktop)

- Playback highlights **one sentence at a time**, in sync with the audio, across all three chapters — including the heading, the list items, the blockquote, and the sentence containing the `<em>`/`<a>`.
- The title page is not part of the narration.
- `1892` is read as a year, the em-dash and ellipsis land as pauses.
- With narration off, the layout/CSS looks like the source (only the highlight style and sentence `<span>`s are added).
