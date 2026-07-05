# Examples

Two kinds of thing live in this directory:

- **Sample books** — small, hand-written EPUBs to run spantrack on. Each `<book>.epub` ships with an unpacked copy in `<book>/` so you can read the source without unzipping. Every book exercises a different corner of the tool.
- **Python scripts** (in [`python/`](python/)) — short, heavily commented programs that drive spantrack's API: some are dry runs, and the full-conversion ones narrate with the real Kokoro voice.

## Prerequisites

- **ffmpeg** on your `PATH` — spantrack shells out to it to encode chapter audio. Dry runs (`--dry-run`) and the pronunciation report (`--oov-report`) do not need it.
- **The package importable** — from the repo root, `pip install -e .` (or `uv sync`). The Python scripts do `import spantrack`.
- **The Kokoro voice** — the full-conversion examples (02 and 05) and the `spantrack` command synthesize with the local Kokoro-82M model. The first run downloads its weights (one-time, a few hundred MB); after that, conversion is fully offline. The sample books are tiny, so a run takes well under a minute on CPU.
- **A JRE** (optional) — only for `--validate`, which runs [epubcheck](https://www.w3.org/publishing/epubcheck/) on the output EPUB.

## Sample books

| Book | EPUB | What it exercises |
| --- | --- | --- |
| `the-lighthouse-keeper` | 2.0 | EPUB 2 → 3 upgrade (NCX, synthesized `nav.xhtml`); guide-based front-matter skip; headings, a blockquote, a list, inline `<em>`/`<a>` |
| `the-clockmakers-apprentice` | 3.0 | Native EPUB 3 (no upgrade); `epub:type` landmark cover skip; clean prose, good for a word-mode A/B |
| `field-notes-on-comets` | 3.0 | Numbers, dates, units, currency, abbreviations — TTS normalization and the `--oov-report` |
| `the-pocket-anthology` | 3.0 | Many documents — `--include` / `--exclude` / `--pick`, and every skip reason (cover, front matter, non-linear) |

Convert any of them with the real voice:

```bash
spantrack examples/the-clockmakers-apprentice.epub --validate
```

The output defaults to `<stem>.narrated.epub`, and a work directory `<input>.spantrack/` holds the cache, sidecars, and per-chapter checkpoints — so a killed run resumes without re-synthesizing. Both are ignored by git under `examples/`.

Useful flags: `--dry-run` (print the plan, synthesize nothing), `--oov-report` (list words the voice may mispronounce), `--voice af_heart`, `--speed 1.1`, `--granularity word` (experimental), `--include`/`--exclude`/`--pick` (steer the narration set), `--progress` (a per-chapter progress bar; needs the optional `pip install 'spantrack[progress]'`).

### the-lighthouse-keeper (EPUB 2)

An EPUB 2 package with an NCX table of contents. spantrack upgrades it to EPUB 3 and synthesizes a `nav.xhtml`. The title page sits before the guide's `text` reference, so it is treated as front matter and skipped; the three chapters narrate. The chapters carry a heading, multi-sentence paragraphs, a `<blockquote>`, a `<ul>`, and inline `<em>`/`<a>` spans, plus a year (`1892`), an em-dash, and an ellipsis for the normalizer.

```text
narration set: 3 document(s)
  narrate  OEBPS/chapter1.xhtml
  narrate  OEBPS/chapter2.xhtml
  narrate  OEBPS/chapter3.xhtml
  skip     OEBPS/title.xhtml (front-matter)
projected: 991 characters, ~59s audio, ~463.6 KB
```

In a reader (Thorium desktop): playback highlights one sentence at a time across all three chapters — heading, list items, blockquote, and the sentence with the `<em>`/`<a>` included. The title page is not part of the narration.

### the-clockmakers-apprentice (EPUB 3)

A native EPUB 3, so there is no upgrade step. Its `nav.xhtml` carries a `landmarks` section whose `cover` entry marks the cover page and whose `bodymatter` entry points at the first chapter. spantrack skips the cover by that landmark and narrates the two chapters.

```text
narration set: 2 document(s)
  narrate  OEBPS/chapter1.xhtml
  narrate  OEBPS/chapter2.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
projected: 650 characters, ~39s audio, ~304.1 KB
```

### field-notes-on-comets (EPUB 3)

One chapter dense with things the TTS normalizer has to handle: a date (`14 March 1997`), a time (`4:32 a.m.`), units (`47.5 km/s`, `3,000 km`), currency (`$1,240`), abbreviations (`Dr.`, `Mt.`), and an em-dash and ellipsis. Pair it with `--oov-report` to see which words fall outside the voice's dictionary (see [example 04](#04--find-mispronounced-words)), or `--lang en-gb` to hear the British pronunciations.

```text
narration set: 1 document(s)
  narrate  OEBPS/chapter1.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
projected: 497 characters, ~30s audio, ~232.5 KB
```

### the-pocket-anthology (EPUB 3)

A cover, a foreword, three short stories, and a colophon marked `linear="no"`. By default spantrack narrates only the three stories: the cover is skipped by its landmark, the foreword is front matter (it precedes the `bodymatter` landmark), and the colophon is non-linear. This is the book to try the narration-set overrides on (see [example 03](#03--choose-what-to-narrate)).

```text
narration set: 3 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 568 characters, ~34s audio, ~265.7 KB
```

## Python examples

Run any of them from the repo root, for example:

```bash
python examples/python/01_inspect_a_book.py
```

| Script | Shows | Needs |
| --- | --- | --- |
| [`01_inspect_a_book.py`](python/01_inspect_a_book.py) | A dry run: the narration set and duration/size projection | — |
| [`02_narrate_with_kokoro.py`](python/02_narrate_with_kokoro.py) | The full pipeline producing a read-along EPUB with the Kokoro voice | ffmpeg + Kokoro |
| [`03_choose_what_to_narrate.py`](python/03_choose_what_to_narrate.py) | `--include` / `--exclude` / `--pick` and the skip reasons | — |
| [`04_find_mispronounced_words.py`](python/04_find_mispronounced_words.py) | The `--oov-report` pronunciation-lexicon starter | — (loads the G2P) |
| [`05_word_by_word_highlighting.py`](python/05_word_by_word_highlighting.py) | Experimental word-level highlighting with the Kokoro voice | ffmpeg + Kokoro |

None of these write into your checkout: each sends its output and work directory to a throwaway temp folder.

### 01 — inspect a book

Dry-runs each sample book and prints what spantrack would narrate and why it skips the rest, plus a rough duration/size estimate — the fast way to check a book before a full run.

```text
# the-clockmakers-apprentice.epub
narration set: 2 document(s)
  narrate  OEBPS/chapter1.xhtml
  narrate  OEBPS/chapter2.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
projected: 650 characters, ~39s audio, ~304.1 KB

# field-notes-on-comets.epub
narration set: 1 document(s)
  narrate  OEBPS/chapter1.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
projected: 497 characters, ~30s audio, ~232.5 KB

# the-pocket-anthology.epub
narration set: 3 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 568 characters, ~34s audio, ~265.7 KB
```

### 02 — narrate with Kokoro

Runs the entire pipeline — unpack, select, synthesize each sentence with the local Kokoro-82M voice, assemble audio, write the Media Overlay, repack — into a genuine, epubcheck-clean read-along EPUB (real speech: ~160 KB of MP3 per chapter here). The clockmaker is tiny, so it finishes in well under a minute on CPU after the one-time weight download.

```text
narrated OEBPS/chapter1.xhtml
narrated OEBPS/chapter2.xhtml
wrote ...\the-clockmakers-apprentice.narrated.epub

wrote the-clockmakers-apprentice.narrated.epub (13 files inside)
  chapter audio  : ['OEBPS/audio/chapter1.mp3', 'OEBPS/audio/chapter2.mp3']
  media overlays : ['OEBPS/chapter1.smil', 'OEBPS/chapter2.smil']
  highlight CSS  : ['OEBPS/spantrack/highlight.css']
```

### 03 — choose what to narrate

Dry-runs the anthology under five different override combinations to show how the narration set responds. `--include` brings back a soft skip (front matter, a non-linear page) but never a cover or nav; `--exclude` drops a document; `--pick` narrows the set to exactly what you name.

The default and one override are shown here; run the script for all five.

```text
## Default: three stories; front matter and colophon skipped
   flags: (none)
narration set: 3 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 568 characters, ~34s audio, ~265.7 KB

## Narrate a single story only
   flags: --pick story-lamp.xhtml
narration set: 1 document(s)
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (not-picked)
  skip     OEBPS/foreword.xhtml (not-picked)
  skip     OEBPS/story-clock.xhtml (not-picked)
  skip     OEBPS/story-tide.xhtml (not-picked)
  skip     OEBPS/colophon.xhtml (not-picked)
projected: 193 characters, ~12s audio, ~90.3 KB
```

<details>
<summary>All five variations</summary>

```text
## Default: three stories; front matter and colophon skipped
   flags: (none)
narration set: 3 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 568 characters, ~34s audio, ~265.7 KB

## Add the foreword back
   flags: --include foreword.xhtml
narration set: 4 document(s)
  narrate  OEBPS/foreword.xhtml
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 724 characters, ~43s audio, ~338.7 KB

## Add the non-linear colophon back
   flags: --include colophon.xhtml
narration set: 4 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-tide.xhtml
  narrate  OEBPS/story-lamp.xhtml
  narrate  OEBPS/colophon.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
projected: 709 characters, ~42s audio, ~331.7 KB

## Drop one story
   flags: --exclude story-tide.xhtml
narration set: 2 document(s)
  narrate  OEBPS/story-clock.xhtml
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
  skip     OEBPS/foreword.xhtml (front-matter)
  skip     OEBPS/story-tide.xhtml (excluded)
  skip     OEBPS/colophon.xhtml (non-linear)
projected: 392 characters, ~23s audio, ~183.4 KB

## Narrate a single story only
   flags: --pick story-lamp.xhtml
narration set: 1 document(s)
  narrate  OEBPS/story-lamp.xhtml
  skip     OEBPS/cover.xhtml (not-picked)
  skip     OEBPS/foreword.xhtml (not-picked)
  skip     OEBPS/story-clock.xhtml (not-picked)
  skip     OEBPS/story-tide.xhtml (not-picked)
  skip     OEBPS/colophon.xhtml (not-picked)
projected: 193 characters, ~12s audio, ~90.3 KB
```

</details>

### 04 — find mispronounced words

Runs `--oov-report` over the comet field notes: the same normalization the real narration would apply, fed through the grapheme-to-phoneme model, listing every out-of-vocabulary word as a commented-out lexicon entry. Save those lines to a file and pass them back with `--lexicon words.toml` to fix the pronunciations.

```text
narration set: 1 document(s)
  narrate  OEBPS/chapter1.xhtml
  skip     OEBPS/cover.xhtml (landmark:cover)
out-of-vocabulary words: 2
[words]
# Okoro = ""
# km = ""
```

### 05 — word by word highlighting

The same real-Kokoro conversion as example 02, but with `--granularity word`: spantrack wraps every word in its own span and the overlay highlights word by word, driven by Kokoro's per-token timestamps. Word mode is experimental — those timings drift and the CLI prints a warning — so prefer sentence mode for anything you ship.

```text
narrated OEBPS/chapter1.xhtml
narrated OEBPS/chapter2.xhtml
wrote ...\the-clockmakers-apprentice.word.narrated.epub

content documents wrapped word-by-word: 2 of 4
first overlay targets word spans: True
```

## Listening to the result

The examples above send their output to a temp folder. To keep a read-along you can open, run the CLI directly on a sample book. A run resumes from its work directory, so re-running after an interruption reuses already-encoded chapters.

```bash
spantrack examples/the-lighthouse-keeper.epub --voice af_heart --validate
```

Then open the `*.narrated.epub` in a Media-Overlay-capable reader (Thorium desktop is a good choice) and press play: the current sentence highlights in time with the audio, and turning narration off leaves the book looking like the source.
