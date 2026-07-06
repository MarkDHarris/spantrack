# spantrack

Convert a text EPUB into a self-contained EPUB 3 with [Media Overlays](https://www.w3.org/TR/epub/#sec-media-overlays).

## Install

```bash
pip install spantrack
```

spantrack also needs the `en_core_web_sm` spaCy model. spaCy models are not published to PyPI, so it cannot be a normal dependency — install the pinned wheel explicitly. Pinning the version keeps the grapheme-to-phoneme step offline and deterministic; otherwise misaki downloads a model at runtime.

```bash
pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

## License

[MIT](LICENSE).
