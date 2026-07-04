"""Independent sentence-count oracle (syntok) for cross-checking pysbd segmentation."""

import syntok.segmenter as syntok  # pyright: ignore[reportMissingTypeStubs]  # no stubs


def syntok_sentence_count(text: str) -> int:
    """Count sentences in text with syntok, as an oracle independent of pysbd."""
    paragraphs = syntok.process(text)  # pyright: ignore[reportUnknownMemberType]

    return sum(1 for paragraph in paragraphs for _sentence in paragraph)
