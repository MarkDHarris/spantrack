"""Factory for `Document` instances shared across narration test modules."""

from spantrack.narration_set import Document


def make_document(href: str, spine_index: int = 0) -> Document:
    """Build a `Document`, deriving a synthetic item id from the spine index."""
    return Document(item_id=f"id{spine_index}", href=href, spine_index=spine_index)
