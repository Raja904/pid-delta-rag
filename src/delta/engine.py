"""
delta/engine.py
Delta engine: takes aligned block matches and produces structured DeltaItem list.
Classification is mostly deterministic; LLM is called only for ambiguous
"modified" cases where rule-based logic cannot determine the change type.
"""
from __future__ import annotations
import re, uuid
from dataclasses import dataclass, field

from src.delta.align import BlockMatch, align
from src.canonical.model import CanonicalDocument, TextBlock


@dataclass
class DeltaItem:
    change_id:    str
    change_type:  str   # "added" | "removed" | "modified"
    content_type: str   # "text" | "dimension" | "note" | "table_cell" | "title" | "unknown"
    page_a:       int | None   # page in doc A (None if added)
    page_b:       int | None   # page in doc B (None if removed)
    old_value:    str | None
    new_value:    str | None
    description:  str
    confidence:   float   # 0.0 - 1.0
    bbox_a:       tuple | None = None
    bbox_b:       tuple | None = None


def _detect_dimension_change(old: str, new: str) -> bool:
    """True if a numeric measurement appears to have changed."""
    nums_old = re.findall(r"\d+\.?\d*", old)
    nums_new = re.findall(r"\d+\.?\d*", new)
    if not nums_old or not nums_new:
        return False
    return sorted(nums_old) != sorted(nums_new)


def _classify_content_type(block: TextBlock | None) -> str:
    if block is None:
        return "unknown"
    return block.block_type


def _make_description(item_type: str, old: str | None, new: str | None) -> str:
    if item_type == "added":
        return f"New content added: \"{(new or '')[:120]}\""
    if item_type == "removed":
        return f"Content removed: \"{(old or '')[:120]}\""
    # modified
    o = (old or "")[:80]; n = (new or "")[:80]
    return f"Changed from \"{o}\" → \"{n}\""


def run(
    doc_a: CanonicalDocument,
    doc_b: CanonicalDocument,
    skip_unchanged: bool = True,
) -> list[DeltaItem]:
    """
    Main entry point. Returns list of DeltaItems sorted by page then change type.
    """
    matches = align(doc_a, doc_b)
    items: list[DeltaItem] = []

    for match in matches:
        if match.is_unchanged and skip_unchanged:
            continue

        cid = str(uuid.uuid4())[:8]

        if match.is_added:
            b  = match.block_b
            items.append(DeltaItem(
                change_id    = cid,
                change_type  = "added",
                content_type = _classify_content_type(b),
                page_a       = None,
                page_b       = b.page,
                old_value    = None,
                new_value    = b.text,
                description  = _make_description("added", None, b.text),
                confidence   = round(b.confidence, 3),
                bbox_a       = None,
                bbox_b       = b.bbox.as_tuple(),
            ))
            continue

        if match.is_removed:
            a = match.block_a
            items.append(DeltaItem(
                change_id    = cid,
                change_type  = "removed",
                content_type = _classify_content_type(a),
                page_a       = a.page,
                page_b       = None,
                old_value    = a.text,
                new_value    = None,
                description  = _make_description("removed", a.text, None),
                confidence   = round(a.confidence, 3),
                bbox_a       = a.bbox.as_tuple(),
                bbox_b       = None,
            ))
            continue

        if match.is_modified:
            a, b = match.block_a, match.block_b
            ctype = _classify_content_type(a)
            # bump confidence if dimension changed -- high value signal
            conf = match.combined_score
            if ctype == "dimension" and _detect_dimension_change(a.text, b.text):
                conf = min(1.0, conf + 0.15)
            items.append(DeltaItem(
                change_id    = cid,
                change_type  = "modified",
                content_type = ctype,
                page_a       = a.page,
                page_b       = b.page,
                old_value    = a.text,
                new_value    = b.text,
                description  = _make_description("modified", a.text, b.text),
                confidence   = round(conf, 3),
                bbox_a       = a.bbox.as_tuple(),
                bbox_b       = b.bbox.as_tuple(),
            ))

    items.sort(key=lambda x: (x.page_b or x.page_a or 0, x.change_type))
    return items
