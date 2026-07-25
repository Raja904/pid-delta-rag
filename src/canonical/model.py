"""
canonical/model.py
Format-agnostic intermediate representation.
Every ingestion adapter normalizes its source into this model.
The delta engine and chat layer never see raw PDF/DWG bytes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


BlockType = Literal["text", "dimension", "note", "table_cell", "geometry_label", "title", "unknown"]
Format    = Literal["native_pdf", "scanned_pdf", "dwg", "unknown"]


@dataclass
class BBox:
    """Bounding box in page-space (points). Top-left origin."""
    x0: float
    y0: float
    x1: float
    y1: float

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def iou(self, other: "BBox") -> float:
        ix0 = max(self.x0, other.x0); iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1); iy1 = min(self.y1, other.y1)
        inter = max(0.0, ix1-ix0) * max(0.0, iy1-iy0)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def as_tuple(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextBlock:
    """A single piece of content extracted from a page."""
    block_id: str                  # unique within document, e.g. "p1-b42"
    text: str
    bbox: BBox
    page: int                      # 1-indexed
    block_type: BlockType = "text"
    confidence: float = 1.0        # 1.0 for native PDF; OCR confidence for scanned


@dataclass
class Page:
    page_num: int          # 1-indexed
    width: float           # points
    height: float          # points
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class CanonicalDocument:
    pid: str               # document identifier (filename or user-supplied)
    format: Format
    revision: str          # e.g. "A", "B", "Rev-2", or ""
    pages: list[Page] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def all_blocks(self) -> list[TextBlock]:
        return [b for p in self.pages for b in p.blocks]

    def block_by_id(self, block_id: str) -> TextBlock | None:
        for b in self.all_blocks():
            if b.block_id == block_id:
                return b
        return None
