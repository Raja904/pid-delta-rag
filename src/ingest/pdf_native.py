"""
ingest/pdf_native.py
Native (born-digital) PDF adapter using PyMuPDF.
Extracts text blocks with coordinates directly from the PDF vector layer.
No OCR needed — confidence is always 1.0.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import fitz  # PyMuPDF

from src.ingest.base import FormatAdapter
from src.canonical.model import (
    BBox, TextBlock, Page, CanonicalDocument, BlockType
)


_DIMENSION_PATTERNS = ["mm", "cm", "m ", "\"", "'", "dia", "dn", "nps", "sch"]
_NOTE_PATTERNS      = ["note:", "general note", "revision", "see", "ref."]
_TITLE_PATTERNS     = ["p&id", "pid", "drawing no", "sheet", "rev."]


def _classify_block(text: str) -> BlockType:
    t = text.lower()
    if any(p in t for p in _TITLE_PATTERNS):
        return "title"
    if any(p in t for p in _NOTE_PATTERNS):
        return "note"
    if any(p in t for p in _DIMENSION_PATTERNS):
        return "dimension"
    return "text"


class NativePDFAdapter(FormatAdapter):
    """Adapter for born-digital PDFs (extractable text layer)."""

    @property
    def supported_extensions(self):
        return (".pdf",)

    def ingest(self, path: Path, pid: str, revision: str = "") -> CanonicalDocument:
        doc_fitz = fitz.open(str(path))
        pages: list[Page] = []

        for page_idx in range(len(doc_fitz)):
            fitz_page = doc_fitz[page_idx]
            page_num  = page_idx + 1
            w, h      = fitz_page.rect.width, fitz_page.rect.height

            # dict extraction gives blocks with bboxes
            raw = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            blocks: list[TextBlock] = []
            block_counter = 0

            for raw_block in raw.get("blocks", []):
                if raw_block.get("type") != 0:   # skip images
                    continue
                # collect all span text in the block
                text_parts = []
                for line in raw_block.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            text_parts.append(t)
                full_text = " ".join(text_parts).strip()
                if not full_text:
                    continue

                x0, y0, x1, y1 = raw_block["bbox"]
                block_id = f"p{page_num}-b{block_counter}"
                blocks.append(TextBlock(
                    block_id   = block_id,
                    text       = full_text,
                    bbox       = BBox(x0, y0, x1, y1),
                    page       = page_num,
                    block_type = _classify_block(full_text),
                    confidence = 1.0,
                ))
                block_counter += 1

            pages.append(Page(page_num=page_num, width=w, height=h, blocks=blocks))

        doc_fitz.close()

        return CanonicalDocument(
            pid      = pid,
            format   = "native_pdf",
            revision = revision,
            pages    = pages,
            metadata = {
                "path":       str(path),
                "page_count": len(pages),
                "sha256":     _sha256(path),
            },
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
