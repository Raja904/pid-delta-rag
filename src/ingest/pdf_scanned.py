"""
ingest/pdf_scanned.py
Scanned-PDF adapter: renders each page to an image then runs Tesseract OCR.
Returns blocks with bounding boxes and per-word OCR confidence.
Falls back gracefully if Tesseract is not installed.
"""
from __future__ import annotations
import hashlib, os
from pathlib import Path

from src.ingest.base import FormatAdapter
from src.canonical.model import BBox, TextBlock, Page, CanonicalDocument, BlockType
from src.ingest.pdf_native import _classify_block

DPI = 200  # render resolution; 200 dpi balances speed vs OCR accuracy


class ScannedPDFAdapter(FormatAdapter):
    """OCR-based adapter for scanned / image-only PDFs."""

    @property
    def supported_extensions(self):
        return ()   # NOT registered by default; caller picks explicitly
        # The registry selects NativePDFAdapter for .pdf by default.
        # To use ScannedPDFAdapter, call it directly.

    def ingest(self, path: Path, pid: str, revision: str = "") -> CanonicalDocument:
        try:
            import fitz
            import pytesseract
            from PIL import Image
            
            # Windows fallback for Tesseract PATH issue
            if os.name == 'nt':
                tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(tess_path):
                    pytesseract.pytesseract.tesseract_cmd = tess_path
                    
        except ImportError as e:
            raise RuntimeError(
                "ScannedPDFAdapter requires PyMuPDF (fitz), pytesseract, and Pillow. "
                f"Ensure Tesseract is on PATH. ({e})"
            )

        doc = fitz.open(str(path))
        pil_pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=DPI)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            pil_pages.append(img)
        pages: list[Page] = []

        for page_idx, pil_image in enumerate(pil_pages):
            page_num = page_idx + 1
            w_pt = pil_image.width  * 72 / DPI
            h_pt = pil_image.height * 72 / DPI

            # OCR with bounding boxes and confidence
            try:
                ocr_data = pytesseract.image_to_data(
                    pil_image, output_type=pytesseract.Output.DICT,
                    config="--psm 6"
                )
            except pytesseract.TesseractNotFoundError:
                raise RuntimeError(
                    "Tesseract is not installed or not on PATH. "
                    "Install from https://github.com/UB-Mannheim/tesseract/wiki"
                )

            # Group words into line-level blocks
            blocks = _group_into_blocks(ocr_data, page_num, w_pt, h_pt, DPI)
            pages.append(Page(page_num=page_num, width=w_pt, height=h_pt, blocks=blocks))

        return CanonicalDocument(
            pid      = pid,
            format   = "scanned_pdf",
            revision = revision,
            pages    = pages,
            metadata = {
                "path":       str(path),
                "page_count": len(pages),
                "dpi":        DPI,
                "sha256":     _sha256(path),
            },
        )


def _group_into_blocks(
    data: dict, page_num: int, w_pt: float, h_pt: float, dpi: int
) -> list[TextBlock]:
    """Merge OCR word-level results into line blocks."""
    scale = 72.0 / dpi
    blocks: list[TextBlock] = []
    current_block_id = None
    current_words: list[str] = []
    current_confs: list[float] = []
    bx0 = bx1 = by0 = by1 = 0.0
    block_counter = 0

    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
        bid  = data["block_num"][i]

        if conf < 0 or not word:
            # flush current block
            if current_words:
                blocks.append(_make_block(
                    current_words, current_confs,
                    bx0, by0, bx1, by1, page_num, block_counter
                ))
                block_counter += 1
                current_words = []; current_confs = []
            current_block_id = None
            continue

        x = data["left"][i]  * scale
        y = data["top"][i]   * scale
        w = data["width"][i] * scale
        h = data["height"][i]* scale

        if bid != current_block_id:
            if current_words:
                blocks.append(_make_block(
                    current_words, current_confs,
                    bx0, by0, bx1, by1, page_num, block_counter
                ))
                block_counter += 1
                current_words = []; current_confs = []
            current_block_id = bid
            bx0, by0, bx1, by1 = x, y, x+w, y+h
        else:
            bx0 = min(bx0, x); by0 = min(by0, y)
            bx1 = max(bx1, x+w); by1 = max(by1, y+h)

        current_words.append(word)
        current_confs.append(conf / 100.0)

    if current_words:
        blocks.append(_make_block(
            current_words, current_confs,
            bx0, by0, bx1, by1, page_num, block_counter
        ))

    return blocks


def _make_block(
    words, confs, x0, y0, x1, y1, page_num, counter
) -> TextBlock:
    text = " ".join(words)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return TextBlock(
        block_id   = f"p{page_num}-b{counter}",
        text       = text,
        bbox       = BBox(x0, y0, x1, y1),
        page       = page_num,
        block_type = _classify_block(text),
        confidence = round(avg_conf, 3),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
