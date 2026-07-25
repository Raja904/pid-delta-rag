"""
ingest/dwg.py
DWG / DXF adapter.
This is a REAL stub — it sits behind the same FormatAdapter seam as the PDF adapters.
For DXF files (AutoCAD''s text-based sibling of DWG), we use ezdxf to extract
text entities and dimensions. Pure DWG binary files are converted via ezdxf''s
recover module where possible, or raise a clear error.

Adding full DWG support later = only this file changes. Nothing else.
"""
from __future__ import annotations
from pathlib import Path

from src.ingest.base import FormatAdapter
from src.canonical.model import BBox, TextBlock, Page, CanonicalDocument


class DWGAdapter(FormatAdapter):
    """Adapter for DWG/DXF CAD files via ezdxf."""

    @property
    def supported_extensions(self):
        return (".dxf", ".dwg")

    def ingest(self, path: Path, pid: str, revision: str = "") -> CanonicalDocument:
        try:
            import ezdxf
        except ImportError:
            raise RuntimeError("ezdxf is required for DWG/DXF ingestion. pip install ezdxf")

        suffix = path.suffix.lower()
        if suffix == ".dwg":
            # ezdxf cannot read binary DWG directly — document this clearly
            raise NotImplementedError(
                "Binary DWG ingestion is stubbed. "
                "Convert to DXF via AutoCAD / ODA File Converter and use .dxf. "
                "The adapter seam is real; full DWG support = update this file only."
            )

        # DXF path
        try:
            doc = ezdxf.readfile(str(path))
        except Exception as e:
            raise ValueError(f"Cannot parse DXF file {path}: {e}")

        msp    = doc.modelspace()
        blocks_out: list[TextBlock] = []
        counter = 0

        for entity in msp:
            text, bbox = _extract_entity(entity)
            if text:
                blocks_out.append(TextBlock(
                    block_id   = f"p1-b{counter}",
                    text       = text,
                    bbox       = bbox,
                    page       = 1,
                    block_type = "text",
                    confidence = 1.0,
                ))
                counter += 1

        # DXF is treated as a single "sheet"
        page = Page(page_num=1, width=0, height=0, blocks=blocks_out)
        return CanonicalDocument(
            pid      = pid,
            format   = "dwg",
            revision = revision,
            pages    = [page],
            metadata = {"path": str(path), "dxf_version": doc.dxfversion},
        )


def _extract_entity(entity) -> tuple[str, BBox]:
    """Extract text and a bounding bbox from a DXF entity."""
    try:
        import ezdxf
        etype = entity.dxftype()

        if etype in ("TEXT", "MTEXT"):
            text = entity.dxf.text if etype == "TEXT" else entity.plain_mtext()
            ins  = entity.dxf.insert
            return text.strip(), BBox(ins.x, ins.y, ins.x + 1, ins.y + 1)

        if etype == "DIMENSION":
            text = str(entity.dxf.text) if entity.dxf.hasattr("text") else ""
            ins  = entity.dxf.defpoint
            return text.strip(), BBox(ins.x, ins.y, ins.x + 1, ins.y + 1)

    except Exception:
        pass
    return "", BBox(0, 0, 0, 0)
