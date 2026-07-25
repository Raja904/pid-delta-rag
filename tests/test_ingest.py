"""
tests/test_ingest.py
Tests for the ingestion layer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.base import AdapterRegistry, FormatAdapter
from src.ingest.pdf_native import NativePDFAdapter
from src.canonical.model import CanonicalDocument


def test_registry_resolves_pdf():
    reg = AdapterRegistry()
    reg.register(NativePDFAdapter())
    adapter = reg.get(Path("test.pdf"))
    assert isinstance(adapter, NativePDFAdapter)


def test_registry_unknown_extension():
    reg = AdapterRegistry()
    reg.register(NativePDFAdapter())
    try:
        reg.get(Path("test.xyz"))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_native_pdf_real_file():
    """Test with one of the real P&ID PDFs."""
    pdf_path = Path(__file__).parent.parent / "eval" / "datasets" / "pair_01" / "doc_a.pdf"
    if not pdf_path.exists():
        import pytest; pytest.skip("Real PDF not available")

    adapter = NativePDFAdapter()
    doc = adapter.ingest(pdf_path, pid="test_pid", revision="A")

    assert isinstance(doc, CanonicalDocument)
    assert doc.pid == "test_pid"
    assert doc.revision == "A"
    assert doc.format == "native_pdf"
    assert len(doc.pages) > 0
    total_blocks = sum(len(p.blocks) for p in doc.pages)
    assert total_blocks > 0, "Should extract at least some text blocks"
    print(f"\n  Pages: {len(doc.pages)}, Blocks: {total_blocks}")
    print(f"  Sample block: {doc.pages[0].blocks[0].text[:60] if doc.pages[0].blocks else None}")
