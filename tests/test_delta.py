"""
tests/test_delta.py
Tests for the delta engine core logic.
These run without any LLM or external services.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical.model import CanonicalDocument, Page, TextBlock, BBox
from src.delta.align import align, _text_sim
from src.delta.engine import run as run_delta


def _make_doc(pid: str, texts: list[str]) -> CanonicalDocument:
    blocks = [
        TextBlock(
            block_id=f"p1-b{i}",
            text=t,
            bbox=BBox(0, i*20, 100, i*20+18),
            page=1,
            block_type="text",
            confidence=1.0,
        )
        for i, t in enumerate(texts)
    ]
    return CanonicalDocument(
        pid=pid, format="native_pdf", revision="",
        pages=[Page(page_num=1, width=595, height=842, blocks=blocks)]
    )


def test_text_sim_identical():
    assert _text_sim("hello world", "hello world") == 1.0


def test_text_sim_different():
    assert _text_sim("hello world", "goodbye") < 0.5


def test_align_identical_docs():
    doc_a = _make_doc("a", ["Line one", "Line two", "Line three"])
    doc_b = _make_doc("b", ["Line one", "Line two", "Line three"])
    matches = align(doc_a, doc_b)
    unchanged = [m for m in matches if m.is_unchanged]
    assert len(unchanged) == 3


def test_delta_added():
    doc_a = _make_doc("a", ["Common text"])
    doc_b = _make_doc("b", ["Common text", "Brand new line added"])
    items = run_delta(doc_a, doc_b)
    added = [i for i in items if i.change_type == "added"]
    assert len(added) >= 1
    assert "Brand new line" in added[0].new_value


def test_delta_removed():
    doc_a = _make_doc("a", ["Common text", "This will be removed"])
    doc_b = _make_doc("b", ["Common text"])
    items = run_delta(doc_a, doc_b)
    removed = [i for i in items if i.change_type == "removed"]
    assert len(removed) >= 1
    assert "removed" in removed[0].old_value.lower()


def test_delta_modified():
    doc_a = _make_doc("a", ["Pressure: 100 PSI"])
    doc_b = _make_doc("b", ["Pressure: 150 PSI"])
    items = run_delta(doc_a, doc_b)
    modified = [i for i in items if i.change_type == "modified"]
    assert len(modified) >= 1
    assert "100 PSI" in modified[0].old_value
    assert "150 PSI" in modified[0].new_value


def test_delta_empty_docs():
    doc_a = _make_doc("a", [])
    doc_b = _make_doc("b", [])
    items = run_delta(doc_a, doc_b)
    assert items == []
