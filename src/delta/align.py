"""
delta/align.py
Content alignment between two canonical documents.
This is THE hard part of the delta engine -- matching blocks across revisions
when content may have moved, been slightly edited, or reformatted.

Strategy (deterministic, no LLM):
  1. Build TF-IDF-like normalized text fingerprints
  2. Cosine similarity on character n-grams for fuzzy text matching
  3. Spatial proximity as a tiebreaker (same page + nearby bbox)
  4. Hungarian algorithm (linear_sum_assignment) for global optimal matching
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.canonical.model import TextBlock, CanonicalDocument


@dataclass
class BlockMatch:
    block_a: TextBlock         # block from doc A (None = added)
    block_b: TextBlock         # block from doc B (None = removed)
    similarity: float          # 0.0 - 1.0; text similarity score
    spatial_score: float       # 0.0 - 1.0; spatial proximity (same page/region)
    combined_score: float      # weighted combo

    @property
    def is_added(self) -> bool:
        return self.block_a is None

    @property
    def is_removed(self) -> bool:
        return self.block_b is None

    @property
    def is_modified(self) -> bool:
        return (self.block_a is not None and self.block_b is not None
                and self.similarity < 0.98)

    @property
    def is_unchanged(self) -> bool:
        return (self.block_a is not None and self.block_b is not None
                and self.similarity >= 0.98)


MATCH_THRESHOLD = 0.35   # below this = not the same block


def align(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[BlockMatch]:
    """
    Align all blocks in doc_a with all blocks in doc_b.
    Returns a list of BlockMatch objects covering every block in both docs.
    """
    blocks_a = doc_a.all_blocks()
    blocks_b = doc_b.all_blocks()

    if not blocks_a and not blocks_b:
        return []

    if not blocks_a:
        return [BlockMatch(None, b, 0.0, 0.0, 0.0) for b in blocks_b]

    if not blocks_b:
        return [BlockMatch(a, None, 0.0, 0.0, 0.0) for a in blocks_a]

    # Build similarity matrix (A rows x B cols)
    n, m = len(blocks_a), len(blocks_b)
    sim_matrix  = _text_similarity_matrix(blocks_a, blocks_b)
    spat_matrix = _spatial_matrix(blocks_a, blocks_b)
    combined    = 0.7 * sim_matrix + 0.3 * spat_matrix

    # Hungarian assignment on the COST matrix (1 - combined)
    cost = 1.0 - combined
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    matches: list[BlockMatch] = []

    for r, c in zip(row_ind, col_ind):
        score = combined[r, c]
        if score >= MATCH_THRESHOLD:
            matches.append(BlockMatch(
                block_a       = blocks_a[r],
                block_b       = blocks_b[c],
                similarity    = round(float(sim_matrix[r, c]), 4),
                spatial_score = round(float(spat_matrix[r, c]), 4),
                combined_score= round(float(score), 4),
            ))
            matched_a.add(r)
            matched_b.add(c)

    # Unmatched in A -> removed
    for r in range(n):
        if r not in matched_a:
            matches.append(BlockMatch(blocks_a[r], None, 0.0, 0.0, 0.0))

    # Unmatched in B -> added
    for c in range(m):
        if c not in matched_b:
            matches.append(BlockMatch(None, blocks_b[c], 0.0, 0.0, 0.0))

    return matches


# ── Text similarity ──────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)   # remove punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _text_sim(a: str, b: str) -> float:
    """SequenceMatcher ratio -- fast, no deps."""
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _text_similarity_matrix(
    blocks_a: list[TextBlock], blocks_b: list[TextBlock]
) -> np.ndarray:
    n, m = len(blocks_a), len(blocks_b)
    mat  = np.zeros((n, m), dtype=np.float32)
    for i, ba in enumerate(blocks_a):
        for j, bb in enumerate(blocks_b):
            mat[i, j] = _text_sim(ba.text, bb.text)
    return mat


# ── Spatial proximity ────────────────────────────────────────────────────────

def _spatial_score(a: TextBlock, b: TextBlock) -> float:
    """1.0 = same page + overlapping bbox; 0.0 = different page or far apart."""
    if a.page != b.page:
        return 0.0
    iou = a.bbox.iou(b.bbox)
    if iou > 0:
        return iou
    # distance fallback: normalised centroid distance on [0,1]
    ax, ay = (a.bbox.x0+a.bbox.x1)/2, (a.bbox.y0+a.bbox.y1)/2
    bx, by = (b.bbox.x0+b.bbox.x1)/2, (b.bbox.y0+b.bbox.y1)/2
    dist = ((ax-bx)**2 + (ay-by)**2) ** 0.5
    return max(0.0, 1.0 - dist / 1000.0)   # 1000 pt ≈ A3 page diagonal


def _spatial_matrix(
    blocks_a: list[TextBlock], blocks_b: list[TextBlock]
) -> np.ndarray:
    n, m = len(blocks_a), len(blocks_b)
    mat  = np.zeros((n, m), dtype=np.float32)
    for i, ba in enumerate(blocks_a):
        for j, bb in enumerate(blocks_b):
            mat[i, j] = _spatial_score(ba, bb)
    return mat
