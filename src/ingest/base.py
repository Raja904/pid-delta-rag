"""
ingest/base.py
FormatAdapter ABC — every format adapter implements this interface.
The rest of the system only calls ingest(path, pid, revision).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from src.canonical.model import CanonicalDocument


class FormatAdapter(ABC):
    """Abstract base for all document ingestion adapters."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """File extensions this adapter handles, e.g. ('.pdf',)"""

    @abstractmethod
    def ingest(self, path: Path, pid: str, revision: str = "") -> CanonicalDocument:
        """
        Read *path* and return a CanonicalDocument.
        Raise ValueError on unrecognisable / corrupt input.
        """


class AdapterRegistry:
    """Resolves a file path to the correct FormatAdapter."""

    def __init__(self):
        self._adapters: list[FormatAdapter] = []

    def register(self, adapter: FormatAdapter) -> None:
        self._adapters.append(adapter)

    def get(self, path: Path) -> FormatAdapter:
        ext = path.suffix.lower()
        for adapter in self._adapters:
            if ext in adapter.supported_extensions:
                return adapter
        raise ValueError(f"No adapter registered for extension '{ext}'")

    def ingest(self, path: Path, pid: str, revision: str = "") -> CanonicalDocument:
        adapter = self.get(path)
        return adapter.ingest(path, pid, revision)
