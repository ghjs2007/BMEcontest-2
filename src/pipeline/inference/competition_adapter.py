"""Isolated adapter boundary for the official competition input/output contract.

The released competition materials audited for this refactor do not define an
official machine input/output schema or runner contract.  This boundary
therefore refuses every official-mode call instead of guessing wire formats.
When the official contract is published, register a concrete adapter here and
rebuild ``dist/submission``; no other component changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

UNREGISTERED_MESSAGE = "official competition input/output adapter is not registered"


@runtime_checkable
class CompetitionAdapter(Protocol):
    """Translate official competition I/O to/from canonical data."""

    def load(self, path: Path) -> tuple[Path, str | None]:
        """Return the canonical raw input path and optional subject id for an official input."""
        ...

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        """Write an official competition output document for a canonical prediction."""
        ...


class UnsupportedCompetitionAdapter:
    """Default adapter: refuses until the official contract is registered."""

    def load(self, path: Path) -> tuple[Path, str | None]:
        raise NotImplementedError(UNREGISTERED_MESSAGE)

    def dump(self, prediction: Mapping[str, object], path: Path) -> None:
        raise NotImplementedError(UNREGISTERED_MESSAGE)


_REGISTERED: dict[str, CompetitionAdapter] = {}


def register_adapter(name: str, adapter: CompetitionAdapter) -> None:
    """Register a concrete adapter under ``name`` (call before building dist/submission)."""
    _REGISTERED[str(name)] = adapter


def registered_adapter(name: str = "official") -> CompetitionAdapter:
    """Return a registered adapter or refuse with the documented error."""
    try:
        return _REGISTERED[str(name)]
    except KeyError:
        raise NotImplementedError(UNREGISTERED_MESSAGE) from None
