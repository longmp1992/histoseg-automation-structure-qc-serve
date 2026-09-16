"""Cell-GPS import helpers.

Cell-GPS used to expose the same API through the historical ``sfplot`` package
name.  HistoSeg prefers the new ``cellgps`` namespace but keeps a legacy fallback
so older editable environments do not break abruptly.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Iterable

CELLGPS_INSTALL_HINT = (
    "Install Cell-GPS with `pip install "
    "\"Cell-GPS @ git+https://github.com/hutaobo/cellgps.git@main\"`."
)


def import_cellgps() -> ModuleType:
    """Import the recommended Cell-GPS namespace, falling back to legacy sfplot."""

    errors: list[BaseException] = []
    try:
        return import_module("cellgps")
    except ModuleNotFoundError as exc:
        if exc.name != "cellgps":
            raise
        errors.append(exc)

    try:
        return import_module("sfplot")
    except ModuleNotFoundError as exc:
        if exc.name != "sfplot":
            raise
        errors.append(exc)

    raise ImportError(
        "Cell-GPS is required for StructureMap/Search-and-Find workflows. "
        f"{CELLGPS_INSTALL_HINT}"
    ) from (errors[-1] if errors else None)


def require_cellgps_attrs(names: Iterable[str]) -> ModuleType:
    """Return Cell-GPS and validate that required public attributes exist."""

    module = import_cellgps()
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Cell-GPS module {module.__name__!r} is missing required API(s): "
            f"{', '.join(missing)}. {CELLGPS_INSTALL_HINT}"
        )
    return module
