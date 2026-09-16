"""HistoSeg: H&E, 2D contour, and 3D contour reconstruction tools."""

from . import contour, he, threed

try:
    from ._version import __version__
except Exception:
    __version__ = "0+unknown"

__all__ = ["__version__", "contour", "he", "threed"]
