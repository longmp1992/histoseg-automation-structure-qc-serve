"""CODA-inspired image registration helpers for HistoSeg 3D.

This module implements a conservative image-guided hard-alignment seed:
tissue-mask Radon rotation followed by phase-correlation translation.  It is
inspired by CODA's image-registration strategy, but it is not a reimplementation
of the full CODA pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import radon, rotate


CODA_METHOD_CREDIT = (
    "CODA-inspired image registration; see Kiemen et al. Nature Methods 2022"
)
CODA_METHOD_REFERENCE_DOI = "10.1038/s41592-022-01650-9"
CODA_METHOD_REFERENCE_URL = "https://www.nature.com/articles/s41592-022-01650-9"
CODA_METHODOLOGY_URL = "https://labs.pathology.jhu.edu/kiemen/coda-3d/"


@dataclass(frozen=True)
class RadonRotationResult:
    """Estimated in-plane rotation to apply to the moving image."""

    rotation_degrees: float
    score: float
    angle_range: tuple[float, float]
    angle_step: float


@dataclass(frozen=True)
class TranslationResult:
    """Estimated pixel shift to apply to the moving image."""

    shift_y: float
    shift_x: float
    error: float
    phase_difference: float
    upsample_factor: int


@dataclass(frozen=True)
class CodaImageRegistrationConfig:
    """Configuration for the conservative CODA-inspired hard-alignment seed."""

    angle_range: tuple[float, float] = (0.0, 180.0)
    angle_step: float = 1.0
    phase_upsample_factor: int = 1


@dataclass(frozen=True)
class CodaImageRegistrationResult:
    """Image-derived similarity seed used before contour TPS refinement."""

    rotation: RadonRotationResult
    translation: TranslationResult
    orientation_disambiguation: str = "translation_iou"
    orientation_candidates: tuple[dict[str, float], ...] = ()
    method_credit: str = CODA_METHOD_CREDIT
    method_reference_doi: str = CODA_METHOD_REFERENCE_DOI
    method_reference_url: str = CODA_METHOD_REFERENCE_URL
    methodology_url: str = CODA_METHODOLOGY_URL


def estimate_radon_rotation(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    angle_range: tuple[float, float] = (0.0, 180.0),
    angle_step: float = 1.0,
) -> RadonRotationResult:
    """Estimate rotation, in degrees, to apply to ``moving_mask``.

    The implementation compares Radon projection-profile signatures across a
    bounded angle grid.  The returned angle is normalized into ``[-90, 90)`` so
    that a moving mask rotated by ``+17`` degrees reports approximately ``-17``.
    """

    fixed = _as_float_image(fixed_mask, name="fixed_mask")
    moving = _as_float_image(moving_mask, name="moving_mask")
    _require_same_shape(fixed, moving)
    angles = _angle_grid(angle_range, angle_step)

    fixed_profile = _radon_signature(fixed, angles)
    moving_profile = _radon_signature(moving, angles)
    scores = np.array(
        [float(np.dot(fixed_profile, np.roll(moving_profile, shift))) for shift in range(len(angles))],
        dtype=float,
    )
    best_shift = int(np.argmax(scores))
    rotation = _normalize_rotation_degrees(float(best_shift) * float(angle_step))
    return RadonRotationResult(
        rotation_degrees=float(rotation),
        score=float(scores[best_shift]),
        angle_range=(float(angle_range[0]), float(angle_range[1])),
        angle_step=float(angle_step),
    )


def estimate_translation(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    *,
    upsample_factor: int = 1,
) -> TranslationResult:
    """Estimate ``(dy, dx)`` pixel shift to apply to ``moving_image``."""

    fixed = _as_float_image(fixed_image, name="fixed_image")
    moving = _as_float_image(moving_image, name="moving_image")
    _require_same_shape(fixed, moving)
    if int(upsample_factor) < 1:
        raise ValueError("upsample_factor must be at least 1.")

    shift, error, phase_difference = phase_cross_correlation(
        fixed,
        moving,
        upsample_factor=int(upsample_factor),
    )
    return TranslationResult(
        shift_y=float(shift[0]),
        shift_x=float(shift[1]),
        error=float(error),
        phase_difference=float(phase_difference),
        upsample_factor=int(upsample_factor),
    )


def estimate_coda_image_registration(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    config: CodaImageRegistrationConfig | None = None,
) -> CodaImageRegistrationResult:
    """Estimate Radon rotation and phase-correlation translation in one pass."""

    cfg = config or CodaImageRegistrationConfig()
    rotation = estimate_radon_rotation(
        fixed_image,
        moving_image,
        angle_range=cfg.angle_range,
        angle_step=cfg.angle_step,
    )
    fixed = _as_float_image(fixed_image, name="fixed_image")
    moving = _as_float_image(moving_image, name="moving_image")
    candidates: list[dict[str, float]] = []
    best_translation: TranslationResult | None = None
    best_angle = float(rotation.rotation_degrees)
    best_score = -math.inf
    for candidate_angle in (
        float(rotation.rotation_degrees),
        _normalize_full_circle_degrees(float(rotation.rotation_degrees) + 180.0),
    ):
        rotated_moving = _rotate_image(moving, candidate_angle)
        translation = estimate_translation(
            fixed,
            rotated_moving,
            upsample_factor=cfg.phase_upsample_factor,
        )
        shifted = _shift_image_no_wrap(
            rotated_moving,
            shift_y=translation.shift_y,
            shift_x=translation.shift_x,
        )
        overlap_score = _binary_iou(fixed, shifted)
        candidates.append(
            {
                "rotation_degrees": float(candidate_angle),
                "phase_shift_y": float(translation.shift_y),
                "phase_shift_x": float(translation.shift_x),
                "translation_iou": float(overlap_score),
                "phase_error": float(translation.error),
            }
        )
        if overlap_score > best_score:
            best_score = float(overlap_score)
            best_angle = float(candidate_angle)
            best_translation = translation

    if best_translation is None:  # pragma: no cover - loop always has two candidates.
        raise RuntimeError("No CODA image registration orientation candidates were evaluated.")

    selected_rotation = RadonRotationResult(
        rotation_degrees=_normalize_full_circle_degrees(best_angle),
        score=rotation.score,
        angle_range=rotation.angle_range,
        angle_step=rotation.angle_step,
    )
    return CodaImageRegistrationResult(
        rotation=selected_rotation,
        translation=best_translation,
        orientation_candidates=tuple(candidates),
    )


def apply_similarity_to_points(
    coords: np.ndarray,
    rotation_degrees: float,
    translate_xy: Sequence[float],
    *,
    origin: str | Sequence[float] = "center",
) -> np.ndarray:
    """Apply a 2D rotate-then-translate similarity transform to point coordinates."""

    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coords must be an array of shape (n, 2).")
    if len(translate_xy) != 2:
        raise ValueError("translate_xy must contain exactly two values: (x, y).")
    origin_xy = _point_origin(points, origin)
    theta = math.radians(float(rotation_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centered = points - origin_xy
    rotated = np.empty_like(centered)
    rotated[:, 0] = centered[:, 0] * cos_t - centered[:, 1] * sin_t
    rotated[:, 1] = centered[:, 0] * sin_t + centered[:, 1] * cos_t
    return rotated + origin_xy + np.asarray(translate_xy, dtype=float)


def _as_float_image(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D image or mask.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    if float(np.max(array)) <= 0:
        raise ValueError(f"{name} must contain at least one positive pixel.")
    return array


def _require_same_shape(fixed: np.ndarray, moving: np.ndarray) -> None:
    if fixed.shape != moving.shape:
        raise ValueError(
            f"fixed and moving images must have the same shape, got {fixed.shape} and {moving.shape}."
        )


def _angle_grid(angle_range: tuple[float, float], angle_step: float) -> np.ndarray:
    start, stop = float(angle_range[0]), float(angle_range[1])
    if not start < stop:
        raise ValueError("angle_range must be an increasing (start, stop) tuple.")
    if float(angle_step) <= 0:
        raise ValueError("angle_step must be greater than 0.")
    angles = np.arange(start, stop, float(angle_step), dtype=float)
    if len(angles) < 3:
        raise ValueError("angle grid must include at least three samples.")
    return angles


def _radon_signature(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    sinogram = radon(image, theta=angles, circle=False)
    profile = np.std(sinogram, axis=0)
    profile = profile - float(np.mean(profile))
    scale = float(np.linalg.norm(profile))
    if scale <= 0:
        return np.zeros_like(profile, dtype=float)
    return profile / scale


def _normalize_rotation_degrees(angle: float) -> float:
    normalized = math.fmod(float(angle), 180.0)
    if normalized < 0:
        normalized += 180.0
    if normalized >= 90.0:
        normalized -= 180.0
    return normalized


def _normalize_full_circle_degrees(angle: float) -> float:
    normalized = math.fmod(float(angle), 360.0)
    if normalized < -180.0:
        normalized += 360.0
    if normalized >= 180.0:
        normalized -= 360.0
    return normalized


def _rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    return rotate(
        image,
        float(angle_degrees),
        resize=False,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True,
    )


def _shift_image_no_wrap(image: np.ndarray, *, shift_y: float, shift_x: float) -> np.ndarray:
    dy = int(round(float(shift_y)))
    dx = int(round(float(shift_x)))
    shifted = np.zeros_like(image)
    src_y0 = max(0, -dy)
    src_y1 = min(image.shape[0], image.shape[0] - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(image.shape[0], image.shape[0] + dy)
    src_x0 = max(0, -dx)
    src_x1 = min(image.shape[1], image.shape[1] - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(image.shape[1], image.shape[1] + dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return shifted
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return shifted


def _binary_iou(fixed: np.ndarray, moving: np.ndarray) -> float:
    fixed_mask = fixed > 0
    moving_mask = moving > 0
    union = np.logical_or(fixed_mask, moving_mask).sum()
    if int(union) == 0:
        return 0.0
    intersection = np.logical_and(fixed_mask, moving_mask).sum()
    return float(intersection / union)


def _point_origin(points: np.ndarray, origin: str | Sequence[float]) -> np.ndarray:
    if isinstance(origin, str):
        if origin != "center":
            raise ValueError("origin must be 'center' or a two-value coordinate.")
        return np.array(
            [
                (float(np.min(points[:, 0])) + float(np.max(points[:, 0]))) / 2.0,
                (float(np.min(points[:, 1])) + float(np.max(points[:, 1]))) / 2.0,
            ],
            dtype=float,
        )
    origin_xy = np.asarray(origin, dtype=float)
    if origin_xy.shape != (2,):
        raise ValueError("origin must be 'center' or a two-value coordinate.")
    return origin_xy


__all__ = [
    "CODA_METHOD_CREDIT",
    "CODA_METHOD_REFERENCE_DOI",
    "CODA_METHOD_REFERENCE_URL",
    "CODA_METHODOLOGY_URL",
    "CodaImageRegistrationConfig",
    "CodaImageRegistrationResult",
    "RadonRotationResult",
    "TranslationResult",
    "apply_similarity_to_points",
    "estimate_coda_image_registration",
    "estimate_radon_rotation",
    "estimate_translation",
]
