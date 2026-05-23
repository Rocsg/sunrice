"""Lighting / shading helpers used by ``marvel-water-conductance``.

Extracted from the historic monolithic ``water_conductance.py``;
behaviour is unchanged.
"""
from __future__ import annotations

import colorsys
import logging

from .constants import (
    AMBIENT,
    BODY_COLOR,
    DIFFUSE,
    OPACITY,
    SPECULAR,
    SPECULAR_COLOR,
    SPECULAR_POWER,
)

logger = logging.getLogger("marvel_view.water_conductance.styling")


def _set_lighting(mesh, ambient, diffuse, specular):
    """Apply Phong lighting with a fixed bluish specular tint."""
    try:
        mesh.lighting(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=SPECULAR_POWER,
            specular_color=[c / 255.0 for c in SPECULAR_COLOR],
        )
    except TypeError:
        mesh.lighting(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            specular_power=SPECULAR_POWER,
        )


def _hue_shifted_rgb(base_rgb, hue_shift):
    """Return ``base_rgb`` rotated by ``hue_shift`` ∈ [0, 1] in HSV space."""
    r, g, b = (c / 255.0 for c in base_rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + hue_shift) % 1.0
    return colorsys.hsv_to_rgb(h, s, v)


def _style_mesh(mesh, body_color=BODY_COLOR) -> None:
    """Apply initial opaque grey body with bluish specular highlights.

    The base RGB used for the diffuse body colour can be overridden via
    ``body_color`` (0-255 triple).  It is also stashed on the mesh as
    ``mesh._body_color`` so later code (hue slider, density-overlay
    restore) can recover it without having to know which mesh it is.
    """
    try:
        mesh.compute_normals(points=True, cells=False,
                             feature_angle=60.0, consistency=True)
    except Exception as exc:  # pragma: no cover - defensive vs vedo API drift
        logger.warning("compute_normals failed: %s", exc)

    try:
        mesh._body_color = tuple(body_color)
    except Exception:  # noqa: BLE001
        pass
    mesh.color([c / 255.0 for c in body_color])
    mesh.alpha(OPACITY)
    _set_lighting(mesh, AMBIENT, DIFFUSE, SPECULAR)
    mesh.name = "Wat_Norm_Cortex"


def _set_shading(mesh, mode_value: int) -> None:
    """Set the VTK interpolation mode (0=Flat, 1=Gouraud, 2=Phong)."""
    prop = getattr(mesh, "properties", None)
    if prop is None:
        try:
            prop = mesh.GetProperty()
        except AttributeError:
            return
    prop.SetInterpolation(int(mode_value))
