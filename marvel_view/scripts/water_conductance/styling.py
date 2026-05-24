"""Lighting / shading helpers used by ``marvel-water-conductance``.

Extracted from the historic monolithic ``water_conductance.py``;
behaviour is unchanged.
"""
from __future__ import annotations

import colorsys
import logging

import numpy as np

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


def _install_scene_lights(plt) -> list:
    """Replace VTK's camera-following headlight with a cinematic 3-point
    scene-fixed lighting rig.

    Used by both the interactive viewer (``marvel-water-conductance``) and
    the movie renderer (``marvel-water-movie``) so both produce the same
    look — letting the flat interactor serve as a reliable VR preview.

    Scene-fixed lights are mandatory for VR cubemap rendering to avoid
    seam artefacts (a camera-attached headlight moves with each cube face
    direction, producing slightly different shading per face and visible
    seams after the equirectangular remap).

    Returns the created :class:`vtkLight` objects; the caller must keep
    them referenced so VTK does not free them mid-render.
    """
    import vtkmodules.vtkRenderingCore as _vtkrc
    created: list = []
    for renderer in plt.renderers:
        try:
            renderer.AutomaticLightCreationOff()
        except Exception:  # noqa: BLE001
            pass
        try:
            renderer.RemoveAllLights()
        except Exception:  # noqa: BLE001
            pass
        try:
            bounds = renderer.ComputeVisiblePropBounds()
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            diag = float(np.linalg.norm(
                [xmax - xmin, ymax - ymin, zmax - zmin]
            ))
            if not np.isfinite(diag) or diag <= 0:
                raise ValueError("bad bounds")
        except Exception:  # noqa: BLE001
            cx = cy = cz = 0.0
            diag = 1.0
        radius = max(diag, 1.0) * 1.8

        # ── 3-point cinematic rig ─────────────────────────────────────
        # The root grows along the world X axis; Y is "up" in the cross-
        # section view.  The key is placed above-right-front so it casts
        # shadows diagonally — very close to what the interactive
        # viewer's camera headlight produces when flying along X.
        light_defs = [
            # (direction from centre, intensity, RGB colour)
            # Key: strong, from above-right, warm white
            (np.array([ 0.55,  1.70,  0.60]), 0.90, (1.00, 0.97, 0.92)),
            # Fill: weak, from below-left, cool blue (matches specular tint)
            (np.array([-0.80, -0.55,  0.35]), 0.22, (0.78, 0.88, 1.00)),
            # Rim: from behind, neutral — edge-separates mesh from bg
            (np.array([-0.25,  0.50, -1.40]), 0.28, (1.00, 1.00, 1.00)),
        ]
        for direction, intensity, color in light_defs:
            n = np.linalg.norm(direction)
            if n < 1e-9:
                continue
            d = direction / n
            lt = _vtkrc.vtkLight()
            lt.SetLightTypeToSceneLight()
            lt.SetPositional(False)
            lt.SetPosition(cx + d[0] * radius,
                           cy + d[1] * radius,
                           cz + d[2] * radius)
            lt.SetFocalPoint(cx, cy, cz)
            lt.SetColor(*color)
            lt.SetIntensity(intensity)
            renderer.AddLight(lt)
            created.append(lt)

        # Ambient fill — very low so it does not flatten the contrast.
        amb = _vtkrc.vtkLight()
        amb.SetLightTypeToSceneLight()
        amb.SetPositional(True)
        amb.SetPosition(cx, cy, cz)
        amb.SetFocalPoint(cx, cy + 1.0, cz)
        amb.SetColor(0.85, 0.90, 1.00)
        amb.SetIntensity(0.08)
        renderer.AddLight(amb)
        created.append(amb)
    return created
