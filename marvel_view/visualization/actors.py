"""
Styling helpers – apply colour, opacity and Phong shading to vedo meshes.

Colours stored in :mod:`marvel_view.config` as RGB (0–255) tuples are
normalised to the 0–1 range expected by vedo's ``color()`` method.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)

# RGB 0-255 tuple or any colour spec accepted by vedo
ColorSpec = Union[str, Tuple[int, int, int]]


def _normalise_color(color: ColorSpec) -> Union[str, List[float]]:
    """Convert an RGB (0-255) tuple to a (0-1) list; pass strings through."""
    if isinstance(color, (tuple, list)):
        return [c / 255.0 for c in color]
    return color  # named colour or hex string – vedo handles these natively


def _ensure_smooth_normals(mesh: "vedo.Mesh", feature_angle: float = 60.0) -> None:
    """Compute per-vertex normals averaged across faces sharing an edge.

    Required for Gouraud / Phong interpolation to produce results that
    differ visibly from Flat shading.  Sharp edges (dihedral angle above
    ``feature_angle`` degrees) are preserved by splitting vertices.
    """
    try:
        mesh.compute_normals(
            points=True,
            cells=False,
            feature_angle=feature_angle,
            consistency=True,
        )
    except Exception as exc:  # pragma: no cover - defensive: vedo API drift
        logger.warning("compute_normals failed on %s: %s", getattr(mesh, "name", "?"), exc)


def style_mesh(
    mesh: "vedo.Mesh",
    color: ColorSpec,
    opacity: float,
    name: str = "",
    specular: float = 0.25,
    diffuse: float = 0.8,
    ambient: float = 0.1,
) -> "vedo.Mesh":
    """Apply colour, opacity and Phong lighting to a vedo mesh (in-place).

    Parameters
    ----------
    mesh:
        The vedo ``Mesh`` to style.
    color:
        RGB tuple (0–255) or named / hex colour string.
    opacity:
        Float in ``[0, 1]``.  0 = fully transparent, 1 = fully opaque.
    name:
        Optional label stored in ``mesh.name`` for later identification.
    specular, diffuse, ambient:
        Phong lighting coefficients.

    Returns
    -------
    The same ``vedo.Mesh`` object (mutated in-place for convenience).
    """
    _ensure_smooth_normals(mesh)
    mesh.color(_normalise_color(color))
    mesh.alpha(float(opacity))
    mesh.lighting(
        specular=specular,
        diffuse=diffuse,
        ambient=ambient,
    )
    if name:
        mesh.name = name
    return mesh


def style_meshes(
    meshes: List["vedo.Mesh"],
    color: ColorSpec,
    opacity: float,
    base_name: str = "",
    specular: float = 0.25,
    diffuse: float = 0.8,
    ambient: float = 0.1,
) -> List["vedo.Mesh"]:
    """Apply the same style to every mesh in a list.

    Parameters
    ----------
    meshes:
        List of vedo meshes to style.
    color:
        Shared colour for all meshes.
    opacity:
        Shared opacity for all meshes.
    base_name:
        If given, each mesh is named ``<base_name>_NNN``.

    Returns
    -------
    The same list of meshes (modified in-place).
    """
    for i, m in enumerate(meshes):
        tag = f"{base_name}_{i:03d}" if base_name else f"mesh_{i:03d}"
        style_mesh(m, color, opacity, tag, specular, diffuse, ambient)
    return meshes
