"""
Surface mesh generation from binary masks via marching cubes.

The pipeline for a single mask is:
  1. (Optional) Gaussian blur   → smoother isosurface for large envelopes.
  2. Marching cubes             → raw triangulated surface.
  3. Light Laplacian pre-smooth → soften marching-cubes staircase.
  4. (Optional) Decimation      → reduce face count for faster rendering.
  5. Main Laplacian post-smooth → finish shape on the smaller mesh.

Smoothing is deliberately split around decimation: most of the smoothing
iterations run on the already-decimated mesh (typically 5-20× smaller),
which is substantially faster for a near-identical visual result.

All heavy imports (skimage, vedo) are deferred to runtime so the module
remains importable in lightweight environments.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Proportion of smoothing iterations to apply BEFORE decimation.
# The rest runs after decimation on the smaller mesh.
_PRE_SMOOTH_FRACTION = 0.2
_PRE_SMOOTH_MAX = 15

# Marching-cubes faces aimed for when auto-picking step_size from a target
# face count (MC should produce ~N× the target so decimation has headroom
# without burning RAM on a 100× oversized mesh).
_AUTO_STEP_TARGET_MULTIPLIER = 3.0
_MAX_AUTO_STEP = 6


def _split_smooth_iters(total: int) -> Tuple[int, int]:
    """Split total smoothing iterations into (pre-decimate, post-decimate)."""
    if total <= 0:
        return 0, 0
    pre = min(_PRE_SMOOTH_MAX, max(1, int(round(total * _PRE_SMOOTH_FRACTION))))
    pre = min(pre, total)
    return pre, total - pre


def _estimate_mc_faces(n_voxels: int) -> float:
    """Rough estimate of marching-cubes face count for a blob of N voxels.

    Uses ``max(25 * V**(2/3),  2 * V)``.  The first term fits compact
    blobs (surface \u221d V**(2/3)).  The second term dominates for
    thin / filament-like structures (iodine stains, membranes) where
    marching cubes produces nearly 2 triangles per voxel because the
    surface weaves through most of the volume.  Calibrated on real
    SunRice segmentations:

        label 2 (aqueous, compact)   : 851M vx \u2192 148M faces  (ratio 0.17)
        label 3 comp 0 (iodine, thin):  20M vx \u2192  42M faces  (ratio 2.0)

    Conservative by design \u2014 over-estimating bumps ``step_size`` up one
    notch (visually harmless after smoothing), while under-estimating
    risks allocating a 100\u00d7-oversized mesh.  Good to within a factor
    of ~2, enough to pick a sensible ``step_size``.
    """
    v = max(1.0, float(n_voxels))
    return max(25.0 * v ** (2.0 / 3.0), 2.0 * v)


def _auto_step_size(n_voxels: int, target_faces: Optional[int]) -> int:
    """Pick a marching-cubes ``step_size`` that keeps RAM under control.

    Face count at step_size ``s`` is ~proportional to ``1 / s**2``, so
    ``s ≈ sqrt(F_est / (target * multiplier))``.  Returns ``1`` when no
    target is given or the mesh is already small enough.
    """
    if target_faces is None or target_faces <= 0:
        return 1
    f_est = _estimate_mc_faces(n_voxels)
    goal = float(target_faces) * _AUTO_STEP_TARGET_MULTIPLIER
    if f_est <= goal:
        return 1
    s = int(round((f_est / goal) ** 0.5))
    return max(1, min(_MAX_AUTO_STEP, s))


def mask_to_mesh(
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth_iter: int = 80,
    decimate_fraction: float = 1.0,
    level: float = 0.5,
    gaussian_sigma: float | Tuple[float, float, float] = 0.0,
    offset: Tuple[int, int, int] = (0, 0, 0),
    target_faces: Optional[int] = None,
    step_size: Optional[int] = None,
) -> Optional["vedo.Mesh"]:
    """Convert a binary 3-D mask to a smoothed surface mesh.

    Parameters
    ----------
    mask:
        Binary (0 / 1) 3-D array shaped ``(Z, Y, X)``.  May be a cropped
        bounding-box view of a larger volume; use ``offset`` to place the
        result back in the global coordinate frame.
    spacing:
        Physical voxel size ``(z, y, x)`` in consistent units.
    smooth_iter:
        Total Laplacian smoothing iterations (0 = skip smoothing).  A small
        fraction runs before decimation to soften the marching-cubes
        staircase; the remainder runs on the decimated mesh.
    decimate_fraction:
        Fraction of faces to **keep** after decimation (1.0 = keep all).
        A value of 0.1 keeps 10% of the faces.  Ignored when ``target_faces``
        is set.
    level:
        Iso-surface threshold for marching cubes (default 0.5).
    gaussian_sigma:
        If > 0, apply a Gaussian blur with this sigma to the floating-point
        mask *before* marching cubes.  Useful for obtaining a smooth outer
        envelope from a noisy or detailed binary region.
    offset:
        ``(z, y, x)`` origin of ``mask`` in the full volume (voxel units).
        Added back to vertex coordinates (times ``spacing``) so meshes from
        cropped sub-volumes end up in the global coordinate system.
    target_faces:
        If given, overrides ``decimate_fraction``: the decimation ratio is
        computed after marching cubes as ``target_faces / n_faces``.  No-op
        when the mesh already has <= ``target_faces`` faces.
    step_size:
        Marching-cubes stride (``skimage.measure.marching_cubes(step_size=…)``).
        Larger values produce coarser meshes in roughly ``1 / step_size**2``
        the time and RAM.  When ``None`` and ``target_faces`` is set, a
        stride is picked automatically so MC emits ~3× the target face
        count — this avoids allocating a 100×-oversized mesh that is
        about to be decimated away.

    Notes
    -----
    Decimation uses ``vtkQuadricDecimation`` (vedo's default ``decimate()``),
    which is faster and more stable at aggressive ratios than ``vtkDecimatePro``.
    For reductions > 90 % a two-pass strategy is applied automatically.

    Returns
    -------
    vedo.Mesh or None
        Smoothed mesh, or ``None`` if the mask is empty or meshing fails.
    """
    from skimage import measure
    import vedo

    if np.count_nonzero(mask) == 0:
        logger.warning("mask_to_mesh received an empty mask – skipping.")
        return None

    n_voxels = int(np.count_nonzero(mask))
    logger.info(
        "  mask_to_mesh  shape=%s  nonzero=%d  spacing=%s  offset=%s",
        mask.shape, n_voxels, spacing, offset,
    )

    volume = mask.astype(np.float32)

    # ``gaussian_sigma`` is either a scalar (isotropic blur) or a per-axis
    # sequence ``(sz, sy, sx)`` for anisotropic blur (e.g. blur only along
    # one axis).  Skip the filter when every sigma is zero.
    if isinstance(gaussian_sigma, (tuple, list)):
        sigma_for_filter: object = tuple(float(s) for s in gaussian_sigma)
        any_blur = any(s > 0.0 for s in sigma_for_filter)
        sigma_repr = "(" + ", ".join(f"{s:.2f}" for s in sigma_for_filter) + ")"
    else:
        sigma_for_filter = float(gaussian_sigma)
        any_blur = sigma_for_filter > 0.0
        sigma_repr = f"{sigma_for_filter:.2f}"

    if any_blur:
        from scipy.ndimage import gaussian_filter
        logger.info("  [1/5] Gaussian blur  sigma=%s …", sigma_repr)
        t = time.perf_counter()
        volume = gaussian_filter(volume, sigma=sigma_for_filter)
        logger.info("  └─ blur done  (%.1f s)", time.perf_counter() - t)

    # Resolve marching-cubes step_size.
    if step_size is None:
        step_size = _auto_step_size(n_voxels, target_faces)
        if step_size > 1:
            logger.info(
                "  Auto step_size=%d  (est. MC faces ≈ %.2e → target %s)",
                step_size, _estimate_mc_faces(n_voxels) / (step_size ** 2),
                target_faces,
            )
    else:
        step_size = max(1, int(step_size))
        if step_size > 1:
            logger.info("  step_size=%d (explicit)", step_size)

    logger.info("  [2/5] Marching cubes  level=%.2f  step=%d …", level, step_size)
    t = time.perf_counter()
    try:
        verts, faces, _normals, _vals = measure.marching_cubes(
            volume, level=level, spacing=spacing, step_size=step_size,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("marching_cubes failed: %s", exc)
        return None
    # Release the float32 volume ASAP — it can be several GiB.
    del volume
    logger.info(
        "  └─ marching cubes done  %d vertices  %d faces  (%.1f s)",
        len(verts), len(faces), time.perf_counter() - t,
    )

    if len(faces) == 0:
        logger.warning("marching_cubes produced zero faces.")
        return None

    # Shift vertices back into the global frame when the input was a
    # bbox-cropped sub-volume.
    if any(o != 0 for o in offset):
        shift = np.asarray(
            [offset[0] * spacing[0], offset[1] * spacing[1], offset[2] * spacing[2]],
            dtype=verts.dtype,
        )
        verts = verts + shift

    mesh = vedo.Mesh([verts, faces])
    # vedo has copied the arrays into VTK structures; drop our refs so the
    # numpy buffers can be freed before the big smoothing/decimation passes.
    del verts, faces

    pre_iter, post_iter = _split_smooth_iters(smooth_iter)

    if pre_iter > 0:
        logger.info("  [3/5] Pre-decimation smoothing  niter=%d …", pre_iter)
        t = time.perf_counter()
        mesh.smooth(niter=pre_iter, boundary=True)
        logger.info(
            "  └─ pre-smooth done  %d faces  (%.1f s)",
            mesh.ncells, time.perf_counter() - t,
        )
    else:
        logger.info("  [3/5] Pre-decimation smoothing skipped.")

    # Resolve decimation fraction — target_faces wins over decimate_fraction.
    if target_faces is not None and target_faces > 0 and mesh.ncells > target_faces:
        effective_fraction = float(target_faces) / float(mesh.ncells)
        logger.info(
            "  target_faces=%d  →  effective fraction=%.4f  (from %d faces)",
            target_faces, effective_fraction, mesh.ncells,
        )
    elif target_faces is not None:
        effective_fraction = 1.0  # mesh already small enough
        logger.info(
            "  target_faces=%d  ≥  current %d faces  —  decimation skipped.",
            target_faces, mesh.ncells,
        )
    else:
        effective_fraction = decimate_fraction

    if 0.0 < effective_fraction < 1.0:
        faces_before = mesh.ncells
        logger.info(
            "  [4/5] Decimation (quadric)  fraction=%.4f  (%d → ~%d faces) …",
            effective_fraction, faces_before,
            int(faces_before * effective_fraction),
        )
        # Multi-pass strategy: a single extreme reduction (> 90%) can hang
        # vtkDecimatePro/vtkQuadricDecimation on large meshes.  Split into
        # two passes when the ratio is very aggressive.
        _SINGLE_PASS_THRESHOLD = 0.1  # fractions below this get two passes
        t = time.perf_counter()
        if effective_fraction < _SINGLE_PASS_THRESHOLD:
            intermediate = min(effective_fraction * 10, 0.5)
            logger.info(
                "  └─ pass 1/2  fraction=%.3f  (%d → ~%d faces) …",
                intermediate, mesh.ncells, int(mesh.ncells * intermediate),
            )
            mesh.decimate(fraction=intermediate)
            logger.info(
                "  └─ pass 1/2 done  %d faces  (%.1f s)",
                mesh.ncells, time.perf_counter() - t,
            )
            second_pass = effective_fraction / intermediate
            logger.info(
                "  └─ pass 2/2  fraction=%.3f  (%d → ~%d faces) …",
                second_pass, mesh.ncells, int(mesh.ncells * second_pass),
            )
            t2 = time.perf_counter()
            mesh.decimate(fraction=second_pass)
            logger.info(
                "  └─ pass 2/2 done  %d faces  (%.1f s)",
                mesh.ncells, time.perf_counter() - t2,
            )
        else:
            mesh.decimate(fraction=effective_fraction)
        logger.info(
            "  └─ decimation total  %d faces  (%.1f s total)",
            mesh.ncells, time.perf_counter() - t,
        )
    else:
        logger.info("  [4/5] Decimation skipped (fraction=1.0).")

    if post_iter > 0:
        logger.info("  [5/5] Post-decimation smoothing  niter=%d …", post_iter)
        t = time.perf_counter()
        mesh.smooth(niter=post_iter, boundary=True)
        logger.info(
            "  └─ post-smooth done  %d faces  (%.1f s)",
            mesh.ncells, time.perf_counter() - t,
        )
    else:
        logger.info("  [5/5] Post-decimation smoothing skipped.")

    logger.info("  mask_to_mesh done  final: %d vertices  %d faces", mesh.npoints, mesh.ncells)
    return mesh


def masks_to_meshes(
    masks: List[np.ndarray],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    smooth_iter: int = 80,
    decimate_fraction: float = 1.0,
    gaussian_sigma: float | Tuple[float, float, float] = 0.0,
) -> List["vedo.Mesh"]:
    """Convert a list of full-volume binary masks to surface meshes.

    ``None`` results (failed marching cubes, empty masks) are filtered out.
    For large numbers of small components, prefer
    :func:`~marvel_view.preprocessing.connected_components.split_components`
    (which returns bbox-cropped sub-masks) and feed those into
    :func:`mask_to_mesh` with their offsets.

    Parameters
    ----------
    masks:
        List of full-size binary ``uint8`` arrays.
    spacing:
        Physical voxel size ``(z, y, x)``.
    smooth_iter:
        Total Laplacian smoothing iterations (split around decimation).
    decimate_fraction:
        Fraction of faces to keep (1.0 = no decimation).
    gaussian_sigma:
        Pre-meshing Gaussian blur sigma (0 = disabled).

    Returns
    -------
    List[vedo.Mesh]
    """
    meshes: list["vedo.Mesh"] = []
    for i, m in enumerate(masks):
        logger.info("  Meshing component %d / %d …", i + 1, len(masks))
        t = time.perf_counter()
        mesh = mask_to_mesh(
            m,
            spacing=spacing,
            smooth_iter=smooth_iter,
            decimate_fraction=decimate_fraction,
            gaussian_sigma=gaussian_sigma,
        )
        if mesh is not None:
            logger.info(
                "  └─ component %d done  %d faces  (%.1f s)",
                i + 1, mesh.ncells, time.perf_counter() - t,
            )
            meshes.append(mesh)
        else:
            logger.warning("  └─ component %d failed or empty.", i + 1)
    return meshes
