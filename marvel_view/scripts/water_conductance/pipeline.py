"""Build / cache helpers for meshes, arrow fields and crown tracks.

Extracted verbatim from the historic monolithic ``water_conductance.py``.
All public function signatures, return types and log messages are
preserved — these helpers are also imported by the sibling scripts
``water_conductance_build_meshes`` and ``water_movie`` via the
backward-compat re-exports in :mod:`marvel_view.scripts.water_conductance.__init__`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from marvel_view.preprocessing import load_float_volume, mask_to_mesh

from .constants import (
    DEFAULT_ARROW_STRIDE,
    DEFAULT_DENSITY_RADIUS_PX,
    DEFAULT_DENSITY_SIGMA,
    DEFAULT_DENSITY_STEP_PX,
    DEFAULT_DENSITY_UPSAMPLE,
    DEFAULT_FINE_STRIDE,
    DEFAULT_LAMES_ALPHA,
    DEFAULT_LONG_AXIS_STRIDE,
    DEFAULT_N_SOURCE_POINTS,
    DEFAULT_OVERLAY_LEVEL,
    DEFAULT_SPACING,
)

logger = logging.getLogger("marvel_view.water_conductance")


def _build_mesh(input_path: Path, level: float, smooth_iter: int):
    """Load the float volume and run marching cubes at ``level``."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")

    volume = load_float_volume(input_path)
    logger.info(
        "Meshing at iso-level %.3f  (no decimation, step_size=1) …", level,
    )
    mesh = mask_to_mesh(
        volume,
        spacing=DEFAULT_SPACING,
        smooth_iter=smooth_iter,
        decimate_fraction=1.0,
        level=level,
        gaussian_sigma=0.0,
        target_faces=None,
        step_size=1,
    )
    if mesh is None:
        raise RuntimeError(
            f"Marching cubes returned no surface at level={level} for {input_path}."
        )
    logger.info(
        "Mesh ready  –  %d vertices  %d faces", mesh.npoints, mesh.ncells,
    )
    return mesh


def _parse_central_axis(path: Path) -> tuple[float, float]:
    """Parse ``Coordinates_central_axis.txt`` (format ``X=…\\nY=…``)."""
    import re
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    cx = cy = None
    for m in re.finditer(r"([XYxy])\s*=\s*([0-9.eE+\-]+)", txt):
        v = float(m.group(2))
        if m.group(1).lower() == "x":
            cx = v
        else:
            cy = v
    if cx is None or cy is None:
        raise ValueError(
            f"Could not parse X= / Y= central-axis coords from {path}"
        )
    return cx, cy


def _build_density_facet_scalars(
    *,
    mask_cortex_path: Path,
    paths_domain_path: Path,
    central_axis_path: Path,
    bridges_mesh=None,
    all_mesh=None,
    radius_px: float = DEFAULT_DENSITY_RADIUS_PX,
    step_px: float = DEFAULT_DENSITY_STEP_PX,
    upsample: int = DEFAULT_DENSITY_UPSAMPLE,
    sigma_up: float = DEFAULT_DENSITY_SIGMA,
) -> dict:
    """Compute per-cell density scalars (∈ [0, 1]) for two meshes.

    The cortex mask is binned in (L, θ) where ``L`` is the normalised
    coordinate along the long volume axis (axis 0 of the TIFF stack) and
    ``θ`` is the azimuth around the central axis read from
    ``Coordinates_central_axis.txt``.  For each bin we count the number
    of cortex voxels (``total``) and how many of those also lie inside
    the possible-paths domain (``on``); the ratio ``on / total`` gives a
    density map in [0, 1].  The map is then upsampled by ``upsample``×
    and smoothed by a Gaussian of ``sigma_up`` (upsampled-pixel units)
    with wrap-around on the θ axis.

    The smoothed grid is finally sampled at every face centroid of both
    input meshes and returned as a dict.  Pass ``None`` for either mesh
    to skip that side.
    """
    import numpy as np
    import tifffile
    from scipy.ndimage import gaussian_filter, zoom

    cx, cy = _parse_central_axis(Path(central_axis_path))
    logger.info("Density: central axis at (X=%.3f, Y=%.3f) px", cx, cy)

    mask = tifffile.imread(str(mask_cortex_path))
    paths = tifffile.imread(str(paths_domain_path))
    if mask.shape != paths.shape:
        raise ValueError(
            f"Density: shape mismatch  mask={mask.shape}  "
            f"paths={paths.shape}"
        )
    if mask.ndim != 3:
        raise ValueError(
            f"Density: expected 3-D volumes, got shape {mask.shape}"
        )
    nz, ny, nx = mask.shape
    long_axis = int(np.argmax(mask.shape))
    if long_axis != 0:
        logger.warning(
            "Density: long volume axis is %d, expected 0; "
            "L coord assumes axis-0 = principal axis.", long_axis,
        )

    # ── Bin counts (square-ish bins at the surface) ──────────────────
    n_L = max(1, int(round(nz / float(step_px))))
    n_T = max(1, int(round(2.0 * np.pi * float(radius_px) / float(step_px))))
    logger.info("Density: L bins=%d  θ bins=%d  (vol shape=%s)",
                n_L, n_T, mask.shape)

    # ── Histogram cortex voxels in (L, θ) ─────────────────────────────
    mask_bool = mask > 0
    paths_bool = paths > 0
    zz, yy, xx = np.nonzero(mask_bool)
    if zz.size == 0:
        raise ValueError(f"Density: empty mask at {mask_cortex_path}")
    Lf = zz.astype(np.float64) / max(nz - 1, 1)
    Lb = np.clip((Lf * n_L).astype(np.int64), 0, n_L - 1)
    dy = yy.astype(np.float64) - float(cy)
    dx = xx.astype(np.float64) - float(cx)
    theta = np.arctan2(dy, dx)                       # [-π, π]
    Tf = (theta + np.pi) / (2.0 * np.pi)
    Tb = np.clip((Tf * n_T).astype(np.int64), 0, n_T - 1)

    flat = Lb * n_T + Tb
    n_total = np.bincount(flat, minlength=n_L * n_T)
    on_mask = paths_bool[zz, yy, xx]
    n_on = np.bincount(flat[on_mask], minlength=n_L * n_T)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(n_total > 0,
                         n_on.astype(np.float32) / np.maximum(n_total, 1),
                         0.0)
    grid = ratio.reshape(n_L, n_T).astype(np.float32)
    logger.info(
        "Density: cortex voxels=%d  on-path voxels=%d  "
        "ratio range=[%.3f, %.3f]",
        int(n_total.sum()), int(n_on.sum()),
        float(grid.min()), float(grid.max()),
    )

    # ── Upsample 3× and Gaussian-smooth (wrap on θ) ──────────────────
    up = max(1, int(upsample))
    grid_up = zoom(grid, zoom=up, order=1, mode="nearest")
    pad_t = int(np.ceil(3.0 * float(sigma_up)))
    if pad_t > 0:
        padded = np.concatenate(
            [grid_up[:, -pad_t:], grid_up, grid_up[:, :pad_t]], axis=1
        )
    else:
        padded = grid_up
    smoothed = gaussian_filter(padded, sigma=(sigma_up, sigma_up),
                                mode="nearest")
    if pad_t > 0:
        smoothed = smoothed[:, pad_t:pad_t + grid_up.shape[1]]
    grid_smooth = np.clip(smoothed, 0.0, 1.0).astype(np.float32)
    n_L_up, n_T_up = grid_smooth.shape
    logger.info(
        "Density: upsampled to (%d, %d), sigma=%.2f px", n_L_up, n_T_up,
        float(sigma_up),
    )

    # ── Sample at face centroids ─────────────────────────────────────
    def _scalars_for(mesh) -> "np.ndarray | None":
        if mesh is None:
            return None
        # Compute one centroid per VTK cell (any type) so the scalar
        # array length matches mesh.ncells exactly — required by
        # vedo's cmap(..., on="cells").  vtkCellCenters does this in C.
        try:
            import vtk
            from vtkmodules.util.numpy_support import vtk_to_numpy
            try:
                pd = mesh.dataset           # vedo ≥ 2024
            except AttributeError:
                pd = mesh.polydata()        # older vedo
            cc = vtk.vtkCellCenters()
            cc.SetInputData(pd)
            cc.Update()
            out = cc.GetOutput()
            cents = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Density: vtkCellCenters failed (%s); falling back to "
                "polygon-only centroids.", exc,
            )
            try:
                verts = np.asarray(mesh.vertices)
            except Exception:  # noqa: BLE001
                verts = np.asarray(mesh.points())
            try:
                cells = np.asarray(mesh.cells)
            except Exception:  # noqa: BLE001
                cells = np.asarray(mesh.cells())
            if cells.ndim != 2 or cells.shape[1] < 3:
                cents = np.array([
                    verts[np.asarray(c, dtype=np.int64)].mean(axis=0)
                    for c in cells
                ], dtype=np.float64)
            else:
                cents = verts[cells[:, :3].astype(np.int64)].mean(axis=1)
        zc = cents[:, 0]
        yc = cents[:, 1]
        xc = cents[:, 2]
        Lfc = np.clip(zc / max(nz - 1, 1), 0.0, 1.0 - 1e-9)
        Lbc = np.clip((Lfc * n_L_up).astype(np.int64), 0, n_L_up - 1)
        th = np.arctan2(yc - float(cy), xc - float(cx))
        Tfc = (th + np.pi) / (2.0 * np.pi)
        Tbc = np.clip((Tfc * n_T_up).astype(np.int64), 0, n_T_up - 1)
        return grid_smooth[Lbc, Tbc].astype(np.float32)

    bridges_sc = _scalars_for(bridges_mesh)
    all_sc     = _scalars_for(all_mesh)
    if bridges_sc is not None:
        logger.info(
            "Density: bridges-mesh scalars shape=%s  range=[%.3f, %.3f]",
            bridges_sc.shape, float(bridges_sc.min()), float(bridges_sc.max()),
        )
    if all_sc is not None:
        logger.info(
            "Density: all-mesh scalars shape=%s  range=[%.3f, %.3f]",
            all_sc.shape, float(all_sc.min()), float(all_sc.max()),
        )
    return {
        "bridges": bridges_sc,
        "all":     all_sc,
        "grid":    grid_smooth,
        "n_L":     int(n_L),
        "n_T":     int(n_T),
        "radius":  float(radius_px),
        "step":    float(step_px),
        "axis_cx": float(cx),
        "axis_cy": float(cy),
    }



def _load_or_build_mesh(
    input_path: Path,
    *,
    level: float,
    smooth_iter: int,
    cache_path: Path | None,
    rebuild: bool = False,
):
    """Load a previously cached .vtk mesh, or run marching cubes.

    When ``cache_path`` is provided and exists (and ``rebuild`` is False),
    the mesh is loaded with vedo and returned immediately.  Otherwise the
    mesh is built from the TIFF input via :func:`_build_mesh`.
    """
    if cache_path is not None and cache_path.exists() and not rebuild:
        logger.info("Loading cached mesh from %s …", cache_path)
        import vedo
        mesh = vedo.Mesh(str(cache_path))
        logger.info(
            "Cached mesh loaded  –  %d vertices  %d faces",
            mesh.npoints, mesh.ncells,
        )
        return mesh
    return _build_mesh(input_path, level=level, smooth_iter=smooth_iter)


def _build_arrow_field(
    geoddist_volume,
    *,
    stride: int = DEFAULT_ARROW_STRIDE,
    long_stride: int = DEFAULT_LONG_AXIS_STRIDE,
    fine_stride: int = DEFAULT_FINE_STRIDE,
    spacing=DEFAULT_SPACING,
):
    """Compute a sub-sampled gradient arrow field from a geodesic distance map.

    The field is first analysed at the finer resolution ``fine_stride`` —
    both the unit gradient direction and the divergence of that unit
    field (a measure of local stream-line *compression*) are evaluated
    there.  Then arrows are emitted with a *per-axis* stride: ``stride``
    on the two short axes, ``long_stride`` on the longest one (so the
    elongated direction is sub-sampled more aggressively).
    """
    import numpy as np

    full = np.asarray(geoddist_volume)
    sp_z, sp_y, sp_x = float(spacing[0]), float(spacing[1]), float(spacing[2])
    long_axis = int(np.argmax(full.shape))

    # ── Fine analysis grid ────────────────────────────────────────────────
    fs = max(1, int(fine_stride))
    sub = full[::fs, ::fs, ::fs].astype(np.float32)

    gz, gy, gx = np.gradient(sub)
    gz /= max(fs * sp_z, 1e-9)
    gy /= max(fs * sp_y, 1e-9)
    gx /= max(fs * sp_x, 1e-9)

    norms = np.sqrt(gz * gz + gy * gy + gx * gx)
    safe_grad = norms > 1e-9
    uz = np.where(safe_grad, gz / np.where(safe_grad, norms, 1.0), 0.0)
    uy = np.where(safe_grad, gy / np.where(safe_grad, norms, 1.0), 0.0)
    ux = np.where(safe_grad, gx / np.where(safe_grad, norms, 1.0), 0.0)

    # Divergence of the unit field.  Negative inside a converging flow
    # (stream-lines piling up).  We work with the convergence rate
    # `compr = -div` (positive when flow concentrates).
    div = (
        np.gradient(uz, axis=0) / max(fs * sp_z, 1e-9)
        + np.gradient(uy, axis=1) / max(fs * sp_y, 1e-9)
        + np.gradient(ux, axis=2) / max(fs * sp_x, 1e-9)
    )
    compr = -div  # high = converging fast (potential singularity)

    # ── Sub-sample fine grid to draw resolution (anisotropic) ────────────
    # Per-axis step in fine-grid voxels.  The longest volume axis gets a
    # coarser draw stride so the elongated direction is not flooded with
    # redundant arrows.
    steps = [max(1, int(round(stride / fs)))] * 3
    steps[long_axis] = max(1, int(round(long_stride / fs)))
    sl = (slice(None, None, steps[0]),
          slice(None, None, steps[1]),
          slice(None, None, steps[2]))
    sub_d   = sub[sl]
    uz_d, uy_d, ux_d = uz[sl], uy[sl], ux[sl]
    compr_d = compr[sl]

    Z, Y, X = sub_d.shape
    # Effective draw stride per axis, in original-volume voxels.
    cells = [fs * steps[0], fs * steps[1], fs * steps[2]]
    zs = np.arange(Z, dtype=np.float32) * (cells[0] * sp_z)
    ys = np.arange(Y, dtype=np.float32) * (cells[1] * sp_y)
    xs = np.arange(X, dtype=np.float32) * (cells[2] * sp_x)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

    pts_world = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    dirs      = np.stack([uz_d, uy_d, ux_d], axis=-1).reshape(-1, 3)
    sub_flat  = sub_d.reshape(-1)
    compr_flat = compr_d.reshape(-1)

    inside = (sub_flat > 1e-3) & (np.linalg.norm(dirs, axis=-1) > 1e-6)
    pts        = pts_world[inside]
    dirs_unit  = dirs[inside]
    compr_keep = compr_flat[inside]

    # Central axis direction.  Centre of perpendicular-to-long-axis plane.
    spans = [
        (Z - 1) * cells[0] * sp_z,
        (Y - 1) * cells[1] * sp_y,
        (X - 1) * cells[2] * sp_x,
    ]
    centre = np.array([0.5 * spans[0], 0.5 * spans[1], 0.5 * spans[2]],
                      dtype=np.float32)
    toward = centre[None, :] - pts
    toward[:, long_axis] = 0.0
    tn = np.linalg.norm(toward, axis=-1)
    safe = tn > 1e-9
    toward_unit = np.zeros_like(toward)
    toward_unit[safe] = toward[safe] / tn[safe, None]

    dot = np.sum(dirs_unit * toward_unit, axis=-1)
    # Drop arrows with angle > 90° (dot < 0): segmentation artefacts.
    keep = (dot >= 0.0) & safe
    pts        = pts[keep]
    dirs_unit  = dirs_unit[keep]
    dot        = dot[keep]
    compr_keep = compr_keep[keep]
    scores = (np.arccos(np.clip(dot, 0.0, 1.0)) / (0.5 * np.pi)).astype(np.float32)

    # Boost ∈ [0, 1] from log of convergence rate.
    # Map percentile 70 → 0 (normal flux concentration → base size),
    # percentile 99 → 1 (very high local compression → tripled).
    log_c = np.log1p(np.maximum(compr_keep, 0.0))
    if len(log_c) > 1:
        lo, hi = np.percentile(log_c, [70.0, 99.0])
        denom = max(float(hi - lo), 1e-6)
        boost = np.clip((log_c - lo) / denom, 0.0, 1.0)
    else:
        boost = np.zeros_like(log_c)
    boost = boost.astype(np.float32)

    if len(scores):
        p05, p50, p95 = np.percentile(scores, [5, 50, 95])
        bp50, bp95 = np.percentile(boost, [50, 95])
        logger.info(
            "Arrow field: fine=%d  draw_cells=%s  kept=%d/%d  "
            "long_axis=col%d  score p5/p50/p95=%.2f/%.2f/%.2f  "
            "boost p50/p95=%.2f/%.2f",
            fs, tuple(cells), int(keep.sum()), int(len(keep)),
            long_axis,
            float(p05), float(p50), float(p95),
            float(bp50), float(bp95),
        )
    else:
        logger.info(
            "Arrow field: fine=%d  draw_cells=%s  kept=0",
            fs, tuple(cells),
        )

    return {
        "pts":         pts.astype(np.float32),
        "dirs":        dirs_unit.astype(np.float32),
        "scores":      scores,
        "boost":       boost,
        "stride":      int(min(cells)),
        # Full-volume compression field at fine_stride resolution.
        # Shape (Zf, Yf, Xf) where each axis = original_axis // fine_stride.
        # Passed through to build_meshes for TIFF export and to
        # crown-track coloring for boost interpolation.
        "compr_volume": compr.astype(np.float32),
        "fine_stride":  fs,
    }


def _load_or_build_arrows(
    geoddist_path: Path,
    *,
    cache_path: Path | None,
    stride: int = DEFAULT_ARROW_STRIDE,
    long_stride: int = DEFAULT_LONG_AXIS_STRIDE,
    fine_stride: int = DEFAULT_FINE_STRIDE,
    rebuild: bool = False,
):
    """Load a cached arrow field (.npz), or compute it from the TIFF.

    Returns ``None`` if neither the cache nor the TIFF is available
    (so the caller can skip the arrow toggle gracefully).
    """
    import numpy as np

    if cache_path is not None and cache_path.exists() and not rebuild:
        try:
            logger.info("Loading cached arrow field from %s …", cache_path)
            data = np.load(cache_path)
            n = len(data["pts"])
            boost = data["boost"] if "boost" in data.files \
                    else (data["sat"] if "sat" in data.files
                          else np.zeros(n, dtype=np.float32))
            return {
                "pts":    data["pts"],
                "dirs":   data["dirs"],
                "scores": data["scores"],
                "boost":  boost.astype(np.float32),
                "stride": int(data["stride"]) if "stride" in data.files else stride,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read arrows cache %s: %s – rebuilding.",
                           cache_path, exc)

    if not geoddist_path.exists():
        logger.info("Geodesic distance TIFF not found at %s – arrows disabled.",
                    geoddist_path)
        return None

    logger.info("Building arrow field from %s …", geoddist_path)
    volume = load_float_volume(geoddist_path)
    return _build_arrow_field(
        volume, stride=stride, long_stride=long_stride,
        fine_stride=fine_stride, spacing=DEFAULT_SPACING,
    )


def _build_crown_dijkstra_tracks(
    source_volume,
    target_volume,
    paths_volume,
    *,
    spacing=DEFAULT_SPACING,
    n_source_points: int = DEFAULT_N_SOURCE_POINTS,
    compr_volume=None,
    fine_stride: int = DEFAULT_FINE_STRIDE,
):
    """Compute geodesic shortest-paths from Source_crown to Target_crown.

    Uses ``skimage.graph.MCP_Geometric`` to propagate a cost field from
    all target voxels through the allowed-domain mask
    (``Source_Target_Possible_Paths.tif``).  Each selected source voxel
    then traces back its optimal path via ``mcp.traceback()``.

    Parameters
    ----------
    source_volume : array-like
        Binary mask (uint8 or float) — voxels >= 127 are potential sources.
    target_volume : array-like
        Binary mask — voxels >= 127 seed the distance propagation.
    paths_volume : array-like
        Binary mask — voxels >= 127 define the traversable domain.
    spacing : tuple of float
        Physical voxel size (z, y, x) in world units.
    n_source_points : int
        Number of source voxels to trace (uniformly sampled from the
        available source voxels that are inside the domain).
    compr_volume : ndarray or None
        Convergence field at ``fine_stride`` resolution, used to
        compute per-segment ``boost`` values.  None → boost = 0.
    fine_stride : int
        Sub-sampling factor used to build ``compr_volume`` (matches
        ``_build_arrow_field``'s ``fine_stride``).

    Returns
    -------
    dict with keys ``segs_start``, ``segs_end``, ``scores``, ``boost``,
    ``pts``, ``dirs``, ``stride``, ``format``.
    Returns ``None`` if tracing yields no segments.
    """
    import numpy as np
    from skimage.graph import MCP_Geometric

    sp_z, sp_y, sp_x = float(spacing[0]), float(spacing[1]), float(spacing[2])

    source_m = np.asarray(source_volume) >= 127
    target_m = np.asarray(target_volume) >= 127
    paths_m  = np.asarray(paths_volume)  >= 127

    n_source = int(source_m.sum())
    n_target = int(target_m.sum())
    n_domain = int(paths_m.sum())
    logger.info(
        "Crown Dijkstra: source=%d  target=%d  domain=%d  shape=%s",
        n_source, n_target, n_domain, source_m.shape,
    )
    if n_target == 0:
        logger.warning("Target_crown has no white voxels – tracks disabled.")
        return None
    if n_source == 0:
        logger.warning("Source_crown has no white voxels – tracks disabled.")
        return None

    # ── Build cost field and propagate from all target voxels ────────────
    # Cost = 1 inside domain, near-infinite outside.  MCP_Geometric
    # scales each edge by the Euclidean distance between voxel centres,
    # so the cumulative cost is approximately the geodesic arc-length.
    costs = np.where(paths_m, 1.0, 1e12).astype(np.float64)
    target_coords = np.argwhere(target_m)
    logger.info("Building MCP cost field (shape=%s) …", costs.shape)
    mcp = MCP_Geometric(costs)
    logger.info("Propagating from %d target voxels …", len(target_coords))
    cumcosts, _ = mcp.find_costs(target_coords)
    logger.info("MCP propagation done.")

    # ── Attainable source voxels ─────────────────────────────────────────
    attainable = (
        source_m
        & np.isfinite(cumcosts)
        & (cumcosts < 1e11)
    )
    att_coords = np.argwhere(attainable)
    n_att = len(att_coords)
    logger.info("Attainable source voxels: %d / %d", n_att, n_source)
    if n_att == 0:
        logger.warning("No source voxel is reachable from target – "
                       "check that Source_crown and Target_crown overlap "
                       "with Source_Target_Possible_Paths.")
        return None

    # ── Source-point selection: random init + fast k-medoids ─────────────
    # Random initialisation followed by a small number of Lloyd iterations
    # with medoid snap (same strategy as the column seeding in
    # _kmeans_medoid_crown / water_lames.py).  Produces a much better
    # spatial spread than the previous spatial-grid approach and avoids
    # the clustering artefacts that appeared near voxel-grid boundaries.
    n_desired = min(int(n_source_points), n_att)
    if n_att <= n_desired:
        selected = att_coords
    else:
        from scipy.spatial import cKDTree as _cKDTree
        _rng = np.random.default_rng(42)
        coords_f = att_coords.astype(np.float64)
        # Random initialisation
        init_idx = _rng.choice(n_att, size=n_desired, replace=False)
        centroids = coords_f[init_idx].copy()
        # Fast k-medoids: 5 Lloyd iterations + medoid snap
        crown_tree = _cKDTree(coords_f)
        for _ki in range(5):
            tree = _cKDTree(centroids)
            _, assign = tree.query(coords_f, k=1)
            counts = np.bincount(assign, minlength=n_desired)
            sum_c = np.empty((n_desired, 3), dtype=np.float64)
            for _d in range(3):
                sum_c[:, _d] = np.bincount(
                    assign, weights=coords_f[:, _d], minlength=n_desired,
                )
            empty = counts == 0
            means = np.empty_like(centroids)
            means[~empty] = sum_c[~empty] / counts[~empty, np.newaxis]
            if empty.any():
                d_any, _ = tree.query(coords_f, k=1)
                n_emp = int(empty.sum())
                far_idx = np.argpartition(d_any, -n_emp)[-n_emp:]
                means[empty] = coords_f[far_idx]
            _, snap_idx = crown_tree.query(means, k=1)
            centroids = coords_f[snap_idx]
        selected = centroids.astype(np.int64)
    logger.info(
        "Tracing %d Dijkstra paths (k-medoids, random init) …",
        len(selected),
    )

    # ── Traceback each path ───────────────────────────────────────────────
    Zf, Yf, Xf = source_m.shape
    long_axis = int(np.argmax([Zf, Yf, Xf]))
    centre = np.array([Zf * sp_z, Yf * sp_y, Xf * sp_x], dtype=np.float32) * 0.5

    segs_start_list = []
    segs_end_list   = []
    scores_list     = []
    compr_list      = []

    report_every = max(1, len(selected) // 20)
    for idx, coord in enumerate(selected):
        if idx % report_every == 0:
            logger.info("  traced %d / %d paths …", idx, len(selected))
        try:
            path = mcp.traceback(coord)   # list of (z, y, x) tuples
        except Exception:  # noqa: BLE001
            continue
        if len(path) < 2:
            continue
        # Convert voxel indices to world coordinates.
        arr = np.array(path, dtype=np.float32)
        arr[:, 0] *= sp_z
        arr[:, 1] *= sp_y
        arr[:, 2] *= sp_x

        s_start = arr[:-1]          # (N_seg, 3)
        s_end   = arr[1:]

        # Score = angle from centripetal direction (same formula as arrows).
        seg_dir = s_end - s_start
        nrm = np.linalg.norm(seg_dir, axis=-1, keepdims=True)
        safe = (nrm > 1e-9).ravel()
        seg_dir_unit = np.zeros_like(seg_dir)
        seg_dir_unit[safe] = seg_dir[safe] / nrm[safe]

        mid = (s_start + s_end) * 0.5
        toward = centre[None, :] - mid
        toward[:, long_axis] = 0.0
        tn = np.linalg.norm(toward, axis=-1, keepdims=True)
        safe_t = (tn > 1e-9).ravel()
        toward_unit = np.zeros_like(toward)
        toward_unit[safe_t] = toward[safe_t] / tn[safe_t]

        dot = np.einsum("ij,ij->i", seg_dir_unit, toward_unit)
        score = (np.arccos(np.clip(dot, 0.0, 1.0)) / (0.5 * np.pi)
                 ).astype(np.float32)

        segs_start_list.append(s_start)
        segs_end_list.append(s_end)
        scores_list.append(score)

        # Convergence boost: sample compr_volume at each segment start.
        if compr_volume is not None:
            fs = max(1, int(fine_stride))
            Zc, Yc, Xc = compr_volume.shape
            # Convert world-coords back to fine-grid indices.
            fi = (arr[:-1] / np.array([sp_z * fs, sp_y * fs, sp_x * fs],
                                      dtype=np.float32)).astype(np.int32)
            fi[:, 0] = np.clip(fi[:, 0], 0, Zc - 1)
            fi[:, 1] = np.clip(fi[:, 1], 0, Yc - 1)
            fi[:, 2] = np.clip(fi[:, 2], 0, Xc - 1)
            compr_list.append(
                compr_volume[fi[:, 0], fi[:, 1], fi[:, 2]].astype(np.float32)
            )
        else:
            compr_list.append(np.zeros(len(s_start), dtype=np.float32))

    logger.info("Traceback complete — %d paths produced segments.",
                len(segs_start_list))
    if not segs_start_list:
        logger.warning("Crown Dijkstra: no segments produced.")
        return None

    segs_start = np.concatenate(segs_start_list, axis=0).astype(np.float32)
    segs_end   = np.concatenate(segs_end_list,   axis=0).astype(np.float32)
    scores_arr = np.concatenate(scores_list,      axis=0).astype(np.float32)
    compr_arr  = np.concatenate(compr_list,       axis=0).astype(np.float32)
    path_lengths = np.asarray(
        [len(s) for s in segs_start_list], dtype=np.int32,
    )

    # Boost from convergence (same log-scale mapping as arrow field).
    log_c = np.log1p(np.maximum(compr_arr, 0.0))
    if len(log_c) > 1:
        lo, hi = np.percentile(log_c, [70.0, 99.0])
        denom = max(float(hi - lo), 1e-6)
        boost = np.clip((log_c - lo) / denom, 0.0, 1.0).astype(np.float32)
    else:
        boost = np.zeros_like(log_c)

    dirs_arr = segs_end - segs_start
    nrm_arr  = np.linalg.norm(dirs_arr, axis=-1, keepdims=True)
    safe_arr = nrm_arr.ravel() > 1e-9
    dirs_unit = np.zeros_like(dirs_arr)
    dirs_unit[safe_arr] = dirs_arr[safe_arr] / nrm_arr[safe_arr]

    logger.info(
        "Crown Dijkstra: %d segments from %d paths.",
        len(segs_start), len(segs_start_list),
    )
    return {
        "segs_start": segs_start,
        "segs_end":   segs_end,
        "scores":     scores_arr,
        "boost":      boost,
        "path_lengths": path_lengths,
        # Backward-compat keys used by _make_arrows_actor fallback.
        "pts":    segs_start,
        "dirs":   dirs_unit.astype(np.float32),
        "stride": 1,
        "format": "crown_v1",
    }


def _build_tracks_polydata(data):
    """Convert a crown-tracks dict to a ``vtkPolyData`` with one
    ``vtkPolyLine`` per Dijkstra path.

    Uses fully-vectorised NumPy operations and ``numpy_to_vtk`` / ``
    numpy_to_vtkIdTypeArray`` to avoid any Python-level
    ``InsertNextCell`` / ``InsertNextPoint`` loops.

    The returned ``vtkPolyData`` has two *CellData* float arrays:
    ``"boost"`` and ``"scores"`` (per-path averages).
    """
    import numpy as np
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    segs_start   = np.asarray(data["segs_start"],   dtype=np.float32)
    segs_end     = np.asarray(data["segs_end"],     dtype=np.float32)
    path_lengths = np.asarray(data["path_lengths"], dtype=np.int64)
    n_segs       = len(segs_start)
    boost_np     = np.asarray(
        data.get("boost",  np.zeros(n_segs, dtype=np.float32)), dtype=np.float32
    )
    scores_np    = np.asarray(
        data.get("scores", np.zeros(n_segs, dtype=np.float32)), dtype=np.float32
    )

    n_paths     = len(path_lengths)
    path_starts = np.concatenate(([0], np.cumsum(path_lengths)[:-1])).astype(np.int64)

    # Each path of L segments contributes L+1 unique points:
    #   points 0..L-1 = segs_start[path_start..path_start+L-1]
    #   point  L      = segs_end[path_start+L-1]
    pt_counts  = (path_lengths + 1).astype(np.int64)         # (n_paths,)
    pt_offsets = np.concatenate(([0], np.cumsum(pt_counts)[:-1])).astype(np.int64)
    total_pts  = int(pt_counts.sum())

    all_pts = np.empty((total_pts, 3), dtype=np.float32)

    # Scatter segs_start into all_pts (vectorised via repeat / arange).
    path_of   = np.repeat(np.arange(n_paths, dtype=np.int64), path_lengths)
    local_idx = np.arange(n_segs, dtype=np.int64) - path_starts[path_of]
    dest_idx  = pt_offsets[path_of] + local_idx
    all_pts[dest_idx] = segs_start

    # Fill the last point of each path (= segs_end of the last segment).
    last_seg_idx  = (path_starts + path_lengths - 1).astype(np.int64)
    last_pt_idx   = (pt_offsets  + path_lengths).astype(np.int64)
    all_pts[last_pt_idx] = segs_end[last_seg_idx]

    # Build vtkPoints.
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetDataTypeToFloat()
    vtk_pts.SetData(numpy_to_vtk(all_pts, deep=True, array_type=vtk.VTK_FLOAT))

    # Build vtkCellArray using the VTK-9 SetData(offsets, connectivity) API.
    # offsets  : (n_paths+1,) – cumulative point count, starting at 0
    # connectivity: (total_pts,) – simply 0, 1, 2, …, total_pts-1
    #   (points are already laid out consecutively per path)
    cell_offsets = np.concatenate(([0], np.cumsum(pt_counts))).astype(np.int64)
    connectivity = np.arange(total_pts, dtype=np.int64)

    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetData(
        numpy_to_vtkIdTypeArray(cell_offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )

    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetLines(vtk_cells)

    # Per-path average boost / scores as CellData arrays.
    path_boost  = np.add.reduceat(boost_np,  path_starts.astype(np.intp)
                                  ) / path_lengths.astype(np.float32)
    path_scores = np.add.reduceat(scores_np, path_starts.astype(np.intp)
                                  ) / path_lengths.astype(np.float32)

    boost_vtk = numpy_to_vtk(path_boost.astype(np.float32), deep=True,
                              array_type=vtk.VTK_FLOAT)
    boost_vtk.SetName("boost")
    scores_vtk = numpy_to_vtk(path_scores.astype(np.float32), deep=True,
                               array_type=vtk.VTK_FLOAT)
    scores_vtk.SetName("scores")

    pd.GetCellData().AddArray(boost_vtk)
    pd.GetCellData().AddArray(scores_vtk)
    pd.GetCellData().SetActiveScalars("boost")

    logger.info(
        "Crown tracks polydata: %d polylines, %d points",
        pd.GetNumberOfCells(), pd.GetNumberOfPoints(),
    )
    return pd


def _write_tracks_vtp(data, vtp_path: Path) -> None:
    """Build a ``vtkPolyData`` of crown-track polylines and write it as a
    binary ``.vtp`` (XML VTK PolyData) file.  At load time the viewer can
    read this with ``vtkXMLPolyDataReader`` — a near-memcopy — instead of
    reconstructing all cells in Python.
    """
    import vtk

    pd = _build_tracks_polydata(data)
    vtp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(pd)
    writer.SetDataModeToBinary()
    writer.Write()
    logger.info(
        "Crown tracks VTP written to %s  (%d polylines, %d points)",
        vtp_path, pd.GetNumberOfCells(), pd.GetNumberOfPoints(),
    )


def _write_tracks_arrows_vtp(vtp_path: Path, arrows_vtp_path: Path,
                              n_arrows: int = 10) -> None:
    """Sample ``n_arrows`` evenly-spaced arrow glyphs per splined path and
    write the result as a binary ``.vtp`` so the viewer can skip the
    expensive Python for-loop at render time.

    The output ``vtkPolyData`` stores:
    * **Points** – glyph centres (``n_paths × n_arrows`` points).
    * **PointData "tangents"** – unit tangent vector at each glyph centre
      (``VTK_DOUBLE``).  Active vectors so ``vtkGlyph3D`` orients glyphs.
    * **PointData "t"** – arc-length fraction in [0, 1] (``VTK_FLOAT``).
      Active scalars so ``vtkGlyph3D`` colours by position along path.
    """
    import numpy as np
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    # ── Load polylines VTP ────────────────────────────────────────────────
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    pd_raw = reader.GetOutput()
    if pd_raw.GetNumberOfCells() == 0:
        logger.warning(
            "_write_tracks_arrows_vtp: polylines VTP has no cells: %s", vtp_path,
        )
        return

    # ── Spline interpolation ─────────────────────────────────────────────
    spline = vtk.vtkSplineFilter()
    spline.SetInputData(pd_raw)
    spline.SetSubdivideToSpecified()
    spline.SetNumberOfSubdivisions(3)
    spline.Update()
    pd_smooth = spline.GetOutput()
    logger.info(
        "Arrows VTP: splined %d pts → %d pts  (%d polylines)",
        pd_raw.GetNumberOfPoints(),
        pd_smooth.GetNumberOfPoints(),
        pd_smooth.GetNumberOfCells(),
    )

    # ── Sample n_arrows evenly-spaced positions per polyline ─────────────
    pts_np   = vtk_to_numpy(pd_smooth.GetPoints().GetData()).astype(np.float64)
    cell_arr = pd_smooth.GetLines()
    offsets  = vtk_to_numpy(cell_arr.GetOffsetsArray()).astype(np.int64)
    conn     = vtk_to_numpy(cell_arr.GetConnectivityArray()).astype(np.int64)
    n_paths  = len(offsets) - 1

    t_samples = (np.arange(n_arrows, dtype=np.float64) + 0.5) / n_arrows

    all_pos = np.empty((n_paths * n_arrows, 3), dtype=np.float64)
    all_tan = np.empty((n_paths * n_arrows, 3), dtype=np.float64)
    all_t   = np.tile(t_samples.astype(np.float32), n_paths)

    for i in range(n_paths):
        s0 = int(offsets[i])
        s1 = int(offsets[i + 1])
        pp = pts_np[conn[s0:s1]]
        L  = len(pp)
        out = slice(i * n_arrows, (i + 1) * n_arrows)
        if L < 2:
            all_pos[out] = pp[0] if L == 1 else 0.0
            all_tan[out] = [1.0, 0.0, 0.0]
            continue
        diffs    = np.diff(pp, axis=0)
        seg_lens = np.maximum(np.linalg.norm(diffs, axis=1), 1e-9)
        cum      = np.concatenate([[0.0], np.cumsum(seg_lens)])
        tot      = cum[-1]
        if tot < 1e-9:
            all_pos[out] = pp[0]
            all_tan[out] = [1.0, 0.0, 0.0]
            continue
        ss = t_samples * tot
        all_pos[out] = np.column_stack([
            np.interp(ss, cum, pp[:, 0]),
            np.interp(ss, cum, pp[:, 1]),
            np.interp(ss, cum, pp[:, 2]),
        ])
        sidx  = np.clip(np.searchsorted(cum[1:], ss, side="left"), 0, L - 2)
        tans  = diffs[sidx].copy()
        tnorm = np.maximum(np.linalg.norm(tans, axis=1, keepdims=True), 1e-9)
        all_tan[out] = tans / tnorm

    # ── Build output vtkPolyData ──────────────────────────────────────────
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetDataTypeToDouble()
    vtk_pts.SetData(numpy_to_vtk(all_pos, deep=True, array_type=vtk.VTK_DOUBLE))
    pd_g = vtk.vtkPolyData()
    pd_g.SetPoints(vtk_pts)

    tan_arr = numpy_to_vtk(all_tan, deep=True, array_type=vtk.VTK_DOUBLE)
    tan_arr.SetName("tangents")
    pd_g.GetPointData().AddArray(tan_arr)
    pd_g.GetPointData().SetActiveVectors("tangents")

    t_arr = numpy_to_vtk(all_t, deep=True, array_type=vtk.VTK_FLOAT)
    t_arr.SetName("t")
    pd_g.GetPointData().AddArray(t_arr)
    pd_g.GetPointData().SetActiveScalars("t")

    # ── Write ─────────────────────────────────────────────────────────────
    arrows_vtp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(arrows_vtp_path))
    writer.SetInputData(pd_g)
    writer.SetDataModeToBinary()
    writer.Write()
    logger.info(
        "Crown tracks arrows VTP written to %s  (%d glyphs, %d paths × %d)",
        arrows_vtp_path, len(all_pos), n_paths, n_arrows,
    )


def _load_or_build_crown_tracks(
    source_path: Path,
    target_path: Path,
    domain_path: Path,
    *,
    cache_path: Path | None,
    vtp_path: Path | None = None,
    arrows_vtp_path: Path | None = None,
    n_source_points: int = DEFAULT_N_SOURCE_POINTS,
    rebuild: bool = False,
    compr_volume=None,
    fine_stride: int = DEFAULT_FINE_STRIDE,
    display_stride: int = 1,
):
    """Load cached crown Dijkstra tracks (.npz), or build them from TIFFs.

    ``display_stride`` keeps only every Nth segment after load -- pure
    display sub-sampling, never affects the cache contents.

    Cache format detection: the new format contains a ``segs_end`` array.
    An old ThinLayer-style cache (no ``segs_end``) triggers a warning and
    a full rebuild.
    """
    import numpy as np

    if cache_path is not None and cache_path.exists() and not rebuild:
        try:
            logger.info("Loading cached crown tracks from %s …", cache_path)
            data = np.load(cache_path, allow_pickle=False)
            if "segs_end" not in data.files:
                logger.warning(
                    "Cache %s is in old ThinLayer format (no 'segs_end') "
                    "– rebuilding with Dijkstra.", cache_path,
                )
            else:
                n = len(data["segs_start"])
                boost = (data["boost"] if "boost" in data.files
                         else np.zeros(n, dtype=np.float32))
                ds = max(1, int(display_stride))
                segs_start = np.asarray(data["segs_start"])
                segs_end   = np.asarray(data["segs_end"])
                scores     = np.asarray(data["scores"])
                boost_arr  = np.asarray(boost).astype(np.float32)
                dirs_full  = (data["dirs"] if "dirs" in data.files
                              else np.zeros((n, 3), dtype=np.float32))
                dirs_arr   = np.asarray(dirs_full)

                # Determine per-path segment boundaries so that stride
                # keeps whole paths instead of cherry-picking every Nth
                # segment across all paths (which produced "snowflakes").
                if "path_lengths" in data.files:
                    path_lengths = np.asarray(data["path_lengths"],
                                              dtype=np.int64)
                else:
                    # Infer paths from continuity: a new path starts
                    # whenever segs_end[i-1] != segs_start[i].
                    if n <= 1:
                        path_lengths = np.array([n], dtype=np.int64)
                    else:
                        breaks = np.any(
                            segs_start[1:] != segs_end[:-1], axis=1,
                        )
                        # Path-start indices: 0 then every i where break.
                        starts = np.concatenate(
                            ([0], np.flatnonzero(breaks) + 1)
                        ).astype(np.int64)
                        ends = np.concatenate(
                            (starts[1:], [n])
                        ).astype(np.int64)
                        path_lengths = (ends - starts).astype(np.int64)
                    logger.info(
                        "Crown tracks: inferred %d paths from segment "
                        "continuity (cache had no 'path_lengths').",
                        len(path_lengths),
                    )

                if ds > 1:
                    # Keep every Nth path, all its segments.
                    n_paths = len(path_lengths)
                    starts = np.concatenate(
                        ([0], np.cumsum(path_lengths)[:-1])
                    ).astype(np.int64)
                    keep_paths = np.arange(0, n_paths, ds, dtype=np.int64)
                    idx_parts = [
                        np.arange(starts[p], starts[p] + path_lengths[p],
                                  dtype=np.int64)
                        for p in keep_paths
                    ]
                    idx = (np.concatenate(idx_parts) if idx_parts
                           else np.empty(0, dtype=np.int64))
                    segs_start = segs_start[idx]
                    segs_end   = segs_end[idx]
                    scores     = scores[idx]
                    boost_arr  = boost_arr[idx]
                    dirs_arr   = dirs_arr[idx]
                    path_lengths = path_lengths[keep_paths]
                    logger.info(
                        "Crown tracks sub-sampled: %d / %d paths "
                        "(%d / %d segments, stride=%d).",
                        len(keep_paths), n_paths,
                        len(idx), n, ds,
                    )
                return {
                    "segs_start": segs_start,
                    "segs_end":   segs_end,
                    "scores":     scores,
                    "boost":      boost_arr,
                    "path_lengths": path_lengths,
                    "pts":        segs_start,
                    "dirs":       dirs_arr,
                    "stride":     1,
                    "format":     "crown_v1",
                    # Forwarded to _make_tracks_actor_curves so the VTP fast
                    # path can apply the same stride at the polyline level.
                    "_display_stride": ds,
                    # Pass the .vtp path so _make_tracks_actor_curves can
                    # load it directly at render time (if it exists).
                    "vtp_path":         vtp_path,
                    # Pre-computed arrow glyphs VTP (positions, tangents, t).
                    # When present, skips the Python for-loop at render time.
                    "arrows_vtp_path":  arrows_vtp_path,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read crown tracks cache %s: %s "
                           "– rebuilding.", cache_path, exc)

    for p, name in (
        (source_path, "Source_crown"),
        (target_path, "Target_crown"),
        (domain_path, "Source_Target_Possible_Paths"),
    ):
        if not p.exists():
            logger.info("%s TIFF not found at %s – crown tracks disabled.",
                        name, p)
            return None

    source_vol = load_float_volume(source_path)
    target_vol = load_float_volume(target_path)
    domain_vol = load_float_volume(domain_path)

    data = _build_crown_dijkstra_tracks(
        source_vol, target_vol, domain_vol,
        spacing=DEFAULT_SPACING,
        n_source_points=n_source_points,
        compr_volume=compr_volume,
        fine_stride=fine_stride,
    )

    if data is not None and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                segs_start=data["segs_start"],
                segs_end=data["segs_end"],
                scores=data["scores"],
                boost=data["boost"],
                dirs=data["dirs"],
                stride=np.int32(data["stride"]),
                format=np.bytes_(data["format"]),
            )
            logger.info(
                "Crown tracks cached to %s  (%d segments)",
                cache_path, len(data["segs_start"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache crown tracks %s: %s",
                           cache_path, exc)
    return data


def _load_or_build_dual_arrows(
    water_harmonic_path: Path,
    wind_field_path: Path,
    *,
    dual_water_cache: Path | None = None,
    dual_air_cache:  Path | None = None,
    fine_stride: int = 4,
    separation_vox: int = 2,
    rebuild: bool = False,
) -> dict | None:
    """Load cached dual-arrow fields (water + air), or build them on the fly.

    Returns a dict ``{"water": arrows_data, "air": arrows_data}`` or ``None``
    if either harmonic field is unavailable.

    The individual ``arrows_data`` dicts have the same structure as
    :func:`_build_arrow_field` (``pts``, ``dirs``, ``scores``, ``boost``,
    ``stride``).
    """
    import numpy as np

    def _load_cache(path: Path) -> dict | None:
        try:
            z = np.load(str(path), allow_pickle=False)
            n = len(z["pts"])
            return {
                "pts":    z["pts"].astype(np.float32),
                "dirs":   z["dirs"].astype(np.float32),
                "scores": z["scores"].astype(np.float32),
                "boost":  z["boost"].astype(np.float32),
                "stride": int(z["stride"][0]) if z["stride"].ndim > 0
                          else int(z["stride"]),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load dual-arrows cache %s: %s", path, exc)
            return None

    # Fast path: both caches present.
    if (dual_water_cache is not None and dual_water_cache.exists()
            and dual_air_cache is not None and dual_air_cache.exists()
            and not rebuild):
        w = _load_cache(dual_water_cache)
        a = _load_cache(dual_air_cache)
        if w is not None and a is not None:
            logger.info(
                "Loaded dual arrows: water=%d  air=%d",
                len(w["pts"]), len(a["pts"]),
            )
            return {"water": w, "air": a}

    # Slow path: build from harmonic fields.
    if not water_harmonic_path.exists():
        logger.info(
            "Water harmonic cache not found (%s) — dual arrows disabled.  "
            "Run marvel-water-harmonic-build first.", water_harmonic_path,
        )
        return None
    if not wind_field_path.exists():
        logger.info(
            "Wind field cache not found (%s) — dual arrows disabled.  "
            "Run marvel-wind-field-build first.", wind_field_path,
        )
        return None

    from marvel_view.preprocessing.water_harmonic import (
        load_water_harmonic_field, build_dual_arrows_filtered,
    )
    from marvel_view.preprocessing.wind_field import load_wind_field

    logger.info("Building dual arrow fields (water + air) …")
    water = load_water_harmonic_field(water_harmonic_path)
    wind  = load_wind_field(wind_field_path)

    w_arrows, a_arrows = build_dual_arrows_filtered(
        water_vec=water["vec"],
        water_area=water["area"],
        air_vec=wind["vec"],
        air_area=wind["area"],
        fine_stride=fine_stride,
        separation_vox=separation_vox,
        spacing=DEFAULT_SPACING,
    )

    # Cache to disk if paths provided.
    for cache_path, arrows in ((dual_water_cache, w_arrows),
                                (dual_air_cache,   a_arrows)):
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(cache_path),
                    pts=arrows["pts"],
                    dirs=arrows["dirs"],
                    scores=arrows["scores"],
                    boost=arrows["boost"],
                    stride=np.array([arrows["stride"]], dtype=np.int32),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not cache dual arrows %s: %s",
                               cache_path, exc)

    return {"water": w_arrows, "air": a_arrows}


def _load_or_build_overlay_mesh(
    input_path: Path,
    *,
    cache_path: Path | None = None,
    level: float = DEFAULT_OVERLAY_LEVEL,
    rebuild: bool = False,
):
    """Return a translucent ghost mesh extracted at ``level`` from an 8-bit TIFF.

    Tries (in order): cached VTK on disk -> marching cubes from ``input_path``.
    Returns ``None`` if neither source is available.
    """
    import vedo

    if cache_path is not None and cache_path.exists() and not rebuild:
        try:
            logger.info("Loading cached overlay mesh from %s …", cache_path)
            return vedo.Mesh(str(cache_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load overlay cache %s: %s – rebuilding.",
                           cache_path, exc)

    if not input_path.exists():
        logger.info("Overlay input TIFF not found at %s – overlay disabled.",
                    input_path)
        return None

    logger.info("Building overlay isosurface from %s (iso=%.2f) …",
                input_path, level)
    volume = load_float_volume(input_path)
    mesh = mask_to_mesh(volume, level=level, smooth_iter=0,
                        spacing=DEFAULT_SPACING)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.write(str(cache_path))
            logger.info("Overlay mesh cached to %s", cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write overlay cache %s: %s",
                           cache_path, exc)
    return mesh


def _load_or_build_membranes(
    vtp_path: Path,
    meta_path: Path,
):
    """Return ``(polydata, meta)`` for the water-membranes animation.

    The cache is *load-only* — it is built by
    ``marvel-water-conductance-build-meshes``.  Returns ``(None, None)``
    when either file is missing.
    """
    import vtk

    if not vtp_path.exists() or not meta_path.exists():
        logger.info(
            "Water-membranes cache not found (vtp=%s, meta=%s) — disabled.",
            vtp_path, meta_path,
        )
        return None, None
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse membranes meta %s: %s", meta_path, exc)
        return None, None
    try:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_path))
        reader.Update()
        pd = reader.GetOutput()
        if pd is None or pd.GetNumberOfCells() == 0:
            logger.warning("Membranes VTP %s contains no cells.", vtp_path)
            return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load membranes VTP %s: %s", vtp_path, exc)
        return None, None
    logger.info(
        "Water membranes loaded: %d cells, n_steps=%d, n_columns=%d",
        pd.GetNumberOfCells(),
        int(meta.get("n_steps", 0)),
        int(meta.get("n_columns", 0)),
    )
    return pd, meta


def _load_lames(vtp_path: Path, meta_path: Path):
    """Return ``(polydata, meta)`` for the V2 lames animation, or
    ``(None, None)`` if either file is missing."""
    import vtk

    if not vtp_path.exists() or not meta_path.exists():
        logger.info(
            "Water-lames (V2) cache not found (vtp=%s, meta=%s) — disabled.",
            vtp_path, meta_path,
        )
        return None, None
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse lames meta %s: %s", meta_path, exc)
        return None, None
    try:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_path))
        reader.Update()
        pd = reader.GetOutput()
        if pd is None or pd.GetNumberOfCells() == 0:
            logger.warning("Lames VTP %s contains no cells.", vtp_path)
            return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load lames VTP %s: %s", vtp_path, exc)
        return None, None
    logger.info(
        "Water lames (V2) loaded: %d cells, n_steps=%d, n_columns=%d",
        pd.GetNumberOfCells(),
        int(meta.get("n_steps", 0)),
        int(meta.get("n_columns", 0)),
    )
    return pd, meta


def _build_lames_step_polydatas(packed_pd, n_steps: int):
    """Split a packed PolyData (cells **already sorted** by ``step_id``)
    into per-step ``vtkPolyData`` objects using numpy slices — O(N) total,
    no ``vtkThreshold`` scans.

    Returns ``(step_pds, actor)``:
      - ``step_pds``: list[vtkPolyData | None] of length ``n_steps``
      - ``actor``:    single pre-configured ``vtkActor`` (hidden, no input)
    """
    import vtk
    from vtk.util import numpy_support as nps
    import numpy as np

    # ── Pull cell-data + connectivity to numpy ONCE ─────────────────
    step_arr = nps.vtk_to_numpy(
        packed_pd.GetCellData().GetArray("step_id")
    ).astype(np.int32)                                   # (N_tris,)
    rgb_arr = nps.vtk_to_numpy(
        packed_pd.GetCellData().GetArray("rgb")
    )                                                     # (N_tris, 3) uint8
    # Legacy flat connectivity [3,p0,p1,p2, 3,p3,p4,p5, ...]
    conn = nps.vtk_to_numpy(
        packed_pd.GetPolys().GetData()
    ).reshape(-1, 4)                                     # (N_tris, 4)
    pts_vtk = packed_pd.GetPoints()                       # shared, no copy

    # ── Step boundaries (data is pre-sorted by step_id) ─────────────
    lo_b = np.searchsorted(step_arr, np.arange(n_steps),       side="left")
    hi_b = np.searchsorted(step_arr, np.arange(1, n_steps + 1), side="left")

    step_pds: list = [None] * int(n_steps)
    for s in range(int(n_steps)):
        a, b = int(lo_b[s]), int(hi_b[s])
        if a >= b:
            continue
        n_s      = b - a
        sub_conn = np.ascontiguousarray(conn[a:b].ravel(), dtype=np.int64)
        sub_rgb  = np.ascontiguousarray(rgb_arr[a:b])

        cells = vtk.vtkCellArray()
        cells.SetCells(
            n_s,
            nps.numpy_to_vtkIdTypeArray(sub_conn, deep=True),
        )
        step_pd = vtk.vtkPolyData()
        step_pd.SetPoints(pts_vtk)                       # shared reference
        step_pd.SetPolys(cells)

        a_rgb = nps.numpy_to_vtk(
            sub_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        a_rgb.SetName("rgb")
        a_rgb.SetNumberOfComponents(3)
        step_pd.GetCellData().AddArray(a_rgb)
        step_pds[s] = step_pd

    n_built = sum(1 for p in step_pds if p is not None)
    logger.info("Lames: pre-split %d / %d steps (numpy, single actor)",
                n_built, n_steps)

    # ── Single actor — geometry swapped at each tick ────────────────
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray("rgb")
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetOpacity(float(DEFAULT_LAMES_ALPHA))
    prop.SetAmbient(0.85)
    prop.SetDiffuse(0.10)
    prop.SetSpecular(0.0)
    prop.BackfaceCullingOff()
    prop.FrontfaceCullingOff()
    actor.SetVisibility(False)

    return step_pds, actor
