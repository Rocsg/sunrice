"""
Water-descent "lames" (V2) precomputation.

Parallel to ``water_membranes.py`` (V1) — kept side-by-side so the old
pipeline / cache / renderer keep working unchanged.

Concept (V2)
------------

Instead of thin iso-surfaces (V1), V2 carves the conducting columns
into *thick* iso-shells whose thickness adapts to the locally-shrinking
cross-section as water descends toward the central axis: shells are
narrow near the periphery (lots of voxels per distance bin) and thicker
deeper inside (fewer voxels).  Each shell is rendered as a single
mesh; per-column phase-shifted Tstart values stagger the columns'
descents so the resulting movie shows water "trickling down" each
column independently.

Pipeline
--------

1. Load 4 input TIFFs (geoddist, bg-dist, object mask, crown mask).
   Cleaning:
     * NaN/inf → 0,
     * < 0     → 0,
     * > max image dim → 0  (sanity).

2. Build the iso-distance grid
   ``levels = arange(d_min+eps, d_max, level_step)`` with at most
   ``N_LEVELS_MAX`` levels (= NT).

3. K-means in 3-D voxel space restricted to the crown, with the
   "medoid-included" constraint: at every Lloyd iteration the floating
   centroid is replaced by the *closest crown voxel* — this guarantees
   the seed lives on the crown.  Seeds = ``n_seeds`` cluster medoids.

4. Geodesic watershed inside the object, weighted by ``geoddist``,
   starting from those medoid seeds → ``label_vol`` (1..n_columns).

5. Bounding boxes of every label in one pass
   (``scipy.ndimage.find_objects``) with a 1-voxel padding.

6. Per-distance-bin voxel histogram inside the object → "shrinkage"
   series ``count[t]``.  Normalised by the mean of the first 5 strictly
   positive bins → ``alpha[t]`` (== 1 baseline near the periphery).
   ``alpha[t] = 1`` where ``count[t] == 0``.

7. For every label (in a thread pool) and every level ``t`` (sequential
   inside a label):
     a. Extract the per-label crop of ``geoddist`` and ``label_vol``.
     b. Clamp thresholds  ``lo = max(0.1, d - 1/α(t-1))``,
                          ``hi = max(lo + 0.1, d + 1/α(t+1))``.
     c. Build ``F = (D - lo) * (hi - D)`` (positive inside the shell).
     d. Marching cubes at ``level = 0 + eps`` on ``F`` masked by the
        current label.
     e. Decimate to ``target_tris_per_shell`` triangles (default 100).
     f. Light Laplacian smoothing (5 iterations) for niceness.
     g. Cache ``(verts, faces)`` to ``iso_cache_dir/L{ll:04d}/T{tt:04d}.npz``
        + record ``t0_per_col[L]``, ``tlast_per_col[L]``.

8. Phase-bin assignment: ``Nstep = phase_factor * NT``, ``tstart[L]``
   drawn uniformly in ``[0, Nstep)``; for column ``L``, level ``t``
   maps to ``step_id = (tstart[L] + (t - t0[L])) % Nstep``.

9. Glow envelope (per cell, baked into RGB): subtle additive brightness
   that fades in over ``fade_frames`` steps after the column's first
   appearance and fades back in over ``fade_frames`` steps before its
   last step.  Same base cyan-blue colour as V1, slightly brighter
   peaks.

10. Pack everything into ONE ``vtkPolyData`` with per-cell arrays
    ``step_id`` / ``column_id`` / ``time_id`` / ``rgb`` (uint8×3,
    glow already baked) — sorted by ``step_id``.  The viewer splits
    this packed PolyData into ``Nstep`` per-step PolyDatas at load
    time and toggles their visibility with ``SetVisibility`` (no
    per-frame topology updates → no flicker).

All heavy imports (skimage, scipy, vtk) are deferred to ``build()`` so
this module stays cheap to import.
"""
from __future__ import annotations

import colorsys
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── defaults ─────────────────────────────────────────────────────────────

DEFAULT_N_SEEDS:               int   = 1000
DEFAULT_PHASE_FACTOR:          int   = 5
DEFAULT_N_LEVELS_MAX:          int   = 200
DEFAULT_LAME_WIDTH:            float = 15.0
DEFAULT_TARGET_TRIS_PER_SHELL: int   = 500
DEFAULT_SMOOTH_ITER:           int   = 5
DEFAULT_FADE_FRAMES:           int   = 5
DEFAULT_KMEANS_ITERS:          int   = 50
DEFAULT_SEED:                  int   = 1234

# Base RGB (cyan-blue) and glow accents.
_BASE_RGB     = np.array((140, 195, 255), dtype=np.float32)
# Peak glow blends this fraction toward (255, 255, 255).
_GLOW_PEAK_T  = 0.45

# Server marker (reused from V1) for adaptive worker count.
_SERVER_MARKER = Path("/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5")


def _resolve_n_workers(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    env = os.environ.get("MARVEL_MAX_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, int((os.cpu_count() or 4) * 2 // 3))


# ── I/O helpers ──────────────────────────────────────────────────────────


def _load_uint8(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(str(path))
    arr = arr.squeeze()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D mask in {path}, got shape {arr.shape}")
    return (arr > 0).astype(np.uint8)


def _load_float32(path: Path) -> np.ndarray:
    import tifffile
    arr = tifffile.imread(str(path))
    arr = arr.squeeze()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D volume in {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _clean_distance(vol: np.ndarray, *, name: str) -> np.ndarray:
    """Replace NaN/inf → 0, negatives → 0, > max(shape) → 0."""
    max_ok = float(max(vol.shape))
    n_nan  = int(np.sum(~np.isfinite(vol)))
    n_neg  = int(np.sum(np.nan_to_num(vol, nan=0.0) < 0.0))
    n_huge = int(np.sum(np.nan_to_num(vol, nan=0.0) > max_ok))
    if n_nan or n_neg or n_huge:
        logger.info(
            "Cleaning %s: NaN/inf=%d  neg=%d  > %.0f=%d",
            name, n_nan, n_neg, max_ok, n_huge,
        )
    out = np.where(np.isfinite(vol), vol, 0.0).astype(np.float32, copy=False)
    out = np.where(out < 0.0, 0.0, out)
    out = np.where(out > max_ok, 0.0, out)
    return out


# ── K-means with "medoid included" constraint ────────────────────────────


def _kmeans_medoid_crown(
    crown_mask: np.ndarray,
    n_seeds:    int,
    *,
    n_iter:     int = DEFAULT_KMEANS_ITERS,
    rng:        np.random.Generator,
) -> np.ndarray:
    """K-means on crown voxel 3-D coords; centroids snapped to the closest
    crown voxel at every iteration (medoid-included)."""
    from scipy.spatial import cKDTree

    coords = np.argwhere(crown_mask > 0).astype(np.float64)
    n_avail = coords.shape[0]
    if n_avail == 0:
        raise ValueError("Crown mask has no positive voxels.")
    n_seeds = int(min(n_seeds, n_avail))
    logger.info("K-means(medoid) on %d crown voxels → %d clusters (%d iters)",
                n_avail, n_seeds, n_iter)

    # ── Init via farthest-point sampling for a well-spread start ────
    first = int(rng.integers(0, n_avail))
    picked = [first]
    diff = coords - coords[first]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    for _ in range(1, n_seeds):
        idx = int(np.argmax(dist2))
        picked.append(idx)
        diff = coords - coords[idx]
        np.minimum(dist2, np.einsum("ij,ij->i", diff, diff), out=dist2)
    centroids = coords[np.asarray(picked, dtype=np.int64)]  # (K, 3)

    # ── Lloyd with medoid snap ──────────────────────────────────────
    crown_tree = cKDTree(coords)
    for it in range(int(n_iter)):
        tree = cKDTree(centroids)
        _, assign = tree.query(coords, k=1)      # O(N log K)

        # Vectorised cluster means via bincount — O(N + K) instead of O(N*K)
        counts  = np.bincount(assign, minlength=n_seeds)          # (K,)
        sum_zyx = np.empty((n_seeds, 3), dtype=np.float64)
        for dim in range(3):
            sum_zyx[:, dim] = np.bincount(
                assign, weights=coords[:, dim], minlength=n_seeds)
        empty = counts == 0
        means = np.empty_like(centroids)
        means[~empty] = sum_zyx[~empty] / counts[~empty, np.newaxis]

        # Empty clusters: reseed at the voxels farthest from any centroid.
        if empty.any():
            d_any, _ = tree.query(coords, k=1)
            n_empty = int(empty.sum())
            far_idx = np.argpartition(d_any, -n_empty)[-n_empty:]
            means[empty] = coords[far_idx]

        # Batch medoid snap: one KD-tree call for all K centroids — O(K log N)
        _, snap_idx  = crown_tree.query(means, k=1)
        new_centroids = coords[snap_idx]

        moved = float(np.linalg.norm(centroids - new_centroids, axis=1).sum())
        centroids = new_centroids
        logger.info("  K-means iter %2d/%d  total drift=%.1f",
                    it + 1, n_iter, moved)
        if moved < 1e-6:
            break

    return centroids.astype(np.int64)


def _geodesic_watershed(
    object_mask: np.ndarray,
    geoddist:    np.ndarray,
    seeds_zyx:   np.ndarray,
) -> np.ndarray:
    from skimage.segmentation import watershed
    markers = np.zeros(object_mask.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(seeds_zyx, start=1):
        z = int(np.clip(z, 0, object_mask.shape[0] - 1))
        y = int(np.clip(y, 0, object_mask.shape[1] - 1))
        x = int(np.clip(x, 0, object_mask.shape[2] - 1))
        markers[z, y, x] = i
    img = np.clip(geoddist, 0.0, None).astype(np.float32, copy=False)
    logger.info("Geodesic watershed: object=%d vox, seeds=%d",
                int(object_mask.sum()), int(markers.max()))
    t = time.perf_counter()
    labels = watershed(img, markers=markers, mask=object_mask.astype(bool),
                       compactness=0.0)
    logger.info("Watershed done in %.1f s (max label=%d)",
                time.perf_counter() - t, int(labels.max()))
    return labels.astype(np.int32)


# ── Per-distance shrinkage / thickening factor ───────────────────────────


def _thickening_factor(
    geoddist: np.ndarray,
    object_mask: np.ndarray,
    levels:   np.ndarray,
    level_step: float,
) -> np.ndarray:
    """Return ``alpha[t]`` of shape ``(NT,)`` such that ``1/alpha[t]`` is
    the local lame-thickness *multiplier*.

    alpha[t] = count[t] / mean(count over first 5 strictly-positive bins)
    alpha[t] = 1.0 where count[t] == 0  (= baseline thickness).
    """
    NT = int(levels.size)
    inside = object_mask.astype(bool)
    d = geoddist[inside]
    d = d[d > 0]
    # bin edges from levels[i] - 0.5*step to levels[i] + 0.5*step
    edges = np.concatenate([
        levels - 0.5 * float(level_step),
        levels[-1:] + 0.5 * float(level_step),
    ])
    count, _ = np.histogram(d, bins=edges)
    count = count.astype(np.float64)

    # Baseline = mean over the first 5 strictly-positive bins.
    pos = np.where(count > 0)[0]
    if pos.size == 0:
        logger.warning("Thickening: no positive bins; falling back to alpha=1.")
        return np.ones(NT, dtype=np.float32)
    base = float(count[pos[:5]].mean())
    alpha = np.where(count > 0, count / max(base, 1.0), 1.0).astype(np.float32)
    alpha = np.clip(alpha, 1e-3, 1e3)  # safety vs. div-by-zero downstream
    logger.info("Thickening factor: base=%.0f vox/bin  α∈[%.3f, %.3f]  pos_bins=%d/%d",
                base, float(alpha.min()), float(alpha.max()), int(pos.size), NT)
    return alpha


# ── ISO-cache layout (per label / per level NPZ) ─────────────────────────


def _shell_cache_path(cache_dir: Path, label: int, ti: int) -> Path:
    return cache_dir / f"L{int(label):04d}" / f"T{int(ti):04d}.npz"


def _shell_cache_save(cache_dir: Path, label: int, ti: int,
                      verts: np.ndarray, faces: np.ndarray) -> None:
    p = _shell_cache_path(cache_dir, label, ti)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p,
                        verts=verts.astype(np.float32),
                        faces=faces.astype(np.int64))


def _shell_cache_load(cache_dir: Path, label: int, ti: int):
    p = _shell_cache_path(cache_dir, label, ti)
    if not p.exists():
        return None, None
    with np.load(p) as z:
        return z["verts"].astype(np.float32), z["faces"].astype(np.int64)


def _cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "lames_cache_meta.json"


def _read_cache_meta(cache_dir: Path) -> dict:
    p = _cache_meta_path(cache_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_cache_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_meta_path(cache_dir).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _cache_is_valid(cache_dir: Path, *, n_columns: int, n_levels: int,
                    level_step: float, target_tris: int,
                    lame_width: float) -> bool:
    m = _read_cache_meta(cache_dir)
    if not m:
        return False
    return (
        m.get("n_columns", -1) == int(n_columns)
        and m.get("n_levels", -1)  == int(n_levels)
        and abs(m.get("level_step", -1.0) - float(level_step))  < 1e-6
        and abs(m.get("lame_width", -1.0) - float(lame_width))  < 1e-6
        and m.get("target_tris_per_shell", -1) == int(target_tris)
    )


# ── Marching cubes + decimation + smoothing for a single shell ──────────


def _extract_shell_mesh(
    F_sub:        np.ndarray,
    label_sub:    np.ndarray,
    label:        int,
    target_tris:  int,
    smooth_iter:  int,
):
    """Marching cubes at level=ε on ``F_sub`` masked to ``label_sub == label``,
    then decimate + Laplacian smooth.  Returns (verts, faces) in **sub-volume
    voxel coords** (caller applies the bounding-box offset)."""
    from skimage.measure import marching_cubes

    # Mask to the current label: outside the label, force F < 0 so the
    # surface stops at the label boundary.
    F = np.where(label_sub == label, F_sub, -1.0).astype(np.float32)

    # Quick gate: surface only exists if F crosses 0 somewhere.
    if not (F.max() > 1e-3 and F.min() < 0.0):
        return None, None
    try:
        verts, faces, _n, _v = marching_cubes(
            F, level=1e-3, step_size=1, allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        logger.debug("MC failed (label=%d): %s", label, exc)
        return None, None
    if verts.size == 0 or faces.size == 0:
        return None, None

    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)
    if faces.shape[0] > target_tris > 0:
        verts, faces = _decimate_and_smooth(verts, faces, target_tris, smooth_iter)
    elif smooth_iter > 0:
        verts, faces = _laplace_smooth(verts, faces, smooth_iter)
    return verts, faces


def _polydata_from_vf(verts: np.ndarray, faces: np.ndarray):
    import vtk
    from vtk.util import numpy_support as nps  # type: ignore
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float32),
                                  deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    n_tri = faces.shape[0]
    conn = np.empty((n_tri, 4), dtype=np.int64)
    conn[:, 0] = 3
    conn[:, 1:] = faces
    cells = vtk.vtkCellArray()
    id_arr = nps.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn.ravel()),
                                         deep=True)
    cells.SetCells(n_tri, id_arr)
    pd.SetPolys(cells)
    return pd


def _vf_from_polydata(pd) -> tuple[np.ndarray, np.ndarray]:
    from vtk.util import numpy_support as nps  # type: ignore
    verts = nps.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float32)
    n_cells = pd.GetNumberOfCells()
    if n_cells == 0:
        return verts, np.empty((0, 3), dtype=np.int64)
    raw = nps.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)
    return verts, raw[:, 1:].astype(np.int64)


def _decimate_and_smooth(verts, faces, target_tris, smooth_iter):
    import vtk
    n_tri = faces.shape[0]
    fraction = 1.0 - (float(target_tris) / float(n_tri))
    fraction = float(np.clip(fraction, 0.0, 0.999))
    pd = _polydata_from_vf(verts, faces)
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(pd)
    dec.SetTargetReduction(fraction)
    dec.SetVolumePreservation(False)
    dec.Update()
    out = dec.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        return verts, faces
    v, f = _vf_from_polydata(out)
    if smooth_iter > 0 and f.shape[0] > 0:
        v, f = _laplace_smooth(v, f, smooth_iter)
    return v, f


def _laplace_smooth(verts, faces, n_iter):
    import vtk
    pd = _polydata_from_vf(verts, faces)
    smo = vtk.vtkSmoothPolyDataFilter()
    smo.SetInputData(pd)
    smo.SetNumberOfIterations(int(n_iter))
    smo.SetRelaxationFactor(0.3)
    smo.FeatureEdgeSmoothingOff()
    smo.BoundarySmoothingOn()
    smo.Update()
    out = smo.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        return verts, faces
    return _vf_from_polydata(out)


# ── Glow envelope ────────────────────────────────────────────────────────


def _glow_curve(s_local: np.ndarray, lifetime: int, fade: int) -> np.ndarray:
    """Per-cell glow envelope in [0, 1].

    Peaks at lifetime onset and offset, ramps linearly over ``fade``
    steps.  ``s_local`` is the per-cell step offset within the column
    (0..lifetime)."""
    fade = max(1, int(fade))
    onset  = np.clip(1.0 - s_local / fade, 0.0, 1.0)
    offset = np.clip(1.0 - (lifetime - s_local) / fade, 0.0, 1.0)
    return np.maximum(onset, offset)


# ── Debug helper: per-column diagnostic TIFFs ────────────────────────────


def _debug_lames(
    *,
    geoddist:      "np.ndarray",
    label_vol:     "np.ndarray",
    active_labels: list,
    levels:        "np.ndarray",
    alpha:         "np.ndarray",
    lame_width:    float,
    output_dir:    "Path",
    n_debug:       int = 3,
    rng:           "np.random.Generator",
) -> None:
    """Generate diagnostic TIFFs for *n_debug* random columns.

    For each selected column:
    * ``imagedistanceColonne<N>.tif`` – cropped distance map (0-255 uint8),
      only voxels belonging to that column; others set to 0.
    * ``colonne<N>/step_NNNN.tif``   – binary shell mask at each iso-level
      (0 or 255 uint8), only non-empty steps are written.
    """
    import tifffile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(int(n_debug), len(active_labels))
    if n == 0:
        logger.warning("Debug: no active labels — nothing to generate.")
        return

    chosen_idx = rng.choice(len(active_labels), size=n, replace=False)
    selected   = [active_labels[int(i)] for i in chosen_idx]
    NT         = int(levels.size)

    for col_num, (label, bbox) in enumerate(selected, start=1):
        pad = 1
        z0 = max(0, bbox[0].start - pad); z1 = min(geoddist.shape[0], bbox[0].stop + pad)
        y0 = max(0, bbox[1].start - pad); y1 = min(geoddist.shape[1], bbox[1].stop + pad)
        x0 = max(0, bbox[2].start - pad); x1 = min(geoddist.shape[2], bbox[2].stop + pad)
        D_sub = geoddist [z0:z1, y0:y1, x0:x1]
        L_sub = label_vol[z0:z1, y0:y1, x0:x1]

        # ── distance map (float32, real values; 0 outside the column) ─
        d_col = np.where(L_sub == label, D_sub, 0.0).astype(np.float32)
        dist_path = output_dir / f"imagedistanceColonne{col_num}.tif"
        tifffile.imwrite(str(dist_path), d_col)
        logger.info("debug colonne%d (L=%d): dist map %s float32 → %s",
                    col_num, label, d_col.shape, dist_path)

        # ── per-column distance range (same logic as _process_label) ──
        d_col_vals = D_sub[L_sub == label]
        d_col_vals = d_col_vals[(d_col_vals > 0) & np.isfinite(d_col_vals)]
        if d_col_vals.size == 0:
            logger.warning("debug colonne%d (L=%d): no positive distances",
                           col_num, label)
            continue
        d_col_min = float(d_col_vals.min())
        d_col_max = float(d_col_vals.max())
        ti_lo = d_col_min + lame_width
        ti_hi = d_col_max - lame_width

        # monotonic lo/hi tracking
        eps = float(levels[1] - levels[0]) * 0.5 if NT >= 2 else 0.1
        lo_prev = -np.inf
        hi_prev = -np.inf

        # ── per-step binary masks + scalar field F ────────────────
        col_dir  = output_dir / f"colonne{col_num}"
        dist_dir = output_dir / f"distancecolonne{col_num}"
        col_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for ti in range(NT):
            d   = float(levels[ti])
            if d < ti_lo or d > ti_hi:
                continue
            a_m = float(alpha[max(0, ti - 1)])
            a_p = float(alpha[min(NT - 1, ti + 1)])
            lo  = max(0.1, d - lame_width / max(1e-3, a_m))
            hi  = max(lo + 0.1, d + lame_width / max(1e-3, a_p))
            lo  = max(lo, lo_prev + eps)
            hi  = max(hi, hi_prev + eps, lo + 0.1)
            lo_prev = lo
            hi_prev = hi
            # scalar field F passed to marching cubes (iso = 0)
            F = ((D_sub - lo) * (hi - D_sub)).astype(np.float32)
            # restrict to the column (outside → negative so iso=0 won't catch it)
            F = np.where(L_sub == label, F, -1.0).astype(np.float32)
            tifffile.imwrite(str(dist_dir / f"step_{ti:04d}.tif"), F)
            mask = (F > 0).astype(np.uint8) * 255
            if mask.max() == 0:
                continue
            tifffile.imwrite(str(col_dir / f"step_{ti:04d}.tif"), mask)
            n_saved += 1
        logger.info("debug colonne%d (L=%d): d=[%.2f, %.2f]  %d step TIFFs → %s  (F field → %s)",
                    col_num, label, d_col_min, d_col_max, n_saved, col_dir, dist_dir)


# ── Per-label worker (a level loop, sequential within one label) ─────────


def _process_label(
    *,
    label:        int,
    bbox,                       # tuple of slices (sub-volume location)
    geoddist:     np.ndarray,
    label_vol:    np.ndarray,
    levels:       np.ndarray,
    alpha:        np.ndarray,
    lame_width:   float,
    iso_cache_dir: Path,
    target_tris:  int,
    smooth_iter:  int,
):
    """Process one column label: marching cubes per level on the
    bounding-box crop, save NPZ caches.  Returns
    ``(label, t0, tlast, n_levels_with_geometry)``."""
    pad = 1
    z0 = max(0, bbox[0].start - pad); z1 = min(geoddist.shape[0], bbox[0].stop + pad)
    y0 = max(0, bbox[1].start - pad); y1 = min(geoddist.shape[1], bbox[1].stop + pad)
    x0 = max(0, bbox[2].start - pad); x1 = min(geoddist.shape[2], bbox[2].stop + pad)
    offset = np.array((z0, y0, x0), dtype=np.float32)

    D_sub  = geoddist [z0:z1, y0:y1, x0:x1]
    L_sub  = label_vol[z0:z1, y0:y1, x0:x1]

    # ── column-local distance range ───────────────────────────────────
    NT = int(levels.size)
    d_col_vals = D_sub[L_sub == label]
    d_col_vals = d_col_vals[(d_col_vals > 0) & np.isfinite(d_col_vals)]
    if d_col_vals.size == 0:
        return int(label), int(NT), -1, 0, 0.0, 0.0, 0
    d_col_min = float(d_col_vals.min())
    d_col_max = float(d_col_vals.max())
    ti_lo = d_col_min + lame_width   # skip levels below this
    ti_hi = d_col_max - lame_width   # skip levels above this
    n_active_steps = int(np.sum((levels >= ti_lo) & (levels <= ti_hi)))

    # monotonic step for lo/hi (prevents shell from receding when α dips)
    eps = float(levels[1] - levels[0]) * 0.5 if NT >= 2 else 0.1
    lo_prev = -np.inf
    hi_prev = -np.inf

    t0 = NT
    tlast = -1
    n_geo = 0

    for ti in range(NT):
        d = float(levels[ti])
        if d < ti_lo or d > ti_hi:
            continue
        a_minus = float(alpha[max(0, ti - 1)])
        a_plus  = float(alpha[min(NT - 1, ti + 1)])
        lo = max(0.1, d - float(lame_width) / max(1e-3, a_minus))
        hi = max(lo + 0.1, d + float(lame_width) / max(1e-3, a_plus))
        # enforce strict monotonic growth
        lo = max(lo, lo_prev + eps)
        hi = max(hi, hi_prev + eps, lo + 0.1)
        lo_prev = lo
        hi_prev = hi
        logger.debug(
            "L=%4d  t=%3d  d=%.2f  lo=%.2f  hi=%.2f",
            label, ti, d, lo, hi,
        )
        F = (D_sub - lo) * (hi - D_sub)
        verts, faces = _extract_shell_mesh(
            F, L_sub, label, target_tris, smooth_iter,
        )
        if verts is None or faces is None or faces.shape[0] == 0:
            continue
        verts = verts + offset  # apply bbox offset → global voxel coords
        _shell_cache_save(iso_cache_dir, label, ti, verts, faces)
        if ti < t0:    t0    = ti
        if ti > tlast: tlast = ti
        n_geo += 1

    return int(label), int(t0), int(tlast), int(n_geo), d_col_min, d_col_max, n_active_steps


# ── Main entry point ─────────────────────────────────────────────────────


def build_water_lames(
    *,
    geoddist_path:   Path,
    bg_dist_path:    Path,
    object_path:     Path,
    crown_path:      Path,
    output_vtp:      Path,
    output_labels:   Path | None = None,
    output_meta:     Path | None = None,
    iso_cache_dir:   Path | None = None,
    rebuild_iso_cache: bool = False,
    iso_only:        bool  = False,
    debug_only:      bool  = False,
    debug_n_cols:    int   = 3,
    n_seeds:         int   = DEFAULT_N_SEEDS,
    phase_factor:    int   = DEFAULT_PHASE_FACTOR,
    n_levels_max:    int   = DEFAULT_N_LEVELS_MAX,
    lame_width:      float = DEFAULT_LAME_WIDTH,
    target_tris_per_shell: int = DEFAULT_TARGET_TRIS_PER_SHELL,
    smooth_iter:     int   = DEFAULT_SMOOTH_ITER,
    fade_frames:     int   = DEFAULT_FADE_FRAMES,
    kmeans_iters:    int   = DEFAULT_KMEANS_ITERS,
    seed:            int   = DEFAULT_SEED,
    n_workers:       int | None = None,
) -> dict:
    """Build the lames (V2) cache.  Mirrors ``build_water_membranes`` but
    with the V2 geometry.  Returns the meta dict (or ``{}`` in
    ``iso_only`` mode)."""
    import tifffile
    import vtk
    from vtk.util import numpy_support as nps  # type: ignore
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
        def _malloc_trim() -> None: _libc.malloc_trim(0)
    except OSError:
        def _malloc_trim() -> None: return

    geoddist_path = Path(geoddist_path)
    bg_dist_path  = Path(bg_dist_path)
    object_path   = Path(object_path)
    crown_path    = Path(crown_path)
    output_vtp    = Path(output_vtp)
    if output_meta is None:
        output_meta = output_vtp.with_suffix("").with_name(
            output_vtp.stem + "_meta.json")
    if iso_cache_dir is None:
        iso_cache_dir = output_vtp.parent / "lames_iso_cache"
    iso_cache_dir = Path(iso_cache_dir)

    logger.info("─" * 60)
    logger.info("Water lames (V2) build")
    logger.info("  geoddist     : %s", geoddist_path)
    logger.info("  bg_dist      : %s", bg_dist_path)
    logger.info("  object       : %s", object_path)
    logger.info("  crown        : %s", crown_path)
    logger.info("  output (.vtp): %s", output_vtp)
    logger.info("  n_seeds=%d  phase_factor=%d  n_levels_max=%d  lame_width=%.3f",
                n_seeds, phase_factor, n_levels_max, lame_width)
    logger.info("  target_tris_per_shell=%d  smooth_iter=%d  fade=%d  seed=%d",
                target_tris_per_shell, smooth_iter, fade_frames, seed)

    rng = np.random.default_rng(int(seed))

    # ── 1. Load + clean inputs ──────────────────────────────────────
    t0_all = time.perf_counter()
    geoddist    = _clean_distance(_load_float32(geoddist_path), name="geoddist")
    bg_dist     = _clean_distance(_load_float32(bg_dist_path),  name="bg_dist")
    object_mask = _load_uint8(object_path)
    crown_mask  = _load_uint8(crown_path)
    if not (geoddist.shape == bg_dist.shape == object_mask.shape == crown_mask.shape):
        raise ValueError(
            f"Input volumes have mismatching shapes: "
            f"geoddist={geoddist.shape}, bg_dist={bg_dist.shape}, "
            f"object={object_mask.shape}, crown={crown_mask.shape}"
        )

    # ── 2. Build iso-distance grid ──────────────────────────────────
    inside = object_mask.astype(bool)
    inside_d = geoddist[inside]
    inside_d = inside_d[inside_d > 0]
    if inside_d.size == 0:
        raise RuntimeError("geoddist has no positive value inside the object.")
    d_min = float(inside_d.min())
    d_max = float(inside_d.max())
    raw_step = (d_max - d_min) / float(max(1, n_levels_max))
    level_step = float(max(raw_step, 1e-3))
    eps = 1e-3
    levels = np.arange(d_min + eps, d_max, level_step, dtype=np.float64)
    if levels.size == 0:
        raise RuntimeError(f"No levels in [{d_min}, {d_max}] step={level_step}")
    if levels.size > n_levels_max:
        levels = levels[:n_levels_max]
    NT = int(levels.size)
    Nstep = int(phase_factor) * NT
    logger.info("geoddist range: [%.3f, %.3f]  NT=%d  Nstep=%d  step=%.3f",
                d_min, d_max, NT, Nstep, level_step)

    # ── 3. K-means (medoid) seeds in crown ──────────────────────────
    seeds = _kmeans_medoid_crown(crown_mask, n_seeds, n_iter=kmeans_iters, rng=rng)
    n_columns = int(seeds.shape[0])

    # ── 4. Geodesic watershed ───────────────────────────────────────
    label_vol = _geodesic_watershed(object_mask, geoddist, seeds)
    if output_labels is not None:
        Path(output_labels).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_labels), label_vol.astype(np.int32),
                         compression="zlib")
        logger.info("Labels TIFF written: %s", output_labels)

    # ── 5. Bounding boxes (one pass) ────────────────────────────────
    from scipy.ndimage import find_objects
    bboxes = find_objects(label_vol)
    # find_objects returns list indexed 0..max_label-1 (label k → bboxes[k-1])
    active_labels = [(k + 1, b) for k, b in enumerate(bboxes) if b is not None]
    logger.info("Active labels with non-empty bbox: %d / %d",
                len(active_labels), n_columns)

    # ── 6. Thickening factor ────────────────────────────────────────
    alpha = _thickening_factor(geoddist, object_mask, levels, level_step)

    # ── debug-only: generate diagnostic TIFFs and exit early ────────
    if debug_only:
        _debug_lames(
            geoddist=geoddist,
            label_vol=label_vol,
            active_labels=active_labels,
            levels=levels,
            alpha=alpha,
            lame_width=lame_width,
            output_dir=Path(output_vtp).parent / "debug",
            n_debug=debug_n_cols,
            rng=rng,
        )
        return {}

    # ── 7. ISO cache validity check ─────────────────────────────────
    _iso_valid = (
        not rebuild_iso_cache
        and iso_cache_dir.exists()
        and _cache_is_valid(
            iso_cache_dir,
            n_columns=n_columns, n_levels=NT,
            level_step=level_step, target_tris=target_tris_per_shell,
            lame_width=lame_width,
        )
    )
    if _iso_valid:
        logger.info("Lames ISO cache valid (%s); skipping per-label MC.",
                    iso_cache_dir)
        # Reconstruct t0/tlast from existing files on disk.
        t0_per_col    = np.full(n_columns + 1, NT, dtype=np.int32)
        tlast_per_col = np.full(n_columns + 1, -1, dtype=np.int32)
        for label, _ in active_labels:
            ldir = iso_cache_dir / f"L{int(label):04d}"
            if not ldir.exists():
                continue
            tids = []
            for entry in os.scandir(ldir):
                name = entry.name
                if name.startswith("T") and name.endswith(".npz"):
                    try:
                        tids.append(int(name[1:5]))
                    except ValueError:
                        pass
            if tids:
                t0_per_col[label]    = int(min(tids))
                tlast_per_col[label] = int(max(tids))
    else:
        if rebuild_iso_cache and iso_cache_dir.exists():
            import shutil
            shutil.rmtree(iso_cache_dir)
            logger.info("Lames ISO cache cleared (rebuild requested).")
        iso_cache_dir.mkdir(parents=True, exist_ok=True)

        t0_per_col    = np.full(n_columns + 1, NT, dtype=np.int32)
        tlast_per_col = np.full(n_columns + 1, -1, dtype=np.int32)

        n_workers_actual = _resolve_n_workers(n_workers)
        logger.info("Per-label shell extraction on %d thread(s) …",
                    n_workers_actual)

        # Sort labels by bbox volume DESC → biggest first (high-water-mark).
        def _bbox_vol(b):
            return ((b[0].stop - b[0].start)
                    * (b[1].stop - b[1].start)
                    * (b[2].stop - b[2].start))
        active_labels.sort(key=lambda lb: _bbox_vol(lb[1]), reverse=True)

        t0_phase = time.perf_counter()
        n_done = 0
        GC_EVERY = n_workers_actual  # free memory every N completions
        with ThreadPoolExecutor(max_workers=n_workers_actual) as ex:
            futures = {
                ex.submit(
                    _process_label,
                    label=lb, bbox=bb,
                    geoddist=geoddist, label_vol=label_vol,
                    levels=levels, alpha=alpha,
                    lame_width=lame_width,
                    iso_cache_dir=iso_cache_dir,
                    target_tris=target_tris_per_shell,
                    smooth_iter=smooth_iter,
                ): lb
                for lb, bb in active_labels
            }
            for fut in as_completed(futures):
                lb, lt0, ltl, ngeo, dmin, dmax, nsteps = fut.result()
                n_done += 1
                if ngeo > 0:
                    t0_per_col[lb]    = lt0
                    tlast_per_col[lb] = ltl
                logger.info(
                    "  [%4d/%d] L=%4d  d=[%.2f, %.2f]  active=%3d  shells=%3d",
                    n_done, len(active_labels), lb, dmin, dmax, nsteps, ngeo,
                )
                if n_done % GC_EVERY == 0:
                    gc.collect()
                    _malloc_trim()
        logger.info("Per-label phase done in %.1f s",
                    time.perf_counter() - t0_phase)

        _write_cache_meta(iso_cache_dir, {
            "n_columns":             int(n_columns),
            "n_levels":              int(NT),
            "level_step":            float(level_step),
            "lame_width":            float(lame_width),
            "target_tris_per_shell": int(target_tris_per_shell),
            "smooth_iter":           int(smooth_iter),
            "seed":                  int(seed),
            "levels":                [float(l) for l in levels],
        })
        logger.info("Lames ISO cache written: %s", iso_cache_dir)

    if iso_only:
        logger.info("iso-only mode: stopping after Phase 1a.")
        return {}

    # ── 8. Phase-bin assignment ─────────────────────────────────────
    active_cols = np.where(tlast_per_col >= 0)[0]
    active_cols = active_cols[active_cols > 0]
    logger.info("Columns with surviving shells: %d / %d",
                int(active_cols.size), n_columns)
    if active_cols.size == 0:
        raise RuntimeError("No surviving shells; check inputs / thresholds.")

    tstart = np.zeros(n_columns + 1, dtype=np.int32)
    tstart[active_cols] = rng.integers(0, Nstep, size=active_cols.size).astype(np.int32)

    # ── 9. Sequential packing pass: build merged_* and bake glow ────
    # First pass: count total verts & tris.
    total_tris = 0
    total_verts = 0
    shell_index: list[tuple[int, int, int]] = []  # (label, ti, n_tris)
    for label in active_cols:
        ldir = iso_cache_dir / f"L{int(label):04d}"
        if not ldir.exists():
            continue
        t0 = int(t0_per_col[label]); tl = int(tlast_per_col[label])
        for ti in range(t0, tl + 1):
            v, f = _shell_cache_load(iso_cache_dir, int(label), ti)
            if v is None or f is None or f.shape[0] == 0:
                continue
            shell_index.append((int(label), int(ti), int(f.shape[0])))
            total_tris  += int(f.shape[0])
            total_verts += int(v.shape[0])

    if total_tris == 0:
        raise RuntimeError("No triangles in any shell cache.")
    logger.info("Packing: total verts=%d  tris=%d  shells=%d",
                total_verts, total_tris, len(shell_index))

    merged_verts = np.empty((total_verts, 3), dtype=np.float32)
    merged_faces = np.empty((total_tris,  3), dtype=np.int64)
    merged_time  = np.empty(total_tris, dtype=np.int32)
    merged_col   = np.empty(total_tris, dtype=np.int32)
    merged_step  = np.empty(total_tris, dtype=np.int32)
    merged_rgb   = np.empty((total_tris, 3), dtype=np.uint8)

    voff = 0
    foff = 0
    for label, ti, _n in shell_index:
        v, f = _shell_cache_load(iso_cache_dir, label, ti)
        n_v = int(v.shape[0]); n_f = int(f.shape[0])
        merged_verts[voff:voff + n_v] = v
        merged_faces[foff:foff + n_f] = f + voff
        merged_time [foff:foff + n_f] = ti
        merged_col  [foff:foff + n_f] = label

        # step_id and glow envelope for this shell.
        t0 = int(t0_per_col[label]); tl = int(tlast_per_col[label])
        s_local = ti - t0
        lifetime = max(0, tl - t0)
        step = int((int(tstart[label]) + s_local) % Nstep)
        merged_step[foff:foff + n_f] = step

        glow_norm = float(_glow_curve(
            np.array([s_local], dtype=np.float32), lifetime, fade_frames
        )[0])  # ∈ [0, 1]
        rgb = _BASE_RGB * (1.0 - _GLOW_PEAK_T * glow_norm) \
              + np.array((255.0, 255.0, 255.0), dtype=np.float32) * (_GLOW_PEAK_T * glow_norm)
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        merged_rgb[foff:foff + n_f] = rgb_u8

        voff += n_v
        foff += n_f
        del v, f
    gc.collect()
    _malloc_trim()

    # ── 10. Sort by step_id for contiguous frame slices ─────────────
    order = np.argsort(merged_step, kind="stable")
    merged_faces = merged_faces[order]
    merged_time  = merged_time [order]
    merged_col   = merged_col  [order]
    merged_step  = merged_step [order]
    merged_rgb   = merged_rgb  [order]
    del order

    # ── 11. Build packed PolyData ───────────────────────────────────
    pd = _polydata_from_vf(merged_verts, merged_faces)
    cd = pd.GetCellData()
    for name, arr, vtype in (
        ("step_id",   merged_step, vtk.VTK_INT),
        ("column_id", merged_col,  vtk.VTK_INT),
        ("time_id",   merged_time, vtk.VTK_INT),
    ):
        a = nps.numpy_to_vtk(np.ascontiguousarray(arr), deep=True,
                             array_type=vtype)
        a.SetName(name)
        cd.AddArray(a)
    a_rgb = nps.numpy_to_vtk(np.ascontiguousarray(merged_rgb),
                             deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    a_rgb.SetName("rgb")
    a_rgb.SetNumberOfComponents(3)
    cd.AddArray(a_rgb)
    del merged_verts, merged_faces, merged_time, merged_col, merged_step, merged_rgb

    # ── 12. Write VTP + meta ────────────────────────────────────────
    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(output_vtp))
    w.SetInputData(pd)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()
    logger.info("Lames .vtp written to %s (%.1f MB)",
                output_vtp, output_vtp.stat().st_size / 1e6)

    meta = {
        "version":              2,
        "n_steps":              int(Nstep),
        "n_times":              int(NT),
        "n_columns":            int(n_columns),
        "n_active_columns":     int(active_cols.size),
        "n_triangles_total":    int(total_tris),
        "level_min":            float(d_min),
        "level_max":            float(d_max),
        "level_step":           float(level_step),
        "phase_factor":         int(phase_factor),
        "lame_width":           float(lame_width),
        "target_tris_per_shell": int(target_tris_per_shell),
        "smooth_iter":          int(smooth_iter),
        "fade_frames":          int(fade_frames),
        "seed":                 int(seed),
        "tstart":               tstart.tolist(),
        "t0":                   t0_per_col.tolist(),
        "tlast":                tlast_per_col.tolist(),
        "alpha":                alpha.tolist(),
    }
    Path(output_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(output_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Lames meta written to %s", output_meta)
    logger.info("Total build time: %.1f s", time.perf_counter() - t0_all)
    return meta
