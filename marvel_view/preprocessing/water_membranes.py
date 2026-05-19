"""
Water-descent "membranes" precomputation.

Builds a packed ``vtkPolyData`` that the viewer can scrub through to
animate the descent of water from the outer crown of the root toward
the central axis, following the conducting columns.

Pipeline
--------

1. Load 4 input TIFFs:
   * ``Wat_norm-geoddist.tif``   — float32 geodesic distance to the
     periphery, defined only inside the conducting columns.
   * ``Source_Target_Possible_Paths-dist.tif`` — float32 distance to
     the background (i.e. distance from any column voxel to the nearest
     non-column voxel).
   * ``Source_Target_Possible_Paths.tif`` — uint8 0/255 mask of the
     conducting columns ("the object").
   * ``Source_crown.tif``         — uint8 0/255 outer crown of the root.

2. Pick ``N_SEEDS`` farthest-point-spaced random seeds inside the crown.

3. Run a geodesic watershed inside the object: each voxel of the object
   gets the label of the nearest seed in the
   ``Wat_norm-geoddist``-weighted graph.

4. For each iso-level ``L ∈ arange(d_min+ε, d_max, level_step)`` of
   ``Wat_norm-geoddist`` (NaN/0 treated as outside):
     a. marching cubes on a level mask;
     b. **decimate the whole level mesh** to a fixed budget
        (``--target-tris-per-level``);
     c. drop any triangle that has a vertex closer than
        ``--membrane-bg-min-dist`` from the background;
     d. for each surviving triangle, vote over its 3 vertices' nearest
        voxel label (smallest label wins on ties) → ``column_id``.

5. For each column, draw a random ``Tstart`` in ``[0, Nstep)`` with
   ``Nstep = phase_factor * NT``; assign each surviving triangle a
   ``step_id = (Tstart[c] + (T - T0[c])) % Nstep`` where ``T0[c]`` is
   the first level at which column ``c`` has any triangle.

6. Merge all surviving triangles into a single ``vtkPolyData``, sorted
   by ``step_id`` so that each animation frame is a contiguous cell
   range.  Per-cell arrays attached:

       step_id   int32   animation phase bin (0 .. Nstep-1)
       column_id int32   watershed label of the column (1 .. N_SEEDS)
       time_id   int32   index of the iso-level the triangle was born at
       rgb       uint8×3 gently jittered RGB so columns are visually distinct

   A small JSON sidecar holds the meta:

       n_steps, n_times, n_columns, n_triangles_total,
       level_min, level_max, level_step,
       tstart, t0, tlast  (per-column lists)

The build is reproducible via a single ``--seed`` integer.

All heavy imports (skimage, vedo, vtk) are deferred to ``build()`` so
this module remains lightweight to import.
"""
from __future__ import annotations

import colorsys
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ── defaults ─────────────────────────────────────────────────────────────

DEFAULT_N_SEEDS:                int   = 1000
DEFAULT_PHASE_FACTOR:           int   = 5
DEFAULT_LEVEL_STEP:             float = 0.6
DEFAULT_TARGET_TRIS_PER_LEVEL:  int   = 100_000
DEFAULT_BG_MIN_DIST:            float = 1.0
DEFAULT_SEED:                   int   = 1234

# Base colour (cyan-blue) used for the per-column hue jitter.
_BASE_RGB_0_255 = (140, 195, 255)
_HUE_JITTER     = 0.04   # ± in HSV [0, 1]
_VAL_JITTER     = 0.05


# ── helpers ──────────────────────────────────────────────────────────────


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


def _farthest_point_seeds(
    crown_mask: np.ndarray,
    n_seeds: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``(n, 3)`` voxel indices of ``n_seeds`` spread-out crown voxels.

    Uses incremental farthest-point sampling: starts from a random crown
    voxel, then repeatedly picks the crown voxel that is farthest (in
    voxel-Euclidean distance) from the already-selected set.  Stops
    early if the crown has fewer voxels than ``n_seeds``.
    """
    coords = np.argwhere(crown_mask > 0)
    n_avail = coords.shape[0]
    if n_avail == 0:
        raise ValueError("Crown mask has no positive voxels.")
    n_seeds = int(min(n_seeds, n_avail))
    logger.info("FPS seed sampling: crown has %d voxels, picking %d seeds",
                n_avail, n_seeds)

    # Distance from every crown voxel to the nearest already-picked seed.
    # Start with the seed = random crown voxel.
    first = int(rng.integers(0, n_avail))
    picked = [first]
    diff = coords - coords[first]
    dist2 = np.einsum("ij,ij->i", diff, diff).astype(np.float64)

    for _ in range(1, n_seeds):
        idx = int(np.argmax(dist2))
        picked.append(idx)
        diff = coords - coords[idx]
        new_d2 = np.einsum("ij,ij->i", diff, diff).astype(np.float64)
        np.minimum(dist2, new_d2, out=dist2)

    return coords[np.asarray(picked, dtype=np.int64)]


def _build_label_volume(
    object_mask: np.ndarray,
    geoddist:    np.ndarray,
    seeds_zyx:   np.ndarray,
) -> np.ndarray:
    """Geodesic watershed inside ``object_mask``, weighted by ``geoddist``.

    Returns an ``int32`` array of the same shape, 0 outside the object,
    1..N_seeds inside.
    """
    from skimage.segmentation import watershed

    markers = np.zeros(object_mask.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(seeds_zyx, start=1):
        markers[z, y, x] = i

    # The watershed function expects a finite, non-negative scalar
    # field as the priority map.  Replace NaN/inf with a large value
    # (so labels can still cross thin gaps if needed) and clamp.
    img = np.where(np.isfinite(geoddist), geoddist, 0.0).astype(np.float32)
    img = np.clip(img, 0.0, None)

    logger.info("Running geodesic watershed (object voxels: %d, seeds: %d) …",
                int(object_mask.sum()), int(markers.max()))
    t = time.perf_counter()
    labels = watershed(img, markers=markers, mask=object_mask.astype(bool),
                       compactness=0.0)
    logger.info("Watershed done in %.1f s (unique labels: %d)",
                time.perf_counter() - t, int(labels.max()))
    return labels.astype(np.int32)


def _per_column_rgb(n_columns: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``(n_columns + 1, 3)`` uint8 RGB lookup table.

    Index 0 is unused (reserved for "no label"); indices 1..n_columns
    are the per-column colours = the base cyan-blue with a gentle HSV
    jitter.
    """
    base = colorsys.rgb_to_hsv(*(c / 255.0 for c in _BASE_RGB_0_255))
    h0, s0, v0 = base
    out = np.zeros((n_columns + 1, 3), dtype=np.uint8)
    out[0] = (255, 255, 255)
    for c in range(1, n_columns + 1):
        dh = float(rng.uniform(-_HUE_JITTER, +_HUE_JITTER))
        dv = float(rng.uniform(-_VAL_JITTER, +_VAL_JITTER))
        h = (h0 + dh) % 1.0
        v = float(np.clip(v0 + dv, 0.0, 1.0))
        r, g, b = colorsys.hsv_to_rgb(h, s0, v)
        out[c] = (int(round(r * 255)), int(round(g * 255)),
                  int(round(b * 255)))
    return out


def _marching_cubes_level(
    geoddist:    np.ndarray,
    object_mask: np.ndarray,
    level:       float,
):
    """Run marching cubes on ``geoddist`` at ``level`` inside the object.

    Returns ``(verts, faces)`` or ``(None, None)`` if no surface is
    produced at this level.

    To keep the iso-surface contained inside the object we substitute a
    sentinel value far above ``level`` for voxels outside the object so
    marching cubes never crosses the boundary.
    """
    from skimage.measure import marching_cubes

    inside = object_mask.astype(bool)
    if not inside.any():
        return None, None

    # Build a "padded" volume: object voxels keep their geoddist value;
    # non-object voxels are replaced by a large value > any iso-level
    # so the surface stays inside the object.
    geo = np.where(np.isfinite(geoddist), geoddist, 0.0).astype(np.float32)
    fill = float(max(np.nanmax(geo[inside]) if inside.any() else level,
                     level) + 10.0)
    padded = np.where(inside, geo, fill)

    try:
        verts, faces, _normals, _values = marching_cubes(
            padded, level=float(level), step_size=1, allow_degenerate=False,
        )
    except (ValueError, RuntimeError) as exc:
        logger.debug("Marching cubes failed at level %.3f: %s", level, exc)
        return None, None
    if verts.size == 0 or faces.size == 0:
        return None, None
    return verts.astype(np.float32), faces.astype(np.int64)


def _decimate_mesh(
    verts: np.ndarray, faces: np.ndarray, target: int,
):
    """Decimate ``(verts, faces)`` down to ~``target`` triangles.

    Uses VTK's :class:`vtkQuadricDecimation`.  Returns the decimated
    ``(verts, faces)``.  If the mesh already has ≤ target triangles,
    returns it unchanged.
    """
    n_tri = faces.shape[0]
    if n_tri <= target or target <= 0:
        return verts, faces

    fraction = 1.0 - (float(target) / float(n_tri))
    fraction = float(np.clip(fraction, 0.0, 0.999))
    try:
        import vtk
        from vtk.util import numpy_support as nps  # type: ignore
    except ImportError as exc:
        logger.warning("Decimation skipped (VTK unavailable): %s", exc)
        return verts, faces

    pd = _polydata_from_vf(verts, faces)
    dec = vtk.vtkQuadricDecimation()
    dec.SetInputData(pd)
    dec.SetTargetReduction(fraction)
    dec.SetVolumePreservation(False)
    dec.Update()
    out = dec.GetOutput()
    if out is None or out.GetNumberOfPoints() == 0:
        return verts, faces
    new_verts = nps.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float32)
    polys = out.GetPolys()
    n_cells = out.GetNumberOfCells()
    if n_cells == 0:
        return verts, faces
    raw = nps.vtk_to_numpy(polys.GetData()).reshape(-1, 4)
    # All cells are triangles → first column is "3", drop it.
    new_faces = raw[:, 1:].astype(np.int64)
    return new_verts, new_faces


def _polydata_from_vf(verts: np.ndarray, faces: np.ndarray):
    """Return a ``vtkPolyData`` for the (verts, faces) pair."""
    import vtk
    from vtk.util import numpy_support as nps  # type: ignore

    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(verts, dtype=np.float32),
                                  deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)

    n_tri = faces.shape[0]
    # Build the connectivity array in VTK's [3, i0, i1, i2, 3, …] format.
    conn = np.empty((n_tri, 4), dtype=np.int64)
    conn[:, 0] = 3
    conn[:, 1:] = faces
    cells = vtk.vtkCellArray()
    id_arr = nps.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn.ravel()),
                                         deep=True)
    cells.SetCells(n_tri, id_arr)
    pd.SetPolys(cells)
    return pd


def _triangle_labels_and_keep(
    verts:     np.ndarray,
    faces:     np.ndarray,
    label_vol: np.ndarray,
    bg_dist:   np.ndarray,
    bg_min:    float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle column vote + background filter.

    Returns ``(column_id, keep_mask)`` of shape ``(n_tri,)``.  Triangles
    with ``keep_mask=False`` should be dropped.  ``column_id`` is set
    to 0 where ``keep_mask`` is False.
    """
    shape = label_vol.shape
    # Per-vertex nearest voxel index, clipped to volume bounds.
    vi = np.clip(np.round(verts).astype(np.int64), 0, None)
    vi[:, 0] = np.minimum(vi[:, 0], shape[0] - 1)
    vi[:, 1] = np.minimum(vi[:, 1], shape[1] - 1)
    vi[:, 2] = np.minimum(vi[:, 2], shape[2] - 1)

    vert_labels = label_vol[vi[:, 0], vi[:, 1], vi[:, 2]]
    vert_bg     = bg_dist  [vi[:, 0], vi[:, 1], vi[:, 2]]

    # Per-triangle: drop if any vertex too close to background or has
    # label 0 (i.e. outside the object).
    f_lab = vert_labels[faces]                     # (n_tri, 3)
    f_bg  = vert_bg    [faces]                     # (n_tri, 3)
    keep = (f_bg >= float(bg_min)).all(axis=1) & (f_lab > 0).all(axis=1)

    # Vote: majority wins, smallest label on ties.
    a, b, c = f_lab[:, 0], f_lab[:, 1], f_lab[:, 2]
    eq_ab = a == b
    eq_ac = a == c
    eq_bc = b == c
    # If all three agree → label = a.
    # If two agree → label = the agreeing pair.
    # If all disagree → smallest of the three.
    pair = np.where(eq_ab | eq_ac, a, np.where(eq_bc, b, np.minimum(np.minimum(a, b), c)))
    column_id = np.where(keep, pair, 0).astype(np.int32)
    return column_id, keep


# ── server detection + worker count ──────────────────────────────────────

_SERVER_MARKER = Path("/mnt/e823c70f-4136-47c9-91be-1ca7901a37b5")


def _resolve_n_workers(requested: int | None) -> int:
    """Return the thread-pool size for the marching-cubes pass.

    Priority: explicit ``requested`` > ``MARVEL_MAX_WORKERS`` env var >
    auto-detect (cpu_count // 2).
    """
    if requested is not None:
        return max(1, int(requested))
    env_val = os.environ.get("MARVEL_MAX_WORKERS")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(1, cpus // 2)


# ── ISO-surface cache helpers ─────────────────────────────────────────────

def _iso_cache_save(cache_dir: Path, ti: int,
                    verts: np.ndarray, faces: np.ndarray) -> None:
    """Persist one level's (verts, faces) to *cache_dir* as a compressed NPZ."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(cache_dir / f"iso_{ti:04d}.npz"),
        verts=verts, faces=faces,
    )


def _iso_cache_load(cache_dir: Path, ti: int):
    """Load (verts, faces) for level *ti*, or return (None, None)."""
    p = cache_dir / f"iso_{ti:04d}.npz"
    if not p.exists():
        return None, None
    data = np.load(str(p))
    return data["verts"].astype(np.float32), data["faces"].astype(np.int64)


def _iso_cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "cache_meta.json"


def _read_cache_meta(cache_dir: Path) -> dict:
    p = _iso_cache_meta_path(cache_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _iso_cache_is_valid(
    cache_dir: Path,
    level_step: float,
    target_tris: int,
    n_levels: int,
) -> bool:
    """True when *cache_dir* was built with exactly the same MC parameters."""
    meta = _read_cache_meta(cache_dir)
    return (
        bool(meta)
        and abs(meta.get("level_step", -1.0) - level_step) < 1e-6
        and meta.get("target_tris_per_level", -1) == int(target_tris)
        and meta.get("n_levels", -1) == int(n_levels)
    )


def _write_cache_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _iso_cache_meta_path(cache_dir).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ── main entry point ─────────────────────────────────────────────────────


def build_water_membranes(
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
    n_seeds:         int   = DEFAULT_N_SEEDS,
    phase_factor:    int   = DEFAULT_PHASE_FACTOR,
    level_step:      float = DEFAULT_LEVEL_STEP,
    target_tris_per_level: int = DEFAULT_TARGET_TRIS_PER_LEVEL,
    bg_min_dist:     float = DEFAULT_BG_MIN_DIST,
    seed:            int   = DEFAULT_SEED,
    n_workers:       int | None = None,
    crop_fraction:   float | None = None,
) -> dict:
    """Build the membranes cache from the 4 input TIFFs.

    Writes ``output_vtp`` (binary compressed) and (when given)
    ``output_meta`` (JSON) and ``output_labels`` (int32 TIFF).  Returns
    the meta dict.
    """
    import tifffile
    import vtk
    from vtk.util import numpy_support as nps  # type: ignore

    geoddist_path = Path(geoddist_path)
    bg_dist_path  = Path(bg_dist_path)
    object_path   = Path(object_path)
    crown_path    = Path(crown_path)
    output_vtp    = Path(output_vtp)
    if output_meta is None:
        output_meta = output_vtp.with_suffix("").with_name(
            output_vtp.stem + "_meta.json")
    if iso_cache_dir is None:
        iso_cache_dir = output_vtp.parent / "membranes_iso_cache"
    iso_cache_dir = Path(iso_cache_dir)

    logger.info("─" * 60)
    logger.info("Water membranes build")
    logger.info("  geoddist     : %s", geoddist_path)
    logger.info("  bg_dist      : %s", bg_dist_path)
    logger.info("  object       : %s", object_path)
    logger.info("  crown        : %s", crown_path)
    logger.info("  output (.vtp): %s", output_vtp)
    logger.info("  n_seeds=%d  phase_factor=%d  level_step=%.3f",
                n_seeds, phase_factor, level_step)
    logger.info("  target_tris_per_level=%d  bg_min_dist=%.3f  seed=%d",
                target_tris_per_level, bg_min_dist, seed)
    if crop_fraction is not None:
        logger.info("  crop_fraction=%.2f  (speed-test mode)", crop_fraction)

    rng = np.random.default_rng(int(seed))

    # ── 1. Load inputs ──────────────────────────────────────────────
    geoddist    = _load_float32(geoddist_path)
    # Neutralise NaN/inf so they are treated as background everywhere
    # downstream (marching cubes, watershed, iso-level range).
    _nan_count = int(np.sum(~np.isfinite(geoddist)))
    if _nan_count:
        logger.info("geoddist: replacing %d non-finite voxels with 0.0", _nan_count)
    geoddist = np.where(np.isfinite(geoddist), geoddist, 0.0).astype(np.float32)
    bg_dist     = _load_float32(bg_dist_path)
    object_mask = _load_uint8(object_path)
    crown_mask  = _load_uint8(crown_path)

    if not (geoddist.shape == bg_dist.shape == object_mask.shape == crown_mask.shape):
        raise ValueError(
            f"Input volumes have mismatching shapes: "
            f"geoddist={geoddist.shape}, bg_dist={bg_dist.shape}, "
            f"object={object_mask.shape}, crown={crown_mask.shape}"
        )

    # ── 1b. Clip sentinel "no-data" values ─────────────────────────
    # Some pipelines write float32 max (~3.4e38) instead of NaN as the
    # background marker.  Detect them: anything > 100× the 99.9th
    # percentile of positive object voxels is treated as background (→ 0).
    _obj_pos = geoddist[(object_mask > 0) & (geoddist > 0)]
    if _obj_pos.size > 0:
        _p999 = float(np.percentile(_obj_pos, 99.9))
        _sentinel_thresh = _p999 * 100.0
        _n_sentinel = int(np.sum(geoddist > _sentinel_thresh))
        if _n_sentinel:
            logger.info(
                "geoddist: zeroing %d sentinel voxels (val > %.1f; "
                "99.9th-pct of object voxels = %.3f)",
                _n_sentinel, _sentinel_thresh, _p999,
            )
            geoddist = np.where(
                geoddist <= _sentinel_thresh, geoddist, 0.0
            ).astype(np.float32)

    # ── 1c. Speed-test crop ─────────────────────────────────────────
    if crop_fraction is not None:
        frac = float(np.clip(crop_fraction, 0.01, 1.0))
        longest_ax = int(np.argmax(geoddist.shape))
        end_idx = max(1, int(round(geoddist.shape[longest_ax] * frac)))
        slc = [slice(None), slice(None), slice(None)]
        slc[longest_ax] = slice(0, end_idx)
        slc = tuple(slc)
        geoddist    = geoddist[slc].copy()
        bg_dist     = bg_dist[slc].copy()
        object_mask = object_mask[slc].copy()
        crown_mask  = crown_mask[slc].copy()
        logger.info(
            "Speed-test crop: axis=%d  kept %d/%d → new shape %s",
            longest_ax, end_idx, geoddist.shape[longest_ax],
            geoddist.shape,
        )

    # Restrict the crown to voxels also inside the object (so seeds are
    # actually labellable by the watershed).
    crown_in_obj = ((crown_mask > 0) & (object_mask > 0)).astype(np.uint8)
    if crown_in_obj.sum() == 0:
        logger.warning("Crown ∩ object is empty; falling back to full crown.")
        crown_in_obj = crown_mask

    # ── 2. Seeds + 3. Watershed ─────────────────────────────────────
    # When the watershed parameters (n_seeds, seed) are unchanged we can
    # reload the label volume from the previous run's TIFF and skip FPS +
    # skimage watershed — which can save several minutes on large volumes.
    _cache_meta = _read_cache_meta(iso_cache_dir)
    _labels_from_cache = (
        not rebuild_iso_cache
        and crop_fraction is None          # cropped runs always recompute
        and output_labels is not None
        and Path(output_labels).exists()
        and _cache_meta.get("n_seeds") == int(n_seeds)
        and _cache_meta.get("seed")    == int(seed)
    )
    if _labels_from_cache:
        logger.info("Loading label volume from cache (%s) …", output_labels)
        label_vol = tifffile.imread(str(output_labels)).squeeze().astype(np.int32)
        n_columns = int(label_vol.max())
        logger.info("Label cache loaded: %d columns", n_columns)
    else:
        seeds = _farthest_point_seeds(crown_in_obj, n_seeds, rng=rng)
        label_vol = _build_label_volume(object_mask, geoddist, seeds)
        n_columns = int(label_vol.max())
        if output_labels is not None:
            try:
                Path(output_labels).parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(str(output_labels), label_vol.astype(np.int32))
                logger.info("Labels written to %s", output_labels)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not write labels TIFF %s: %s",
                               output_labels, exc)

    # ── 4. Iso-levels ───────────────────────────────────────────────
    finite_inside = geoddist[(object_mask > 0) & np.isfinite(geoddist) & (geoddist > 0)]
    if finite_inside.size == 0:
        raise RuntimeError("No positive finite values in geoddist inside object.")
    d_min = float(finite_inside.min())
    d_max = float(finite_inside.max())
    eps = 1e-3
    _MAX_LEVELS = 300
    _raw_n = max(1, int((d_max - d_min) / float(level_step)))
    if _raw_n > _MAX_LEVELS:
        level_step = (d_max - d_min) / _MAX_LEVELS
        logger.warning(
            "geoddist range [%.1f, %.1f] with requested level_step would produce "
            "%d levels (> %d); auto-adjusting level_step to %.3f.",
            d_min, d_max, _raw_n, _MAX_LEVELS, level_step,
        )
    levels = np.arange(d_min + eps, d_max, float(level_step), dtype=np.float64)
    NT = int(levels.size)
    if NT == 0:
        raise RuntimeError(
            f"No iso-levels in range [{d_min}, {d_max}] with step {level_step}."
        )
    Nstep = int(phase_factor) * NT
    logger.info("geoddist range: [%.3f, %.3f]  NT=%d  Nstep=%d",
                d_min, d_max, NT, Nstep)

    # ── 5. Per-level marching cubes → decimate → filter + vote ─────
    # Collect surviving triangles across all levels.
    all_verts:   list[np.ndarray] = []
    all_faces:   list[np.ndarray] = []   # face indices into THAT level's verts
    all_levels:  list[np.ndarray] = []   # per-triangle T index
    all_columns: list[np.ndarray] = []   # per-triangle column id

    # Per-column min/max time index (for Tstart phase).
    t0_per_col    = np.full(n_columns + 1, NT, dtype=np.int32)
    tlast_per_col = np.full(n_columns + 1, -1, dtype=np.int32)

    t0_total = time.perf_counter()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── ISO cache validity check ────────────────────────────────────
    _iso_valid = (
        not rebuild_iso_cache
        and iso_cache_dir.exists()
        and _iso_cache_is_valid(iso_cache_dir, level_step, target_tris_per_level, NT)
    )
    if _iso_valid:
        logger.info(
            "ISO cache valid (%s, %d levels); skipping marching cubes.",
            iso_cache_dir, NT,
        )
    else:
        if rebuild_iso_cache and iso_cache_dir.exists():
            import shutil
            shutil.rmtree(iso_cache_dir)
            logger.info("ISO cache cleared (rebuild requested).")
        iso_cache_dir.mkdir(parents=True, exist_ok=True)

    n_workers_actual = _resolve_n_workers(n_workers)
    logger.info(
        "Per-level marching cubes + filter on %d thread(s) %s …",
        n_workers_actual,
        "(from ISO cache — MC skipped)" if _iso_valid else "(full recompute)",
    )

    def _process_level(ti: int, lvl: float):
        if _iso_valid:
            # Load pre-decimated geometry from cache; re-apply the
            # bg/column filter (which may have changed parameters).
            verts, faces = _iso_cache_load(iso_cache_dir, ti)
            if verts is None:
                return ti, lvl, -1, -1, None, None, None
            column_id, keep = _triangle_labels_and_keep(
                verts, faces, label_vol, bg_dist, bg_min_dist,
            )
            if not bool(keep.any()):
                return ti, lvl, -1, int(faces.shape[0]), None, None, None
            return ti, lvl, -1, int(faces.shape[0]), verts, faces[keep], column_id[keep]

        # Full computation.
        verts, faces = _marching_cubes_level(geoddist, object_mask, float(lvl))
        if verts is None:
            return ti, lvl, 0, 0, None, None, None
        n_raw = int(faces.shape[0])
        verts, faces = _decimate_mesh(verts, faces, target_tris_per_level)
        n_dec = int(faces.shape[0])
        # Save (verts, faces) BEFORE the bg/column filter so that future
        # runs with different bg_min_dist / n_seeds can reuse this cache.
        _iso_cache_save(iso_cache_dir, ti, verts, faces)
        column_id, keep = _triangle_labels_and_keep(
            verts, faces, label_vol, bg_dist, bg_min_dist,
        )
        if not bool(keep.any()):
            return ti, lvl, n_raw, n_dec, None, None, None
        return ti, lvl, n_raw, n_dec, verts, faces[keep], column_id[keep]

    # Slots indexed by ti so we can later assemble in deterministic order.
    slot_verts:   list[np.ndarray | None] = [None] * NT
    slot_faces:   list[np.ndarray | None] = [None] * NT
    slot_columns: list[np.ndarray | None] = [None] * NT

    n_done = 0
    with ThreadPoolExecutor(max_workers=n_workers_actual) as ex:
        futures = [ex.submit(_process_level, ti, float(lvl))
                   for ti, lvl in enumerate(levels)]
        for fut in as_completed(futures):
            ti, lvl, n_raw, n_dec, verts, kept_faces, kept_cols = fut.result()
            n_done += 1
            if kept_faces is None:
                logger.info("  [%3d/%d done] T=%3d level=%6.3f  raw=%7d  dec=%7d  keep=      0",
                            n_done, NT, ti, lvl, n_raw, n_dec)
                continue
            n_keep = int(kept_faces.shape[0])
            slot_verts[ti]   = verts
            slot_faces[ti]   = kept_faces
            slot_columns[ti] = kept_cols
            unique_cols = np.unique(kept_cols)
            unique_cols = unique_cols[unique_cols > 0]
            if unique_cols.size:
                t0_per_col[unique_cols]    = np.minimum(t0_per_col[unique_cols],    ti)
                tlast_per_col[unique_cols] = np.maximum(tlast_per_col[unique_cols], ti)
            logger.info("  [%3d/%d done] T=%3d level=%6.3f  raw=%7d  dec=%7d  keep=%7d",
                        n_done, NT, ti, lvl, n_raw, n_dec, n_keep)

    # Assemble in ti order so downstream code sees deterministic results.
    for ti in range(NT):
        if slot_faces[ti] is None:
            continue
        all_verts.append(slot_verts[ti])
        all_faces.append(slot_faces[ti])
        all_levels.append(np.full(slot_faces[ti].shape[0], ti, dtype=np.int32))
        all_columns.append(slot_columns[ti])

    logger.info("Marching cubes + filter done in %.1f s",
                time.perf_counter() - t0_total)

    # Persist cache metadata so the next run can validate the cache.
    if not _iso_valid:
        _write_cache_meta(iso_cache_dir, {
            "level_step":            float(level_step),
            "target_tris_per_level": int(target_tris_per_level),
            "n_levels":              int(NT),
            "levels":                [float(l) for l in levels],
            "n_seeds":               int(n_seeds),
            "seed":                  int(seed),
        })
        logger.info("ISO cache written: %s (%d levels)", iso_cache_dir, NT)

    if not all_faces:
        raise RuntimeError("No triangles survived filtering; check inputs.")

    # ── 6. Phase-bin assignment ─────────────────────────────────────
    active_cols = np.where(tlast_per_col >= 0)[0]
    active_cols = active_cols[active_cols > 0]
    logger.info("Columns with surviving triangles: %d / %d",
                int(active_cols.size), n_columns)

    tstart = np.zeros(n_columns + 1, dtype=np.int32)
    tstart[active_cols] = rng.integers(0, Nstep, size=active_cols.size).astype(np.int32)

    # Per-column RGB lookup table (gentle hue jitter).
    rgb_lut = _per_column_rgb(n_columns, rng)

    # ── 7. Merge into a single packed PolyData ──────────────────────
    # Offset face indices by the running vertex count.
    total_verts = sum(v.shape[0] for v in all_verts)
    total_tris  = sum(f.shape[0] for f in all_faces)
    logger.info("Merging packed mesh: total verts=%d  total tris=%d",
                total_verts, total_tris)

    merged_verts = np.empty((total_verts, 3), dtype=np.float32)
    merged_faces = np.empty((total_tris,  3), dtype=np.int64)
    merged_time  = np.empty(total_tris, dtype=np.int32)
    merged_col   = np.empty(total_tris, dtype=np.int32)

    voff = 0
    foff = 0
    for v, f, t_arr, c_arr in zip(all_verts, all_faces, all_levels, all_columns):
        n_v = v.shape[0]
        n_f = f.shape[0]
        merged_verts[voff:voff + n_v] = v
        merged_faces[foff:foff + n_f] = f + voff
        merged_time [foff:foff + n_f] = t_arr
        merged_col  [foff:foff + n_f] = c_arr
        voff += n_v
        foff += n_f

    # step_id = (Tstart[c] + (T - T0[c])) % Nstep
    t0_arr_per_tri = t0_per_col[merged_col]
    tst_arr_per_tri = tstart[merged_col]
    step_id = ((tst_arr_per_tri + (merged_time - t0_arr_per_tri)) % Nstep).astype(np.int32)

    # Per-triangle RGB (gentle hue jitter per column).
    rgb_per_tri = rgb_lut[merged_col]   # (n_tri, 3) uint8

    # Sort triangles by step_id so each animation frame is a contiguous
    # cell range — minimises VTK per-tick work.
    order = np.argsort(step_id, kind="stable")
    merged_faces = merged_faces[order]
    merged_time  = merged_time [order]
    merged_col   = merged_col  [order]
    step_id      = step_id     [order]
    rgb_per_tri  = rgb_per_tri [order]

    # Build the PolyData.
    pd = _polydata_from_vf(merged_verts, merged_faces)
    cd = pd.GetCellData()

    arr_step = nps.numpy_to_vtk(np.ascontiguousarray(step_id),  deep=True,
                                array_type=vtk.VTK_INT)
    arr_step.SetName("step_id")
    cd.AddArray(arr_step)

    arr_col = nps.numpy_to_vtk(np.ascontiguousarray(merged_col), deep=True,
                               array_type=vtk.VTK_INT)
    arr_col.SetName("column_id")
    cd.AddArray(arr_col)

    arr_time = nps.numpy_to_vtk(np.ascontiguousarray(merged_time), deep=True,
                                array_type=vtk.VTK_INT)
    arr_time.SetName("time_id")
    cd.AddArray(arr_time)

    arr_rgb = nps.numpy_to_vtk(np.ascontiguousarray(rgb_per_tri),  deep=True,
                               array_type=vtk.VTK_UNSIGNED_CHAR)
    arr_rgb.SetName("rgb")
    arr_rgb.SetNumberOfComponents(3)
    cd.AddArray(arr_rgb)

    # ── 8. Write ────────────────────────────────────────────────────
    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_vtp))
    writer.SetInputData(pd)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    writer.Write()
    logger.info("Membranes .vtp written to %s (%.1f MB)",
                output_vtp, output_vtp.stat().st_size / 1e6)

    meta = {
        "n_steps":           int(Nstep),
        "n_times":           int(NT),
        "n_columns":         int(n_columns),
        "n_active_columns":  int(active_cols.size),
        "n_triangles_total": int(total_tris),
        "level_min":         float(d_min),
        "level_max":         float(d_max),
        "level_step":        float(level_step),
        "phase_factor":      int(phase_factor),
        "target_tris_per_level": int(target_tris_per_level),
        "bg_min_dist":       float(bg_min_dist),
        "seed":              int(seed),
        "tstart":            tstart.tolist(),
        "t0":                t0_per_col.tolist(),
        "tlast":             tlast_per_col.tolist(),
    }
    Path(output_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(output_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Membranes meta written to %s", output_meta)

    return meta
