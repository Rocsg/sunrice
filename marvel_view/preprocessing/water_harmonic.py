"""
Water harmonic potential field — pre-processing module.

Builds a *harmonic potential* :math:`u` on the water-conducting mask with
Dirichlet boundary conditions

* :math:`u = 0` on the **source** mask (crown / outer cortex),
* :math:`u = 1` on the **target** mask (central axis / stele),

and homogeneous Neumann conditions on cell walls.  The resulting field drives:

1. **Stream tracks** — polylines integrated along :math:`+\\nabla u` from
   source to target, replacing Dijkstra / MCP (faster, smoother, hydraulically
   meaningful).

2. **Lames / iso-shells** — animated iso-shells sorted by passage time
   :math:`T` instead of geodesic distance.  Two passage-time maps are
   provided:

   * ``T_simple  = 1 - u``        — immediate, monotone, always available.
   * ``T_eikonal = |\\nabla T|=1``  — eikonal equation solved along :math:`u`
     iso-surfaces; gives true "time" for a front propagating at speed
     :math:`|\\nabla u|`.  Requires ``scikit-fmm``; skip with
     ``skip_eikonal=True``.

3. **Dual-arrow overlay** — water + air arrow fields shown simultaneously,
   spatially separated by a 2-voxel EDT to avoid overlap.

Outputs saved as ``water_harmonic.npz``:

    u, vec, speed, area, source, target, T_simple[, T_eikonal]

The stream tracks are saved separately as ``crown_tracks.vtp``
(same cache path as the Dijkstra tracks → viewer unchanged).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_CG_TOL:    float = 1e-3
DEFAULT_CG_MAXITER: int  = 4000
DEFAULT_GRAD_SIGMA: float = 0.9   # masked-Gaussian sigma before gradient

# Stream-line integration
DEFAULT_STEP_VOX:  float = 0.5    # integration step in voxels
DEFAULT_MAX_STEPS:  int  = 5000   # safety cap per track
DEFAULT_N_SEEDS:    int  = 500    # stream-track seed count (k-medoids)

# Dual-arrow spatial separation (voxels)
DEFAULT_SEPARATION_VOX: int = 2


# ── Internal helpers (copied / adapted from wind_field.py) ──────────────────

def _largest_connected_component(
    area: np.ndarray,
    src:  np.ndarray,
    tgt:  np.ndarray,
) -> np.ndarray:
    """Keep only the connected component of ``area`` that touches both
    ``src`` and ``tgt``.  Returns a refined boolean mask."""
    from scipy.ndimage import label

    labelled, n_lab = label(area.astype(bool))
    if n_lab == 0:
        raise ValueError("water area mask is empty.")

    src_labels = set(int(v) for v in np.unique(labelled[src]) if v != 0)
    tgt_labels = set(int(v) for v in np.unique(labelled[tgt]) if v != 0)
    valid = src_labels & tgt_labels
    if not valid:
        raise ValueError(
            "No connected component of the water area touches both source and target."
        )
    keep = np.zeros_like(labelled, dtype=bool)
    for v in valid:
        keep |= (labelled == v)
    logger.info(
        "Connected components: kept %d/%d voxels (%.1f%%) in %d traversing labels.",
        int(keep.sum()), int(area.sum()),
        100.0 * int(keep.sum()) / max(int(area.sum()), 1), len(valid),
    )
    return keep


def _masked_gaussian(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised-convolution Gaussian smooth of ``arr`` inside ``mask``."""
    from scipy.ndimage import gaussian_filter
    a = arr.astype(np.float32, copy=False)
    m = mask.astype(np.float32, copy=False)
    num = gaussian_filter(a * m, sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter(m,     sigma=sigma, mode="constant", cval=0.0)
    out = np.zeros_like(a)
    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out


# ── Laplace solver ──────────────────────────────────────────────────────────

def solve_harmonic_potential(
    area: np.ndarray,
    src:  np.ndarray,
    tgt:  np.ndarray,
    tol:     float = DEFAULT_CG_TOL,
    maxiter: int   = DEFAULT_CG_MAXITER,
) -> np.ndarray:
    """Solve :math:`\\nabla^2 u = 0` on ``area`` with ``u=0`` on ``src`` and
    ``u=1`` on ``tgt`` (homogeneous Neumann on walls).

    Uses a 7-point finite-difference stencil with PyAMG (smoothed
    aggregation) and a SciPy CG fallback.

    Returns
    -------
    u : float32 ``(Z, Y, X)``
        Potential field, zero outside ``area``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator, cg

    area = np.asarray(area, dtype=bool)
    src  = np.asarray(src,  dtype=bool) & area
    tgt  = np.asarray(tgt,  dtype=bool) & area
    interior = area & ~src & ~tgt
    n = int(interior.sum())
    if n == 0:
        raise ValueError("solve_harmonic_potential: no interior voxels.")

    idx = np.full(area.shape, -1, dtype=np.int64)
    idx[interior] = np.arange(n, dtype=np.int64)

    interior_coords = np.array(np.nonzero(interior), dtype=np.int64).T  # (n, 3)

    diag = np.zeros(n, dtype=np.float64)
    b    = np.zeros(n, dtype=np.float64)
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []

    sz, sy, sx = area.shape
    for (dz, dy, dx) in ((1,0,0), (-1,0,0),
                         (0,1,0), (0,-1,0),
                         (0,0,1), (0,0,-1)):
        zz = interior_coords[:, 0] + dz
        yy = interior_coords[:, 1] + dy
        xx = interior_coords[:, 2] + dx
        in_bounds = (
            (zz >= 0) & (zz < sz) &
            (yy >= 0) & (yy < sy) &
            (xx >= 0) & (xx < sx)
        )
        ok = np.where(in_bounds)[0]
        zz_o = zz[ok]; yy_o = yy[ok]; xx_o = xx[ok]
        nbr_in_area = area[zz_o, yy_o, xx_o]
        active_local = ok[nbr_in_area]
        np.add.at(diag, active_local, 1.0)

        zz_a = interior_coords[active_local, 0] + dz
        yy_a = interior_coords[active_local, 1] + dy
        xx_a = interior_coords[active_local, 2] + dx
        nbr_is_tgt = tgt[zz_a, yy_a, xx_a]
        nbr_is_int = interior[zz_a, yy_a, xx_a]

        np.add.at(b, active_local[nbr_is_tgt], 1.0)

        i_sel = np.where(nbr_is_int)[0]
        rows_list.append(active_local[i_sel])
        cols_list.append(idx[zz_a[i_sel], yy_a[i_sel], xx_a[i_sel]])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = -np.ones(len(rows), dtype=np.float64)
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([vals, diag])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Linear initial guess along source→target axis.
    src_c = np.array(np.nonzero(src), dtype=np.float64).mean(axis=1)
    tgt_c = np.array(np.nonzero(tgt), dtype=np.float64).mean(axis=1)
    direction = tgt_c - src_c
    length = np.linalg.norm(direction)
    if length > 1e-9:
        direction /= length
        proj = (interior_coords - src_c).dot(direction)
        x0 = np.clip(proj / length, 0.0, 1.0)
    else:
        x0 = np.full(n, 0.5)

    diag_safe = np.where(diag > 0, diag, 1.0)
    M = LinearOperator(
        (n, n), matvec=lambda r, _d=diag_safe: r / _d, dtype=np.float64,
    )

    u_int: np.ndarray
    try:
        import pyamg  # type: ignore[import]
        logger.info(
            "Solving Laplace (PyAMG): n_unknowns=%d  tol=%.1e  maxiter=%d",
            n, tol, maxiter,
        )
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(A)
        residuals: list[float] = []
        u_int = ml.solve(b, x0=x0.copy(), tol=tol, maxiter=maxiter,
                         residuals=residuals, accel="cg")
        dt = time.perf_counter() - t0
        final_res = residuals[-1] if residuals else float("nan")
        if final_res <= tol * (np.linalg.norm(b) or 1.0):
            logger.info(
                "PyAMG converged in %d V-cycles, %.1fs  (final_res=%.2e).",
                len(residuals), dt, final_res,
            )
        else:
            logger.warning(
                "PyAMG reached maxiter=%d without full convergence "
                "(%d V-cycles, final_res=%.2e, %.1fs).",
                maxiter, len(residuals), final_res, dt,
            )
    except ImportError:
        logger.info(
            "PyAMG not installed — falling back to SciPy CG + Jacobi.  "
            "Install with: pip install pyamg"
        )
        logger.info(
            "Solving Laplace (CG): n_unknowns=%d  tol=%.1e  maxiter=%d",
            n, tol, maxiter,
        )
        t0 = time.perf_counter()
        try:
            u_int, info = cg(A, b, x0=x0, rtol=tol, maxiter=maxiter, M=M)
        except TypeError:
            u_int, info = cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        dt = time.perf_counter() - t0
        if info > 0:
            logger.warning(
                "CG did not fully converge after %d iterations (info=%d, %.1fs).",
                maxiter, info, dt,
            )
        elif info < 0:
            logger.error("CG illegal input: info=%d", info)
        else:
            logger.info("CG converged in %.1fs.", dt)

    u = np.zeros(area.shape, dtype=np.float32)
    u[interior] = u_int.astype(np.float32)
    u[tgt] = 1.0
    return u


# ── Field computation ────────────────────────────────────────────────────────

def compute_water_harmonic_field(
    area: np.ndarray,
    src:  np.ndarray,
    tgt:  np.ndarray,
    grad_sigma: float = DEFAULT_GRAD_SIGMA,
    cg_tol:     float = DEFAULT_CG_TOL,
    cg_maxiter: int   = DEFAULT_CG_MAXITER,
) -> dict:
    """Solve the Laplace equation on the water-conducting domain and compute
    the unit gradient field.

    Returns a dict with keys ``u``, ``vec``, ``speed``, ``area``,
    ``source``, ``target``.
    """
    area = np.asarray(area, dtype=bool)
    src  = np.asarray(src,  dtype=bool) & area
    tgt  = np.asarray(tgt,  dtype=bool) & area

    area = _largest_connected_component(area, src, tgt)
    src  = src & area
    tgt  = tgt & area

    u_raw = solve_harmonic_potential(area, src, tgt, tol=cg_tol, maxiter=cg_maxiter)

    if grad_sigma > 0.0:
        u = _masked_gaussian(u_raw, area, sigma=grad_sigma).astype(np.float32)
        u[src] = 0.0
        u[tgt] = 1.0
        u[~area] = 0.0
    else:
        u = u_raw.astype(np.float32, copy=False)

    gz, gy, gx = np.gradient(u)
    grad = np.stack([gz, gy, gx], axis=-1).astype(np.float32)
    grad[~area] = 0.0

    speed = np.linalg.norm(grad, axis=-1).astype(np.float32)
    safe  = speed > 1e-9
    unit  = np.zeros_like(grad)
    unit[safe] = grad[safe] / speed[safe, None]

    logger.info(
        "Water harmonic field: area=%d voxels  |∇u| mean=%.4g  range=[%.3g, %.3g]"
        " (p50=%.3g, p95=%.3g)",
        int(area.sum()),
        float(speed[area].mean()) if area.any() else 0.0,
        float(speed[area].min()) if area.any() else 0.0,
        float(speed[area].max()) if area.any() else 0.0,
        float(np.percentile(speed[area], 50)) if area.any() else 0.0,
        float(np.percentile(speed[area], 95)) if area.any() else 0.0,
    )

    return {
        "u":      u,
        "vec":    unit,
        "speed":  speed,
        "area":   area,
        "source": src,
        "target": tgt,
    }


# ── Passage-time maps ────────────────────────────────────────────────────────

def build_passage_time_simple(u: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Return ``T_simple = 1 - u``, clamped to [0, 1] inside ``area``.

    Values outside the domain are set to 0.
    """
    T = np.zeros_like(u, dtype=np.float32)
    T[area] = np.clip(1.0 - u[area], 0.0, 1.0).astype(np.float32)
    return T


def build_passage_time_eikonal(
    u:       np.ndarray,
    area:    np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Solve the eikonal equation :math:`|\\nabla T| = 1 / |\\nabla u|` inside
    ``area``, seeded at the source boundary (``u ≈ 0``).

    Uses ``scikit-fmm`` (fast marching method).  The returned map gives the
    true "front-arrival time" for a front that travels at local speed
    :math:`|\\nabla u|`.  Normalised to [0, 1].

    .. note::
        This step can be slow (several minutes) on large volumes.
        Pass ``skip_eikonal=True`` to :func:`build_water_harmonic_field_from_tifs`
        to skip it.
    """
    print(
        "Computing Eikonal passage time (slow — use --skip-eikonal to skip)…"
    )
    try:
        import skfmm  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "scikit-fmm is required for Eikonal passage time.  "
            "Install with: pip install scikit-fmm"
        ) from exc

    u = np.asarray(u, dtype=np.float64)
    area = np.asarray(area, dtype=bool)

    # Signed-distance seed: negative at source (u≈0), positive elsewhere.
    phi = np.where(area, u - 0.0, -1.0).astype(np.float64)
    # Mask out the non-domain: skfmm ignores masked cells.
    phi_masked = np.ma.MaskedArray(phi, mask=~area)

    # Speed field: |∇u|, floor at 1e-6 to avoid division by zero.
    gz, gy, gx = np.gradient(u)
    speed_raw = np.sqrt(gz**2 + gy**2 + gx**2).astype(np.float64)
    speed_raw[~area] = 0.0
    speed_safe = np.where(area, np.maximum(speed_raw, 1e-6), 1e-6)

    dx = float(spacing[0])  # assume isotropic or use first axis spacing
    T_raw = skfmm.travel_time(phi_masked, speed_safe, dx=dx)

    # Convert masked array → ndarray.
    T_arr = np.zeros(area.shape, dtype=np.float32)
    if hasattr(T_raw, "filled"):
        T_arr[area] = np.asarray(T_raw.filled(0.0)[area], dtype=np.float32)
    else:
        T_arr[area] = np.asarray(T_raw[area], dtype=np.float32)

    # Normalise to [0, 1] inside the domain, clipped at p99 to avoid outlier
    # compression (near-zero |∇u| voxels can push T to very large values).
    T_inside = T_arr[area]
    T_max  = float(T_inside.max())  if T_inside.size > 0 else 1.0
    T_norm = float(np.percentile(T_inside, 99)) if T_inside.size > 0 else T_max
    if T_norm < 1e-9:
        T_norm = T_max
    T_arr[area] = np.clip(T_inside / T_norm, 0.0, 1.0)

    logger.info(
        "Eikonal passage time done.  T_max=%.3g  T_p99=%.3g (normalisation scale)",
        T_max, T_norm,
    )
    return T_arr


# ── TIFF-based convenience entry point ──────────────────────────────────────

def build_water_harmonic_field_from_tifs(
    area_path:    Path,
    src_path:     Path,
    tgt_path:     Path,
    spacing:      tuple[float, float, float] = (1.0, 1.0, 1.0),
    grad_sigma:   float = DEFAULT_GRAD_SIGMA,
    cg_tol:       float = DEFAULT_CG_TOL,
    cg_maxiter:   int   = DEFAULT_CG_MAXITER,
    skip_eikonal: bool  = False,
) -> dict:
    """Load three binary 8-bit TIFF masks (threshold 127) and build the full
    water harmonic field including passage-time maps.

    Returns a dict with keys:
        ``u``, ``vec``, ``speed``, ``area``, ``source``, ``target``,
        ``T_simple``, and (unless ``skip_eikonal``) ``T_eikonal``.
    """
    from tifffile import imread

    area = imread(str(area_path))
    src  = imread(str(src_path))
    tgt  = imread(str(tgt_path))

    if area.shape != src.shape or area.shape != tgt.shape:
        raise ValueError(
            f"Water masks have mismatched shapes: "
            f"area={area.shape}  source={src.shape}  target={tgt.shape}"
        )

    area_b = area > 127
    src_b  = src  > 127
    tgt_b  = tgt  > 127
    logger.info(
        "Water masks loaded: shape=%s  area=%d  source=%d  target=%d",
        area_b.shape, int(area_b.sum()), int(src_b.sum()), int(tgt_b.sum()),
    )

    field = compute_water_harmonic_field(
        area_b, src_b, tgt_b,
        grad_sigma=grad_sigma,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
    )

    field["T_simple"] = build_passage_time_simple(field["u"], field["area"])

    if not skip_eikonal:
        try:
            field["T_eikonal"] = build_passage_time_eikonal(
                field["u"], field["area"], spacing=spacing
            )
        except ImportError as exc:
            logger.warning("Eikonal skipped: %s", exc)
    else:
        logger.info("Eikonal passage time skipped (--skip-eikonal).")

    return field


# ── Persistence ─────────────────────────────────────────────────────────────

def save_water_harmonic_field(field: dict, path: Path) -> None:
    """Save the output of :func:`compute_water_harmonic_field` (+ passage
    times) to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "u":      field["u"].astype(np.float32),
        "vec":    field["vec"].astype(np.float16),
        "speed":  field["speed"].astype(np.float32),
        "area":   field["area"].astype(np.uint8),
        "source": field["source"].astype(np.uint8),
        "target": field["target"].astype(np.uint8),
        "T_simple": field["T_simple"].astype(np.float32),
    }
    if "T_eikonal" in field:
        arrays["T_eikonal"] = field["T_eikonal"].astype(np.float32)
    np.savez_compressed(str(path), **arrays)
    logger.info("Water harmonic field saved → %s", path)


def load_water_harmonic_field(path: Path) -> dict:
    """Reverse of :func:`save_water_harmonic_field`."""
    z = np.load(str(path))
    area = z["area"].astype(bool)
    out = {
        "u":        z["u"].astype(np.float32),
        "vec":      z["vec"].astype(np.float32),
        "speed":    z["speed"].astype(np.float32),
        "area":     area,
        "source":   z["source"].astype(bool),
        "target":   z["target"].astype(bool),
        "T_simple": z["T_simple"].astype(np.float32),
    }
    if "T_eikonal" in z.files:
        out["T_eikonal"] = z["T_eikonal"].astype(np.float32)
    return out


# ── Stream-line track building ───────────────────────────────────────────────

def build_water_stream_tracks(
    vec_field:   np.ndarray,
    u_field:     np.ndarray,
    source_vol:  np.ndarray,
    *,
    spacing:     tuple[float, float, float] = (1.0, 1.0, 1.0),
    n_seeds:     int   = DEFAULT_N_SEEDS,
    step_vox:    float = DEFAULT_STEP_VOX,
    max_steps:   int   = DEFAULT_MAX_STEPS,
    u_stop:      float = 0.98,
) -> dict | None:
    """Integrate stream lines along ``vec_field`` (∇u direction) from seeds
    on ``source_vol`` to the target (u ≈ 1).

    Seeds are selected with the same k-medoids strategy as the Dijkstra
    crown tracks in ``pipeline.py``.

    Returns a dict with the same keys as ``_build_crown_dijkstra_tracks``
    (``segs_start``, ``segs_end``, ``scores``, ``boost``, ``path_lengths``,
    ``pts``, ``dirs``, ``stride``, ``format``) so the VTP-writing pipeline
    is unchanged.

    Returns ``None`` if no tracks could be integrated.
    """
    from scipy.spatial import cKDTree

    vec   = np.asarray(vec_field,  dtype=np.float32)   # (Z, Y, X, 3)
    u     = np.asarray(u_field,    dtype=np.float32)   # (Z, Y, X)
    area  = (u > 1e-6) | (np.linalg.norm(vec, axis=-1) > 1e-6)
    src_m = np.asarray(source_vol) >= 127

    sp_z, sp_y, sp_x = float(spacing[0]), float(spacing[1]), float(spacing[2])
    Z, Y, X = u.shape
    long_axis = int(np.argmax([Z, Y, X]))

    # ── Seed selection: k-medoids on source voxels ───────────────────────
    src_coords = np.argwhere(src_m & area).astype(np.float64)
    n_att = len(src_coords)
    if n_att == 0:
        logger.warning("build_water_stream_tracks: no source voxels in domain.")
        return None

    n_desired = min(int(n_seeds), n_att)
    if n_att <= n_desired:
        selected = src_coords.astype(np.int64)
    else:
        rng = np.random.default_rng(42)
        init_idx = rng.choice(n_att, size=n_desired, replace=False)
        centroids = src_coords[init_idx].copy()
        crown_tree = CKDTree = cKDTree(src_coords)  # noqa: N806 — stored for snapping
        for _ in range(5):
            tree = cKDTree(centroids)
            _, assign = tree.query(src_coords, k=1)
            counts = np.bincount(assign, minlength=n_desired)
            sum_c = np.empty((n_desired, 3), dtype=np.float64)
            for _d in range(3):
                sum_c[:, _d] = np.bincount(
                    assign, weights=src_coords[:, _d], minlength=n_desired,
                )
            empty = counts == 0
            means = np.empty_like(centroids)
            means[~empty] = sum_c[~empty] / counts[~empty, np.newaxis]
            if empty.any():
                d_any, _ = tree.query(src_coords, k=1)
                n_emp = int(empty.sum())
                far_idx = np.argpartition(d_any, -n_emp)[-n_emp:]
                means[empty] = src_coords[far_idx]
            _, snap_idx = crown_tree.query(means, k=1)
            centroids = src_coords[snap_idx]
        selected = centroids.astype(np.int64)

    logger.info(
        "Stream tracks: %d seeds (k-medoids) from %d source voxels",
        len(selected), n_att,
    )

    # ── Vectorised Euler integration ──────────────────────────────────────
    # Loop over time steps; all N particles advance simultaneously at each step.
    # At step t: read field at current positions → update positions → record.
    sp_arr   = np.array([sp_z, sp_y, sp_x], dtype=np.float32)
    centre   = np.array([Z * sp_z, Y * sp_y, X * sp_x], dtype=np.float32) * 0.5

    # Transpose vec to (3, Z, Y, X) for O(1) channel read via fancy indexing.
    vec_czyx = np.ascontiguousarray(vec.transpose(3, 0, 1, 2))  # (3, Z, Y, X)

    N  = len(selected)
    lo = np.zeros(3, dtype=np.int32)
    hi = np.array([Z - 1, Y - 1, X - 1], dtype=np.int32)

    # Initialise all particle positions in world-space — shape (N, 3).
    pos = selected.astype(np.float32) * sp_arr[None, :]

    # Store every frame: all_paths[p, t] = world position of particle p at step t.
    all_paths = np.empty((N, max_steps + 1, 3), dtype=np.float32)
    all_paths[:, 0] = pos

    alive     = np.ones(N, dtype=bool)
    stop_step = np.full(N, max_steps, dtype=np.int32)

    for t in range(max_steps):
        # Trilinear interpolation of the vector field: avoids the sharp angular
        # kinks that nearest-neighbor lookup produces at voxel-boundary crossings.
        pos_vox = pos / sp_arr[None, :]                                   # (N, 3) fractional vox
        i0 = np.clip(np.floor(pos_vox).astype(np.int32), lo, hi)         # lower corner
        i1 = np.clip(i0 + 1, lo, hi)                                     # upper corner (clamped)
        frac = (pos_vox - i0).astype(np.float32)                          # (N, 3) in [0, 1)
        fz = frac[:, 0:1]; fy = frac[:, 1:2]; fx = frac[:, 2:3]          # (N, 1) each
        z0, y0, x0 = i0[:, 0], i0[:, 1], i0[:, 2]
        z1, y1, x1 = i1[:, 0], i1[:, 1], i1[:, 2]

        v = (
            vec_czyx[:, z0, y0, x0].T * ((1 - fz) * (1 - fy) * (1 - fx)) +
            vec_czyx[:, z0, y0, x1].T * ((1 - fz) * (1 - fy) *      fx ) +
            vec_czyx[:, z0, y1, x0].T * ((1 - fz) *      fy  * (1 - fx)) +
            vec_czyx[:, z0, y1, x1].T * ((1 - fz) *      fy  *      fx ) +
            vec_czyx[:, z1, y0, x0].T * (     fz  * (1 - fy) * (1 - fx)) +
            vec_czyx[:, z1, y0, x1].T * (     fz  * (1 - fy) *      fx ) +
            vec_czyx[:, z1, y1, x0].T * (     fz  *      fy  * (1 - fx)) +
            vec_czyx[:, z1, y1, x1].T * (     fz  *      fy  *      fx )
        )  # (N, 3)

        # Nearest-neighbor is fine for the stopping criterion (u is slowly varying).
        u_vals = u[z0, y0, x0]                                            # (N,)

        # Update alive; record step at which each particle stops.
        still_ok   = alive & (np.linalg.norm(v, axis=-1) > 1e-6) & (u_vals < u_stop)
        newly_dead = alive & ~still_ok
        stop_step[newly_dead] = t
        alive = still_ok

        # Dead particles contribute zero displacement (stay at last position).
        v[~alive] = 0.0

        # Advance all particles (one step).
        pos = pos + v * (step_vox * sp_arr[None, :])
        all_paths[:, t + 1] = pos

        if not alive.any():
            break

    # ── Reconstruct per-particle segments ────────────────────────────────
    segs_start_list:   list[np.ndarray] = []
    segs_end_list:     list[np.ndarray] = []
    scores_list:       list[np.ndarray] = []
    path_lengths_list: list[int]        = []

    for i in range(N):
        T = int(stop_step[i])   # number of valid steps for this particle
        if T < 1:
            continue
        arr     = all_paths[i, :T + 1]   # (T+1, 3)
        s_start = arr[:-1]               # (T, 3)
        s_end   = arr[1:]                # (T, 3)

        # Score = centripetal angle.
        seg_dir  = s_end - s_start
        nrm      = np.linalg.norm(seg_dir, axis=-1, keepdims=True)
        safe     = nrm.ravel() > 1e-9
        seg_unit = np.zeros_like(seg_dir)
        seg_unit[safe] = seg_dir[safe] / nrm[safe]

        mid    = (s_start + s_end) * 0.5
        toward = centre[None, :] - mid
        toward[:, long_axis] = 0.0
        tn     = np.linalg.norm(toward, axis=-1, keepdims=True)
        safe_t = tn.ravel() > 1e-9
        toward_unit = np.zeros_like(toward)
        toward_unit[safe_t] = toward[safe_t] / tn[safe_t]

        dot   = np.einsum("ij,ij->i", seg_unit, toward_unit)
        score = (np.arccos(np.clip(dot, 0.0, 1.0)) / (0.5 * np.pi)).astype(np.float32)

        segs_start_list.append(s_start)
        segs_end_list.append(s_end)
        scores_list.append(score)
        path_lengths_list.append(T)

    if not segs_start_list:
        logger.warning("build_water_stream_tracks: no tracks produced.")
        return None

    segs_start  = np.concatenate(segs_start_list, axis=0).astype(np.float32)
    segs_end    = np.concatenate(segs_end_list,   axis=0).astype(np.float32)
    scores_arr  = np.concatenate(scores_list,     axis=0).astype(np.float32)
    path_lengths = np.asarray(path_lengths_list, dtype=np.int32)
    boost       = np.zeros(len(segs_start), dtype=np.float32)

    dirs_arr  = segs_end - segs_start
    nrm_arr   = np.linalg.norm(dirs_arr, axis=-1, keepdims=True)
    safe_arr  = nrm_arr.ravel() > 1e-9
    dirs_unit = np.zeros_like(dirs_arr)
    dirs_unit[safe_arr] = dirs_arr[safe_arr] / nrm_arr[safe_arr]

    logger.info(
        "Stream tracks: %d segments from %d paths.",
        len(segs_start), len(segs_start_list),
    )

    return {
        "segs_start":   segs_start,
        "segs_end":     segs_end,
        "scores":       scores_arr,
        "boost":        boost,
        "path_lengths": path_lengths,
        "pts":          segs_start,
        "dirs":         dirs_unit.astype(np.float32),
        "stride":       1,
        "format":       "stream_v1",
    }


# ── Dual-arrow field building ────────────────────────────────────────────────

def build_dual_arrows_filtered(
    water_vec:  np.ndarray,
    water_area: np.ndarray,
    air_vec:    np.ndarray,
    air_area:   np.ndarray,
    fine_stride: int  = 4,
    separation_vox: int = DEFAULT_SEPARATION_VOX,
    spacing:    tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[dict, dict]:
    """Build two spatially-separated arrow-field dicts (water and air) for
    the dual-overlay mode.

    Each dict has the same structure as ``_build_arrow_field``'s return
    value (``pts``, ``dirs``, ``scores``, ``boost``, ``stride``), but uses
    pre-computed gradient vectors instead of recomputing them.

    Spatial separation: voxels within ``separation_vox`` of the other
    domain (measured by EDT of that domain's mask) are dropped, preventing
    visual overlap between water and air arrows.

    Parameters
    ----------
    water_vec / air_vec : (Z, Y, X, 3) float arrays
        Unit gradient vectors from the respective harmonic fields.
    water_area / air_area : (Z, Y, X) bool arrays
        Domain masks.
    fine_stride : int
        Sub-sampling step for arrow placement (same as ``_build_arrow_field``).
    separation_vox : int
        Minimum voxel distance from the opposite domain.
    spacing : (sz, sy, sx) float
        Physical voxel size in world units.
    """
    from scipy.ndimage import distance_transform_edt

    w_vec  = np.asarray(water_vec,  dtype=np.float32)
    w_area = np.asarray(water_area, dtype=bool)
    a_vec  = np.asarray(air_vec,    dtype=np.float32)
    a_area = np.asarray(air_area,   dtype=bool)

    sp_z, sp_y, sp_x = float(spacing[0]), float(spacing[1]), float(spacing[2])

    # EDT of each mask (distance to nearest voxel *in* that domain).
    # We use distance to the *interior* of the opposite domain.
    water_to_air_dist  = distance_transform_edt(~a_area).astype(np.float32)
    air_to_water_dist  = distance_transform_edt(~w_area).astype(np.float32)

    long_axis = int(np.argmax(w_area.shape))

    def _subsample(vec, area, exclude_dist, name: str) -> dict:
        fs = max(1, int(fine_stride))
        sv = vec[::fs, ::fs, ::fs]
        sa = area[::fs, ::fs, ::fs]
        sd = exclude_dist[::fs, ::fs, ::fs]

        Z, Y, X = sv.shape[:3]
        cells = [fs * sp_z, fs * sp_y, fs * sp_x]
        zs = np.arange(Z, dtype=np.float32) * cells[0]
        ys = np.arange(Y, dtype=np.float32) * cells[1]
        xs = np.arange(X, dtype=np.float32) * cells[2]
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        pts_world = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
        dirs_flat = sv.reshape(-1, 3)
        area_flat = sa.reshape(-1)
        dist_flat = sd.reshape(-1)

        nrm = np.linalg.norm(dirs_flat, axis=-1)
        inside = area_flat & (nrm > 1e-6) & (dist_flat >= separation_vox)

        pts_k   = pts_world[inside].astype(np.float32)
        dirs_k  = dirs_flat[inside].astype(np.float32)
        nrm_k   = nrm[inside]
        safe_k  = nrm_k > 1e-9
        dirs_unit = np.zeros_like(dirs_k)
        dirs_unit[safe_k] = dirs_k[safe_k] / nrm_k[safe_k, None]

        # Score = centripetal angle (same as arrow field).
        spans = [(Z - 1) * cells[0], (Y - 1) * cells[1], (X - 1) * cells[2]]
        centre = np.array([s * 0.5 for s in spans], dtype=np.float32)
        toward = centre[None, :] - pts_k
        toward[:, long_axis] = 0.0
        tn = np.linalg.norm(toward, axis=-1)
        safe_t = tn > 1e-9
        toward_unit = np.zeros_like(toward)
        toward_unit[safe_t] = toward[safe_t] / tn[safe_t, None]
        dot = np.einsum("ij,ij->i", dirs_unit, toward_unit)
        keep = (dot >= 0.0) & safe_t
        pts_k  = pts_k[keep]
        dirs_unit = dirs_unit[keep]
        scores = (np.arccos(np.clip(dot[keep], 0.0, 1.0)) / (0.5 * np.pi)
                  ).astype(np.float32)

        logger.info(
            "Dual arrows [%s]: fine_stride=%d  kept=%d  separated by >=%d vox",
            name, fs, len(pts_k), separation_vox,
        )
        return {
            "pts":    pts_k,
            "dirs":   dirs_unit,
            "scores": scores,
            "boost":  np.zeros(len(pts_k), dtype=np.float32),
            "stride": int(fine_stride),
        }

    water_arrows = _subsample(w_vec, w_area, water_to_air_dist,  "water")
    air_arrows   = _subsample(a_vec, a_area, air_to_water_dist,  "air")

    return water_arrows, air_arrows
