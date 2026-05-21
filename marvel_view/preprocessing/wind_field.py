"""
Wind / gas-particles visualization — pre-processing module.

Builds a *harmonic potential* :math:`u` on the air-mask volume (``wind_area``)
with Dirichlet boundary conditions

* :math:`u = 0` on the **source** mask (``wind_source``, at numpy axis-0 = high X),
* :math:`u = 1` on the **target** mask (``wind_target``, at numpy axis-0 = low  X),

and homogeneous Neumann conditions on the cell walls (= voxels outside the air
mask).  The resulting field is :math:`C^\\infty` inside the domain, so its
gradient is a clean, smooth navigation field that:

* points along the open passages from source to target,
* vanishes in dead-end branches (Neumann tubes carry no stationary flux),
* shares the flux between parallel channels in proportion to their conductance.

The viewer's "wind" toggles advect short-lived particles along
:math:`+\\nabla u`  (CH₄, source side → target side reversed below) or
:math:`-\\nabla u` (O₂) using **pre-computed trajectory templates** so the
runtime cost is essentially a table-lookup per frame.

Outputs of :func:`build_wind_field` (saved as a single ``.npz``):

* ``vec`` — float32 array ``(Z, Y, X, 3)`` of unit gradient vectors
  :math:`\\hat n = \\nabla u / \\lVert\\nabla u\\rVert` (zero outside the mask).
* ``speed`` — float32 array ``(Z, Y, X)`` of :math:`\\lVert\\nabla u\\rVert`
  (raw, not normalised); the **reliability** of the local flow.
* ``wall_dist`` — float32 ``(Z, Y, X)``, Euclidean distance transform of the
  air mask (= distance to the nearest wall).
* ``area`` — bool ``(Z, Y, X)``, the air mask actually used (largest connected
  component that touches both source and target).
* ``source`` / ``target`` — bool ``(Z, Y, X)``, the seed masks restricted to
  the same connected component.

The two outputs of :func:`build_particle_templates` (one ``.npz`` per species):

* ``positions`` — float32 ``(N_TEMPLATES, N_FRAMES, 3)``, voxel coords
  ``(x_numpy, y_numpy, z_numpy)`` = ``(axis0, axis1, axis2)`` of each frame.
* ``alive`` — uint8 ``(N_TEMPLATES, N_FRAMES)``, 1 if particle is alive at
  that frame, 0 otherwise (dead particles freeze in place and are rendered
  invisible by the viewer).
* ``meta`` — small JSON dict (``species``, ``n_frames``, ``fps``,
  ``life_frames``, ``direction_sign``, ``shape``).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_FPS: int = 25
DEFAULT_SECONDS: float = 40.0
DEFAULT_LIFESPAN_S: float = 4.0
DEFAULT_N_TEMPLATES: int = 4000  # shared trajectory pool (per species)

# Laplace solver
# 1e-3 is sufficient for a visual flow field (saves ~10× CG iterations).
# PyAMG is tried first; it solves in 10–30 iterations instead of ~500.
DEFAULT_CG_TOL: float = 1e-3
DEFAULT_CG_MAXITER: int = 4000
DEFAULT_GRAD_SIGMA: float = 0.9  # gaussian smoothing of u before gradient

# Advection
# Speed of a particle in voxels per frame, applied as
#     dx = direction_sign * unit_vec * (speed_voxels_per_frame * relative_speed)
# where ``relative_speed`` is ``|∇u|`` normalised by its 95th percentile.
DEFAULT_SPEED_VOX_PER_FRAME: float = 1.4

# Seeding / death
# O2 walls "consume": probability of dying per frame ~ wall_kill_rate * exp(-wd/wall_kill_tau)
DEFAULT_O2_WALL_KILL_TAU: float = 4.0      # voxels
DEFAULT_O2_WALL_KILL_RATE: float = 0.05    # per frame, at the wall
# CH4 walls "emit": seed pdf weight = (wall_emit_bias / (wall_dist + 1))**wall_emit_power
DEFAULT_CH4_WALL_EMIT_TAU: float = 4.0
DEFAULT_CH4_WALL_EMIT_BIAS: float = 1.3    # multiplier on the wall-prefer seeding


# ── Laplace solver ──────────────────────────────────────────────────────────

def _largest_connected_component(area: np.ndarray,
                                 src: np.ndarray,
                                 tgt: np.ndarray) -> np.ndarray:
    """Keep only the connected component of ``area`` that touches **both**
    ``src`` and ``tgt``.  Returns a refined boolean mask of the same shape."""
    from scipy.ndimage import label

    labelled, n_lab = label(area.astype(bool))
    if n_lab == 0:
        raise ValueError("wind_area mask is empty.")

    src_labels = set(int(v) for v in np.unique(labelled[src]) if v != 0)
    tgt_labels = set(int(v) for v in np.unique(labelled[tgt]) if v != 0)
    valid = src_labels & tgt_labels
    if not valid:
        raise ValueError(
            "No connected component of wind_area touches both source and target."
        )
    keep = np.zeros_like(labelled, dtype=bool)
    for v in valid:
        keep |= (labelled == v)
    n_kept = int(keep.sum())
    n_total = int(area.sum())
    logger.info(
        "Connected components: kept %d/%d voxels (%.1f%%) in %d traversing labels.",
        n_kept, n_total, 100.0 * n_kept / max(n_total, 1), len(valid),
    )
    return keep


def solve_harmonic_potential(
    area: np.ndarray,
    src:  np.ndarray,
    tgt:  np.ndarray,
    tol: float = DEFAULT_CG_TOL,
    maxiter: int = DEFAULT_CG_MAXITER,
) -> np.ndarray:
    """Solve :math:`\\nabla^2 u = 0` on ``area`` with ``u=0`` on ``src`` and
    ``u=1`` on ``tgt`` (homogeneous Neumann on walls, i.e. outside ``area``).

    Uses a 7-point finite-difference stencil and ``scipy.sparse.linalg.cg``
    with diagonal (Jacobi) preconditioning.  Walls (=voxels not in ``area``)
    are simply omitted from the stencil, which is the discrete equivalent
    of a zero-flux (Neumann) boundary condition.

    Returns
    -------
    u : float32 ``(Z, Y, X)``
        Potential field, zero outside ``area``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator, cg

    area = np.asarray(area, dtype=bool)
    src = np.asarray(src,  dtype=bool) & area
    tgt = np.asarray(tgt,  dtype=bool) & area
    interior = area & ~src & ~tgt
    n = int(interior.sum())
    if n == 0:
        raise ValueError("solve_harmonic_potential: no interior voxels.")

    # Linear indexing for interior voxels.
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
        active_local = ok[nbr_in_area]  # positions in [0,n) whose neighbour is in area
        # Diagonal += 1 for each active neighbour
        np.add.at(diag, active_local, 1.0)

        # Of these, partition into {interior, src, tgt}
        zz_a = interior_coords[active_local, 0] + dz
        yy_a = interior_coords[active_local, 1] + dy
        xx_a = interior_coords[active_local, 2] + dx
        nbr_is_tgt = tgt[zz_a, yy_a, xx_a]
        nbr_is_int = interior[zz_a, yy_a, xx_a]

        # tgt neighbour contributes +u_tgt = +1 to the RHS
        np.add.at(b, active_local[nbr_is_tgt], 1.0)
        # src neighbour contributes 0 (u_src = 0) — nothing to do.

        # Interior neighbour: off-diagonal entry of value -1.
        i_sel = np.where(nbr_is_int)[0]
        rows_list.append(active_local[i_sel])
        cols_list.append(idx[zz_a[i_sel], yy_a[i_sel], xx_a[i_sel]])

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = -np.ones(len(rows), dtype=np.float64)
    # Add diagonal entries.
    rows = np.concatenate([rows, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([cols, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([vals, diag])
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Linear initial guess along the source→target axis (best axis = the one
    # along which source and target centroids differ the most).
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

    # Jacobi preconditioner.
    diag_safe = np.where(diag > 0, diag, 1.0)
    M = LinearOperator(
        (n, n), matvec=lambda r, _d=diag_safe: r / _d, dtype=np.float64,
    )

    # ── Solver: try PyAMG (smoothed aggregation) first, fall back to CG ────
    # PyAMG converges in 10–30 iterations vs ~500 for plain CG + Jacobi on
    # a 3-D Laplacian, with identical accuracy at the requested tolerance.
    # ``pip install pyamg`` once to unlock the fast path.
    u_int: np.ndarray
    try:
        import pyamg  # type: ignore[import]
        logger.info(
            "Solving Laplace (PyAMG smoothed-aggregation): "
            "n_unknowns=%d  tol=%.1e  maxiter=%d",
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
            "PyAMG not installed — falling back to SciPy CG + Jacobi."
            "  Install with: pip install pyamg"
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
    u[src] = 0.0
    u[tgt] = 1.0
    # Outside area: remains 0 (we'll mask wherever we use it).
    return u


# ── Vector / speed / wall field extraction ──────────────────────────────────

def _masked_gaussian(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth ``arr`` while ignoring contributions from ``~mask``.

    Implements the Normalized Convolution trick: smooth ``arr * mask`` and
    ``mask`` separately, then divide.  Prevents the zeros outside the mask
    from bleeding in near the boundary."""
    from scipy.ndimage import gaussian_filter
    a = arr.astype(np.float32, copy=False)
    m = mask.astype(np.float32, copy=False)
    num = gaussian_filter(a * m, sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter(m,     sigma=sigma, mode="constant", cval=0.0)
    out = np.zeros_like(a)
    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out


def compute_wind_field(
    area:   np.ndarray,
    src:    np.ndarray,
    tgt:    np.ndarray,
    grad_sigma: float = DEFAULT_GRAD_SIGMA,
    cg_tol:     float = DEFAULT_CG_TOL,
    cg_maxiter: int   = DEFAULT_CG_MAXITER,
) -> dict:
    """End-to-end: connected-component clip, Laplace solve, smoothing,
    gradient extraction, wall-distance transform.

    Parameters
    ----------
    area, src, tgt : bool arrays of identical shape
        Binary masks (any input dtype is cast to bool).
    grad_sigma : float
        Sigma (voxels) for the masked Gaussian smoothing of ``u`` applied
        before the gradient is computed.  Use ~0.8–1.0 to suppress the
        slight near-wall staircase of the 7-point stencil.

    Returns a dict with keys ``vec``, ``speed``, ``wall_dist``, ``area``,
    ``source``, ``target``, ``shape``.
    """
    from scipy.ndimage import distance_transform_edt

    area = np.asarray(area, dtype=bool)
    src  = np.asarray(src,  dtype=bool) & area
    tgt  = np.asarray(tgt,  dtype=bool) & area

    # Restrict to the largest "traversing" connected component.
    area = _largest_connected_component(area, src, tgt)
    src  = src & area
    tgt  = tgt & area

    # Harmonic potential.
    u_raw = solve_harmonic_potential(area, src, tgt, tol=cg_tol, maxiter=cg_maxiter)

    # Masked smoothing of u inside the domain.
    if grad_sigma > 0.0:
        u = _masked_gaussian(u_raw, area, sigma=grad_sigma).astype(np.float32)
        # Re-impose Dirichlet (smoothing may have slightly altered the seeds).
        u[src] = 0.0
        u[tgt] = 1.0
        u[~area] = 0.0
    else:
        u = u_raw.astype(np.float32, copy=False)

    # Central-difference gradient on the smoothed u.
    gz, gy, gx = np.gradient(u)  # each (Z, Y, X)
    grad = np.stack([gz, gy, gx], axis=-1).astype(np.float32)  # (Z, Y, X, 3)
    # Zero out anywhere not in area.
    grad[~area] = 0.0

    # Magnitude = local conductance / reliability.
    speed = np.linalg.norm(grad, axis=-1).astype(np.float32)
    # Unit-direction vector field.
    safe = speed > 1e-9
    unit = np.zeros_like(grad)
    unit[safe] = grad[safe] / speed[safe, None]

    # Distance to nearest wall (used by seeding / wall-death).
    wall_dist = distance_transform_edt(area).astype(np.float32)

    logger.info(
        "Wind field: area=%d voxels  |∇u| range=[%.3g, %.3g] (p50=%.3g, p95=%.3g)",
        int(area.sum()),
        float(speed[area].min()) if area.any() else 0.0,
        float(speed[area].max()) if area.any() else 0.0,
        float(np.percentile(speed[area], 50)) if area.any() else 0.0,
        float(np.percentile(speed[area], 95)) if area.any() else 0.0,
    )

    return {
        "vec":       unit,
        "speed":     speed,
        "wall_dist": wall_dist,
        "area":      area,
        "source":    src,
        "target":    tgt,
        "shape":     area.shape,
        "u":         u,
    }


# ── Particle templates ──────────────────────────────────────────────────────

def _build_seed_pdf_o2(field: dict) -> tuple[np.ndarray, np.ndarray]:
    """O₂: seed proportional to ``speed`` (|∇u|) — particles spawn where the
    flow is most reliable.  Returns (flat_indices, flat_cumulative_weights)."""
    area = field["area"]
    w = field["speed"].copy()
    w[~area] = 0.0
    # Floor: avoid zero PDF in mid-trunk where ∇u is locally small.
    w_med = float(np.percentile(w[area], 50)) if area.any() else 1.0
    w[area] = np.maximum(w[area], 0.10 * w_med)
    flat = w.ravel()
    return _build_cdf(flat)


def _build_seed_pdf_ch4(field: dict, bias: float = DEFAULT_CH4_WALL_EMIT_BIAS) -> tuple[np.ndarray, np.ndarray]:
    """CH₄: seed preferentially near walls (emission by the surrounding
    medium).  Weight = ``bias / (wall_dist + 1)``, but only inside ``area``.
    Floor so that "tronc" spawns aren't completely suppressed."""
    area = field["area"]
    w = np.zeros(area.shape, dtype=np.float32)
    w[area] = bias / (field["wall_dist"][area] + 1.0)
    # Floor so particles don't form a thin sheet at the wall only.
    w_med = float(np.percentile(w[area], 50)) if area.any() else 1.0
    w[area] = np.maximum(w[area], 0.4 * w_med)
    return _build_cdf(w.ravel())


def _build_cdf(weights_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a (indices, cumulative) pair for weighted sampling of nonzero
    flat positions.  ``cumulative`` is normalised to [0, 1]."""
    nz = np.flatnonzero(weights_flat > 0)
    w = weights_flat[nz].astype(np.float64)
    cum = np.cumsum(w)
    total = cum[-1] if cum.size else 1.0
    cum /= total
    return nz, cum.astype(np.float32)


def _sample_seeds(cdf_pair: tuple[np.ndarray, np.ndarray],
                  shape: tuple,
                  n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` voxel coordinates from a flat-CDF.  Returns (n, 3) float32
    voxel coords plus a small uniform jitter so particles don't all start on
    integer cells (smoother visual)."""
    nz, cum = cdf_pair
    r = rng.random(n).astype(np.float32)
    pick = np.searchsorted(cum, r)
    pick = np.clip(pick, 0, len(nz) - 1)
    flat = nz[pick]
    sz, sy, sx = shape
    z = (flat // (sy * sx)).astype(np.float32)
    y = ((flat % (sy * sx)) // sx).astype(np.float32)
    x = (flat % sx).astype(np.float32)
    # Sub-voxel jitter.
    z += rng.random(n).astype(np.float32) - 0.5
    y += rng.random(n).astype(np.float32) - 0.5
    x += rng.random(n).astype(np.float32) - 0.5
    return np.stack([z, y, x], axis=-1)


def _trilinear_batch(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Trilinearly sample a scalar or vector ``field`` at ``pts`` (N, 3).

    ``field.shape`` = ``(Z, Y, X)`` (scalar) or ``(Z, Y, X, C)`` (vector).
    Out-of-bounds samples return 0.
    """
    from scipy.ndimage import map_coordinates
    coords = pts.T  # (3, N)
    if field.ndim == 3:
        return map_coordinates(field, coords, order=1,
                               mode="constant", cval=0.0).astype(np.float32)
    elif field.ndim == 4:
        C = field.shape[-1]
        out = np.zeros((pts.shape[0], C), dtype=np.float32)
        for c in range(C):
            out[:, c] = map_coordinates(field[..., c], coords, order=1,
                                        mode="constant", cval=0.0)
        return out
    else:
        raise ValueError(f"unexpected field.ndim={field.ndim}")


def build_particle_templates(
    field: dict,
    species: str,
    n_templates: int = DEFAULT_N_TEMPLATES,
    fps: int = DEFAULT_FPS,
    seconds: float = DEFAULT_SECONDS,
    lifespan_s: float = DEFAULT_LIFESPAN_S,
    speed_vox_per_frame: float = DEFAULT_SPEED_VOX_PER_FRAME,
    seed: int = 0,
) -> dict:
    """Build the (positions, alive) template tables for one species.

    Parameters
    ----------
    field : dict
        Output of :func:`compute_wind_field`.
    species : {"o2", "ch4"}
        Selects:

        * direction (``"o2"`` follows ``+∇u`` from source@high-X to target@low-X;
          ``"ch4"`` follows ``-∇u``, i.e. travels backwards source ↔ target),
        * seeding distribution (O₂ ∝ speed, CH₄ ∝ wall preference),
        * wall-interaction (O₂ may die near walls; CH₄ does not).
    """
    species = species.lower()
    if species not in {"o2", "ch4"}:
        raise ValueError(f"species must be 'o2' or 'ch4', got {species!r}")

    rng = np.random.default_rng(seed)
    n_frames = int(round(fps * seconds))
    life_frames = max(2, int(round(fps * lifespan_s)))
    n_lives = max(1, int(np.ceil(n_frames / life_frames)))
    # Pad n_frames so that it is an exact multiple of life_frames (cleaner loop).
    n_frames_padded = n_lives * life_frames
    if n_frames_padded != n_frames:
        n_frames = n_frames_padded

    direction_sign = +1.0 if species == "o2" else -1.0
    if species == "o2":
        cdf = _build_seed_pdf_o2(field)
        wall_kill_rate = DEFAULT_O2_WALL_KILL_RATE
        wall_kill_tau  = DEFAULT_O2_WALL_KILL_TAU
    else:
        cdf = _build_seed_pdf_ch4(field)
        wall_kill_rate = 0.0
        wall_kill_tau  = 1.0  # unused

    shape = field["area"].shape
    area = field["area"]
    speed_field = field["speed"]
    vec_field   = field["vec"]
    wall_dist   = field["wall_dist"]

    # Normalise speed so that the 95th percentile maps to 1.0 — gives
    # ``speed_vox_per_frame`` voxels per frame in the trunk.
    s_p95 = float(np.percentile(speed_field[area], 95)) if area.any() else 1.0
    s_p95 = max(s_p95, 1e-9)

    logger.info(
        "Templates[%s]: %d × %d frames (life=%d) → %.1fs @ %d fps",
        species.upper(), n_templates, n_frames, life_frames,
        n_frames / fps, fps,
    )
    logger.info("  speed scale (p95)=%.4g vox⁻¹ ; vox/frame at trunk=%.2f",
                s_p95, speed_vox_per_frame)

    positions = np.zeros((n_templates, n_frames, 3), dtype=np.float32)
    alive     = np.zeros((n_templates, n_frames), dtype=np.uint8)

    # Initial seed for the first life of each template.
    cur = _sample_seeds(cdf, shape, n_templates, rng)
    is_alive = np.ones(n_templates, dtype=bool)

    t0 = time.perf_counter()
    for life_i in range(n_lives):
        if life_i > 0:
            cur = _sample_seeds(cdf, shape, n_templates, rng)
            is_alive[:] = True

        for sub in range(life_frames):
            f = life_i * life_frames + sub
            positions[:, f, :] = cur
            alive[:, f] = is_alive.astype(np.uint8)

            # Trilinear sample of the unit-direction vector and speed.
            v = _trilinear_batch(vec_field, cur)   # (N, 3)
            sp = _trilinear_batch(speed_field, cur)  # (N,)
            sp_norm = np.clip(sp / s_p95, 0.0, 2.0).astype(np.float32)

            step = v * (direction_sign * speed_vox_per_frame * sp_norm[:, None])
            nxt = cur + step

            # Bounds check.
            sz, sy, sx = shape
            in_bounds = (
                (nxt[:, 0] >= 0) & (nxt[:, 0] < sz - 1) &
                (nxt[:, 1] >= 0) & (nxt[:, 1] < sy - 1) &
                (nxt[:, 2] >= 0) & (nxt[:, 2] < sx - 1)
            )
            # Mask check (must remain in air).
            in_area = np.zeros(n_templates, dtype=bool)
            ok = in_bounds
            if ok.any():
                in_area[ok] = _trilinear_batch(
                    area.astype(np.float32), nxt[ok]) > 0.5
            new_dead = is_alive & (~in_bounds | ~in_area)

            # Stall check: if speed too small, gradually fade out (treat as death).
            stalled = is_alive & (sp_norm < 0.02)
            new_dead |= stalled

            # Wall-death (O2 consumption).
            if wall_kill_rate > 0.0:
                wd = _trilinear_batch(wall_dist, np.where(
                    in_bounds[:, None], nxt, cur))  # (N,)
                p_die = wall_kill_rate * np.exp(-wd / wall_kill_tau)
                roll = rng.random(n_templates).astype(np.float32)
                wall_dies = is_alive & (roll < p_die)
                new_dead |= wall_dies

            is_alive &= ~new_dead
            # Frozen particles keep their last valid position (we already
            # wrote ``cur`` to this frame).
            cur = np.where(is_alive[:, None], nxt, cur).astype(np.float32)

    dt = time.perf_counter() - t0
    n_lived = int(alive.sum())
    total = n_templates * n_frames
    logger.info(
        "Templates[%s] built in %.2fs. Live-frame ratio = %d/%d = %.1f%%",
        species.upper(), dt, n_lived, total, 100.0 * n_lived / max(total, 1),
    )

    return {
        "positions":     positions,           # (T, F, 3)
        "alive":         alive,               # (T, F)
        "n_templates":   n_templates,
        "n_frames":      n_frames,
        "life_frames":   life_frames,
        "fps":           int(fps),
        "species":       species,
        "direction_sign": float(direction_sign),
        "shape":         tuple(int(s) for s in shape),
    }


# ── I/O helpers ─────────────────────────────────────────────────────────────

def save_wind_field(field: dict, path: Path) -> None:
    """Save the output of :func:`compute_wind_field` to a ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        vec=field["vec"].astype(np.float16),       # 16-bit is enough for unit vec
        speed=field["speed"].astype(np.float32),
        wall_dist=field["wall_dist"].astype(np.float32),
        area=field["area"].astype(np.uint8),
        source=field["source"].astype(np.uint8),
        target=field["target"].astype(np.uint8),
        u=field["u"].astype(np.float32),
        shape=np.asarray(field["shape"], dtype=np.int64),
    )
    logger.info("Wind field saved → %s", path)


def load_wind_field(path: Path) -> dict:
    """Reverse of :func:`save_wind_field`."""
    z = np.load(str(path))
    area = z["area"].astype(bool)
    return {
        "vec":       z["vec"].astype(np.float32),
        "speed":     z["speed"].astype(np.float32),
        "wall_dist": z["wall_dist"].astype(np.float32),
        "area":      area,
        "source":    z["source"].astype(bool),
        "target":    z["target"].astype(bool),
        "u":         z["u"].astype(np.float32),
        "shape":     tuple(int(s) for s in z["shape"]),
    }


def save_particle_templates(tpl: dict, path: Path) -> None:
    """Save the output of :func:`build_particle_templates` to a ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_templates":    int(tpl["n_templates"]),
        "n_frames":       int(tpl["n_frames"]),
        "life_frames":    int(tpl["life_frames"]),
        "fps":            int(tpl["fps"]),
        "species":        str(tpl["species"]),
        "direction_sign": float(tpl["direction_sign"]),
        "shape":          list(tpl["shape"]),
    }
    np.savez_compressed(
        str(path),
        positions=tpl["positions"].astype(np.float32),
        alive=tpl["alive"].astype(np.uint8),
        meta=np.asarray([json.dumps(meta)], dtype=object),
    )
    logger.info("Particle templates [%s] saved → %s   (%d × %d frames)",
                tpl["species"].upper(), path,
                tpl["n_templates"], tpl["n_frames"])


def load_particle_templates(path: Path) -> dict:
    """Reverse of :func:`save_particle_templates`."""
    z = np.load(str(path), allow_pickle=True)
    meta = json.loads(str(z["meta"][0]))
    return {
        "positions":     z["positions"].astype(np.float32),
        "alive":         z["alive"].astype(np.uint8),
        "n_templates":   int(meta["n_templates"]),
        "n_frames":      int(meta["n_frames"]),
        "life_frames":   int(meta["life_frames"]),
        "fps":           int(meta["fps"]),
        "species":       str(meta["species"]),
        "direction_sign": float(meta["direction_sign"]),
        "shape":         tuple(int(s) for s in meta["shape"]),
    }


# ── Top-level convenience ───────────────────────────────────────────────────

def build_wind_field_from_tifs(
    wind_area_path:   Path,
    wind_source_path: Path,
    wind_target_path: Path,
    grad_sigma: float = DEFAULT_GRAD_SIGMA,
    cg_tol:     float = DEFAULT_CG_TOL,
    cg_maxiter: int   = DEFAULT_CG_MAXITER,
) -> dict:
    """Load the three TIFF masks (binary 8-bit, threshold 127) and build
    the harmonic wind field.  Convenience wrapper used by the CLI."""
    from tifffile import imread

    area = imread(str(wind_area_path))
    src  = imread(str(wind_source_path))
    tgt  = imread(str(wind_target_path))
    if area.shape != src.shape or area.shape != tgt.shape:
        raise ValueError(
            f"Wind masks have mismatched shapes: area={area.shape}  "
            f"source={src.shape}  target={tgt.shape}"
        )
    area_b = area > 127
    src_b  = src  > 127
    tgt_b  = tgt  > 127
    logger.info(
        "Wind masks loaded: shape=%s  area=%d  source=%d  target=%d",
        area_b.shape, int(area_b.sum()), int(src_b.sum()), int(tgt_b.sum()),
    )
    return compute_wind_field(area_b, src_b, tgt_b,
                              grad_sigma=grad_sigma,
                              cg_tol=cg_tol, cg_maxiter=cg_maxiter)
