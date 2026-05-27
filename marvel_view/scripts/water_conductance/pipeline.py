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
    DEFAULT_RADIAL_LT_BIN_SCALE,
    DEFAULT_RADIAL_N_RBINS,
    DEFAULT_RADIAL_SIGMA,
    DEFAULT_SPACING,
    TORTUOSITY_CMAP_STOPS,
    TORTUOSITY_VMAX,
    TORTUOSITY_VMIN,
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


def _build_radial_gradient_facet_scalars(
    *,
    paths_domain_path: Path,
    central_axis_path: Path,
    bridges_mesh=None,
    all_mesh=None,
    radius_px: float = DEFAULT_DENSITY_RADIUS_PX,
    step_px: float = DEFAULT_DENSITY_STEP_PX,
    n_rbins: int = DEFAULT_RADIAL_N_RBINS,
    lt_bin_scale: float = DEFAULT_RADIAL_LT_BIN_SCALE,
    upsample: int = DEFAULT_DENSITY_UPSAMPLE,
    sigma_up: float = DEFAULT_RADIAL_SIGMA,
) -> dict:
    """Compute per-cell radial-gradient scalars (centred at 0) for two meshes.

    For each (L, θ) sector the possible-paths voxels are binned by radial
    distance R from the central axis into ``n_rbins`` shells.  A linear
    regression of count vs R gives ``slope_actual``; the scalar is then

        scalar(L, θ) = slope_actual / expected_slope(L) − 1

    where ``expected_slope(L)`` is the mean of all non-zero slopes in the
    same L-band.  The result is centred at 0: positive values indicate
    sectors with *better-than-average* radial connectivity (more paths
    toward the centre than expected), negative values flag bottlenecks.

    The scalar grid is upsampled and smoothed with the same parameters
    as :func:`_build_density_facet_scalars`, then sampled at face
    centroids of both meshes.
    """
    import numpy as np
    import tifffile
    from scipy.ndimage import gaussian_filter, zoom

    cx, cy = _parse_central_axis(Path(central_axis_path))
    logger.info(
        "Radial gradient: central axis at (X=%.3f, Y=%.3f) px", cx, cy,
    )

    paths = tifffile.imread(str(paths_domain_path))
    if paths.ndim != 3:
        raise ValueError(
            f"Radial gradient: expected 3-D volume, got shape {paths.shape}"
        )
    nz, ny, nx = paths.shape

    # ── Angular and longitudinal bin counts (larger bins via lt_bin_scale) ─
    eff_step = float(step_px) * float(lt_bin_scale)
    n_L = max(1, int(round(nz / eff_step)))
    n_T = max(1, int(round(2.0 * np.pi * float(radius_px) / eff_step)))
    n_R = max(2, int(n_rbins))
    logger.info(
        "Radial gradient: L bins=%d  θ bins=%d  R bins=%d  lt_bin_scale=%.2f  (vol shape=%s)",
        n_L, n_T, n_R, float(lt_bin_scale), paths.shape,
    )

    # ── Collect all possible-paths voxels and compute their coords ──────
    paths_bool = paths > 0
    zz, yy, xx = np.nonzero(paths_bool)
    if zz.size == 0:
        logger.warning(
            "Radial gradient: empty possible-paths domain at %s — "
            "returning zero scalars.", paths_domain_path,
        )
        return {"bridges": None, "all": None, "grid": np.zeros((n_L, n_T), dtype=np.float32)}

    # Longitudinal bin
    Lf = zz.astype(np.float64) / max(nz - 1, 1)
    Lb = np.clip((Lf * n_L).astype(np.int64), 0, n_L - 1)

    # Azimuthal bin
    dy = yy.astype(np.float64) - float(cy)
    dx = xx.astype(np.float64) - float(cx)
    theta = np.arctan2(dy, dx)
    Tf = (theta + np.pi) / (2.0 * np.pi)
    Tb = np.clip((Tf * n_T).astype(np.int64), 0, n_T - 1)

    # ── Radial distance per voxel (px from central axis) ─────────────────
    R = np.sqrt(dy ** 2 + dx ** 2).astype(np.float64)

    # Flat sector index: position in the (n_L × n_T) grid
    flat_2d = (Lb * n_T + Tb).astype(np.int64)              # (n_voxels,)

    # ── Per-sector R_min and R_max (vectorised scatter) ───────────────────
    # Each (L, θ) sector gets its own radial range from its actual voxels.
    sector_R_max = np.full(n_L * n_T, -np.inf)
    np.maximum.at(sector_R_max, flat_2d, R)
    sector_R_min = np.full(n_L * n_T, np.inf)
    np.minimum.at(sector_R_min, flat_2d, R)
    sector_R_max   = sector_R_max.reshape(n_L, n_T)
    sector_R_min   = sector_R_min.reshape(n_L, n_T)
    sector_R_range = (sector_R_max - sector_R_min).clip(min=0.0)  # (n_L, n_T)

    # ── Per-voxel bin index within its sector ────────────────────────────
    # Bin 0 = outermost (sclerenchyma side, R close to sector_R_max)
    # Bin n_R-1 = innermost (stele side,    R close to sector_R_min)
    R_max_v   = sector_R_max.ravel()[flat_2d]               # (n_voxels,)
    R_range_v = sector_R_range.ravel()[flat_2d]             # (n_voxels,)
    # Normalised position: 0 at outer edge, 1 at inner edge
    with np.errstate(divide="ignore", invalid="ignore"):
        Rf_sect = np.where(
            R_range_v > 1e-6,
            (R_max_v - R) / (R_range_v + 1e-12),
            0.0,
        )
    Rb_sect = np.clip((Rf_sect * n_R).astype(np.int64), 0, n_R - 1)

    # ── 3-D histogram (L, θ, radial_bin) ─────────────────────────────────
    flat_3d   = flat_2d * n_R + Rb_sect
    counts_3d = np.bincount(flat_3d, minlength=n_L * n_T * n_R)
    counts_3d = counts_3d.reshape(n_L, n_T, n_R).astype(np.float64)

    # Representative R of each per-sector bin (px):
    #   R_centre[l, t, k] = R_max[l,t] − (k + 0.5)/n_R × R_range[l,t]
    bin_frac    = (np.arange(n_R, dtype=np.float64) + 0.5) / n_R   # (n_R,)
    R_max_3d    = sector_R_max[:, :, np.newaxis]                    # (n_L, n_T, 1)
    R_range_3d  = sector_R_range[:, :, np.newaxis]                  # (n_L, n_T, 1)
    R_centre_3d = R_max_3d - bin_frac * R_range_3d                  # (n_L, n_T, n_R)
    R_centre_3d = np.where(R_centre_3d > 0, R_centre_3d, 1.0)      # avoid /0

    # ── Sanity check: aggregate all per-sector bin counts ────────────────
    # 1. Sum bin counts across all valid sectors  → one "aggregate sector"
    # 2. R_min / R_max come from the median of per-sector values
    #    (robust to corner sectors with tiny R ranges)
    # 3. Re-bin voxels into N_SANITY bins using the median range,
    #    normalise count/R_centre, then normalise so mean = 1.
    # The result should be roughly flat if the count/R geometry is correct.
    _N_SANITY  = 20
    _valid_2d  = counts_3d.sum(axis=2) > 0                              # (n_L, n_T)
    _rmin_med  = float(np.median(sector_R_min[_valid_2d])) if _valid_2d.any() else 0.0
    _rmax_med  = float(np.median(sector_R_max[_valid_2d])) if _valid_2d.any() else 1.0
    if _valid_2d.any():
        _rrange = _rmax_med - _rmin_med
        if _rrange > 1e-6:
            _bf_s    = (np.arange(_N_SANITY, dtype=np.float64) + 0.5) / _N_SANITY
            _Rf_s    = (R - _rmin_med) / (_rrange + 1e-12)             # 0=inner 1=outer
            _Rf_s    = np.clip(1.0 - _Rf_s, 0.0, 1.0)                 # flip: 0=outer 1=inner
            _Rb_s    = np.clip((_Rf_s * _N_SANITY).astype(np.int64), 0, _N_SANITY - 1)
            _gc_s    = np.bincount(_Rb_s, minlength=_N_SANITY).astype(np.float64)
            _Rc_s    = _rmax_med - _bf_s * _rrange                     # (N_SANITY,)
            _Rc_s    = np.where(_Rc_s > 0, _Rc_s, 1.0)
            _nc_s    = _gc_s / _Rc_s
            _mean_s  = float(_nc_s.mean()) if _nc_s.mean() > 0 else 1.0
            _nc_norm = _nc_s / _mean_s
            _xs      = np.linspace(0.0, 1.0, _N_SANITY)
            _xd      = _xs - _xs.mean()
            _sanity_slope = float(
                ((_nc_norm - _nc_norm.mean()) * _xd).sum() / ((_xd ** 2).sum() + 1e-30)
            )
            logger.info(
                "Radial gradient SANITY  "
                "(%d bins; aggregate of %d valid sectors; "
                "R_min_med=%.1f  R_max_med=%.1f px; "
                "count/R normalised so mean=1; slope=%+.4f; "
                "r0=outer/sclerenchyma  r%d=inner/stele — "
                "flat ≈ uniform density): %s",
                _N_SANITY, int(_valid_2d.sum()), _rmin_med, _rmax_med,
                _sanity_slope, _N_SANITY - 1,
                "  ".join(f"r{i}={v:.3f}" for i, v in enumerate(_nc_norm)),
            )

    # ── Normalise per-sector bins by their representative R ──────────────
    # In the uniform camembert model count ∝ R, so count/R is flat.
    norm_counts = counts_3d / R_centre_3d                           # (n_L, n_T, n_R)

    # ── Normalise per-sector so the mean density = 1 ─────────────────────
    # This makes the regression slopes dimensionless and comparable across
    # sectors and datasets: a slope of +0.1 means "density rises by 10 % of
    # the sector mean per unit of normalised radius".
    nc_sector_mean = norm_counts.mean(axis=2, keepdims=True)        # (n_L, n_T, 1)
    # Guard against empty sectors (mean ≈ 0).
    nc_sector_mean = np.where(nc_sector_mean > 1e-30, nc_sector_mean, 1.0)
    norm_counts = norm_counts / nc_sector_mean                      # mean = 1 per sector

    # ── OLS regression of norm_counts vs normalised radius ──────────────
    # x ∈ [0, 1]: 0 = outermost shell (sclerenchyma side, R ≈ R_max)
    #             1 = innermost shell  (stele side,        R ≈ R_min)
    # y mean = 1 per sector (normalised above).
    # Positive slope → density increases inward → more paths near stele.
    # Negative slope → density decreases inward → more paths near sclerenchyma.
    # Slope unit: fraction of sector-mean density per unit normalised radius.
    x     = np.linspace(0.0, 1.0, n_R, dtype=np.float64)           # (n_R,)
    x_dev = x - x.mean()                                           # (n_R,)
    denom = float((x_dev ** 2).sum())

    nc_mean = norm_counts.mean(axis=2)                             # (n_L, n_T)
    nc_dev  = norm_counts - nc_mean[..., np.newaxis]               # (n_L, n_T, n_R)
    slopes  = (nc_dev * x_dev).sum(axis=2) / (denom + 1e-30)      # (n_L, n_T)

    # Valid mask: sectors with at least one voxel
    total_per_sector = counts_3d.sum(axis=2)                # (n_L, n_T)
    valid_mask = total_per_sector > 0

    # Scalar = raw slope (no expected_slope normalisation).
    # Sectors without data get 0.
    scalar_grid = np.where(valid_mask, slopes, 0.0).astype(np.float32)

    logger.info(
        "Radial gradient: slope range=[%.4f, %.4f]  "
        "scalar range=[%.4f, %.4f]  valid sectors=%d/%d",
        float(slopes[valid_mask].min()) if valid_mask.any() else 0.0,
        float(slopes[valid_mask].max()) if valid_mask.any() else 0.0,
        float(scalar_grid.min()), float(scalar_grid.max()),
        int(valid_mask.sum()), n_L * n_T,
    )
    if valid_mask.any():
        _sv = slopes[valid_mask]
        _pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        _vals = np.percentile(_sv, _pcts)
        logger.info(
            "Radial gradient slope distribution (valid sectors only):\n"
            "  p1=%+.3f  p5=%+.3f  p10=%+.3f  p25=%+.3f  p50=%+.3f\n"
            "  p75=%+.3f  p90=%+.3f  p95=%+.3f  p99=%+.3f\n"
            "  mean=%+.3f  std=%.3f",
            *_vals, float(_sv.mean()), float(_sv.std()),
        )
        # Histogram with ~12 equal-width bins across [p1, p99] to show shape.
        _lo, _hi = float(_vals[0]), float(_vals[-1])
        if _hi > _lo:
            _edges = np.linspace(_lo, _hi, 21)
            _counts, _ = np.histogram(_sv, bins=_edges)
            _bar_max = int(_counts.max()) or 1
            _bar_w = 30
            _hist_lines = ["  Histogram of slopes  [p1 … p99]:"]
            for _i, (_c, _e0, _e1) in enumerate(
                zip(_counts, _edges[:-1], _edges[1:])
            ):
                _bar = "#" * int(round(_c / _bar_max * _bar_w))
                _hist_lines.append(
                    f"  [{_e0:+.3f}, {_e1:+.3f})  {_bar:<{_bar_w}}  {_c}"
                )
            logger.info("\n".join(_hist_lines))

    # ── Upsample × upsample and Gaussian-smooth (wrap on θ) ──────────────
    up = max(1, int(upsample))
    grid_up = zoom(scalar_grid, zoom=up, order=1, mode="nearest")
    pad_t = int(np.ceil(3.0 * float(sigma_up)))
    if pad_t > 0:
        padded = np.concatenate(
            [grid_up[:, -pad_t:], grid_up, grid_up[:, :pad_t]], axis=1
        )
    else:
        padded = grid_up
    smoothed = gaussian_filter(padded, sigma=(sigma_up, sigma_up), mode="nearest")
    if pad_t > 0:
        smoothed = smoothed[:, pad_t:pad_t + grid_up.shape[1]]
    grid_smooth = smoothed.astype(np.float32)
    n_L_up, n_T_up = grid_smooth.shape
    logger.info(
        "Radial gradient: upsampled to (%d, %d), sigma=%.2f px",
        n_L_up, n_T_up, float(sigma_up),
    )
    # Distribution of the smoothed grid values (what the mesh colours will be).
    _gs_flat = grid_smooth.ravel()
    _gs_pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    _gs_vals = np.percentile(_gs_flat, _gs_pcts)
    logger.info(
        "Radial gradient smoothed-grid distribution:\n"
        "  p1=%+.3f  p5=%+.3f  p10=%+.3f  p25=%+.3f  p50=%+.3f\n"
        "  p75=%+.3f  p90=%+.3f  p95=%+.3f  p99=%+.3f\n"
        "  mean=%+.3f  std=%.3f",
        *_gs_vals, float(_gs_flat.mean()), float(_gs_flat.std()),
    )

    # ── Sample at face centroids (same helper as density) ─────────────────
    def _scalars_for(mesh) -> "np.ndarray | None":
        if mesh is None:
            return None
        try:
            import vtk
            from vtkmodules.util.numpy_support import vtk_to_numpy
            try:
                pd = mesh.dataset
            except AttributeError:
                pd = mesh.polydata()
            cc = vtk.vtkCellCenters()
            cc.SetInputData(pd)
            cc.Update()
            out = cc.GetOutput()
            cents = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Radial gradient: vtkCellCenters failed (%s); falling back.", exc,
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
                cents = np.array(
                    [verts[np.asarray(c, dtype=np.int64)].mean(axis=0) for c in cells],
                    dtype=np.float64,
                )
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
            "Radial gradient: bridges-mesh scalars shape=%s  range=[%.3f, %.3f]",
            bridges_sc.shape, float(bridges_sc.min()), float(bridges_sc.max()),
        )
    if all_sc is not None:
        logger.info(
            "Radial gradient: all-mesh scalars shape=%s  range=[%.3f, %.3f]",
            all_sc.shape, float(all_sc.min()), float(all_sc.max()),
        )
    return {
        "bridges": bridges_sc,
        "all":     all_sc,
        "grid":    grid_smooth,
        "n_L":     int(n_L),
        "n_T":     int(n_T),
        "n_R":     int(n_R),
        "R_max":   _rmax_med,
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
    # Base cost = 1 inside domain, near-infinite outside.  MCP_Geometric
    # scales each edge by the Euclidean distance between voxel centres,
    # so the cumulative cost is approximately the geodesic arc-length.
    #
    # With a perfectly uniform cost field, many paths share the exact same
    # total length → the traceback breaks ties arbitrarily, producing
    # lateral oscillations (zigzags) along otherwise straight segments.
    # To eliminate this, we add a small distance-to-wall gradient: voxels
    # near the centre of the domain cost 5 % less than those at the
    # boundary.  This preference is too weak to force any geometric detour
    # (a 1-voxel detour costs ~1 extra unit while the savings are < 0.05),
    # but it cleanly breaks ties in favour of the anatomically meaningful
    # centre-of-tissue path.
    from scipy.ndimage import distance_transform_edt as _dist_edt
    dist_to_wall = _dist_edt(paths_m).astype(np.float64)
    max_d = float(dist_to_wall.max()) or 1.0
    _cost_inside = 1.0 - 0.05 * (dist_to_wall / max_d)   # 0.95 … 1.0
    costs = np.where(paths_m, _cost_inside, 1e12).astype(np.float64)
    logger.info(
        "MCP cost field: inside range [%.4f, %.4f]  (centre-preference "
        "tie-breaker, 5%% gradient).",
        float(_cost_inside[paths_m].min()),
        float(_cost_inside[paths_m].max()),
    )
    del dist_to_wall, _cost_inside  # free memory before the heavy propagation
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


def _compute_per_path_tortuosity(
    segs_start,
    segs_end,
    path_lengths,
    cx: float,
    cy: float,
) -> "np.ndarray":
    """Compute per-path *radial* tortuosity = arc_length / |R_start − R_end|.

    The denominator is the absolute change in radial distance from the root
    central axis: this is the minimum possible displacement a water molecule
    must cover to travel from its start radius to its end radius.  A
    perfectly straight radial path gives tortuosity = 1; any detour
    (circumferential, tangential, back-tracking) inflates the ratio.

    Parameters
    ----------
    segs_start, segs_end : (n_segs, 3) float32 arrays — world coords (z, y, x).
    path_lengths         : (n_paths,) int64 — number of segments per path.
    cx, cy               : float — central-axis x/y image coordinates.

    Returns
    -------
    (n_paths,) float32 — tortuosity values ≥ 1.0.
    """
    import numpy as np

    segs_start   = np.asarray(segs_start,   dtype=np.float64)
    segs_end     = np.asarray(segs_end,     dtype=np.float64)
    path_lengths = np.asarray(path_lengths, dtype=np.int64)

    n_paths     = len(path_lengths)
    path_starts = np.concatenate(([0], np.cumsum(path_lengths)[:-1])).astype(np.intp)

    # Arc length: sum of segment Euclidean distances per path.
    seg_lengths  = np.linalg.norm(segs_end - segs_start, axis=1)  # (n_segs,)
    path_arc_len = np.add.reduceat(seg_lengths, path_starts)       # (n_paths,)

    # Radial distance from the central axis at each path endpoint.
    # Coordinates are (z, y, x) → columns 1 and 2 are y and x.
    first_pts = segs_start[path_starts]                            # (n_paths, 3)
    last_idx  = (path_starts + path_lengths - 1).astype(np.intp)
    last_pts  = segs_end[last_idx]                                 # (n_paths, 3)

    r_start = np.sqrt((first_pts[:, 1] - cy) ** 2
                      + (first_pts[:, 2] - cx) ** 2)              # (n_paths,)
    r_end   = np.sqrt((last_pts[:, 1]  - cy) ** 2
                      + (last_pts[:, 2] - cx) ** 2)               # (n_paths,)

    # |ΔR|: minimum radial displacement a perfectly radial path would cover.
    radial_delta = np.abs(r_start - r_end)

    # Paths with very small radial displacement (circumferential / near-flat)
    # are genuinely tortuous from a water-transport perspective; clamp to
    # 1.0 voxel minimum so we do not produce inf/nan.
    small = radial_delta < 1.0
    if small.any():
        logger.info(
            "Tortuosity: %d path(s) have |ΔR| < 1 voxel (near-circumferential); "
            "clamping denominator to 1.0 for those.",
            int(small.sum()),
        )
    radial_delta = np.where(small, 1.0, radial_delta)

    tortuosity = (path_arc_len / radial_delta).astype(np.float32)
    tortuosity = np.maximum(tortuosity, 1.0)  # floating-point noise clamp

    # Log distribution statistics + histogram.
    med = float(np.median(tortuosity))
    mad = float(np.median(np.abs(tortuosity - med)))
    logger.info(
        "Tortuosity distribution (%d paths): "
        "min=%.3f  max=%.3f  mean=%.3f  std=%.3f  median=%.3f  MAD=%.3f",
        n_paths,
        float(tortuosity.min()), float(tortuosity.max()),
        float(tortuosity.mean()), float(tortuosity.std()),
        med, mad,
    )
    _hist_edges  = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, np.inf]
    _hist_labels = ["1.0-1.05", "1.05-1.1", "1.1-1.2", "1.2-1.3",
                    "1.3-1.5", "1.5-1.75", "1.75-2.0", "2.0+"]
    _counts = np.histogram(tortuosity, bins=_hist_edges)[0]
    logger.info(
        "Tortuosity histogram: %s",
        "  ".join(f"{lbl}:{cnt}" for lbl, cnt in zip(_hist_labels, _counts)),
    )
    return tortuosity


def _build_tracks_polydata(data, *, cx: float | None = None, cy: float | None = None):
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

    # Optional per-path tortuosity = arc_length / (R_start − R_end).
    if cx is not None and cy is not None:
        tort = _compute_per_path_tortuosity(segs_start, segs_end, path_lengths, cx, cy)
        tort_vtk = numpy_to_vtk(tort, deep=True, array_type=vtk.VTK_FLOAT)
        tort_vtk.SetName("tortuosity")
        pd.GetCellData().AddArray(tort_vtk)
        pd.GetCellData().SetActiveScalars("tortuosity")
    else:
        pd.GetCellData().SetActiveScalars("boost")

    logger.info(
        "Crown tracks polydata: %d polylines, %d points",
        pd.GetNumberOfCells(), pd.GetNumberOfPoints(),
    )
    return pd


def _write_tracks_vtp(data, vtp_path: Path, *,
                      cx: float | None = None, cy: float | None = None) -> None:
    """Build a ``vtkPolyData`` of crown-track polylines and write it as a
    binary ``.vtp`` (XML VTK PolyData) file.  At load time the viewer can
    read this with ``vtkXMLPolyDataReader`` — a near-memcopy — instead of
    reconstructing all cells in Python.

    When *cx* and *cy* are provided the polydata will also carry a
    ``"tortuosity"`` CellData array (see :func:`_compute_per_path_tortuosity`).
    """
    import vtk

    pd = _build_tracks_polydata(data, cx=cx, cy=cy)
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


def _dp_mask(pts: "np.ndarray", epsilon: float) -> "np.ndarray":
    """Return a boolean keep-mask for one polyline using Douglas-Peucker.

    ``pts`` is an (N, 3) float64 array of control points; the first and last
    points are always kept.  The algorithm is iterative (stack-based) to
    avoid Python recursion limits on long paths.
    """
    import numpy as np

    n = len(pts)
    kept = np.zeros(n, dtype=bool)
    kept[0] = kept[-1] = True
    if n <= 2:
        return kept
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg  = pts[i1] - pts[i0]
        seg_sq = float(np.dot(seg, seg))
        inner  = pts[i0 + 1:i1]  # (k, 3)
        if seg_sq < 1e-12:
            dists = np.linalg.norm(inner - pts[i0], axis=1)
        else:
            t = np.clip(
                np.einsum("ij,j->i", inner - pts[i0], seg) / seg_sq,
                0.0, 1.0,
            )
            proj  = pts[i0] + t[:, np.newaxis] * seg
            dists = np.linalg.norm(inner - proj, axis=1)
        mi_local = int(np.argmax(dists))
        if dists[mi_local] > epsilon:
            mi = i0 + 1 + mi_local
            kept[mi] = True
            stack.append((i0, mi))
            stack.append((mi, i1))
    return kept


def _write_splined_tracks_vtp(
    vtp_path: Path,
    splined_vtp_path: Path,
    *,
    n_subdivisions: int = 64,
    dp_epsilon: float = 1.0,
) -> None:
    """Read the raw polylines VTP, simplify with Douglas-Peucker, run
    ``vtkSplineFilter`` and write the result as a binary ``.vtp`` cache.

    **Why Douglas-Peucker before splining?**
    Dijkstra paths on a voxel grid step between adjacent voxel centres at
    right angles, producing staircase control-point sequences.  Splining
    *through* those points yields a smooth curve that still faithfully
    follows the staircase shape.  D-P (epsilon = 1 voxel by default)
    removes collinear / near-collinear intermediate voxels first; the
    remaining skeleton points capture only real turns.  The spline then
    interpolates between those skeleton points, giving genuinely curved
    paths rather than rounded staircases.

    ``vtkSplineFilter`` preserves CellData (one value per polyline cell),
    so the ``"tortuosity"``, ``"boost"`` and ``"scores"`` arrays survive.
    """
    import numpy as np
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    pd_raw = reader.GetOutput()
    if pd_raw.GetNumberOfCells() == 0:
        logger.warning(
            "_write_splined_tracks_vtp: polylines VTP has no cells: %s", vtp_path,
        )
        return

    # ── Douglas-Peucker simplification ───────────────────────────────────
    # Remove near-collinear control points (voxel-grid staircase) so the
    # spline interpolates between genuine turning-points only.
    if dp_epsilon > 0.0:
        pts     = vtk_to_numpy(pd_raw.GetPoints().GetData()).astype(np.float64)
        ca      = pd_raw.GetLines()
        offsets = vtk_to_numpy(ca.GetOffsetsArray()).astype(np.intp)
        conn    = vtk_to_numpy(ca.GetConnectivityArray()).astype(np.intp)
        n_cells = pd_raw.GetNumberOfCells()

        new_conn_parts: list = []
        new_off = [0]
        n_kept_total = 0
        for ci in range(n_cells):
            s, e   = offsets[ci], offsets[ci + 1]
            ids    = conn[s:e]
            if len(ids) <= 2:
                new_conn_parts.append(ids)
            else:
                mask = _dp_mask(pts[ids], dp_epsilon)
                new_conn_parts.append(ids[mask])
            n_kept_total += len(new_conn_parts[-1])
            new_off.append(n_kept_total)

        new_conn_arr = np.concatenate(new_conn_parts).astype(np.int64)
        new_off_arr  = np.array(new_off, dtype=np.int64)

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_to_vtk(pts, deep=True))
        ca_dp = vtk.vtkCellArray()
        ca_dp.SetData(
            numpy_to_vtk(new_off_arr,  deep=True, array_type=vtk.VTK_ID_TYPE),
            numpy_to_vtk(new_conn_arr, deep=True, array_type=vtk.VTK_ID_TYPE),
        )
        pd_dp = vtk.vtkPolyData()
        pd_dp.SetPoints(vtk_pts)
        pd_dp.SetLines(ca_dp)
        pd_dp.GetCellData().ShallowCopy(pd_raw.GetCellData())

        n_orig = int(offsets[-1])
        logger.info(
            "Douglas-Peucker (eps=%.2f): %d → %d control points (%.1f%% kept).",
            dp_epsilon, n_orig, n_kept_total,
            100.0 * n_kept_total / max(1, n_orig),
        )
        pd_raw = pd_dp

    # Use Kochanek spline with moderate tension (0.5) to limit overshoot at
    # sharp turns while keeping smooth curves between D-P skeleton points.
    # Tension=0 is Catmull-Rom (can overshoot); tension=1 is fully linear.
    _ksp = vtk.vtkKochanekSpline()
    _ksp.SetDefaultTension(0.5)
    spl = vtk.vtkSplineFilter()
    spl.SetSpline(_ksp)
    spl.SetInputData(pd_raw)
    spl.SetSubdivideToSpecified()
    spl.SetNumberOfSubdivisions(n_subdivisions)
    spl.Update()
    pd_sm = spl.GetOutput()

    splined_vtp_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(splined_vtp_path))
    writer.SetInputData(pd_sm)
    writer.SetDataModeToBinary()
    writer.Write()
    logger.info(
        "Splined crown tracks VTP written to %s  (%d polylines, %d points, "
        "%d subdivisions)",
        splined_vtp_path,
        pd_sm.GetNumberOfCells(),
        pd_sm.GetNumberOfPoints(),
        n_subdivisions,
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
    splined_vtp_path: Path | None = None,
    central_axis_path: Path | None = None,
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
                    # Pre-splined polylines VTP (64 subdivisions).
                    # When present, the viewer/movie skip vtkSplineFilter.
                    "splined_vtp_path": splined_vtp_path,
                    # Path to the central-axis coordinates file (forwarded
                    # so callers can parse cx/cy for colourmap purposes).
                    "central_axis_path": central_axis_path,
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


# ── Normals-cache helpers ─────────────────────────────────────────────────────

_LAME2_NORMALS_META_NAME = "_normals_meta.json"


def _lame2_cache_subdir(base_dir: Path, n_steps: int) -> Path:
    """Return the n_steps-specific subdirectory inside *base_dir*.

    Using a subdirectory keyed by step-count lets different lame2 VTP
    variants (e.g. 1000-step test vs 2290-step full) share the same base
    cache directory without conflicts.
    """
    return base_dir / f"steps_{n_steps}"


def _lame2_step_vtp_path(normals_dir: Path, step: int) -> Path:
    return normals_dir / f"step_{step:04d}.vtp"


def _compute_normals_one_step(args):
    """Worker: compact unused points, run vtkPolyDataNormals, write to disk."""
    import vtk
    s, pd, normals_dir = args
    # Each step_pd shares the full packed point array — strip unused points
    # first so the VTP only contains the points referenced by this step's cells.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd)
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.PointMergingOff()
    cleaner.Update()
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputConnection(cleaner.GetOutputPort())
    nf.ComputePointNormalsOn()
    nf.ComputeCellNormalsOff()
    nf.SplittingOff()
    nf.ConsistencyOn()
    nf.SetFeatureAngle(60.0)
    nf.Update()
    pd_with_normals = nf.GetOutput()
    p = _lame2_step_vtp_path(normals_dir, s)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(p))
    w.SetInputData(pd_with_normals)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()
    return s


def save_lame2_normals_cache(step_pds: list, normals_dir: Path) -> None:
    """Compute point normals on each per-step PolyData and write them to
    *normals_dir/steps_{n}/* as individual binary VTPs.

    A small ``_normals_meta.json`` sentinel is written last so that an
    interrupted save is never mistaken for a complete one.
    """
    import os
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_steps = len(step_pds)
    normals_dir = _lame2_cache_subdir(normals_dir, n_steps)   # versioned subdir
    normals_dir.mkdir(parents=True, exist_ok=True)
    n_steps   = len(step_pds)
    n_workers = min(os.cpu_count() or 4, n_steps)
    logger.info(
        "Lame2 normals: processing %d steps with %d parallel workers …",
        n_steps, n_workers,
    )

    n_written = 0
    _t0 = _time.perf_counter()

    tasks = [(s, pd, normals_dir) for s, pd in enumerate(step_pds) if pd is not None]
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_compute_normals_one_step, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            fut.result()   # re-raise any worker exception
            n_written += 1
            _elapsed = _time.perf_counter() - _t0
            _rate    = _elapsed / n_written
            _remain  = _rate * (n_steps - n_written)
            logger.info(
                "Lame2 normals: step %d/%d  —  elapsed %.1f s  —  ETA ~%.0f s",
                n_written, n_steps, _elapsed, _remain,
            )

    import json as _json
    (_normals_meta_path := normals_dir / _LAME2_NORMALS_META_NAME).write_text(
        _json.dumps({"n_steps": n_steps, "n_written": n_written}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Lame2 normals cache: wrote %d/%d step VTPs → %s",
        n_written, n_steps, normals_dir,
    )


def load_lame2_normals_cache(normals_dir: Path, n_steps: int):
    """Load pre-baked per-step PolyDatas from *normals_dir/steps_{n_steps}/*.

    Returns ``(step_pds, actor)`` in the same form as
    :func:`_build_lames_step_polydatas`, or ``(None, None)`` if the cache
    is absent / incomplete.
    """
    import vtk

    normals_dir = _lame2_cache_subdir(normals_dir, n_steps)   # versioned subdir
    meta_path = normals_dir / _LAME2_NORMALS_META_NAME
    if not meta_path.exists():
        logger.info("Lame2 normals cache not found at %s — will compute.", normals_dir)
        return None, None

    try:
        import json as _json
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read lame2 normals meta: %s", exc)
        return None, None

    cached_n = int(meta.get("n_steps", 0))
    if cached_n != n_steps:
        logger.warning(
            "Lame2 normals cache at %s has n_steps=%d but expected %d — ignoring.",
            normals_dir, cached_n, n_steps,
        )
        return None, None

    step_pds: list = [None] * n_steps
    n_loaded = 0
    for s in range(n_steps):
        p = _lame2_step_vtp_path(normals_dir, s)
        if not p.exists():
            continue
        try:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(p))
            reader.Update()
            pd = reader.GetOutput()
            if pd is not None and pd.GetNumberOfCells() > 0:
                step_pds[s] = pd
                n_loaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load step VTP %s: %s", p, exc)

    if n_loaded == 0:
        logger.warning("Lame2 normals cache is empty — ignoring.")
        return None, None

    logger.info(
        "Lame2 normals cache loaded: %d/%d steps from %s",
        n_loaded, n_steps, normals_dir,
    )

    # Build a single actor with the same configuration as
    # _build_lames_step_polydatas (geometry swapped at each tick).
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
