#!/usr/bin/env python3
"""
Marvel Water-Conductance – Standalone viewer for ``Wat_Norm_Cortex.tif``.

Loads a single 32-bit "norm" TIFF (cortex water-conductance field), runs
marching cubes at iso-level ``0.2`` directly on the float values, and
opens an interactive vedo window showing the resulting surface as a
dense, opaque grey mesh with subtle bluish specular highlights.

Interactive controls
--------------------
* **Shading** button (top-right) – cycles Phong → Gouraud → Flat.
* **Sliders** (bottom) – ``opacity``, ``ambient``, ``diffuse``,
  ``specular``, ``hue`` (rotates the base colour around the colour
  wheel while keeping its lightness / saturation).

The mesh is intentionally kept at full resolution (no decimation,
``step_size = 1``) so the displayed surface has plenty of vertices.

Usage
-----
::

    python -m marvel_view.scripts.water_conductance
    # or, after pip install -e .:
    marvel-water-conductance
"""
from __future__ import annotations

import argparse
import colorsys
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import aerench_config as acfg
from marvel_view.preprocessing import load_float_volume, mask_to_mesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.water_conductance")


# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_INPUT_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Wat_Norm_Cortex.tif"
DEFAULT_RAW_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Raw.tif"
DEFAULT_LEVEL: float = 0.2
DEFAULT_SPACING = (1.0, 1.0, 1.0)
DEFAULT_SMOOTH_ITER: int = 20  # gentle smoothing, keeps vertices dense

# Répertoire racine des données. Seule variable à exporter dans ~/.bashrc.
_DATA_BASE: Path = Path(
    os.environ.get("MARVEL_DATA_DIR", "/home/rfernandez/Data/Arize/Hollow_test")
)

DEFAULT_VTK_OUTPUT_DIR: Path = _DATA_BASE / "2_Vtk_files"
DEFAULT_POSITIONS_DIR:  Path = _DATA_BASE / "positions"
DEFAULT_MP4_DIR:        Path = _DATA_BASE / "mp4"

# Sub-directory holding the per-level ISO-surface cache (NPZ files).
# Built once; reused across runs that only change phase/bg parameters.
DEFAULT_MEMBRANES_ISO_CACHE_DIR: Path = DEFAULT_VTK_OUTPUT_DIR / "membranes_iso_cache"

# Cached marching-cubes output: built once by
# ``marvel-water-conductance-build-meshes``, loaded by both the viewer
# (``marvel-water-conductance``) and the movie tool (``marvel-water-movie``)
# so they don't have to re-mesh on every launch.
DEFAULT_MESH_CACHE_PATH: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "cortex.vtk"
)

# Alternate mesh: "All watered tissues" extracted from Wat_Norm_All.tif
# using the same iso-level / smoothing as the cortex mesh.  In mesh-view
# mode an extra button lets the user swap between the two.
DEFAULT_ALL_INPUT_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Wat_Norm_All.tif"
DEFAULT_ALL_MESH_CACHE_PATH: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "all.vtk"
)

# Glowing green "pillars" isosurface superimposed on the cortex (Cortical
# bridges) mesh.  Extracted from the 8-bit Pillars.tif at iso=122 and
# rendered translucent with lighting off so it reads as a faint glow.
DEFAULT_PILLARS_TIFF_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Pillars.tif"
DEFAULT_PILLARS_CACHE_PATH: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "pillars_iso.vtk"
)
DEFAULT_PILLARS_LEVEL: float = 220.0
DEFAULT_PILLARS_ALPHA: float = 0.12

# Water "membranes" descending animation.  A packed vtkPolyData (one
# triangle per cell, with per-cell ``step_id``/``column_id``/``time_id``/
# ``rgb`` arrays) and a JSON sidecar are built by
# ``marvel-water-conductance-build-meshes`` via the new
# ``marvel_view.preprocessing.water_membranes`` module.  The viewer toggles
# them on/off with a button and animates by stepping a ``vtkThreshold``
# over the ``step_id`` cell array on a ~60 ms timer.
DEFAULT_MEMBRANES_VTP_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "membranes.vtp"
)
DEFAULT_MEMBRANES_META_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "membranes_meta.json"
)
DEFAULT_MEMBRANES_LABELS_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "membranes_labels.tif"
)
DEFAULT_MEMBRANES_BG_DIST_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Source_Target_Possible_Paths-dist.tif"
)
# Soft cyan-blue base colour (used by the build to seed the per-column
# hue jitter, and by the viewer if rgb cell-data is missing).
DEFAULT_MEMBRANES_COLOR: tuple[int, int, int] = (140, 195, 255)
DEFAULT_MEMBRANES_ALPHA: float = 0.25
# Animation tick in milliseconds (~16 fps).  Each tick advances the
# visible ``step_id`` slice by +1.
MEMBRANE_TICK_MS: int = 60

# Status-line titles shown at the top-center on every rendering change.
STATUS_TITLE_MESH_CORTEX = "Cortical bridges between sclerenchyma and stele"
STATUS_TITLE_MESH_ALL    = "All watered tissues"
STATUS_TITLE_ARROWS_GRID = "Vector field on regular grid"
STATUS_TITLE_ARROWS_TRACKS = "Vector field track from sclerenchyma to surface"
STATUS_DURATION_S: float = 5.0

# Crown geodesic tracks (Arrows view #2) — Dijkstra shortest-paths from
# Source_crown.tif white voxels to Target_crown.tif through the domain
# defined by Source_Target_Possible_Paths.tif.
DEFAULT_SOURCE_CROWN_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Source_crown.tif"
DEFAULT_TARGET_CROWN_PATH: Path = acfg.DEFAULT_INPUT_DIR / "Target_crown.tif"
DEFAULT_PATHS_DOMAIN_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Source_Target_Possible_Paths.tif"
)
DEFAULT_CROWN_TRACKS_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "crown_tracks.npz"
)
# Binary VTK PolyData with one vtkPolyLine per Dijkstra path.  Built by
# ``marvel-water-conductance-build-meshes`` alongside the .npz; the viewer
# loads this at runtime for near-instant actor construction (no Python-side
# InsertNextCell loop, just a VTK binary read).
DEFAULT_CROWN_TRACKS_VTP_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "crown_tracks.vtp"
)
# Pre-computed arrow glyphs for the tracks view: positions, tangents and
# arc-length fraction ``t`` (0→1) for 10 evenly-spaced arrows per splined
# path.  Built by ``marvel-water-conductance-build-meshes`` alongside the
# polylines .vtp so the viewer can skip the expensive Python for-loop at
# render time.
DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "crown_tracks_arrows.vtp"
)
DEFAULT_N_SOURCE_POINTS: int = 20_000
DEFAULT_TRACK_MAX_STEPS: int = 300
DEFAULT_TRACK_STEP_VOX: float = 4.0

# Geodesic distance-to-exterior map used by the "arrows" view: each
# voxel's gradient points toward the deepest interior point along the
# geodesic.  Sub-sampled and cached as an .npz for instant load.
DEFAULT_GEODDIST_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Wat_norm-geoddist.tif"
)
DEFAULT_ARROWS_CACHE_PATH: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "geoddist_arrows.npz"
)
# Dilatation (= convergence) scalar field saved as a float32 multipage TIFF
# alongside the arrow field.  Set to None to skip the export.
DEFAULT_DILATATION_TIFF_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "dilatation_scalar_field.tiff"
)
DEFAULT_ARROW_STRIDE: int = 8          # draw stride on perpendicular axes (voxels)
DEFAULT_LONG_AXIS_STRIDE: int = 15     # draw stride on the long volume axis
DEFAULT_FINE_STRIDE: int = 3           # analyse field (∇, div) at this finer resolution
DEFAULT_ARROW_LENGTH: float = 7.0      # world units (≈ 7 voxels at spacing=1)
DEFAULT_ARROW_THICKNESS: float = 6.3   # vedo.Arrows `thickness=` base multiplier
# Optional translucent "shadow" mesh that is overlaid on every view.
DEFAULT_OVERLAY_TIFF_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Mask_cortex_Gradient.tif"
)
DEFAULT_OVERLAY_CACHE_PATH: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "cortex_gradient_iso.vtk"
)
DEFAULT_OVERLAY_LEVEL: float = 122.5   # iso-value (8-bit input)
DEFAULT_OVERLAY_ALPHA: float = 0.05    # very subtle ghost

# ── "Density colormap" overlay ───────────────────────────────────────────
# Pre-computed per-cell density scalar built by
# ``marvel-water-conductance-build-meshes`` from ``Mask_cortex.tif`` and
# ``Source_Target_Possible_Paths.tif``.  Voxels of the cortex mask are
# binned in (L, θ) — long-axis fraction × azimuth — and the fraction of
# them that also lie inside the possible-paths volume gives a density
# map.  That map is upsampled and smoothed, then sampled at each mesh
# face centroid and stored as a small .npy file next to the mesh cache.
# At view time, a toggle button shows / hides the colormap on both the
# Cortical-bridges and All-watered-tissues meshes.
DEFAULT_MASK_CORTEX_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Mask_cortex.tif"
)
DEFAULT_CENTRAL_AXIS_PATH: Path = (
    acfg.DEFAULT_INPUT_DIR / "Coordinates_central_axis.txt"
)
DEFAULT_DENSITY_BRIDGES_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "density_facets_bridges.npy"
)
DEFAULT_DENSITY_ALL_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "density_facets_all.npy"
)
DEFAULT_DENSITY_RADIUS_PX: float = 130.0   # object radius (px) for θ-bin sizing
DEFAULT_DENSITY_STEP_PX: float = 3.0       # ≈ bin edge length at the surface (px)
DEFAULT_DENSITY_UPSAMPLE: int = 3
DEFAULT_DENSITY_SIGMA: float = 5.0         # Gaussian sigma in upsampled pixels

# Cool light grey body with a slight blue cast in the highlights.
BODY_COLOR = (185, 188, 195)        # RGB 0-255 — neutral grey (default / "all" mesh)
# Cortical-bridges mesh: very slightly bluer to distinguish it visually
# from the watered-tissues mesh at first glance.
BODY_COLOR_BRIDGES = (178, 188, 205)
SPECULAR_COLOR = (110, 165, 235)    # bluish reflection tint
AMBIENT = 0.18
DIFFUSE = 0.78
SPECULAR = 0.55
SPECULAR_POWER = 35
OPACITY = 1.0

BACKGROUND = "black"

# VTK interpolation modes (vtkProperty::SetInterpolation)
#   0 = Flat, 1 = Gouraud, 2 = Phong
_SHADING_MODES = [
    ("Phong",   2),
    ("Gouraud", 1),
    ("Flat",    0),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone viewer for Wat_Norm_Cortex.tif (single grey mesh).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT_PATH),
                   help="Path to the Wat_Norm_Cortex.tif file.")
    p.add_argument("--raw", default=str(DEFAULT_RAW_PATH),
                   help="Path to the raw greyscale volume (Raw.tif) used "
                        "for the orthogonal-slice locator panel.")
    p.add_argument("--mesh-cache", default=str(DEFAULT_MESH_CACHE_PATH),
                   help="Path to a cached .vtk mesh.  If the file exists "
                        "it is loaded directly (instant) instead of "
                        "running marching cubes.  Build it once with "
                        "`marvel-water-conductance-build-meshes`.")
    p.add_argument("--rebuild-mesh", action="store_true",
                   help="Ignore --mesh-cache and re-run marching cubes "
                        "from the TIFF input.")
    p.add_argument("--geoddist", default=str(DEFAULT_GEODDIST_PATH),
                   help="Path to the geodesic distance-to-exterior TIFF "
                        "(`Wat_norm-geoddist.tif`).  Used to build the "
                        "gradient-arrows view toggled by the 4th button.")
    p.add_argument("--arrows-cache", default=str(DEFAULT_ARROWS_CACHE_PATH),
                   help="Cached .npz with pre-computed arrow positions, "
                        "directions and convergence scores.  Built once by "
                        "`marvel-water-conductance-build-meshes`.")
    p.add_argument("--arrow-stride", type=int, default=DEFAULT_ARROW_STRIDE,
                   help="Sub-sampling stride on the two short axes (one "
                        "arrow every N voxels).")
    p.add_argument("--long-stride", type=int, default=DEFAULT_LONG_AXIS_STRIDE,
                   help="Sub-sampling stride on the longest volume axis "
                        "(coarser → fewer redundant arrows along the root).")
    p.add_argument("--fine-stride", type=int, default=DEFAULT_FINE_STRIDE,
                   help="Finer stride used internally to compute the "
                        "gradient and its divergence (compression rate). "
                        "Should be ≤ --arrow-stride.")
    p.add_argument("--rebuild-arrows", action="store_true",
                   help="Ignore --arrows-cache and recompute the arrow "
                        "field from the geodesic-distance TIFF.")
    p.add_argument("--overlay-input", default=str(DEFAULT_OVERLAY_TIFF_PATH),
                   help="8-bit TIFF (Mask_cortex_Gradient.tif) used to build "
                        "the translucent ghost mesh overlaid on every view.")
    p.add_argument("--overlay-cache", default=str(DEFAULT_OVERLAY_CACHE_PATH),
                   help="Cached VTK of the overlay isosurface.")
    p.add_argument("--overlay-level", type=float, default=DEFAULT_OVERLAY_LEVEL,
                   help="Iso-value used to extract the overlay isosurface.")
    p.add_argument("--overlay-alpha", type=float, default=DEFAULT_OVERLAY_ALPHA,
                   help="Opacity of the overlay ghost mesh.")
    p.add_argument("--rebuild-overlay", action="store_true",
                   help="Ignore --overlay-cache and recompute the overlay "
                        "isosurface.")
    p.add_argument("--no-overlay", action="store_true",
                   help="Don't render the translucent overlay mesh.")
    # Alternate "All watered tissues" mesh (Wat_Norm_All.tif).
    p.add_argument("--all-input", default=str(DEFAULT_ALL_INPUT_PATH),
                   help="TIFF used to build the alternate 'All watered tissues' mesh.")
    p.add_argument("--all-mesh-cache", default=str(DEFAULT_ALL_MESH_CACHE_PATH),
                   help="Cached .vtk of the 'All watered tissues' mesh.")
    p.add_argument("--rebuild-all-mesh", action="store_true",
                   help="Ignore --all-mesh-cache and re-run marching cubes.")
    p.add_argument("--no-all-mesh", action="store_true",
                   help="Disable the 'All watered tissues' alternate mesh "
                        "(hides the mesh-choice toggle).")
    # Density colormap caches (built by water_conductance_build_meshes).
    p.add_argument("--density-bridges-cache",
                   default=str(DEFAULT_DENSITY_BRIDGES_CACHE),
                   help="Per-cell density scalar .npy for the cortical-bridges mesh.")
    p.add_argument("--density-all-cache",
                   default=str(DEFAULT_DENSITY_ALL_CACHE),
                   help="Per-cell density scalar .npy for the all-watered-tissues mesh.")
    # Glowing green "Pillars" overlay (only shown on the cortex mesh).
    p.add_argument("--pillars-input", default=str(DEFAULT_PILLARS_TIFF_PATH),
                   help="8-bit TIFF (Pillars.tif) used to build the glowing green overlay.")
    p.add_argument("--pillars-cache", default=str(DEFAULT_PILLARS_CACHE_PATH),
                   help="Cached .vtk of the Pillars isosurface.")
    p.add_argument("--pillars-level", type=float, default=DEFAULT_PILLARS_LEVEL,
                   help="Iso-value used to extract the Pillars isosurface.")
    p.add_argument("--pillars-alpha", type=float, default=DEFAULT_PILLARS_ALPHA,
                   help="Opacity of the glowing Pillars overlay.")
    p.add_argument("--rebuild-pillars", action="store_true",
                   help="Ignore --pillars-cache and recompute the Pillars isosurface.")
    p.add_argument("--no-pillars", action="store_true",
                   help="Don't render the glowing green Pillars overlay.")
    # Water "membranes" descending animation.
    p.add_argument("--membranes-vtp-cache",
                   default=str(DEFAULT_MEMBRANES_VTP_CACHE),
                   help="Cached binary .vtp of the water-membranes packed "
                        "mesh (produced by marvel-water-conductance-build-meshes).")
    p.add_argument("--membranes-meta-cache",
                   default=str(DEFAULT_MEMBRANES_META_CACHE),
                   help="Cached JSON sidecar holding the membranes animation "
                        "meta (n_steps, level range, per-column phases, …).")
    p.add_argument("--no-membranes", action="store_true",
                   help="Don't load the water-membranes animation (hides the "
                        "Water ON/OFF toggle).")
    # Crown geodesic tracks (Arrows view #2) — built by build-meshes, loaded here.
    p.add_argument("--tracks-cache", default=str(DEFAULT_CROWN_TRACKS_CACHE),
                   help="Cached .npz of pre-computed Dijkstra track segments "
                        "(produced by marvel-water-conductance-build-meshes).")
    p.add_argument("--tracks-vtp-cache", default=str(DEFAULT_CROWN_TRACKS_VTP_CACHE),
                   help="Cached binary .vtp of crown-track polylines "
                        "(produced alongside the .npz by build-meshes). "
                        "When present, loaded instead of reconstructing cells "
                        "from the .npz, giving near-instant actor construction.")
    p.add_argument("--tracks-arrows-vtp-cache",
                   default=str(DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE),
                   help="Cached binary .vtp of pre-computed arrow glyph centres, "
                        "tangents and arc-length fractions (produced by "
                        "build-meshes).  When present, skips the Python for-loop "
                        "in _make_tracks_actor_curves.")
    p.add_argument("--no-tracks", action="store_true",
                   help="Disable the crown-tracks view.")
    p.add_argument("--tracks-stride", type=int, default=1,
                   help="Keep only every Nth track segment from the cache "
                        "(display sub-sampling -- doesn't recompute anything). "
                        "Use e.g. 100 to render 1%% of the segments.")
    p.add_argument("--no-ortho-panel", action="store_true",
                   help="Disable the orthogonal-slice locator panel.")
    p.add_argument("--level", "-l", type=float, default=DEFAULT_LEVEL,
                   help="Marching-cubes iso-level on the float volume.")
    p.add_argument("--smooth-iter", type=int, default=DEFAULT_SMOOTH_ITER,
                   help="Laplacian smoothing iterations applied to the mesh.")
    p.add_argument("--save-vtk", default=None,
                   help="Optional path to also save the generated mesh as a VTK file.")
    p.add_argument("--width", type=int, default=1280,
                   help="Viewer window width in pixels.")
    p.add_argument("--height", type=int, default=720,
                   help="Viewer window height in pixels (16:9 matches "
                        "the default movie aspect).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


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

    # ── Uniform spatial sampling ─────────────────────────────────────────
    # Divide the bounding box into a grid of cells of edge length
    # `cell_size` voxels; keep one random voxel per occupied cell.
    n_desired = min(int(n_source_points), n_att)
    if n_att <= n_desired:
        selected = att_coords
    else:
        cell_size = max(1, int(np.ceil(n_att ** (1 / 3) /
                                       n_desired ** (1 / 3))))
        bk = att_coords // cell_size
        _, first_idx = np.unique(
            bk[:, 0] * 100_000_000 + bk[:, 1] * 10_000 + bk[:, 2],
            return_index=True,
        )
        if len(first_idx) > n_desired:
            rng = np.random.default_rng(42)
            first_idx = rng.choice(first_idx, size=n_desired, replace=False)
        selected = att_coords[first_idx]
    logger.info(
        "Tracing %d Dijkstra paths (cell_size=%d) …",
        len(selected),
        max(1, int(np.ceil(n_att ** (1 / 3) / n_desired ** (1 / 3))))
        if n_att > n_desired else 1,
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


def _attach_controls(
    plt, mesh,
    arrows_data=None,
    ortho_overlay=None,
    alt_mesh=None,
    pillars_mesh=None,
    tracks_data=None,
    density_scalars=None,
    membranes_data=None,
) -> None:
    """Add shading button + opacity / lighting / hue sliders.

    When ``arrows_data`` is provided (output of
    :func:`_load_or_build_arrows`), a 4th top-right button toggles
    between the mesh view and a 3-D gradient-arrows view computed from
    the geodesic distance map.  The bottom slider toolbar is swapped to
    expose the arrow-rendering parameters (arrow length, colormap
    range, colormap cycle button)."""
    # Pool of meshes that share the *same* rendering parameters (shading
    # mode, sliders).  Sliders & shading toggles always apply to every
    # mesh in this pool so swapping between them is seamless.
    all_meshes = [m for m in (mesh, alt_mesh) if m is not None]
    active = {"mesh": mesh}
    # Tracks Mesh ↔ Arrows view; consulted by `_pillars_should_show`.
    view_mode = {"name": "mesh"}

    state = {
        "opacity":  OPACITY,
        "ambient":  AMBIENT,
        "diffuse":  DIFFUSE,
        "specular": SPECULAR,
        "hue":      0.0,
    }
    shading_idx = [0]
    for _m in all_meshes:
        _set_shading(_m, _SHADING_MODES[0][1])

    # ── Snapshot of the initial camera so "Restart navigation" can
    #    restore the starting framing (position / focal point / view-up
    #    / parallel-scale / clipping range / view-angle).
    _initial_cam: dict = {}
    try:
        _cam0 = plt.camera
        _initial_cam = {
            "pos":   tuple(_cam0.GetPosition()),
            "fp":    tuple(_cam0.GetFocalPoint()),
            "up":    tuple(_cam0.GetViewUp()),
            "pscale": float(_cam0.GetParallelScale()),
            "vangle": float(_cam0.GetViewAngle()),
            "clip":  tuple(_cam0.GetClippingRange()),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not snapshot initial camera: %s", exc)

    # ── Status text (top-center) — auto-hides after STATUS_DURATION_S ─
    import vedo as _vedo_mod
    import time as _time
    status_state = {
        "text2d": None,
        "expire_at": 0.0,
        "seq": 0,
    }
    try:
        status_state["text2d"] = _vedo_mod.Text2D(
            "", pos="top-center", s=1.1, c="white", bg="black", alpha=0.65,
        )
        plt.add(status_state["text2d"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create status Text2D: %s", exc)
        status_state["text2d"] = None

    def _hide_status():
        t = status_state["text2d"]
        if t is None:
            return
        try:
            t.text("")
        except Exception:  # noqa: BLE001
            pass

    def _show_status(msg: str):
        t = status_state["text2d"]
        if t is None:
            return
        try:
            t.text(msg)
        except Exception:  # noqa: BLE001
            return
        status_state["seq"] += 1
        my_seq = status_state["seq"]
        status_state["expire_at"] = _time.time() + STATUS_DURATION_S
        iren = getattr(plt, "interactor", None)
        if iren is not None:
            try:
                iren.CreateOneShotTimer(int(STATUS_DURATION_S * 1000))
            except Exception:  # noqa: BLE001
                pass
        # Stash sequence for the timer observer to check.
        status_state["_pending_seq"] = my_seq
        plt.render()

    def _timer_cb(_obj=None, _event=None):
        # Only clear if no newer status message superseded this timer
        # and the expiry has actually passed.
        if _time.time() + 0.05 < status_state["expire_at"]:
            return
        _hide_status()
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    _iren_for_timer = getattr(plt, "interactor", None)
    if _iren_for_timer is not None:
        try:
            _iren_for_timer.AddObserver("TimerEvent", _timer_cb)
        except Exception:  # noqa: BLE001
            pass

    def _apply_hue():
        for _m in all_meshes:
            base = getattr(_m, "_body_color", BODY_COLOR)
            rgb = _hue_shifted_rgb(base, state["hue"])
            _m.color(list(rgb))

    def _slider_cb(key):
        def _cb(widget, _event):
            state[key] = float(widget.value)
            if key == "opacity":
                for _m in all_meshes:
                    _m.alpha(state["opacity"])
            elif key == "hue":
                _apply_hue()
            else:
                for _m in all_meshes:
                    _set_lighting(
                        _m,
                        ambient=state["ambient"],
                        diffuse=state["diffuse"],
                        specular=state["specular"],
                    )
            plt.render()
        return _cb

    x1, x2 = 0.30, 0.85
    slider_defs = [
        ("opacity",  0.0, 1.0, 0.04),
        ("ambient",  0.0, 1.0, 0.09),
        ("diffuse",  0.0, 1.0, 0.14),
        ("specular", 0.0, 1.0, 0.19),
        ("hue",      0.0, 1.0, 0.24),
    ]
    mesh_sliders: list = []
    for key, vmin, vmax, y in slider_defs:
        sw = plt.add_slider(
            _slider_cb(key),
            vmin, vmax, value=state[key],
            pos=((x1, y), (x2, y)),
            title=key,
            title_size=0.7,
            show_value=True,
        )
        mesh_sliders.append(sw)

    def _set_widget_visible(widget, visible: bool) -> None:
        """Show / hide a vtkSliderWidget (used to swap the bottom toolbar)."""
        if widget is None:
            return
        try:
            widget.SetEnabled(1 if visible else 0)
            rep = widget.GetRepresentation()
            if rep is not None:
                rep.SetVisibility(1 if visible else 0)
        except Exception:  # noqa: BLE001
            pass

    def _set_button_visible(button, visible: bool) -> None:
        """Show / hide a vedo button actor."""
        if button is None:
            return
        actor = getattr(button, "actor", None)
        if actor is None:
            return
        try:
            actor.SetVisibility(1 if visible else 0)
        except Exception:  # noqa: BLE001
            pass

    def _shading_cb(*_a, **_kw):
        shading_idx[0] = (shading_idx[0] + 1) % len(_SHADING_MODES)
        mode_name, mode_val = _SHADING_MODES[shading_idx[0]]
        for _m in all_meshes:
            _set_shading(_m, mode_val)
        _shading_btn.switch()
        _show_status(f"Shading: {mode_name}")
        plt.render()

    _shading_btn = plt.add_button(
        _shading_cb,
        states=[f"Shading ▸ {name}" for name, _ in _SHADING_MODES],
        c=["white"] * len(_SHADING_MODES),
        bc=["#3b5b8c", "#6b8e23", "#8c6b3b"],
        pos=(0.88, 0.95),
        size=16,
        bold=True,
    )

    # ─── interaction style switch ───────────────────────────────────────
    # Two modes only:
    #  - Trackball Cam: standard VTK mouse navigation, plus w/s keys to
    #    *really* move forward/backward (camera AND focal point translate
    #    together along the view direction).  The default VTK mouse wheel
    #    only dollies toward a fixed focal point, which feels like zoom
    #    when exploring inside a long object like a root.
    #  - Custom (buttons): VTK navigation is disabled; the on-screen
    #    movement panel below is the only way to move the camera.
    import vtk
    _INTERACTION_STYLES = [
        ("Trackball Cam",    vtk.vtkInteractorStyleTrackballCamera),
        ("Custom (buttons)", vtk.vtkInteractorStyleUser),
    ]
    style_state = {
        "idx": 0,
        "instances": [cls() for _, cls in _INTERACTION_STYLES],
        "custom_idx": len(_INTERACTION_STYLES) - 1,
    }

    def _apply_style(i: int) -> None:
        iren = getattr(plt, "interactor", None)
        if iren is None:
            return
        iren.SetInteractorStyle(style_state["instances"][i])

    _apply_style(0)

    def _cycle_style_cb(*_a, **_kw):
        style_state["idx"] = (style_state["idx"] + 1) % len(_INTERACTION_STYLES)
        _apply_style(style_state["idx"])
        _style_btn.switch()
        _refresh_nav_panel_visibility()
        plt.render()

    _style_btn = plt.add_button(
        _cycle_style_cb,
        states=[f"Move ▸ {n}" for n, _ in _INTERACTION_STYLES],
        c=["white"] * len(_INTERACTION_STYLES),
        bc=["#2e7d32", "#455a64"],
        pos=(0.88, 0.89),
        size=16,
        bold=True,
    )

    # ─── w/s keys: walk forward/backward along view direction ───────────
    # Active only when the Trackball Cam style is selected.  Translates
    # camera *and* focal point together, so this is real "walking" inside
    # the scene, not a zoom toward a fixed pivot.
    def _walk_camera(step_frac: float) -> None:
        cam = plt.camera
        pos = list(cam.GetPosition())
        fp = list(cam.GetFocalPoint())
        direction = cam.GetDirectionOfProjection()  # unit cam→focal vector
        # Step length scaled by current cam→focal distance, so the feel is
        # consistent at different zoom levels.
        step = float(cam.GetDistance()) * step_frac
        new_pos = [pos[i] + step * direction[i] for i in range(3)]
        new_fp = [fp[i] + step * direction[i] for i in range(3)]
        cam.SetPosition(*new_pos)
        cam.SetFocalPoint(*new_fp)
        try:
            plt.renderer.ResetCameraClippingRange()
        except Exception:  # noqa: BLE001
            pass
        plt.render()

    def _keypress_cb(obj, _event):
        # Only walk when Trackball is the active style; Custom mode owns
        # navigation through the on-screen buttons.
        if style_state["idx"] != 0:
            return
        try:
            key = obj.GetKeySym()
        except Exception:  # noqa: BLE001
            return
        if key in ("w", "W"):
            _walk_camera(+0.05)
        elif key in ("s", "S"):
            _walk_camera(-0.05)

    iren = getattr(plt, "interactor", None)
    if iren is not None:
        iren.AddObserver("KeyPressEvent", _keypress_cb)

    # ─── on-screen navigation panel (Custom mode only) ──────────────────
    # 6 translation directions × 3 speeds (factor ×5 between speeds) +
    # 3 rotation axes × 3 speeds × 2 directions.  All button actors are
    # stored so they can be hidden when not in Custom mode.
    import numpy as np

    _nav_buttons: list = []

    # Translation: 5 speed levels (×5 between each).
    #   very-very-fine / very-slow / slow / fast / very-fast
    SPEED_FACTORS = [0.04, 0.2, 1.0, 5.0, 25.0]
    TRANS_BASE_FRAC = 0.008            # nominal step = 0.8% of cam→focal dist

    # Rotation: 5 speed levels (×5 between each), degrees per click.
    ROT_SPEED_FACTORS = [0.024, 0.12, 0.6, 3.0, 15.0]
    # Around-camera rotation: pivot at camera position, not focal point.
    # (Camera pos stays fixed; focal point and view-up are rotated.)

    def _camera_axes():
        cam = plt.camera
        pos = np.asarray(cam.GetPosition(), dtype=float)
        fp = np.asarray(cam.GetFocalPoint(), dtype=float)
        vu = np.asarray(cam.GetViewUp(), dtype=float)
        forward = fp - pos
        fn = float(np.linalg.norm(forward))
        if fn < 1e-12:
            forward = np.array([0.0, 0.0, -1.0]); fn = 1.0
        forward /= fn
        right = np.cross(forward, vu)
        rn = float(np.linalg.norm(right))
        if rn < 1e-12:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        up = np.cross(right, forward)
        un = float(np.linalg.norm(up))
        if un < 1e-12:
            up = vu / max(np.linalg.norm(vu), 1e-12)
        else:
            up /= un
        return cam, forward, right, up, fn

    def _refresh_ortho():
        if ortho_overlay is None:
            logger.debug("_refresh_ortho: overlay is None, skipping.")
            return
        try:
            cam = plt.camera
            pos = np.asarray(cam.GetPosition(), dtype=float)
            foc = np.asarray(cam.GetFocalPoint(), dtype=float)
            direction = foc - pos
            n = float(np.linalg.norm(direction))
            if n > 1e-9:
                direction = direction / n
            else:
                direction = np.array([0.0, 0.0, -1.0])
            logger.info(
                "_refresh_ortho  cam.pos=(%.2f, %.2f, %.2f)  cam.foc=(%.2f, %.2f, %.2f)",
                pos[0], pos[1], pos[2], foc[0], foc[1], foc[2],
            )
            ortho_overlay.update(pos, direction)
        except Exception as exc:  # noqa: BLE001
            logger.warning("_refresh_ortho failed: %s", exc)

    def _translate(axis: str, sign: int, speed_factor: float):
        cam, fwd, right, up, dist = _camera_axes()
        vec = {"F": fwd, "R": right, "U": up}[axis] * float(sign)
        step = dist * TRANS_BASE_FRAC * speed_factor
        delta = vec * step
        pos = np.asarray(cam.GetPosition()) + delta
        fp = np.asarray(cam.GetFocalPoint()) + delta
        cam.SetPosition(*pos)
        cam.SetFocalPoint(*fp)
        try:
            plt.renderer.ResetCameraClippingRange()
        except Exception:  # noqa: BLE001
            pass
        _refresh_ortho()
        plt.render()

    def _rotate_around_camera(axis: str, deg: float):
        """Rotate the camera in place: pivot is the camera position.

        - yaw:   rotate look direction & view_up around world-style up
                 (use the camera's current 'up' axis).
        - pitch: rotate look direction & view_up around the camera's
                 'right' axis.
        - roll:  rotate view_up around the look direction.
        """
        cam, fwd, right, up, dist = _camera_axes()
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        if axis == "yaw":
            axis_vec = up
        elif axis == "pitch":
            axis_vec = right
        elif axis == "roll":
            axis_vec = fwd
        else:
            return
        # Rodrigues' rotation of fwd and up around axis_vec.
        def _rot(v):
            return (v * c
                    + np.cross(axis_vec, v) * s
                    + axis_vec * np.dot(axis_vec, v) * (1.0 - c))
        new_fwd = _rot(fwd)
        new_up = _rot(up)
        pos = np.asarray(cam.GetPosition())
        # Keep cam→focal distance unchanged so the focal point stays at
        # the same logical depth in front of the camera.
        new_fp = pos + new_fwd * dist
        cam.SetFocalPoint(*new_fp)
        cam.SetViewUp(*new_up)
        try:
            plt.renderer.ResetCameraClippingRange()
        except Exception:  # noqa: BLE001
            pass
        _refresh_ortho()
        plt.render()

    def _mk_btn(cb, label, x, y, bc):
        b = plt.add_button(
            lambda *_a, _f=cb, **_kw: _f(),
            states=[label],
            c=["white"],
            bc=[bc],
            pos=(x, y),
            size=14,
            bold=True,
        )
        _nav_buttons.append(b)
        return b

    # Layout: right side, below the three existing top-right buttons
    # (Shading / Move / Save pos at y ≈ 0.95 / 0.89 / 0.83).
    # Columns: 5 speed columns for translations, 10 for rotations
    # (5 speeds × 2 directions).
    COL_X = [0.69, 0.74, 0.79, 0.84, 0.89]                    # 5 speed cols
    ROT_X = [0.69, 0.73, 0.77, 0.81, 0.85,
             0.90, 0.94, 0.98, 1.02, 1.06]                    # 5×2 dirs

    # Translation rows: (label, axis_key, sign, color)
    trans_rows = [
        ("Fwd",  "F", +1, "#1b5e20"),
        ("Bwd",  "F", -1, "#1b5e20"),
        ("Rgt",  "R", +1, "#0d47a1"),
        ("Lft",  "R", -1, "#0d47a1"),
        ("Up",   "U", +1, "#4a148c"),
        ("Dn",   "U", -1, "#4a148c"),
    ]
    y_top = 0.77
    dy = 0.044
    speed_tags = ["·", "··", "···", "····", "·····"]
    for ri, (lbl, axis, sign, color) in enumerate(trans_rows):
        y = y_top - ri * dy
        for ci, factor in enumerate(SPEED_FACTORS):
            tag = speed_tags[ci]
            _mk_btn(
                lambda _a=axis, _s=sign, _f=factor: _translate(_a, _s, _f),
                f"{lbl}{tag}",
                COL_X[ci], y, color,
            )

    # Rotation rows: yaw / pitch / roll.
    # 5 speeds × 2 directions = 10 buttons per row.
    rot_rows = [
        ("Yaw",   "yaw",   "◄", "►", "#bf360c"),
        ("Pitch", "pitch", "▲", "▼", "#33691e"),
        ("Roll",  "roll",  "↺", "↻", "#01579b"),
    ]
    y_rot_top = y_top - len(trans_rows) * dy - 0.01
    for ri, (lbl, axis, lsym, rsym, color) in enumerate(rot_rows):
        y = y_rot_top - ri * dy
        # Negative direction, fast → slow → very-very-fine (reversed).
        for ci, factor in enumerate(ROT_SPEED_FACTORS[::-1]):
            tag = speed_tags[::-1][ci]
            _mk_btn(
                lambda _a=axis, _f=factor: _rotate_around_camera(_a, -_f),
                f"{lsym}{tag}", ROT_X[ci], y, color,
            )
        # Positive direction, very-very-fine → fast.
        for ci, factor in enumerate(ROT_SPEED_FACTORS):
            tag = speed_tags[ci]
            _mk_btn(
                lambda _a=axis, _f=factor: _rotate_around_camera(_a, +_f),
                f"{rsym}{tag}", ROT_X[5 + ci], y, color,
            )

    def _refresh_nav_panel_visibility():
        show = (style_state["idx"] == style_state["custom_idx"])
        for b in _nav_buttons:
            actor = getattr(b, "actor", None)
            if actor is None:
                continue
            try:
                actor.SetVisibility(1 if show else 0)
            except Exception:  # noqa: BLE001
                pass

    _refresh_nav_panel_visibility()

    # ─── save current camera position to ./positions/ ───────────────────
    # Each click appends one JSON entry with the full camera state plus a
    # timestamp, so the sequence can later be replayed as a fly-through.
    positions_state = {
        "dir":   DEFAULT_POSITIONS_DIR,
        "file":  None,  # type: ignore[var-annotated]
        "count": 0,
    }

    # Registry of zero-arg callables that return a partial UI-state dict.
    # Each control section (view-mode, pillars, arrows, fog/SSAO, …)
    # appends its own capturer here when it sets up.  At save time we
    # merge them all into the entry's ``ui_state`` block so the movie
    # renderer can replay the full visual configuration (and interpolate
    # between consecutive keyframes).
    _ui_capturers: list = []

    def _save_position_cb(*_a, **_kw):
        try:
            positions_state["dir"].mkdir(exist_ok=True)
            if positions_state["file"] is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                positions_state["file"] = (
                    positions_state["dir"] / f"positions_{stamp}.json"
                )
            cam = plt.camera
            # Capture the underlying vtkActor's transform.  In actor-style
            # interaction modes, the user drags the mesh, which writes back
            # into Position / Orientation / Scale (and optionally a
            # UserMatrix).  Saving the camera alone would discard all that.
            actor = getattr(active["mesh"], "actor", None) or active["mesh"]
            try:
                user_mat = actor.GetUserMatrix()
                if user_mat is not None:
                    user_mat_list = [
                        [user_mat.GetElement(r, c) for c in range(4)]
                        for r in range(4)
                    ]
                else:
                    user_mat_list = None
            except Exception:  # noqa: BLE001
                user_mat_list = None

            try:
                win_size = list(plt.window.GetSize())
            except Exception:  # noqa: BLE001
                win_size = None

            entry = {
                "index":     positions_state["count"],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "window_size": win_size,
                "camera": {
                    "position":       list(cam.GetPosition()),
                    "focal_point":    list(cam.GetFocalPoint()),
                    "view_up":        list(cam.GetViewUp()),
                    "view_angle":     float(cam.GetViewAngle()),
                    "clipping_range": list(cam.GetClippingRange()),
                    "parallel_scale": float(cam.GetParallelScale()),
                    "distance":       float(cam.GetDistance()),
                },
                "actor": {
                    "position":    list(actor.GetPosition()),
                    "orientation": list(actor.GetOrientation()),
                    "scale":       list(actor.GetScale()),
                    "origin":      list(actor.GetOrigin()),
                    "user_matrix": user_mat_list,
                },
            }
            # Collect any UI state contributed by other sections.  Each
            # capturer returns a small dict; we merge them shallowly into
            # ``ui_state``.  Errors in individual capturers are isolated
            # so a single buggy widget cannot break the save.
            ui_state: dict = {}
            for cap in _ui_capturers:
                try:
                    chunk = cap()
                    if isinstance(chunk, dict):
                        ui_state.update(chunk)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("UI capturer failed: %s", exc)
            if ui_state:
                entry["ui_state"] = ui_state
            fpath = positions_state["file"]
            if fpath.exists():
                try:
                    data = json.loads(fpath.read_text())
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
            else:
                data = []
            data.append(entry)
            fpath.write_text(json.dumps(data, indent=2))
            positions_state["count"] += 1
            logger.info(
                "Saved position #%d → %s  "
                "(cam.pos=%s, actor.pos=%s, actor.ori=%s)",
                entry["index"], fpath,
                tuple(round(v, 2) for v in entry["camera"]["position"]),
                tuple(round(v, 2) for v in entry["actor"]["position"]),
                tuple(round(v, 2) for v in entry["actor"]["orientation"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Save position failed: %s", exc)

    plt.add_button(
        _save_position_cb,
        states=["Save pos"],
        c=["white"],
        bc=["#00838f"],
        pos=(0.88, 0.83),
        size=16,
        bold=True,
    )

    # ─── Mesh-choice toggle (Cortical bridges ↔ All watered tissues) ────
    # Only meaningful when the alternate mesh is available.  The button
    # is hidden whenever we are in the Arrows view.  Switching also
    # shows/hides the green-glow Pillars overlay (Pillars are only
    # displayed on the Cortical bridges mesh).
    pillars_state = {"visible": False, "style": None, "hue_shift": 0.0}

    # Base colours (teal-leaning green-blue) for the two pillars styles.
    # The user-facing "pillars hue" slider rotates these in HSV space so
    # the change applies coherently to both the transparent (glow, mesh
    # view) and opaque (solid, arrows view) styles.
    _PILLARS_BASE_GLOW  = (0.20, 0.95, 0.75)   # bright teal ghost
    _PILLARS_BASE_SOLID = (0.05, 0.50, 0.45)   # darker teal body

    def _pillars_rgb(base_rgb):
        """Return ``base_rgb`` rotated by ``pillars_state['hue_shift']``."""
        return _hue_shifted_rgb(
            tuple(int(round(c * 255)) for c in base_rgb),
            pillars_state["hue_shift"],
        )

    # Initial pillars style as set by `main()`.  We restore it whenever
    # we leave the Arrows view.
    _initial_pillars_alpha = None
    if pillars_mesh is not None:
        try:
            _initial_pillars_alpha = float(pillars_mesh.alpha())
        except Exception:  # noqa: BLE001
            _initial_pillars_alpha = None

    def _apply_pillars_style(style: str | None = None) -> None:
        """Switch pillars look between ``"glow"`` (mesh view) and
        ``"solid"`` (arrows view: opaque, darker teal, specular).
        Passing ``style=None`` re-applies the current style — useful to
        refresh the colour after a hue change."""
        if pillars_mesh is None:
            return
        if style is None:
            style = pillars_state["style"] or "glow"
        # Don't early-return on style equality: we also use this entry
        # point to refresh the hue, so always re-apply.
        try:
            if style == "solid":
                # Re-activate per-actor lighting that was turned OFF by
                # the previous "glow" style.  Without this, subsequent
                # `lighting(ambient=..., diffuse=...)` calls only update
                # the coefficients but the OpenGL pipeline keeps
                # bypassing the lighting equation -> flat, dull look.
                try:
                    prop = getattr(pillars_mesh, "properties", None)
                    if prop is None:
                        prop = pillars_mesh.GetProperty()
                    prop.LightingOn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Pillars LightingOn() failed: %s", exc)
                # Teal body + crisp white highlights for a "wet shiny"
                # look.  Phong shading + recomputed normals (done at
                # load time in main()) make the highlights glide
                # smoothly over the surface instead of facetting.
                pillars_mesh.c(list(_pillars_rgb(_PILLARS_BASE_SOLID)))
                pillars_mesh.alpha(1.0)
                try:
                    pillars_mesh.lighting(
                        ambient=0.30, diffuse=0.65,
                        specular=1.0, specular_power=60,
                        specular_color=(1.0, 1.0, 1.0),
                    )
                except TypeError:
                    pillars_mesh.lighting(
                        ambient=0.30, diffuse=0.65,
                        specular=1.0, specular_power=60,
                    )
                _set_shading(pillars_mesh, 2)  # Phong
            else:  # "glow"
                pillars_mesh.c(list(_pillars_rgb(_PILLARS_BASE_GLOW)))
                pillars_mesh.alpha(_initial_pillars_alpha
                                   if _initial_pillars_alpha is not None
                                   else 0.12)
                pillars_mesh.lighting("off")
            pillars_state["style"] = style
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not switch pillars style → %s: %s", style, exc)

    # Initialise to glow (mesh view is the default startup mode).
    _apply_pillars_style("glow")

    def _pillars_should_show() -> bool:
        if pillars_mesh is None:
            return False
        # Visible whenever we're in any Arrows view, or when the
        # Cortical bridges mesh is the active mesh.  Hidden on the
        # 'All watered tissues' mesh.
        if view_mode.get("name") in ("arrows_grid", "arrows_tracks"):
            return True
        return view_mode.get("name") == "mesh_bridges"

    def _refresh_pillars_visibility():
        if pillars_mesh is None:
            return
        # Determine the target style *before* any visibility change.
        in_arrows = view_mode.get("name") in ("arrows_grid", "arrows_tracks")
        target_style = "solid" if in_arrows else "glow"
        want = _pillars_should_show()
        # --- visibility change FIRST -----------------------------------
        # When `want` is True and the mesh was absent, vedo's plt.add()
        # may reset the mapper's ScalarVisibility / LUT, overriding any
        # colour we set before the add.  By adding first and styling
        # afterwards we guarantee the final colour is always correct.
        if want and not pillars_state["visible"]:
            try:
                plt.add(pillars_mesh)
                pillars_state["visible"] = True
            except Exception:  # noqa: BLE001
                pass
        elif (not want) and pillars_state["visible"]:
            try:
                plt.remove(pillars_mesh)
                pillars_state["visible"] = False
            except Exception:  # noqa: BLE001
                pass
        # --- style / colour AFTER add ----------------------------------
        _apply_pillars_style(target_style)

    def _active_mesh_title() -> str:
        return (STATUS_TITLE_MESH_CORTEX
                if view_mode.get("name") == "mesh_bridges"
                else STATUS_TITLE_MESH_ALL)

    # Default starting mode is the cortex-bridges mesh.
    view_mode["name"] = "mesh_bridges"
    active["mesh"] = mesh
    # Show Pillars initially (we start on the cortex mesh).
    _refresh_pillars_visibility()

    # ─── 4-button radio: mesh_bridges / mesh_all / arrows_grid /
    #     arrows_tracks ──────────────────────────────────────────────────
    # Replaces the previous cycling "View" button + separate "Mesh"
    # toggle.  At most one button is active; clicking enters that mode.
    # Two independent arrow-field data sets can be displayed:
    #   * "arrows_grid"   – :func:`_build_arrow_field`: a sub-sampled
    #                       grid of gradient directions sampled on the
    #                       whole volume (the original arrows view).
    #   * "arrows_tracks" – :func:`_build_track_arrow_field`: integrated
    #                       stream-lines starting from ThinLayer.tif and
    #                       stopping at Pillars / undefined / >90° turns.
    # Both share the same length slider and the same Angle/Convergence
    # colormap toggle; switching modes hot-swaps the actor in the scene.
    astate = {
        "length":    DEFAULT_ARROW_LENGTH,
        "score_min": 0.0,
        "score_max": 1.0,
        "cmap_mode": "angle",   # or "convergence"
    }
    arrows_actor = None
    tracks_actor = None          # may be a list [lines_act, path_arrows_act]
    _track_cam_obs_tag = [None]  # mutable container for camera observer tag
    arrow_sliders: list = []

    # ── Density colormap (per-cell scalars) ──────────────────────────────
    _density_btn = None  # populated below if scalars are present
    _density_state = {"on": False}
    if density_scalars is None:
        _density_has_data = False
    else:
        _b = density_scalars.get("bridges")
        _a = density_scalars.get("all")
        _ok_b = (_b is not None and mesh is not None
                 and getattr(mesh, "ncells", -1) == int(_b.shape[0]))
        _ok_a = (_a is not None and alt_mesh is not None
                 and getattr(alt_mesh, "ncells", -1) == int(_a.shape[0]))
        if _b is not None and not _ok_b:
            logger.warning(
                "Density: bridges scalars length=%d does not match mesh "
                "(%d faces); disabling.",
                int(_b.shape[0]), getattr(mesh, "ncells", -1),
            )
        if _a is not None and not _ok_a:
            logger.warning(
                "Density: all-mesh scalars length=%d does not match mesh "
                "(%d faces); disabling.",
                int(_a.shape[0]), getattr(alt_mesh, "ncells", -1),
            )
        _density_has_data = bool(_ok_b or _ok_a)

    def _plt_add(actor_or_list) -> None:
        """Add a single actor or a list of actors to the plotter."""
        if actor_or_list is None:
            return
        items = actor_or_list if isinstance(actor_or_list, list) else [actor_or_list]
        for a in items:
            if a is not None:
                try:
                    plt.add(a)
                except Exception:  # noqa: BLE001
                    pass

    def _plt_remove(actor_or_list) -> None:
        """Remove a single actor or a list of actors from the plotter."""
        if actor_or_list is None:
            return
        items = actor_or_list if isinstance(actor_or_list, list) else [actor_or_list]
        for a in items:
            if a is not None:
                try:
                    plt.remove(a)
                except Exception:  # noqa: BLE001
                    pass

    _ui_capturers.append(lambda: {
        "view_mode": view_mode.get("name"),
        "arrow_length":   float(astate["length"]),
        "arrow_score_min": float(astate["score_min"]),
        "arrow_score_max": float(astate["score_max"]),
        "cmap_mode":  astate["cmap_mode"],
    })
    _ui_capturers.append(lambda: {
        "pillars_visible":   bool(pillars_state["visible"]),
        "pillars_style":     pillars_state["style"],
        "pillars_hue_shift": float(pillars_state["hue_shift"]),
    })

    # ─── Water "membranes" descending animation ─────────────────────────
    # ``membranes_data``: (polydata, meta) or (None, None) when disabled.
    # A toggle button shows/hides the cyan-blue glowing membranes; while
    # visible, a dedicated repeating timer advances a ``vtkThreshold``
    # over the per-cell ``step_id`` array so only the cells of the
    # current step are drawn.  Cells are pre-sorted by ``step_id`` so
    # each tick selects a contiguous range — minimising VTK work.
    #
    # EMISSIVE BLOOM: this is currently approximated by a saturated
    # cyan-blue colour + high ambient lighting against the dark
    # background.  A real bloom would attach a ``vtkOpenGLBloomPass``
    # render-pass (sketch left in the plan notes).
    membranes_state = {
        "visible":  False,
        "step":     0,
        "n_steps":  0,
        "timer_id": [None],
        "actor":    None,
        "threshold": None,
        "mapper":   None,
    }
    _membranes_pd, _membranes_meta = (None, None)
    if membranes_data is not None:
        _membranes_pd, _membranes_meta = membranes_data

    if _membranes_pd is not None and _membranes_meta is not None:
        try:
            import vtk as _vtk_m
            membranes_state["n_steps"] = int(_membranes_meta.get("n_steps", 0))

            # vtkThreshold over the cell-data 'step_id' field.
            _thr = _vtk_m.vtkThreshold()
            _thr.SetInputData(_membranes_pd)
            _thr.SetInputArrayToProcess(
                0, 0, 0,
                _vtk_m.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                "step_id",
            )
            # Initial step = 0; "between" so cells with step_id == step
            # are kept (lower==upper).
            try:
                _thr.SetThresholdFunction(_vtk_m.vtkThreshold.THRESHOLD_BETWEEN)
            except AttributeError:
                # Older VTK: default is "between".
                pass
            _thr.SetLowerThreshold(0.0)
            _thr.SetUpperThreshold(0.0)
            _thr.Update()

            # vtkThreshold outputs an unstructured grid → wrap back to
            # PolyData via vtkGeometryFilter so we can use a PolyData
            # mapper (cheaper than DataSetMapper for triangles).
            _geo = _vtk_m.vtkGeometryFilter()
            _geo.SetInputConnection(_thr.GetOutputPort())

            _mapper = _vtk_m.vtkPolyDataMapper()
            _mapper.SetInputConnection(_geo.GetOutputPort())
            # Use the per-cell ``rgb`` array directly as colours.
            _mapper.SetScalarModeToUseCellFieldData()
            _mapper.SelectColorArray("rgb")
            _mapper.SetColorModeToDirectScalars()
            _mapper.ScalarVisibilityOn()

            _actor = _vtk_m.vtkActor()
            _actor.SetMapper(_mapper)
            _prop = _actor.GetProperty()
            _prop.SetOpacity(float(DEFAULT_MEMBRANES_ALPHA))
            # Emissive-ish look: high ambient, zero diffuse/specular, on
            # the saturated cyan-blue rgb cell data → reads as glow on
            # the dark background.
            _prop.SetAmbient(0.85)
            _prop.SetDiffuse(0.10)
            _prop.SetSpecular(0.0)
            # Backface culling off so internal-facing triangles still
            # contribute to the "water film" feel.
            _prop.BackfaceCullingOff()
            _prop.FrontfaceCullingOff()

            membranes_state["actor"]     = _actor
            membranes_state["threshold"] = _thr
            membranes_state["mapper"]    = _mapper

            logger.info(
                "Water membranes ready: n_steps=%d  (toggle via 'Water' button)",
                membranes_state["n_steps"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise water membranes: %s", exc)
            membranes_state["actor"] = None

    _iren_mem = getattr(plt, "interactor", None)

    def _membrane_apply_step() -> None:
        thr = membranes_state["threshold"]
        if thr is None:
            return
        s = float(membranes_state["step"])
        try:
            thr.SetLowerThreshold(s)
            thr.SetUpperThreshold(s)
            thr.Modified()
            thr.Update()  # force pipeline execution before render
        except Exception as exc:  # noqa: BLE001
            logger.debug("Membranes threshold update failed: %s", exc)

    def _membrane_tick(_obj=None, _ev=None) -> None:
        if not membranes_state["visible"] or membranes_state["n_steps"] <= 0:
            return
        membranes_state["step"] = (
            (int(membranes_state["step"]) + 1) % int(membranes_state["n_steps"])
        )
        _membrane_apply_step()
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    if (_iren_mem is not None
            and membranes_state["actor"] is not None):
        try:
            _iren_mem.AddObserver("TimerEvent", _membrane_tick)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Membranes timer observer could not attach: %s", exc)

    def _membrane_start_timer() -> None:
        iren = getattr(plt, "interactor", None) or _iren_mem
        if (iren is None
                or membranes_state["timer_id"][0] is not None
                or membranes_state["actor"] is None):
            return
        try:
            membranes_state["timer_id"][0] = (
                iren.CreateRepeatingTimer(int(MEMBRANE_TICK_MS))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Membranes timer could not start: %s", exc)

    def _membrane_stop_timer() -> None:
        iren = getattr(plt, "interactor", None) or _iren_mem
        if iren is None or membranes_state["timer_id"][0] is None:
            return
        try:
            iren.DestroyTimer(membranes_state["timer_id"][0])
        except Exception:  # noqa: BLE001
            pass
        membranes_state["timer_id"][0] = None

    def _toggle_membranes(*_args, **_kwargs) -> None:
        logger.info("_toggle_membranes called (actor=%s, visible=%s)",
                    membranes_state["actor"], membranes_state["visible"])
        if membranes_state["actor"] is None:
            logger.warning("_toggle_membranes: actor is None — ignoring click")
            return
        want = not membranes_state["visible"]
        logger.info("_toggle_membranes: want=%s", "ON" if want else "OFF")
        if want:
            try:
                # Add the actor FIRST (mirrors the pillars trick: vedo /
                # vtk may reset mapper state on add, so style after add).
                plt.renderer.AddActor(membranes_state["actor"])
                membranes_state["visible"] = True
                _membrane_apply_step()
                _membrane_start_timer()
                logger.info("_toggle_membranes: Water ON — actor added, step=%d, timer=%s",
                            membranes_state["step"], membranes_state["timer_id"][0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Show membranes failed: %s", exc)
        else:
            try:
                _membrane_stop_timer()
                plt.renderer.RemoveActor(membranes_state["actor"])
                membranes_state["visible"] = False
                logger.info("_toggle_membranes: Water OFF — actor removed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hide membranes failed: %s", exc)
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    if membranes_state["actor"] is not None:
        plt.add_button(
            _toggle_membranes,
            states=["Water OFF", "Water ON"],
            c=["white", "white"],
            bc=["#22336e", "#3aa0ff"],
            pos=(0.88, 0.53),
            size=14,
            bold=True,
        )

    _ui_capturers.append(lambda: {
        "membranes_visible": bool(membranes_state["visible"]),
        "membranes_step":    int(membranes_state["step"]),
    })

    def _make_arrows_actor(data):
        """Build a uniform-length / uniform-thickness vedo Arrows actor
        from a data dict produced by :func:`_build_arrow_field` or
        :func:`_build_track_arrow_field`.  Only the colour varies, driven
        by ``astate["cmap_mode"]`` (``"angle"`` or ``"convergence"``)."""
        if data is None:
            return None
        try:
            import numpy as np
            import vedo as _vedo
            pts    = np.asarray(data["pts"],    dtype=np.float32)
            dirs   = np.asarray(data["dirs"],   dtype=np.float32)
            scores = np.asarray(data["scores"], dtype=np.float32)
            boost  = np.asarray(data.get("boost",
                        np.zeros(len(pts), dtype=np.float32)),
                                dtype=np.float32)
            if len(pts) == 0:
                return None
            base_length = float(astate["length"])
            ends = pts + dirs * base_length

            # 5-stop perceptual ramp blue → green → yellow → orange → red.
            stops = np.array([
                [0.45, 0.70, 1.00],   # light blue
                [0.20, 0.80, 0.30],   # green
                [0.95, 0.90, 0.15],   # yellow (slightly dark)
                [1.00, 0.55, 0.10],   # orange
                [0.90, 0.10, 0.10],   # red
            ], dtype=np.float32)

            if astate["cmap_mode"] == "convergence" and len(boost) > 1:
                # Defensive: scrub NaN/Inf that may sneak in from older
                # caches (np.percentile propagates NaN, which then makes
                # every t = NaN -> rgb = NaN -> uint8 = 0 -> BLACK arrows).
                b = np.nan_to_num(boost, nan=0.0, posinf=1.0, neginf=0.0)
                lo, hi = np.percentile(b, [5.0, 95.0])
                rng = float(hi - lo)
                if rng < 1e-6:
                    # All boost values are (essentially) equal — fall
                    # back to a constant mid-ramp colour rather than
                    # exploding the division.
                    t = np.full(len(b), 0.5, dtype=np.float32)
                else:
                    t = np.clip((b - lo) / rng, 0.0, 1.0).astype(np.float32)
            else:
                s = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
                t = np.clip(s, 0.0, 1.0).astype(np.float32)

            seg = t * (len(stops) - 1)
            i0 = np.clip(seg.astype(int), 0, len(stops) - 2)
            f  = (seg - i0)[:, None]
            rgb = stops[i0] * (1.0 - f) + stops[i0 + 1] * f
            colors = (rgb * 255).astype(np.uint8)

            act = _vedo.Arrows(
                pts, ends,
                c=colors,
                thickness=DEFAULT_ARROW_THICKNESS,
            )
            try:
                act.lighting("off")
            except Exception:  # noqa: BLE001
                pass
            try:
                act.linewidth(1).linecolor("black")
            except Exception:  # noqa: BLE001
                pass
            return act
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build arrows actor: %s", exc)
            return None

    def _make_tracks_actor_curves(data):
        """Build vedo actors from crown Dijkstra track data.

        Returns a list ``[lines_act, path_arrows_act]`` (fast path) or a
        single actor (slow fallback), or ``None`` on failure.

        Fast path (binary .vtp available):
          * Loads the pre-built polyline polydata, strides and splines it.
          * ``lines_act``  – translucent white polylines (alpha=0.30, lw=5).
          * ``path_arrows_act`` – 10 evenly-spaced ``vtkGlyph3D`` arrows per
            path coloured by arc-length fraction (blue→red ramp, t 0→1).
          * Both actors share a ``vtkPlane`` clip plane updated by a camera
            observer so that geometry more than half the volume's short-axis
            beyond the focal point is clipped away.

        Slow fallback (no .vtp): ``vedo.Lines`` from raw segment arrays.
        """
        if data is None:
            return None
        try:
            import time as _t
            import numpy as np
            import vedo as _vedo
            import vtk
            from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

            vtp_path = data.get("vtp_path")
            if vtp_path is not None and Path(vtp_path).exists():
                # ── Fast path: load pre-built binary .vtp ────────────────
                _t0 = _t.perf_counter()
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(str(vtp_path))
                reader.Update()
                pd = reader.GetOutput()
                n_cells = pd.GetNumberOfCells()
                logger.info(
                    "Crown tracks VTP loaded in %.3fs  (%d polylines, %d pts)",
                    _t.perf_counter() - _t0,
                    n_cells, pd.GetNumberOfPoints(),
                )
                if n_cells == 0:
                    logger.warning("Crown tracks VTP has no cells: %s", vtp_path)
                    return None

                # Subsample paths if stride > 1.
                ds = max(1, int(data.get("_display_stride", 1)))
                if ds > 1:
                    id_list = vtk.vtkIdList()
                    for cid in range(0, n_cells, ds):
                        id_list.InsertNextId(cid)
                    extract = vtk.vtkExtractCells()
                    extract.SetInputData(pd)
                    extract.SetCellList(id_list)
                    extract.Update()
                    geo = vtk.vtkGeometryFilter()
                    geo.SetInputConnection(extract.GetOutputPort())
                    geo.Update()
                    pd = geo.GetOutput()
                    logger.info(
                        "Crown tracks strided to %d / %d polylines (stride=%d)",
                        id_list.GetNumberOfIds(), n_cells, ds,
                    )

                # ── Spline interpolation: replace voxel-grid staircase ────
                # ``SetSubdivideToSpecified`` with N=3 emits only 3
                # segments for the *whole* polyline — that's why tracks
                # looked angular instead of softly curved.  64 gives a
                # genuinely smooth Kochanek spline at negligible cost.
                spline = vtk.vtkSplineFilter()
                spline.SetInputData(pd)
                spline.SetSubdivideToSpecified()
                spline.SetNumberOfSubdivisions(64)
                spline.Update()
                pd_smooth = spline.GetOutput()
                logger.info(
                    "Crown tracks splined: %d pts → %d pts",
                    pd.GetNumberOfPoints(), pd_smooth.GetNumberOfPoints(),
                )

                # ── Lines actor (translucent water-blue) ─────────────────
                # Pale cyan-blue to read as "water" rather than highlight.
                WATER_RGB = (0.72, 0.90, 1.00)
                lines_act = _vedo.Mesh(pd_smooth)
                try:
                    lines_act.mapper().ScalarVisibilityOff()
                except Exception:  # noqa: BLE001
                    pass
                lines_act.c(WATER_RGB).lw(4).alpha(0.18)
                try:
                    lines_act.lighting("off")
                except Exception:  # noqa: BLE001
                    pass

                # ── Path arrows: load pre-built .vtp or compute on-the-fly ──
                # Coloured with the same 5-stop blue→red ramp as the grid
                # arrows, but t = arc-length fraction along the path (0→1).
                path_arrows_act = None
                try:
                    arrows_vtp_path = data.get("arrows_vtp_path")
                    if (arrows_vtp_path is not None
                            and Path(arrows_vtp_path).exists()):
                        # ── Fast path: load pre-computed glyph centres ────
                        _ta0 = _t.perf_counter()
                        ar_reader = vtk.vtkXMLPolyDataReader()
                        ar_reader.SetFileName(str(arrows_vtp_path))
                        ar_reader.Update()
                        pd_g = ar_reader.GetOutput()
                        n_glyphs = pd_g.GetNumberOfPoints()
                        logger.info(
                            "Crown tracks arrows VTP loaded in %.3fs  "
                            "(%d glyph centres)",
                            _t.perf_counter() - _ta0, n_glyphs,
                        )
                    else:
                        # ── Slow path: compute arrow positions from splined
                        #    polylines (Python for-loop per path) ───────────
                        pts_np   = vtk_to_numpy(
                            pd_smooth.GetPoints().GetData()
                        ).astype(np.float64)                      # (total_pts, 3)
                        cell_arr = pd_smooth.GetLines()
                        offsets  = vtk_to_numpy(
                            cell_arr.GetOffsetsArray()
                        ).astype(np.int64)                         # (n_paths+1,)
                        conn     = vtk_to_numpy(
                            cell_arr.GetConnectivityArray()
                        ).astype(np.int64)                         # (total_pts,)
                        n_paths  = len(offsets) - 1
                        N_ARROWS  = 10
                        t_samples = (
                            np.arange(N_ARROWS, dtype=np.float64) + 0.5
                        ) / N_ARROWS                              # [0.05 … 0.95]
                        all_pos  = np.empty(
                            (n_paths * N_ARROWS, 3), dtype=np.float64
                        )
                        all_tan  = np.empty(
                            (n_paths * N_ARROWS, 3), dtype=np.float64
                        )
                        all_t    = np.tile(
                            t_samples.astype(np.float32), n_paths
                        )
                        for i in range(n_paths):
                            s0 = int(offsets[i])
                            s1 = int(offsets[i + 1])
                            pp = pts_np[conn[s0:s1]]           # (L, 3)
                            L  = len(pp)
                            out = slice(i * N_ARROWS, (i + 1) * N_ARROWS)
                            if L < 2:
                                all_pos[out] = pp[0] if L == 1 else 0.0
                                all_tan[out] = [1.0, 0.0, 0.0]
                                continue
                            diffs    = np.diff(pp, axis=0)
                            seg_lens = np.maximum(
                                np.linalg.norm(diffs, axis=1), 1e-9
                            )
                            cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
                            tot = cum[-1]
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
                            sidx = np.clip(
                                np.searchsorted(cum[1:], ss, side="left"),
                                0, L - 2,
                            )
                            tans  = diffs[sidx].copy()
                            tnorm = np.maximum(
                                np.linalg.norm(tans, axis=1, keepdims=True),
                                1e-9,
                            )
                            all_tan[out] = tans / tnorm

                        # Build glyph-input vtkPolyData.
                        vtk_pts_g = vtk.vtkPoints()
                        vtk_pts_g.SetDataTypeToDouble()
                        vtk_pts_g.SetData(
                            numpy_to_vtk(all_pos, deep=True,
                                         array_type=vtk.VTK_DOUBLE)
                        )
                        pd_g = vtk.vtkPolyData()
                        pd_g.SetPoints(vtk_pts_g)

                        tan_arr = numpy_to_vtk(
                            all_tan, deep=True, array_type=vtk.VTK_DOUBLE
                        )
                        tan_arr.SetName("tangents")
                        pd_g.GetPointData().AddArray(tan_arr)
                        pd_g.GetPointData().SetActiveVectors("tangents")

                        t_arr = numpy_to_vtk(
                            all_t, deep=True, array_type=vtk.VTK_FLOAT
                        )
                        t_arr.SetName("t")
                        pd_g.GetPointData().AddArray(t_arr)
                        pd_g.GetPointData().SetActiveScalars("t")
                        n_paths_str = str(n_paths)
                        n_glyphs = n_paths * N_ARROWS
                        logger.info(
                            "Crown path arrows computed from splined polylines: "
                            "%d glyphs (%s paths × %d)",
                            n_glyphs, n_paths_str, N_ARROWS,
                        )

                    if pd_g.GetNumberOfPoints() > 0:
                        # ── Arrow source ──────────────────────────────────
                        arrow_src = vtk.vtkArrowSource()
                        arrow_src.SetTipLength(0.35)
                        arrow_src.SetTipRadius(0.12)
                        arrow_src.SetShaftRadius(0.06)
                        arrow_src.Update()

                        # ── Glyph3D ───────────────────────────────────────
                        glyph = vtk.vtkGlyph3D()
                        glyph.SetSourceConnection(arrow_src.GetOutputPort())
                        glyph.SetInputData(pd_g)
                        glyph.SetVectorModeToUseVector()
                        glyph.OrientOn()
                        glyph.SetScaleModeToNoDataScaling()
                        glyph.SetScaleFactor(DEFAULT_ARROW_LENGTH * 1.2)
                        glyph.SetColorModeToColorByScalar()
                        glyph.Update()

                        # ── 5-stop LUT: blue → green → yellow → orange → red
                        stops_lut = np.array([
                            [0.45, 0.70, 1.00],
                            [0.20, 0.80, 0.30],
                            [0.95, 0.90, 0.15],
                            [1.00, 0.55, 0.10],
                            [0.90, 0.10, 0.10],
                        ], dtype=np.float64)
                        lut = vtk.vtkLookupTable()
                        lut.SetNumberOfTableValues(256)
                        lut.SetRange(0.0, 1.0)
                        for k in range(256):
                            t_k = k / 255.0
                            seg = t_k * 4.0
                            i0  = min(int(seg), 3)
                            f   = seg - i0
                            rgb = (
                                stops_lut[i0] * (1.0 - f)
                                + stops_lut[i0 + 1] * f
                            )
                            lut.SetTableValue(k, rgb[0], rgb[1], rgb[2], 1.0)
                        lut.Build()

                        path_arrows_act = _vedo.Mesh(glyph.GetOutput())
                        m = path_arrows_act.mapper()
                        m.SetLookupTable(lut)
                        m.SetScalarRange(0.0, 1.0)
                        m.ScalarVisibilityOn()
                        m.SetColorModeToMapScalars()
                        try:
                            path_arrows_act.lighting("off")
                        except Exception:  # noqa: BLE001
                            pass
                        logger.info(
                            "Crown path arrows actor: %d glyphs", n_glyphs,
                        )
                except Exception as exc_arr:  # noqa: BLE001
                    logger.warning(
                        "Could not build path arrows: %s", exc_arr
                    )
                    path_arrows_act = None

                # ── Bin-based visibility along the long axis ─────────────
                # Group polylines (and path-arrow glyph centres) into
                # N_BINS equal-width slabs along the volume's longest
                # axis, and toggle whole sub-actors visible/invisible
                # based on the camera's coordinate along that axis.
                # Only bins within ±BIN_RADIUS of the camera bin are
                # rendered (≈ 1/6 of bins ahead + 1/6 behind for N=12).
                # ~1/3 of bins visible total (matches the previous 12/2
                # ratio); the inner half is rendered at full opacity, the
                # outer half fades linearly to 0.
                N_BINS          = 100
                BIN_RADIUS_FULL = 8    # full opacity within ±FULL
                BIN_RADIUS_FADE = 16   # linear fade from FULL to FADE

                bounds  = pd_smooth.GetBounds()
                extents = np.array([
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                ], dtype=np.float64)
                long_axis_idx = int(np.argmax(extents))
                axis_col = long_axis_idx
                axis_min = float(bounds[2 * long_axis_idx])
                axis_max = float(bounds[2 * long_axis_idx + 1])
                axis_len = max(axis_max - axis_min, 1e-9)
                axis_vec = np.zeros(3, dtype=np.float64)
                axis_vec[long_axis_idx] = 1.0

                # ── Per-cell bin assignment for the splines ──────────────
                pts_sm = vtk_to_numpy(
                    pd_smooth.GetPoints().GetData()
                )                                             # (Npts, 3)
                ca_sm = pd_smooth.GetLines()
                offs_sm = vtk_to_numpy(
                    ca_sm.GetOffsetsArray()
                ).astype(np.int64)                            # (n_cells+1,)
                conn_sm = vtk_to_numpy(
                    ca_sm.GetConnectivityArray()
                ).astype(np.int64)                            # (Npts_used,)
                coord_per_pt = pts_sm[conn_sm, axis_col].astype(np.float64)
                # Sum per cell via np.add.reduceat on cell start offsets.
                seg_sum    = np.add.reduceat(coord_per_pt, offs_sm[:-1])
                seg_counts = np.maximum(np.diff(offs_sm).astype(np.float64),
                                        1.0)
                mean_coord = seg_sum / seg_counts
                cell_bin = np.clip(
                    ((mean_coord - axis_min) / axis_len * N_BINS).astype(int),
                    0, N_BINS - 1,
                )

                line_bin_actors: list = []
                for bi in range(N_BINS):
                    ids = np.where(cell_bin == bi)[0]
                    if len(ids) == 0:
                        continue
                    id_list = vtk.vtkIdList()
                    id_list.SetNumberOfIds(len(ids))
                    for k, cid in enumerate(ids):
                        id_list.SetId(k, int(cid))
                    extr = vtk.vtkExtractCells()
                    extr.SetInputData(pd_smooth)
                    extr.SetCellList(id_list)
                    extr.Update()
                    geo = vtk.vtkGeometryFilter()
                    geo.SetInputConnection(extr.GetOutputPort())
                    geo.Update()
                    sub = geo.GetOutput()
                    a = _vedo.Mesh(sub)
                    try:
                        a.mapper().ScalarVisibilityOff()
                    except Exception:  # noqa: BLE001
                        pass
                    a.c(WATER_RGB).lw(4).alpha(0.18)
                    try:
                        a.lighting("off")
                    except Exception:  # noqa: BLE001
                        pass
                    a.name = f"track_lines_bin_{bi}"
                    a._bin_idx = bi
                    a._base_alpha = 0.18
                    line_bin_actors.append(a)
                # Drop the full-resolution actor — bin actors replace it.
                lines_act = None

                # ── Per-point bin assignment for path arrows ─────────────
                arrow_bin_actors: list = []
                _pd_g = locals().get("pd_g", None)
                _arrow_src = locals().get("arrow_src", None)
                _lut = locals().get("lut", None)
                if (path_arrows_act is not None
                        and _pd_g is not None
                        and _arrow_src is not None
                        and _lut is not None
                        and _pd_g.GetNumberOfPoints() > 0):
                    pts_g = vtk_to_numpy(_pd_g.GetPoints().GetData())
                    tan_full = _pd_g.GetPointData().GetArray("tangents")
                    t_full   = _pd_g.GetPointData().GetArray("t")
                    if tan_full is not None and t_full is not None:
                        tan_g = vtk_to_numpy(tan_full)
                        t_g   = vtk_to_numpy(t_full)
                        pt_bin = np.clip(
                            ((pts_g[:, axis_col] - axis_min)
                             / axis_len * N_BINS).astype(int),
                            0, N_BINS - 1,
                        )
                        for bi in range(N_BINS):
                            idxs = np.where(pt_bin == bi)[0]
                            if len(idxs) == 0:
                                continue
                            sub_pd = vtk.vtkPolyData()
                            sub_pts = vtk.vtkPoints()
                            sub_pts.SetDataTypeToDouble()
                            sub_pts.SetData(numpy_to_vtk(
                                pts_g[idxs].astype(np.float64),
                                deep=True, array_type=vtk.VTK_DOUBLE,
                            ))
                            sub_pd.SetPoints(sub_pts)
                            ta = numpy_to_vtk(
                                tan_g[idxs].astype(np.float64),
                                deep=True, array_type=vtk.VTK_DOUBLE,
                            )
                            ta.SetName("tangents")
                            sub_pd.GetPointData().AddArray(ta)
                            sub_pd.GetPointData().SetActiveVectors("tangents")
                            sa = numpy_to_vtk(
                                t_g[idxs].astype(np.float32),
                                deep=True, array_type=vtk.VTK_FLOAT,
                            )
                            sa.SetName("t")
                            sub_pd.GetPointData().AddArray(sa)
                            sub_pd.GetPointData().SetActiveScalars("t")
                            gl = vtk.vtkGlyph3D()
                            gl.SetSourceConnection(
                                _arrow_src.GetOutputPort()
                            )
                            gl.SetInputData(sub_pd)
                            gl.SetVectorModeToUseVector()
                            gl.OrientOn()
                            gl.SetScaleModeToNoDataScaling()
                            gl.SetScaleFactor(DEFAULT_ARROW_LENGTH * 1.2)
                            gl.SetColorModeToColorByScalar()
                            gl.Update()
                            a = _vedo.Mesh(gl.GetOutput())
                            mm = a.mapper()
                            mm.SetLookupTable(_lut)
                            mm.SetScalarRange(0.0, 1.0)
                            mm.ScalarVisibilityOn()
                            mm.SetColorModeToMapScalars()
                            try:
                                a.lighting("off")
                            except Exception:  # noqa: BLE001
                                pass
                            a.name = f"track_arrows_bin_{bi}"
                            a._bin_idx = bi
                            a._base_alpha = 1.0
                            arrow_bin_actors.append(a)
                # Drop the full-resolution arrows actor.
                path_arrows_act = None

                all_bin_actors = line_bin_actors + arrow_bin_actors

                # ── Camera observer: toggle bin visibility ───────────────
                if _track_cam_obs_tag[0] is not None:
                    try:
                        plt.renderer.GetActiveCamera().RemoveObserver(
                            _track_cam_obs_tag[0]
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    _track_cam_obs_tag[0] = None

                def _update_bins(obj, event):  # noqa: ANN001
                    cam = plt.renderer.GetActiveCamera()
                    pos = np.array(cam.GetPosition(), dtype=np.float64)
                    cam_a = float(np.dot(pos, axis_vec))
                    cam_a = max(axis_min, min(axis_max, cam_a))
                    cam_bin = int(np.clip(
                        int((cam_a - axis_min) / axis_len * N_BINS),
                        0, N_BINS - 1,
                    ))
                    fade_span = max(BIN_RADIUS_FADE - BIN_RADIUS_FULL, 1)
                    for a in all_bin_actors:
                        d = abs(a._bin_idx - cam_bin)
                        if d <= BIN_RADIUS_FULL:
                            mult = 1.0
                        elif d <= BIN_RADIUS_FADE:
                            mult = 1.0 - (d - BIN_RADIUS_FULL) / fade_span
                        else:
                            mult = 0.0
                        vis = 1 if mult > 0.0 else 0
                        try:
                            a.actor.SetVisibility(vis)
                        except Exception:  # noqa: BLE001
                            try:
                                a.SetVisibility(vis)
                            except Exception:  # noqa: BLE001
                                pass
                        if vis:
                            try:
                                a.alpha(a._base_alpha * mult)
                            except Exception:  # noqa: BLE001
                                pass

                tag = plt.renderer.GetActiveCamera().AddObserver(
                    "ModifiedEvent", _update_bins
                )
                _track_cam_obs_tag[0] = tag
                _update_bins(None, None)   # initialise with current camera

                logger.info(
                    "Crown tracks binned: %d line-bins + %d arrow-bins "
                    "(axis=%d, N_BINS=%d, full=%d, fade=%d)",
                    len(line_bin_actors), len(arrow_bin_actors),
                    long_axis_idx, N_BINS,
                    BIN_RADIUS_FULL, BIN_RADIUS_FADE,
                )
                return all_bin_actors

            # ── Slow fallback: raw segment arrays (no .vtp available) ────
            segs_start = np.asarray(data["segs_start"], dtype=np.float32)
            segs_end   = np.asarray(data["segs_end"],   dtype=np.float32)
            if len(segs_start) == 0:
                logger.warning("Crown tracks: empty segments array.")
                return None
            act = _vedo.Lines(segs_start, segs_end, c=(0.72, 0.90, 1.00), lw=4)
            act.alpha(0.18)
            try:
                act.lighting("off")
            except Exception:  # noqa: BLE001
                pass
            return act
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build tracks curves actor: %s", exc)
            return None

    def _rebuild_grid_arrows():
        nonlocal arrows_actor
        if arrows_data is None:
            return
        new = _make_arrows_actor(arrows_data)
        if new is None:
            return
        new.name = "geoddist_arrows"
        if view_mode["name"] == "arrows_grid" and arrows_actor is not None:
            try: plt.remove(arrows_actor)
            except Exception: pass
            try: plt.add(new)
            except Exception: pass
        arrows_actor = new

    def _rebuild_tracks_arrows():
        nonlocal tracks_actor
        if tracks_data is None:
            return
        # Crown Dijkstra format has segs_end; old ThinLayer format has pts/dirs.
        if "segs_end" in tracks_data:
            new = _make_tracks_actor_curves(tracks_data)
        else:
            new = _make_arrows_actor(tracks_data)
        if new is None:
            return
        # Set debug name(s) on the returned actor(s).
        if isinstance(new, list):
            for i, a in enumerate(new):
                if a is not None:
                    try:
                        a.name = f"track_arrows_{i}"
                    except Exception:  # noqa: BLE001
                        pass
        else:
            new.name = "track_arrows"
        if view_mode["name"] == "arrows_tracks" and tracks_actor is not None:
            _plt_remove(tracks_actor)
            _plt_add(new)
        tracks_actor = new

    def _rebuild_arrows_actor():
        """Called by the length slider and the colormap toggle: rebuild
        both arrow data sets (each becomes visible at most once)."""
        _rebuild_grid_arrows()
        # Only rebuild tracks if the lazy-build has already happened —
        # otherwise the slider would force the expensive first build.
        if tracks_actor is not None:
            _rebuild_tracks_arrows()

    import time as _time_diag
    _t0 = _time_diag.perf_counter()
    _rebuild_grid_arrows()
    logger.info(
        "Grid arrows actor built in %.2fs (%d pts).",
        _time_diag.perf_counter() - _t0,
        len(arrows_data["pts"]) if arrows_data is not None else 0,
    )
    # Tracks actor is built lazily on first entry into "arrows_tracks" mode:
    # building vedo.Lines with millions of segments may take a while and
    # would freeze the interactor if done eagerly here.

    any_arrows = (arrows_actor is not None) or (tracks_data is not None)
    if any_arrows:
        def _arrow_slider_cb(key):
            def _cb(widget, _event):
                astate[key] = float(widget.value)
                _rebuild_arrows_actor()
                plt.render()
            return _cb

        arrow_slider_defs = [
            ("length", 0.5, 15.0, 0.04),
        ]
        for key, vmin, vmax, y in arrow_slider_defs:
            sw = plt.add_slider(
                _arrow_slider_cb(key),
                vmin, vmax, value=astate[key],
                pos=((x1, y), (x2, y)),
                title=f"arr·{key}",
                title_size=0.7,
                show_value=True,
            )
            arrow_sliders.append(sw)
            _set_widget_visible(sw, False)  # hidden until Arrows mode

        # ── Pillars hue slider (above the arrow-length slider) ─────────
        # Rotates the base teal colour of the Pillars mesh in HSV space.
        # Applies to both the transparent "glow" style (mesh view) and
        # the opaque "solid" style (arrows view) — the actual colour is
        # always derived from `pillars_state["hue_shift"]` via
        # `_apply_pillars_style`.
        if pillars_mesh is not None:
            def _pillars_hue_cb(widget, _event):
                pillars_state["hue_shift"] = float(widget.value)
                _apply_pillars_style(None)  # refresh current style colour
                plt.render()
            _pillars_hue_slider = plt.add_slider(
                _pillars_hue_cb,
                0.0, 1.0, value=pillars_state["hue_shift"],
                pos=((x1, 0.10), (x2, 0.10)),
                title="pillars·hue",
                title_size=0.7,
                show_value=True,
            )
            arrow_sliders.append(_pillars_hue_slider)
            _set_widget_visible(_pillars_hue_slider, False)

        # Build the list of available view modes (4 max).
        # Order matters: it drives the vertical stacking of the radio
        # buttons (top → bottom).
        modes: list[str] = ["mesh_bridges"]
        if alt_mesh is not None:
            modes.append("mesh_all")
        if arrows_actor is not None:
            modes.append("arrows_grid")
        if tracks_data is not None:
            modes.append("arrows_tracks")

        def _actor_for(name):
            if name == "mesh_bridges":
                return mesh
            if name == "mesh_all":
                return alt_mesh
            if name == "arrows_grid":
                return arrows_actor
            if name == "arrows_tracks":
                return tracks_actor
            return None

        def _is_arrows_mode(name: str) -> bool:
            return name in ("arrows_grid", "arrows_tracks")

        def _enter_mode(new_mode: str) -> None:
            cur = view_mode["name"]
            if cur == new_mode:
                return
            cur_actor = _actor_for(cur)
            _plt_remove(cur_actor)
            view_mode["name"] = new_mode
            # Keep `active["mesh"]` in sync (used by `_pillars_should_show`
            # and the status banner).
            if new_mode == "mesh_bridges":
                active["mesh"] = mesh
            elif new_mode == "mesh_all":
                active["mesh"] = alt_mesh
            # Lazy-build the tracks actor on first entry into this mode.
            if new_mode == "arrows_tracks" and tracks_actor is None:
                _show_status("Building tracks lines (first time)…")
                try: plt.render()
                except Exception: pass
                _t = _time_diag.perf_counter()
                _rebuild_tracks_arrows()
                logger.info(
                    "Tracks lines actor built in %.2fs (%d segments).",
                    _time_diag.perf_counter() - _t,
                    len(tracks_data["segs_start"])
                    if tracks_data is not None else 0,
                )
            new_actor = _actor_for(new_mode)
            _plt_add(new_actor)
            _refresh_pillars_visibility()
            in_arrows = _is_arrows_mode(new_mode)
            for s in mesh_sliders:
                _set_widget_visible(s, not in_arrows)
            for s in arrow_sliders:
                _set_widget_visible(s, in_arrows)
            _set_button_visible(_cmap_btn, in_arrows)
            if _density_btn is not None:
                _set_button_visible(
                    _density_btn,
                    (new_mode in ("mesh_bridges", "mesh_all"))
                    and _density_has_data,
                )
            if new_mode in ("mesh_bridges", "mesh_all"):
                _show_status(_active_mesh_title())
            elif new_mode == "arrows_grid":
                _show_status(STATUS_TITLE_ARROWS_GRID)
            elif new_mode == "arrows_tracks":
                _show_status(STATUS_TITLE_ARROWS_TRACKS)
            plt.render()

        # ─── 4 exclusive radio buttons ──────────────────────────────────
        # One button per available view mode.  Clicking a button enters
        # that mode; the other three are visually dimmed.
        _view_buttons: dict = {}

        _btn_positions = {
            "mesh_bridges":  (0.88, 0.77),
            "mesh_all":      (0.88, 0.71),
            "arrows_grid":   (0.88, 0.65),
            "arrows_tracks": (0.88, 0.59),
        }
        _btn_active_bg = {
            "mesh_bridges":  "#37474f",   # blue-grey
            "mesh_all":      "#5d4037",   # brown
            "arrows_grid":   "#ad1457",   # magenta
            "arrows_tracks": "#6a1b9a",   # purple
        }
        _btn_inactive_bg = "#263238"      # darker neutral
        _btn_label = {
            "mesh_bridges":  "Mesh ▸ Cortical bridges",
            "mesh_all":      "Mesh ▸ All watered tissues",
            "arrows_grid":   "Arrows ▸ grid",
            "arrows_tracks": "Arrows ▸ tracks",
        }

        def _refresh_view_buttons():
            cur = view_mode["name"]
            for _name, _btn in _view_buttons.items():
                try:
                    _btn.status(0 if _name == cur else 1)
                except Exception:  # noqa: BLE001
                    # Older vedo: fallback via switch() until state matches.
                    try:
                        want = 0 if _name == cur else 1
                        if _btn.status() != _btn.states[want]:
                            _btn.switch()
                    except Exception:  # noqa: BLE001
                        pass

        def _make_view_cb(target_mode):
            def _cb(*_a, **_kw):
                if view_mode["name"] == target_mode:
                    return
                _enter_mode(target_mode)
                _refresh_view_buttons()
            return _cb

        for _name in modes:
            _label = _btn_label[_name]
            _active_bg = _btn_active_bg[_name]
            _btn = plt.add_button(
                _make_view_cb(_name),
                # Two visual states: 0 = active (bright), 1 = inactive (dim).
                states=[_label, _label],
                c=["white", "#9e9e9e"],
                bc=[_active_bg, _btn_inactive_bg],
                pos=_btn_positions[_name],
                size=14,
                bold=True,
            )
            _view_buttons[_name] = _btn

        _refresh_view_buttons()

        # ─── Colormap-mode toggle (Angle ↔ Convergence) ─────────────────
        # Only visible in any Arrows view.  Drives which scalar feeds
        # the 5-stop perceptual ramp.
        _cmap_status = {
            "angle":       "Arrow colormap: angle to central axis",
            "convergence": "Arrow colormap: local convergence rate",
        }

        def _toggle_cmap_cb(*_a, **_kw):
            astate["cmap_mode"] = (
                "convergence" if astate["cmap_mode"] == "angle" else "angle"
            )
            _rebuild_arrows_actor()
            _cmap_btn.switch()
            _show_status(_cmap_status[astate["cmap_mode"]])
            plt.render()

        _cmap_btn = plt.add_button(
            _toggle_cmap_cb,
            states=["Arrow color ▸ Angle", "Arrow color ▸ Convergence"],
            c=["white", "white"],
            bc=["#283593", "#bf360c"],
            pos=(0.88, 0.47),
            size=14,
            bold=True,
        )
        # Hidden until an Arrows view is active.
        _set_button_visible(_cmap_btn, False)
        if arrows_actor is not None:
            logger.info("Arrows (grid) ready (%d arrows).",
                        len(arrows_data["pts"]))
        if tracks_data is not None:
            logger.info("Crown tracks loaded (%d segments) -- actor built on first view.",
                        len(tracks_data["segs_start"]))
    else:
        logger.info("No arrow field available – mesh/arrows toggle disabled.")

    # ─── Density colormap toggle ────────────────────────────────────────
    # Applies a viridis colormap to per-face density scalars on the
    # cortical-bridges mesh and on the all-watered-tissues mesh. Visible
    # only in the corresponding mesh modes.
    if _density_has_data:
        _density_body_rgb = tuple(c / 255.0 for c in BODY_COLOR)

        def _apply_density_to(target_mesh, scalars):
            if target_mesh is None or scalars is None:
                return
            try:
                target_mesh.cmap("viridis", scalars, on="cells",
                                 vmin=0.0, vmax=1.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Density: could not apply cmap: %s", exc)

        def _restore_solid(target_mesh):
            if target_mesh is None:
                return
            # 1) Turn the scalar mapping off so the cmap stops driving
            #    the actor's colour.  We also detach the per-cell scalar
            #    array from active selection so VTK does not try to keep
            #    a LUT alive on the mapper.
            try:
                mp = target_mesh.mapper
                if callable(mp):
                    mp = mp()
                mp.ScalarVisibilityOff()
            except Exception:  # noqa: BLE001
                pass
            try:
                poly = target_mesh.dataset
                cd = poly.GetCellData()
                cd.SetActiveScalars(None)
            except Exception:  # noqa: BLE001
                pass
            # 2) Restore the solid body colour + lighting *without*
            #    recomputing normals (the original mesh normals are
            #    still valid, and recomputing them on the full mesh
            #    can freeze the UI for several seconds — that was the
            #    "density-toggle freeze" bug).
            base = getattr(target_mesh, "_body_color", BODY_COLOR)
            try:
                rgb = _hue_shifted_rgb(base, state.get("hue", 0.0))
                target_mesh.color(list(rgb))
            except Exception:  # noqa: BLE001
                try:
                    target_mesh.c([c / 255.0 for c in base])
                except Exception:  # noqa: BLE001
                    pass
            try:
                target_mesh.alpha(state.get("opacity", OPACITY))
            except Exception:  # noqa: BLE001
                pass
            try:
                _set_lighting(
                    target_mesh,
                    state.get("ambient",  AMBIENT),
                    state.get("diffuse",  DIFFUSE),
                    state.get("specular", SPECULAR),
                )
            except Exception:  # noqa: BLE001
                pass

        def _toggle_density_cb(_obj=None, _ev=None):
            new_on = not _density_state["on"]
            _density_state["on"] = new_on
            if new_on:
                _apply_density_to(mesh, density_scalars.get("bridges"))
                _apply_density_to(alt_mesh, density_scalars.get("all"))
                _show_status("Density colormap: on (viridis)")
            else:
                _restore_solid(mesh)
                _restore_solid(alt_mesh)
                _show_status("Density colormap: off")
            try:
                _density_btn.switch()
            except Exception:  # noqa: BLE001
                pass
            plt.render()

        _density_btn = plt.add_button(
            _toggle_density_cb,
            states=["Density colormap ▸ off", "Density colormap ▸ on"],
            c=["white", "white"],
            bc=["#263238", "#0277bd"],
            pos=(0.72, 0.77),
            size=14,
            bold=True,
        )
        # Visible only in Mesh modes; default mode is mesh_bridges so
        # show it immediately.
        _set_button_visible(_density_btn, True)
    else:
        logger.info("Density colormap data unavailable – toggle disabled.")

    # ─── Spaceship navigation (bottom-left, translation + rotation) ────
    # Two 4-row × 3-col control panels stacked in the bottom-left corner
    # of the window (well clear of the right-side mesh / arrow buttons
    # and of the top-left ortho-panel overlay).
    #
    # Spaceship physics:
    #   * Translation velocity is stored in the *world* frame.  Each
    #     thrust button (FWD / BCK / ▲ / ▼ / ◀ / ▶) applies a fixed
    #     velocity impulse along the corresponding camera-local axis,
    #     evaluated at the moment of the click, and accumulated into
    #     the world-frame velocity vector.  Once gained, that velocity
    #     persists (no friction) until either an opposite impulse
    #     cancels it or STOP zeroes it.
    #   * Rotation velocity (yaw / pitch / roll) is stored in the
    #     camera-local frame and integrated the same way.
    #   * Each tick (≈30 Hz) advances the camera by ``velocity * dt``.
    import numpy as _np_ship
    import time as _time_ship

    # Thrust model:
    #   * 1st click (from rest) adds a *very small* impulse — about
    #     1/500 of the scene diagonal per second — so you start gently.
    #   * Subsequent clicks add an impulse whose magnitude is
    #     proportional to the *current* velocity magnitude
    #     (≈ ``_SHIP_THRUST_K`` × |v|), so the spaceship accelerates
    #     geometrically the more you press in the same direction.
    #   * Pressing the opposite direction subtracts the same impulse,
    #     so two consecutive opposite presses cancel down towards zero.
    #
    # Rotation uses the same idea per axis (yaw / pitch / roll
    # treated independently), with a 1 °/s base impulse.
    _SHIP_BASE_T_FRAC = 1.0 / 500.0   # diag-fraction / s at rest
    _SHIP_BASE_R_DPS  = 1.0           # deg / s at rest
    _SHIP_THRUST_K    = 0.30          # impulse ≈ 30 % of current speed
    try:
        _xmn, _xmx, _ymn, _ymx, _zmn, _zmx = mesh.bounds()
        _scene_diag_ship = float(
            ((_xmx - _xmn) ** 2 + (_ymx - _ymn) ** 2 + (_zmx - _zmn) ** 2) ** 0.5
        )
    except Exception:  # noqa: BLE001
        _scene_diag_ship = 1000.0
    _SHIP_BASE_T = _SHIP_BASE_T_FRAC * _scene_diag_ship
    _ship = {
        # World-frame translation velocity (x, y, z) in scene units / s.
        "v_world": _np_ship.zeros(3, dtype=_np_ship.float64),
        # Camera-local rotation velocity:
        #   axis 0 = yaw   (+ = view-up axis)
        #   axis 1 = pitch (+ = right axis)
        #   axis 2 = roll  (+ = forward axis)
        "v_r": [0.0, 0.0, 0.0],
        "last_t": None,
    }

    def _ship_camera_axes():
        """Return (forward, right, up) unit vectors in world frame.

        Returns ``None`` if the camera basis is degenerate."""
        cam = plt.camera
        try:
            fp  = _np_ship.array(cam.GetFocalPoint(), dtype=_np_ship.float64)
            pos = _np_ship.array(cam.GetPosition(),   dtype=_np_ship.float64)
            up_in = _np_ship.array(cam.GetViewUp(),   dtype=_np_ship.float64)
        except Exception:  # noqa: BLE001
            return None
        fwd = fp - pos
        fn = float(_np_ship.linalg.norm(fwd))
        if fn < 1e-9:
            return None
        fwd_u = fwd / fn
        right = _np_ship.cross(fwd_u, up_in)
        rn = float(_np_ship.linalg.norm(right))
        if rn < 1e-9:
            return None
        right /= rn
        up = _np_ship.cross(right, fwd_u)
        un = float(_np_ship.linalg.norm(up))
        if un < 1e-9:
            return None
        up /= un
        return fwd_u, right, up

    def _ship_any_active() -> bool:
        if any(abs(v) > 0.0 for v in _ship["v_world"]):
            return True
        return any(abs(v) > 0.0 for v in _ship["v_r"])

    def _ship_impulse(kind: str, axis: int, sign: int) -> None:
        """Apply a progressive thrust impulse.

        From rest, the impulse magnitude is a small base value
        (``_SHIP_BASE_T`` for translation, ``_SHIP_BASE_R_DPS`` for
        rotation).  Once moving, the impulse becomes proportional to
        the current speed (``_SHIP_THRUST_K`` × |v|), so each click in
        the same direction accelerates the ship geometrically.

        ``kind == 't'`` adds a translation impulse along the
        camera-local axis (0=forward, 1=right, 2=up), converted to the
        world frame at click time.  ``kind == 'r'`` adds a rotation
        impulse around the camera-local axis (0=yaw, 1=pitch, 2=roll).
        """
        if kind == "t":
            axes = _ship_camera_axes()
            if axes is None:
                return
            fwd_u, right, up = axes
            local = (fwd_u, right, up)[axis]
            v_w = _ship["v_world"]
            v_norm = float(_np_ship.linalg.norm(v_w))
            mag = max(_SHIP_BASE_T, _SHIP_THRUST_K * v_norm)
            _ship["v_world"] = v_w + float(sign) * mag * local
            v = _ship["v_world"]
            _show_status(
                f"Navigating observer thrust (+{mag:.2f}/s) → "
                f"|v|={float(_np_ship.linalg.norm(v)):.2f}/s"
            )
            _ensure_ship_timer()
        else:
            cur = _ship["v_r"][axis]
            mag = max(_SHIP_BASE_R_DPS, _SHIP_THRUST_K * abs(cur))
            _ship["v_r"][axis] = cur + float(sign) * mag
            label = ["yaw", "pitch", "roll"][axis]
            _show_status(
                f"Navigating observer thrust {label} (+{mag:.2f} °/s) → "
                f"{_ship['v_r'][axis]:+.2f} °/s"
            )
            _ensure_ship_timer()

    def _ship_stop(kind: str) -> None:
        if kind == "t":
            _ship["v_world"] = _np_ship.zeros(3, dtype=_np_ship.float64)
        else:
            for i in range(3):
                _ship["v_r"][i] = 0.0
        _show_status(
            "Navigating observer thrust "
            + ("translation" if kind == "t" else "rotation")
            + " stop"
        )
        _ship["last_t"] = None
        if not _ship_any_active():
            _stop_ship_timer()

    def _ship_tick(_obj=None, _ev=None) -> None:
        if not _ship_any_active():
            _ship["last_t"] = None
            # Kill the repeating timer when idle so VTK's trackball
            # interactor stops receiving TimerEvents (which would
            # otherwise re-apply the last rotate/pan delta and feel
            # like inertia during a click-and-drag).
            _stop_ship_timer()
            return
        now = _time_ship.perf_counter()
        if _ship["last_t"] is None:
            _ship["last_t"] = now
            return
        dt = now - _ship["last_t"]
        _ship["last_t"] = now
        if dt <= 0.0 or dt > 1.0:
            return
        cam = plt.camera

        # Translation: world-frame velocity integrated directly.
        v_w = _ship["v_world"]
        moved = False
        if v_w[0] or v_w[1] or v_w[2]:
            try:
                fp  = _np_ship.array(cam.GetFocalPoint(),
                                     dtype=_np_ship.float64)
                pos = _np_ship.array(cam.GetPosition(),
                                     dtype=_np_ship.float64)
            except Exception:  # noqa: BLE001
                return
            delta = v_w * dt
            cam.SetPosition(*(pos + delta))
            cam.SetFocalPoint(*(fp + delta))
            moved = True

        # Rotation: camera-local angular velocity.
        vr = _ship["v_r"]
        if vr[0]:
            cam.Yaw(vr[0] * dt)
            moved = True
        if vr[1]:
            cam.Pitch(vr[1] * dt)
            moved = True
        if vr[2]:
            cam.Roll(vr[2] * dt)
            moved = True

        # Keep near/far planes sane while the spaceship is moving —
        # otherwise the object gets clipped when you fly close to it.
        if moved:
            try:
                for _ren in plt.renderers:
                    _ren.ResetCameraClippingRange()
            except Exception:  # noqa: BLE001
                try:
                    plt.renderer.ResetCameraClippingRange()
                except Exception:  # noqa: BLE001
                    pass

        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    _iren_ship = getattr(plt, "interactor", None)
    _ship_timer_id: list = [None]

    def _ensure_ship_timer() -> None:
        """Create the repeating timer on demand.

        We *only* keep the timer alive while the ship is actually
        moving, because VTK's ``vtkInteractorStyleTrackballCamera``
        responds to every ``TimerEvent`` by re-applying the last
        rotate/pan/zoom delta when a mouse button is held — which
        makes the trackball feel like it has inertia.  Killing the
        timer when v ≡ 0 restores the vanilla trackball behaviour.
        """
        if _iren_ship is None or _ship_timer_id[0] is not None:
            return
        try:
            _ship_timer_id[0] = _iren_ship.CreateRepeatingTimer(40)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Navigating-observer-thrust timer could not start: %s", exc,
            )

    def _stop_ship_timer() -> None:
        if _iren_ship is None or _ship_timer_id[0] is None:
            return
        try:
            _iren_ship.DestroyTimer(_ship_timer_id[0])
        except Exception:  # noqa: BLE001
            pass
        _ship_timer_id[0] = None

    if _iren_ship is not None:
        try:
            _iren_ship.AddObserver("TimerEvent", _ship_tick)
            logger.info(
                "Navigating-observer-thrust observer attached "
                "(timer created on demand)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Navigating-observer-thrust observer could not attach: %s",
                exc,
            )

    # Button factories (vedo passes (button, event) -- accept *args).
    def _mk_bump(kind, axis, sign):
        def _cb(*_a, **_kw):
            _ship_impulse(kind, axis, sign)
        return _cb

    def _mk_stop(kind):
        def _cb(*_a, **_kw):
            _ship_stop(kind)
        return _cb

    _SHIP_SIZE   = 18      # button font size (mesh-mode buttons use 14)
    _SHIP_T_FG   = "white"
    _SHIP_T_BG   = "#0d47a1"  # deep blue for translation
    _SHIP_R_FG   = "white"
    _SHIP_R_BG   = "#1b5e20"  # deep green for rotation
    _SHIP_STOP_FG = "white"
    _SHIP_STOP_BG = "#b71c1c"  # red for STOP

    # Layout: columns at x ∈ {0.04, 0.10, 0.16}, well clear of the
    # right-side button column (x ≈ 0.72 – 0.88) and below the
    # top-left ortho-panel overlay (y ≥ 0.5).
    _SHIP_X_L, _SHIP_X_C, _SHIP_X_R = 0.04, 0.10, 0.16

    # Translation panel (upper of the two).  Cross + bottom row.
    #       [ ↑ ]
    # [ ← ]       [ → ]
    #       [ ↓ ]
    # [FWD][STOP][BCK]
    _t_rows = [0.40, 0.36, 0.32, 0.28]
    plt.add_button(_mk_bump("t", 2, +1), states=["▲"],
                   pos=(_SHIP_X_C, _t_rows[0]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])
    plt.add_button(_mk_bump("t", 1, -1), states=["◀"],
                   pos=(_SHIP_X_L, _t_rows[1]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])
    plt.add_button(_mk_bump("t", 1, +1), states=["▶"],
                   pos=(_SHIP_X_R, _t_rows[1]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])
    plt.add_button(_mk_bump("t", 2, -1), states=["▼"],
                   pos=(_SHIP_X_C, _t_rows[2]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])
    plt.add_button(_mk_bump("t", 0, +1), states=["FWD"],
                   pos=(_SHIP_X_L, _t_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])
    plt.add_button(_mk_stop("t"),        states=["STOP"],
                   pos=(_SHIP_X_C, _t_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_STOP_FG], bc=[_SHIP_STOP_BG])
    plt.add_button(_mk_bump("t", 0, -1), states=["BCK"],
                   pos=(_SHIP_X_R, _t_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_T_FG], bc=[_SHIP_T_BG])

    # Rotation panel (lower).
    #       [ P↑ ]
    # [ Y← ]       [ Y→ ]
    #       [ P↓ ]
    # [R⟲ ][STOP ][ R⟳]
    _r_rows = [0.20, 0.16, 0.12, 0.08]
    plt.add_button(_mk_bump("r", 1, +1), states=["P▲"],
                   pos=(_SHIP_X_C, _r_rows[0]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])
    # Yaw signs flipped so that the arrow direction matches the
    # actual rotation of the view (cam.Yaw(+deg) swings view left).
    plt.add_button(_mk_bump("r", 0, +1), states=["Y◀"],
                   pos=(_SHIP_X_L, _r_rows[1]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])
    plt.add_button(_mk_bump("r", 0, -1), states=["Y▶"],
                   pos=(_SHIP_X_R, _r_rows[1]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])
    plt.add_button(_mk_bump("r", 1, -1), states=["P▼"],
                   pos=(_SHIP_X_C, _r_rows[2]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])
    plt.add_button(_mk_bump("r", 2, -1), states=["R↺"],
                   pos=(_SHIP_X_L, _r_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])
    plt.add_button(_mk_stop("r"),        states=["STOP"],
                   pos=(_SHIP_X_C, _r_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_STOP_FG], bc=[_SHIP_STOP_BG])
    plt.add_button(_mk_bump("r", 2, +1), states=["R↻"],
                   pos=(_SHIP_X_R, _r_rows[3]),
                   size=_SHIP_SIZE, bold=True,
                   c=[_SHIP_R_FG], bc=[_SHIP_R_BG])

    # ─── Restart navigation ────────────────────────────────────────────
    # Zeroes all thrust velocities and restores the initial camera
    # framing captured right after ``plt.show(...)``.
    def _restart_nav_cb(*_a, **_kw):
        # Kill all thrust velocities first.
        try:
            _ship["v_world"] = _np_ship.zeros(3, dtype=_np_ship.float64)
            for _i in range(3):
                _ship["v_r"][_i] = 0.0
            _ship["last_t"] = None
        except Exception:  # noqa: BLE001
            pass
        # Also kill the repeating timer so the trackball recovers its
        # vanilla, inertia-free feel immediately.
        try:
            _stop_ship_timer()
        except Exception:  # noqa: BLE001
            pass
        # Restore the initial camera.
        if _initial_cam:
            try:
                cam = plt.camera
                cam.SetPosition(*_initial_cam["pos"])
                cam.SetFocalPoint(*_initial_cam["fp"])
                cam.SetViewUp(*_initial_cam["up"])
                cam.SetParallelScale(_initial_cam["pscale"])
                cam.SetViewAngle(_initial_cam["vangle"])
                cam.SetClippingRange(*_initial_cam["clip"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Restart navigation: %s", exc)
        # Recompute clipping range against the current scene bounds so
        # nothing is missing if the snapshot is now too tight.
        try:
            for _ren in plt.renderers:
                _ren.ResetCameraClippingRange()
        except Exception:  # noqa: BLE001
            try:
                plt.renderer.ResetCameraClippingRange()
            except Exception:  # noqa: BLE001
                pass
        _show_status("Navigation restarted (camera + thrust reset)")
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    plt.add_button(
        _restart_nav_cb,
        states=["Restart navigation"],
        c=["white"],
        bc=["#4527a0"],   # deep indigo, distinct from the STOP red
        pos=(_SHIP_X_C, 0.44),   # just above the translation panel
        size=14,
        bold=True,
    )

    # ─── Fog (depth) + SSAO toggles ─────────────────────────────────────
    # Both are GPU effects driven from the bottom-right button column.
    # They are independent and either / both / none may be active.
    _depth_state = {
        "fog_on":  False,
        "ssao_on": False,
        # Cached SSAO pass + the previous renderer pass so we can
        # restore it when toggled off.
        "ssao_pass":     None,
        "ssao_keepalive": [],
        "prev_pass":     {},   # renderer -> previous SetPass() value
        # All mesh actors fog may touch (mesh + alt_mesh + pillars_mesh).
        "fog_actors": [a for a in (mesh, alt_mesh, pillars_mesh) if a is not None],
    }

    def _scene_diag() -> float:
        """Approximate scene diagonal for choosing fog near/far."""
        try:
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds()
            import math as _math
            return _math.sqrt(
                (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
            )
        except Exception:  # noqa: BLE001
            return 1000.0

    def _enable_fog():
        diag = _scene_diag()
        fog_near = diag * 0.20
        fog_far  = diag * 0.90
        # Use the renderer's background colour so fog blends invisibly
        # with the void at the back of the scene.
        bg = BACKGROUND
        if isinstance(bg, str):
            # Named colour: defer to black as a safe fallback.
            fog_r, fog_g, fog_b = 0.0, 0.0, 0.0
        else:
            try:
                fog_r, fog_g, fog_b = float(bg[0]), float(bg[1]), float(bg[2])
                if max(fog_r, fog_g, fog_b) > 1.5:  # 0..255 form
                    fog_r /= 255.0; fog_g /= 255.0; fog_b /= 255.0
            except Exception:  # noqa: BLE001
                fog_r = fog_g = fog_b = 0.0

        # Fragment shader replacement: keep the original lighting
        # placeholder, then post-multiply colour by a depth-based fog
        # factor.  ``vertexVC`` is the view-space position varying
        # provided by VTK's default pipeline.
        snippet = (
            "//VTK::Light::Impl\n"
            f"  float _fog_d = length(vertexVC.xyz);\n"
            f"  float _fog_f = clamp((_fog_d - {fog_near:.3f}) / "
            f"({fog_far:.3f} - {fog_near:.3f}), 0.0, 1.0);\n"
            f"  gl_FragData[0].rgb = mix(gl_FragData[0].rgb, "
            f"vec3({fog_r:.3f}, {fog_g:.3f}, {fog_b:.3f}), _fog_f);\n"
        )
        for actor in _depth_state["fog_actors"]:
            try:
                raw = getattr(actor, "actor", actor)  # vedo wraps vtkActor
                sp = raw.GetShaderProperty()
                sp.AddFragmentShaderReplacement(
                    "//VTK::Light::Impl", True, snippet, False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fog attach failed on actor %r: %s",
                               getattr(actor, "name", actor), exc)
        _depth_state["fog_on"] = True
        logger.info("Fog enabled: near=%.1f far=%.1f", fog_near, fog_far)

    def _disable_fog():
        for actor in _depth_state["fog_actors"]:
            try:
                raw = getattr(actor, "actor", actor)
                sp = raw.GetShaderProperty()
                # Remove only the fog replacement, keep any user code.
                sp.ClearFragmentShaderReplacement(
                    "//VTK::Light::Impl", True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fog detach failed on actor %r: %s",
                               getattr(actor, "name", actor), exc)
        _depth_state["fog_on"] = False
        logger.info("Fog disabled.")

    def _toggle_fog_cb(*_a, **_kw):
        if _depth_state["fog_on"]:
            _disable_fog()
        else:
            _enable_fog()
        _fog_btn.switch()
        plt.render()

    _fog_btn = plt.add_button(
        _toggle_fog_cb,
        states=["Fog ▸ off", "Fog ▸ on"],
        c=["#9e9e9e", "white"],
        bc=["#263238", "#1976d2"],
        pos=(0.88, 0.41),
        size=14,
        bold=True,
    )

    def _enable_ssao():
        # Use vtkRenderer's built-in SSAO API (VTK 9.2+) instead of
        # replacing the pipeline with vtkSSAOPass.  The built-in path
        # adds SSAO as a post-process on top of the *default* pipeline,
        # which leaves VTK's internal pick/interaction machinery intact
        # → buttons and sliders keep responding to clicks.
        diag = _scene_diag()
        fallback_needed = False
        for renderer in plt.renderers:
            try:
                renderer.UseSSAOOn()
                renderer.SetSSAORadius(diag * 0.005)
                renderer.SetSSAOBias(diag * 0.0005)
                renderer.SetSSAOKernelSize(32)
                renderer.SetSSAOBlur(True)
            except AttributeError:
                # VTK < 9.2: fall back to the pass-chain approach only
                # when the native API is absent.
                fallback_needed = True
                break
        if fallback_needed:
            # Legacy path — same pass chain as before but we can't avoid
            # the button-hit issue on older VTK builds.
            try:
                from vtkmodules.vtkRenderingOpenGL2 import (
                    vtkCameraPass, vtkLightsPass, vtkOpaquePass,
                    vtkOverlayPass, vtkTranslucentPass,
                    vtkRenderPassCollection, vtkSequencePass, vtkSSAOPass,
                )
            except ImportError as exc:  # pragma: no cover
                logger.warning("SSAO unavailable: %s", exc)
                return
            keepalive: list = []
            for renderer in plt.renderers:
                try:
                    _depth_state["prev_pass"][renderer] = renderer.GetPass()
                except Exception:  # noqa: BLE001
                    _depth_state["prev_pass"][renderer] = None
                lights = vtkLightsPass()
                opaque = vtkOpaquePass()
                translucent = vtkTranslucentPass()
                overlay = vtkOverlayPass()
                coll = vtkRenderPassCollection()
                coll.AddItem(lights); coll.AddItem(opaque)
                coll.AddItem(translucent); coll.AddItem(overlay)
                seq = vtkSequencePass(); seq.SetPasses(coll)
                cam_pass = vtkCameraPass(); cam_pass.SetDelegatePass(seq)
                ssao = vtkSSAOPass()
                ssao.SetRadius(diag * 0.005); ssao.SetBias(diag * 0.0005)
                ssao.SetKernelSize(32); ssao.SetBlur(True)
                ssao.SetDelegatePass(cam_pass)
                renderer.SetPass(ssao)
                keepalive.extend([ssao, cam_pass, seq, coll,
                                   opaque, lights, translucent, overlay])
            _depth_state["ssao_keepalive"] = keepalive
        _depth_state["ssao_on"] = True
        logger.info("SSAO enabled (radius=%.3f).", diag * 0.005)

    def _disable_ssao():
        for renderer in plt.renderers:
            try:
                renderer.UseSSAOOff()
            except AttributeError:
                pass  # VTK < 9.2 — handled by pass restore below
            prev = _depth_state["prev_pass"].get(renderer)
            if prev is not None or renderer in _depth_state["prev_pass"]:
                try:
                    renderer.SetPass(prev)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SSAO restore failed: %s", exc)
        _depth_state["prev_pass"].clear()
        _depth_state["ssao_keepalive"] = []
        _depth_state["ssao_on"] = False
        logger.info("SSAO disabled.")

    def _toggle_ssao_cb(*_a, **_kw):
        if _depth_state["ssao_on"]:
            _disable_ssao()
        else:
            _enable_ssao()
        _ssao_btn.switch()
        plt.render()

    _ssao_btn = plt.add_button(
        _toggle_ssao_cb,
        states=["SSAO ▸ off", "SSAO ▸ on"],
        c=["#9e9e9e", "white"],
        bc=["#263238", "#6a1b9a"],
        pos=(0.88, 0.35),
        size=14,
        bold=True,
    )

    _ui_capturers.append(lambda: {
        "fog_on":  bool(_depth_state["fog_on"]),
        "ssao_on": bool(_depth_state["ssao_on"]),
    })


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input).expanduser().resolve()

    logger.info("═" * 64)
    logger.info("Marvel Water-Conductance Viewer")
    logger.info("  Input    : %s", input_path)
    logger.info("  Iso-level: %.3f", args.level)
    logger.info("  Smooth   : %d iterations", args.smooth_iter)
    logger.info("═" * 64)

    mesh = _load_or_build_mesh(
        input_path,
        level=args.level,
        smooth_iter=args.smooth_iter,
        cache_path=Path(args.mesh_cache).expanduser().resolve()
                   if args.mesh_cache else None,
        rebuild=args.rebuild_mesh,
    )
    _style_mesh(mesh, body_color=BODY_COLOR_BRIDGES)
    mesh.name = "Wat_Norm_Cortex"

    # Alternate "All watered tissues" mesh (Wat_Norm_All.tif).
    alt_mesh = None
    if not args.no_all_mesh:
        try:
            alt_input = Path(args.all_input).expanduser().resolve()
            alt_mesh = _load_or_build_mesh(
                alt_input,
                level=args.level,
                smooth_iter=args.smooth_iter,
                cache_path=Path(args.all_mesh_cache).expanduser().resolve()
                           if args.all_mesh_cache else None,
                rebuild=args.rebuild_all_mesh,
            )
            _style_mesh(alt_mesh)
            alt_mesh.name = "Wat_Norm_All"
            logger.info("Alternate 'All watered tissues' mesh ready.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build alternate 'All' mesh: %s", exc)
            alt_mesh = None

    if args.save_vtk:
        out = Path(args.save_vtk).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(str(out))
        logger.info("Mesh written to %s", out)

    import vedo
    plt = vedo.Plotter(
        title="Marvel · Wat_Norm_Cortex",
        bg=BACKGROUND,
        axes=1,
        size=(args.width, args.height),
    )
    plt.show(mesh, viewup="z", interactive=False)

    # Load the raw greyscale volume once: it is reused by both the
    # orthogonal-slice panel and the (optional) volume-rendering toggle.
    raw_volume = None
    raw_path = Path(args.raw).expanduser().resolve()
    if raw_path.exists():
        try:
            from marvel_view.visualization.ortho_panel import load_raw_volume
            raw_volume = load_raw_volume(raw_path)
            logger.info("Raw volume loaded (%s, shape=%s).",
                        raw_path, raw_volume.shape)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load raw volume %s: %s", raw_path, exc)
    else:
        logger.info("Raw volume not found at %s.", raw_path)

    # Optional orthogonal-slice locator panel.
    ortho_overlay = None
    if not args.no_ortho_panel and raw_volume is not None:
        try:
            from marvel_view.visualization.ortho_panel import OrthoPanelOverlay
            ortho_overlay = OrthoPanelOverlay(
                plt, raw_volume,
                viewport=(0.0, 0.5, 0.4, 1.0),  # top-left, ~40% wide × 50% tall
                cell_pixels=320,
            )
            # Refresh on standard mouse interactions too.
            iren = getattr(plt, "interactor", None)
            if iren is not None:
                import math as _math
                def _ortho_cb(_caller=None, _event=None, *_a, **_kw):
                    try:
                        cam = plt.camera
                        px, py, pz = cam.GetPosition()
                        fx, fy, fz = cam.GetFocalPoint()
                        dx, dy, dz = fx - px, fy - py, fz - pz
                        n = _math.sqrt(dx * dx + dy * dy + dz * dz)
                        if n > 1e-9:
                            direction = (dx / n, dy / n, dz / n)
                        else:
                            direction = (0.0, 0.0, -1.0)
                        ortho_overlay.update((px, py, pz), direction)
                        plt.render()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("ortho mouse/key cb failed: %s", exc)
                iren.AddObserver("EndInteractionEvent", _ortho_cb)
                iren.AddObserver("KeyPressEvent", _ortho_cb)
                logger.info("Ortho mouse/key observers attached.")
            logger.info("Ortho-panel overlay attached.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to attach ortho panel: %s", exc)

    # Build (or load cached) gradient-arrow field for the 4th toggle.
    arrows_data = _load_or_build_arrows(
        Path(args.geoddist).expanduser().resolve() if args.geoddist else Path(),
        cache_path=Path(args.arrows_cache).expanduser().resolve()
                   if args.arrows_cache else None,
        stride=args.arrow_stride,
        long_stride=args.long_stride,
        fine_stride=args.fine_stride,
        rebuild=args.rebuild_arrows,
    )

    # Load cached crown Dijkstra tracks (built by build-meshes).
    tracks_data = None
    if not args.no_tracks:
        tracks_data = _load_or_build_crown_tracks(
            Path(),  # source/target/domain not used when cache exists
            Path(),
            Path(),
            cache_path=Path(args.tracks_cache).expanduser().resolve()
                       if args.tracks_cache else None,
            vtp_path=Path(args.tracks_vtp_cache).expanduser().resolve()
                     if args.tracks_vtp_cache else None,
            arrows_vtp_path=(
                Path(args.tracks_arrows_vtp_cache).expanduser().resolve()
                if args.tracks_arrows_vtp_cache else None
            ),
            rebuild=False,  # viewer never rebuilds; use build-meshes for that
            display_stride=args.tracks_stride,
        )

    # Persistent translucent "ghost" mesh overlaid on every view.
    if not args.no_overlay:
        overlay_mesh = _load_or_build_overlay_mesh(
            Path(args.overlay_input).expanduser().resolve(),
            cache_path=Path(args.overlay_cache).expanduser().resolve()
                       if args.overlay_cache else None,
            level=args.overlay_level,
            rebuild=args.rebuild_overlay,
        )
        if overlay_mesh is not None:
            try:
                overlay_mesh.alpha(args.overlay_alpha)
                overlay_mesh.c("white")
                overlay_mesh.lighting("off")
                overlay_mesh.name = "overlay_ghost"
                plt.add(overlay_mesh)
                logger.info("Overlay ghost mesh added (alpha=%.3f).",
                            args.overlay_alpha)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to add overlay mesh: %s", exc)

    # Glowing green "Pillars" overlay — only displayed when the Cortical
    # bridges mesh is the active mesh (toggled by `_attach_controls`).
    pillars_mesh = None
    if not args.no_pillars:
        try:
            pillars_mesh = _load_or_build_overlay_mesh(
                Path(args.pillars_input).expanduser().resolve(),
                cache_path=Path(args.pillars_cache).expanduser().resolve()
                           if args.pillars_cache else None,
                level=args.pillars_level,
                rebuild=args.rebuild_pillars,
            )
            if pillars_mesh is not None:
                # Smooth surface + recomputed normals → Phong shading
                # has something to work with (no faceted look in the
                # 'solid' Arrows-view style).
                try:
                    pillars_mesh.smooth(niter=15)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Pillars smooth() failed: %s", exc)
                try:
                    pillars_mesh.compute_normals(
                        points=True, cells=False,
                        feature_angle=60.0, consistency=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Pillars compute_normals failed: %s", exc)
                # Default = glow style (lighting off, bright teal ghost).
                # The Arrows-view 'solid' style is applied later by
                # _attach_controls when entering that view.
                pillars_mesh.c((0.20, 0.95, 0.75))
                pillars_mesh.alpha(args.pillars_alpha)
                pillars_mesh.lighting("off")
                pillars_mesh.name = "pillars_glow"
                logger.info("Pillars glow overlay ready (alpha=%.3f, iso=%.1f).",
                            args.pillars_alpha, args.pillars_level)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not build Pillars overlay: %s", exc)
            pillars_mesh = None

    # ── Try to load pre-computed per-cell density scalars ──────────────
    density_scalars: dict | None = None
    try:
        import numpy as np
        _b_cache = Path(args.density_bridges_cache).expanduser().resolve()
        _a_cache = Path(args.density_all_cache).expanduser().resolve()
        _b_arr = None
        _a_arr = None
        if _b_cache.exists():
            _b_arr = np.load(_b_cache)
            logger.info("Loaded density bridges scalars from %s (%d cells).",
                        _b_cache, int(_b_arr.shape[0]))
        if _a_cache.exists():
            _a_arr = np.load(_a_cache)
            logger.info("Loaded density all-mesh scalars from %s (%d cells).",
                        _a_cache, int(_a_arr.shape[0]))
        if _b_arr is not None or _a_arr is not None:
            density_scalars = {"bridges": _b_arr, "all": _a_arr}
        else:
            logger.info(
                "Density caches not found (%s, %s) – toggle disabled.",
                _b_cache, _a_cache,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load density scalar caches: %s", exc)
        density_scalars = None

    # ── Water "membranes" descending animation (load-only) ─────────────
    membranes_data = None
    if not args.no_membranes:
        try:
            membranes_data = _load_or_build_membranes(
                Path(args.membranes_vtp_cache).expanduser().resolve(),
                Path(args.membranes_meta_cache).expanduser().resolve(),
            )
            if membranes_data == (None, None):
                membranes_data = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load membranes cache: %s", exc)
            membranes_data = None

    _attach_controls(
        plt, mesh,
        arrows_data=arrows_data,
        ortho_overlay=ortho_overlay,
        alt_mesh=alt_mesh,
        pillars_mesh=pillars_mesh,
        tracks_data=tracks_data,
        density_scalars=density_scalars,
        membranes_data=membranes_data,
    )
    plt.interactive()
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

