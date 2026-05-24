"""Default paths, colours and animation tuning constants.

Extracted from the historic monolithic ``water_conductance.py`` so that
the viewer/pipeline modules can ``from .constants import …`` without
pulling the full controller code.

The values themselves are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

from marvel_view import aerench_config as acfg


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
MEMBRANE_TICK_MS: int = 40

# ── Water "lames" (V2) descending animation ─────────────────────────────
# Parallel V2 pipeline: thick adaptive iso-shells, one rendered actor per
# step toggled with SetVisibility (no per-frame topology updates).  Built
# by ``marvel-water-conductance-build-meshes`` via
# ``marvel_view.preprocessing.water_lames``.
DEFAULT_LAMES_ISO_CACHE_DIR: Path = DEFAULT_VTK_OUTPUT_DIR / "lames_iso_cache"
DEFAULT_LAMES_VTP_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lames.vtp"
)
DEFAULT_LAMES_META_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lames_meta.json"
)
DEFAULT_LAMES_LABELS_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lames_labels.tif"
)
# Reuse the membranes BG-dist TIFF by default — same physical input.
DEFAULT_LAMES_BG_DIST_PATH: Path = DEFAULT_MEMBRANES_BG_DIST_PATH
DEFAULT_LAMES_COLOR: tuple[int, int, int] = (140, 195, 255)
DEFAULT_LAMES_ALPHA: float = 0.30
# Target ≈ 25 fps for the lames animation.
LAMES_TICK_MS: int = 40

# ── Water "lame2" (V3) descending animation ─────────────────────────────
# Parallel V3 pipeline: per-column 3-phase (grow d1 / hold / grow d0)
# schedule.  Outputs sit alongside the V2 lames cache and do NOT
# overwrite it; the viewer loads V3 by default and falls back to V2.
DEFAULT_LAME2_ISO_CACHE_DIR: Path = DEFAULT_VTK_OUTPUT_DIR / "lame2_iso_cache"
DEFAULT_LAME2_VTP_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lame2.vtp"
)
DEFAULT_LAME2_META_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lame2_meta.json"
)
DEFAULT_LAME2_LABELS_CACHE: Path = (
    DEFAULT_VTK_OUTPUT_DIR / "lame2_labels.tif"
)
DEFAULT_LAME2_BG_DIST_PATH: Path = DEFAULT_MEMBRANES_BG_DIST_PATH
DEFAULT_LAME2_ALPHA: float = 0.30
LAME2_TICK_MS: int = 40

# ── Wind / gas particles (O₂ + CH₄) ─────────────────────────────────────
# Two toggles drive small "Shadow-of-Colossus" particles that flow
# through the aerenchyma along the gradient of a *harmonic potential*
# solved in the air mask (Dirichlet at high-X / low-X faces, Neumann on
# walls).  Source masks, the field and the per-frame particle positions
# are all pre-computed by ``marvel-wind-field-build``.
#
# Convention: numpy axis-0 = the long axis of the cylinder.  Source mask
# (``wind_source.tif``) lives at axis-0 = high X (cylinder end "A");
# target mask (``wind_target.tif``) lives at axis-0 = low X (end "B").
# O₂ flows A → B (decreasing X, +∇u direction).
# CH₄ flows B → A (increasing X, -∇u direction).
DEFAULT_WIND_AREA_PATH:   Path = acfg.DEFAULT_INPUT_DIR / "wind_area.tif"
DEFAULT_WIND_SOURCE_PATH: Path = acfg.DEFAULT_INPUT_DIR / "wind_source.tif"
DEFAULT_WIND_TARGET_PATH: Path = acfg.DEFAULT_INPUT_DIR / "wind_target.tif"
DEFAULT_WIND_FIELD_CACHE: Path = DEFAULT_VTK_OUTPUT_DIR / "wind_field.npz"
DEFAULT_WIND_O2_CACHE:    Path = DEFAULT_VTK_OUTPUT_DIR / "wind_o2.npz"
DEFAULT_WIND_CH4_CACHE:   Path = DEFAULT_VTK_OUTPUT_DIR / "wind_ch4.npz"
# Number of *displayed* particles per species.  Templates are reused at
# different phase offsets so you can display more particles than were
# precomputed.  All positions for all frames are pre-indexed at startup
# into a pos_phased[n_frames, n_display, 3] tensor (float32); memory is
# roughly n_display × n_frames × 12 bytes per species.
#   5 000 ×  1 000 × 12 ≈  60 MB   (default, comfortable)
#  20 000 ×  1 000 × 12 ≈ 240 MB   (medium budget)
#  50 000 ×  1 000 × 12 ≈ 600 MB   (high budget, heavy startup)
DEFAULT_WIND_O2_DISPLAY:  int = 5_000
DEFAULT_WIND_CH4_DISPLAY: int = 2_500
# Sphere radii (voxels) — world-space radius of each sphere glyph.
DEFAULT_WIND_O2_SPHERE_RADIUS:  float = 0.51

# ── Water harmonic potential field ──────────────────────────────────────────
# Harmonic field + passage-time maps + stream tracks pre-built by
# ``marvel-water-harmonic-build``.  Stream tracks are saved to the same
# ``crown_tracks.vtp`` path so the viewer is unchanged.  Dual-arrow caches
# store spatially-separated water and air arrow fields.
DEFAULT_WATER_AREA_PATH:         Path = acfg.DEFAULT_INPUT_DIR / "Source_Target_Possible_Paths.tif"
DEFAULT_WATER_SOURCE_PATH:       Path = acfg.DEFAULT_INPUT_DIR / "Source_crown.tif"
DEFAULT_WATER_TARGET_PATH:       Path = acfg.DEFAULT_INPUT_DIR / "Target_crown.tif"
DEFAULT_WATER_HARMONIC_CACHE:    Path = DEFAULT_VTK_OUTPUT_DIR / "water_harmonic.npz"
DEFAULT_WATER_DUAL_ARROWS_CACHE: Path = DEFAULT_VTK_OUTPUT_DIR / "water_dual_arrows.npz"
DEFAULT_AIR_DUAL_ARROWS_CACHE:   Path = DEFAULT_VTK_OUTPUT_DIR / "air_dual_arrows.npz"
# Colours used for the dual-arrow overlay.
DUAL_WATER_COLOR: tuple[float, float, float] = (0.3, 0.6, 1.0)   # blue
DUAL_AIR_COLOR:   tuple[float, float, float] = (0.3, 0.9, 0.3)   # green
DEFAULT_WIND_CH4_SPHERE_RADIUS: float = 0.51
# Camera-based distance culling: particles farther than this fraction of
# the scene diagonal from the camera are given alpha = 0 (transparent).
# 0.0 disables culling (show all).  0.35 shows roughly a 1/3-diagonal
# sphere around the camera — enough to see the full local neighbourhood.
DEFAULT_WIND_CULL_RADIUS_FRAC: float = 0.35
# Soft colours.
DEFAULT_WIND_O2_COLOR  = (245, 250, 255)   # near-white
DEFAULT_WIND_CH4_COLOR = (180, 168,  80)   # khaki / olive
# Runtime tick (~25 fps to match the precomputed sequence).
WIND_TICK_MS: int = 40
# Playback speed multiplier — kept for backward compatibility but no longer
# used by the 3-speed system below.
DEFAULT_WIND_SPEED_MULT: int = 2

# ── Three pre-baked trajectory speed levels ──────────────────────────────
# Each level is a separate .npz built with a different n_substeps value
# (10 / 20 / 40 Euler steps per stored frame).  At runtime the speed
# button swaps between pre-phased tensors — always 1 frame/tick,
# always smooth, always the full 4 s lifespan.
WIND_SPEED_LABELS: tuple = ("slow", "med", "fast")
WIND_SUBSTEP_VALUES: tuple = (10, 20, 40)      # n_substeps: slow / med / fast
DEFAULT_WIND_SPEED_LEVEL_IDX: int = 0           # start on "slow"
DEFAULT_WIND_O2_CACHES: list = [
    DEFAULT_VTK_OUTPUT_DIR / f"wind_o2_{lbl}.npz"
    for lbl in WIND_SPEED_LABELS
]
DEFAULT_WIND_CH4_CACHES: list = [
    DEFAULT_VTK_OUTPUT_DIR / f"wind_ch4_{lbl}.npz"
    for lbl in WIND_SPEED_LABELS
]

# ── Mask overlays (Stele + Outside) ─────────────────────────────────────
# Two binary 8-bit masks (0/255) iso-surfaced at 127.5 and displayed
# as static translucent actors, toggled on/off with the lames button.
DEFAULT_STELE_TIFF_PATH:    Path = acfg.DEFAULT_INPUT_DIR / "Stele_mask.tif"
DEFAULT_OUTSIDE_TIFF_PATH:  Path = acfg.DEFAULT_INPUT_DIR / "Outside_mask.tif"
DEFAULT_STELE_MASK_CACHE:   Path = DEFAULT_VTK_OUTPUT_DIR / "stele_mask_iso.vtk"
DEFAULT_OUTSIDE_MASK_CACHE: Path = DEFAULT_VTK_OUTPUT_DIR / "outside_mask_iso.vtk"
DEFAULT_MASK_ISO_LEVEL: float = 127.5

# Status-line titles shown at the top-center on every rendering change.
STATUS_TITLE_MESH_CORTEX = "Cortical bridges between sclerenchyma and stele"
STATUS_TITLE_MESH_ALL    = "All watered tissues"
STATUS_TITLE_ARROWS_GRID = "Vector field on regular grid"
STATUS_TITLE_ARROWS_TRACKS = "Conduction shorter paths (Dijkstra)"
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
DEFAULT_ARROW_LENGTH: float = 7.0 / 3  # world units – default 3× smaller than the original 7.0
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

# ── Info panel ─────────────────────────────────────────────────────────────────
# Physical voxel edge length in µm — used to convert voxels/s → µm/s.
VOXEL_SIZE_UM: float = 6.71

# Fixed title shown at the top of every rendering mode (interactor + movie).
PANEL_TITLE = "Specimen Dolores. 3 cm from apex of root 4."

# Mode subtitle shown on the second line of the info panel.
VIEW_MODE_SUBTITLES: dict = {
    "mesh_bridges":  "Cortical bridges mesh",
    "mesh_all":      "All watered tissues",
    "arrows_grid":   "Water gradient field",
    "arrows_tracks": "Conduction shortest paths",
}
