"""
Configuration for the **roots** extension (small_roots-2 dataset).

Kept separate from :mod:`marvel_view.config` so it does not shadow the
original SunRice/marvel pipeline.  Used by:

* :mod:`marvel_view.scripts.roots_preprocess`   – mesh generation
* :mod:`marvel_view.scripts.roots_visualize`    – interactive viewer
                                                  (mesh + volumetric mode)

Notes
-----
``seal_borders`` (in this module's settings, applied at preprocessing time)
forces every voxel of the 6 outer faces of the segmentation volume to
value ``2`` (background).  This makes objects of any other label end
"closed" at the volume boundary instead of producing open / singular
surfaces in the marching-cubes output.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

# ──────────────────────────────── paths ───────────────────────────────────────

DATA_DIR: Path = Path("/home/rfernandez/Data/Arize/SunRice/3D_marvel_01/small")

# Multi-label segmentation TIFF (values 1..5).
DEFAULT_SEG_PATH: Path = DATA_DIR / "small_roots-2_Simple Segmentation.tiff"

# Source 8-bit grayscale volume used for volumetric rendering.
DEFAULT_SOURCE_PATH: Path = DATA_DIR / "small_roots-2.tif"

# Where the per-label VTK meshes are written / read from.
# Override with env var MARVEL_ROOTS_OUTPUT_DIR when running on a remote server.
DEFAULT_OUTPUT_DIR: Path = Path(
    os.environ.get(
        "MARVEL_ROOTS_OUTPUT_DIR",
        "/home/rfernandez/Data/Arize/Hollow_test/roots_output",
    )
)

# Default settings file (camera, sliders, visibility…).  Lives next to
# the data so it travels with the dataset, no path-asking needed.
DEFAULT_SETTINGS_PATH: Path = DATA_DIR / "small_roots-2_marvel_settings.json"

# ──────────────────────────────── imaging ─────────────────────────────────────

# Voxel size (z, y, x).  Adjust to actual acquisition; 1,1,1 keeps things
# isotropic for marching cubes.
DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)

# Apply once at preprocess time: paint the 6 outer faces of the
# segmentation volume with this label.  Choose the "background" label
# (here: 2) so non-background labels are always strictly interior and
# their meshes close cleanly at the boundary.
SEAL_BORDERS_LABEL: int = 2

# ──────────────────────────────── meshing ─────────────────────────────────────

SMOOTH_ITERATIONS: int = 80
DECIMATE_FRACTION: float = 1.0    # only used when face_budget = None
DEFAULT_NUM_WORKERS: int | None = None
DEFAULT_MIN_VOXELS: int = 200

# Per-label face budget targets are set explicitly inside LABEL_CONFIG
# below (the user gave concrete numbers per label), so no global budget.
DEFAULT_FACE_BUDGET: int | None = None

# ──────────────────────────────── label specs ─────────────────────────────────
#
# Voxel value 2 is treated as background and not meshed.
# Other labels each have a hard ``face_budget`` that overrides any global.
# ──────────────────────────────────────────────────────────────────────────────

LABEL_CONFIG: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "iodine",
        "description": "Iodine deposits — light turquoise, opaque",
        # Light turquoise blue
        "color": (90, 220, 230),
        "opacity": 1.0,
        "processing": "all_cc",
        "min_voxels": 200,
        "face_budget": 10_000_000,
        # Gentle Gaussian on the binary mask before marching cubes — removes
        # voxel-aliasing spikes / staircases on thin filaments.
        "gaussian_sigma": 0.8,
        "ambient": 0.15,
        "diffuse": 0.80,
        "specular": 0.30,
    },
    3: {
        "name": "lamellae",
        "description": "Lamellae — orange, slightly transparent",
        # Vivid orange
        "color": (255, 140, 30),
        "opacity": 0.55,
        "processing": "all_cc",
        "min_voxels": 100,
        "face_budget": 20_000_000,
        "gaussian_sigma": 1.0,
        "ambient": 0.15,
        "diffuse": 0.85,
        "specular": 0.20,
    },
    4: {
        "name": "stele",
        "description": "Stele — brown, very transparent",
        # Warm brown
        "color": (140, 85, 45),
        "opacity": 0.20,
        "processing": "full_mask",
        "min_voxels": 0,
        "face_budget": 4_000_000,
        "gaussian_sigma": 1.5,
        "ambient": 0.20,
        "diffuse": 0.80,
        "specular": 0.10,
    },
    5: {
        "name": "medium",
        "description": "Surrounding medium — light grey, opaque",
        # Light cool grey
        "color": (210, 212, 218),
        "opacity": 1.0,
        "processing": "full_mask",
        "min_voxels": 0,
        "face_budget": 40_000_000,
        "gaussian_sigma": 1.5,
        "ambient": 0.20,
        "diffuse": 0.85,
        "specular": 0.15,
    },
}

# ──────────────────────────────── viewer defaults ─────────────────────────────

# Initial threshold band for the volumetric view (8-bit values).
# The two sliders move ``volume_thr_low`` and ``volume_thr_high``.
VOLUME_THR_LOW_DEFAULT: int = 60
VOLUME_THR_HIGH_DEFAULT: int = 230
VOLUME_OPACITY_DEFAULT: float = 0.25

# View mode: "mesh", "volume", or "split"
VIEW_MODE_DEFAULT: str = "split"

# Background colour for the viewer.
BACKGROUND_DEFAULT: str = "black"
