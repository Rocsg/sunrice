"""
Configuration for the **aerench** extension (Unrolled / Extract2).

Inputs are *three* 32-bit "norm" TIFFs, one per class.  In each volume
positive values mean the voxel belongs to the class and negative values
mean it does not — the iso-surface at ``level=0`` is therefore the
class boundary (sub-voxel accurate, no Gaussian blur required).

* 1 — *gas*    (``Aerench_norm.tif``)  — large cavities
* 2 — *meats*  (``Meat_norm.tif``)     — fine gas filaments
* 3 — *water*  (``Wat_norm.tif``)      — tissues / everything else

In addition, a shared 8-bit ``Outer.tif`` flags the exterior region of
the sample (value ``255`` outside, ``0`` inside).  Voxels at ``255`` are
excluded from every label *before* connected-component analysis — this
replaces the older "drop largest CC" / per-slice opening tricks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

# ──────────────────────────────── paths ───────────────────────────────────────
# Répertoire racine des données. Seule variable à exporter dans ~/.bashrc.
DEFAULT_INPUT_DIR: Path = (
    Path(os.environ.get("MARVEL_DATA_DIR", "/home/rfernandez/Data/Arize/Hollow_test"))
    / "1_Intermediate_computed_images"
)

# 8-bit mask of the *outside* of the sample — voxels at 255 are excluded
# from every label before any further processing.
DEFAULT_OUTER_PATH: Path = DEFAULT_INPUT_DIR / "Outer.tif"

# Kept for API compatibility with shared code (viewer, settings file).
# Points to the gas norm file by convention.
DEFAULT_SEG_PATH: Path = DEFAULT_INPUT_DIR / "Aerench_norm.tif"

DEFAULT_OUTPUT_DIR: Path = Path("./aerench_output")

DEFAULT_SETTINGS_PATH: Path = (
    DEFAULT_INPUT_DIR / "Extract2_aerench_settings.json"
)

# ──────────────────────────────── imaging ─────────────────────────────────────

# (Z, Y, X) voxel spacing.
DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)

# Extra voxels of padding around each component's bbox so the smoothed
# iso-surface closes cleanly at the crop boundary.
DEFAULT_BBOX_PAD: int = 3

# Iso-surface threshold for the float "norm" inputs (positive = inside).
DEFAULT_LEVEL: float = 0.0

# ──────────────────────────────── meshing ─────────────────────────────────────

SMOOTH_ITERATIONS: int = 80
DECIMATE_FRACTION: float = 1.0
DEFAULT_NUM_WORKERS: int | None = None
DEFAULT_MIN_VOXELS: int = 200
DEFAULT_FACE_BUDGET: int | None = None  # explicit per-label budgets below

# ──────────────────────────────── label specs ─────────────────────────────────
#
# Per-label strategy:
#   1. Load the float "norm" volume for this label.
#   2. Zero out voxels where ``Outer.tif == 255``.
#   3. Connected components on ``vol > 0``.
#   4. For each CC: crop the float sub-volume to the CC's bbox (+ padding)
#      and run marching cubes at ``level = 0``.
#
# No Gaussian blur is needed since the input is already a smooth signed
# field — running it would only round off fine features.
# ──────────────────────────────────────────────────────────────────────────────

LABEL_CONFIG: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "gas",
        "description": "Gas — main cavities (Aerench_norm)",
        "input_file": "Aerench_norm.tif",
        # Deep electric "thunder" blue — opaque
        "color": (15, 35, 165),
        "opacity": 1.0,
        "min_voxels": 500,
        "face_budget": 25_000_000,
        "gaussian_sigma": 0.0,
        "ambient":  0.18,
        "diffuse":  0.80,
        "specular": 0.25,
    },
    2: {
        "name": "meats",
        "description": "Aerenchyma méats — fine gas filaments (Meat_norm)",
        "input_file": "Meat_norm.tif",
        # Light spectrum blue leaning green (cyan-aqua)
        "color": (90, 215, 200),
        "opacity": 1.0,
        "min_voxels": 100,
        "face_budget": 50_000_000,
        "gaussian_sigma": 0.0,
        "ambient":  0.18,
        "diffuse":  0.85,
        "specular": 0.25,
    },
    3: {
        "name": "water",
        "description": "Tissus aqueux (Wat_norm) — orange, very transparent",
        "input_file": "Wat_norm.tif",
        # Vivid orange, alpha = 0.03
        "color": (255, 140, 30),
        "opacity": 0.03,
        "min_voxels": 500,
        "face_budget": 20_000_000,
        "gaussian_sigma": 0.0,
        "ambient":  0.30,
        "diffuse":  0.75,
        "specular": 0.10,
    },
}

# ──────────────────────────────── viewer defaults ─────────────────────────────
#
# The aerench viewer reuses :class:`RootsViewer` in mesh-only mode.  The
# volume / split sliders are not wired up (no source 8-bit TIFF in this
# pipeline), so VOLUME_* defaults below are placeholders kept for API
# compatibility with the shared viewer code.

VOLUME_THR_LOW_DEFAULT: int = 60
VOLUME_THR_HIGH_DEFAULT: int = 230
VOLUME_OPACITY_DEFAULT: float = 0.25
VIEW_MODE_DEFAULT: str = "mesh"
BACKGROUND_DEFAULT: str = "black"
