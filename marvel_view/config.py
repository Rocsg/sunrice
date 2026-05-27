"""
Global configuration: paths, label definitions, colours and mesh parameters.

Edit this file to match your dataset and rendering preferences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

# ──────────────────────────────── paths ───────────────────────────────────────

DEFAULT_IMAGE_PATH: Path = Path(
    "/home/rfernandez/Data/Arize/SunRice/3D_marvel_01/"
    "W4_IR_aftercavit-short_short.tif"
)

# Directory where preprocessed VTK meshes are saved / loaded from.
DEFAULT_OUTPUT_DIR: Path = Path("./marvel_output")

# ──────────────────────────────── leaderboard ─────────────────────────────────
#
# The shared leaderboard lives in a GitHub repository as leaderboard.json.
# Scores are submitted via the GitHub Contents API (no extra server needed).
#
# Required on every machine that will submit scores
# ──────────────────────────────────────────────────
#   export SUNRICE_LB_TOKEN=<your-PAT>
#
#   The PAT must have "Contents: Read and Write" permission scoped to
#   SUNRICE_LB_REPO only.  Generate a fine-grained PAT at:
#   https://github.com/settings/personal-access-tokens
#
# Optional overrides
# ───────────────────
#   export SUNRICE_LB_REPO=owner/repo       (default below)
#   export SUNRICE_LB_LOCAL_DB=/path/to.db  (default: ~/.cache/sunrice/…)
#
# The client (marvel_view/leaderboard/client.py) reads these env vars at
# import time.  Values here are kept as documentation only; the actual
# defaults live in client.py.

LEADERBOARD_GITHUB_REPO: str = "rfernandez/sunrice-leaderboard"

# ──────────────────────────────── imaging ─────────────────────────────────────

# Physical voxel size (z, y, x) in µm – adjust to the real acquisition.
# Used by marching cubes to produce correctly-scaled meshes.
DEFAULT_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)

# ──────────────────────────────── meshing ─────────────────────────────────────

# Number of Laplacian smoothing iterations applied after marching cubes.
# Set to 0 to skip smoothing entirely.
SMOOTH_ITERATIONS: int = 80

# Mesh decimation: fraction of faces to **keep** (1.0 = no decimation).
# Values < 1 speed up rendering at the cost of detail.
# Ignored for a given label when a `face_budget` is resolved for it
# (either via per-label override or the global DEFAULT_FACE_BUDGET).
DECIMATE_FRACTION: float = 1.0

# Default global face budget (total triangles across ALL labels / components).
# When set, overrides per-label decimate_fraction: each label gets a share
# proportional to its voxel count, and `all_cc` labels further distribute
# their share across components.  Set to ``None`` to disable and fall back
# to per-label decimate_fraction values.
DEFAULT_FACE_BUDGET: int | None = 10_000_000

# Number of parallel worker processes for `all_cc` component meshing.
# ``None`` → use os.cpu_count(); ``1`` disables multiprocessing (useful for
# debugging).
DEFAULT_NUM_WORKERS: int | None = None

# Global minimum voxel count for a connected component to be meshed.
DEFAULT_MIN_VOXELS: int = 200

# ──────────────────────────────── label specs ─────────────────────────────────
#
# Processing strategies
# ─────────────────────
#  "full_mask"  : mesh the entire label as one surface (no CC analysis).
#  "interior_cc": connected-component analysis; keep only CCs that do NOT
#                 touch any face of the image bounding box (= closed cavities).
#  "all_cc"     : mesh every connected component independently.
#
# Per-label overrides (all optional)
# ────────────────────────────────────
#  min_voxels       – skip components smaller than this (default: DEFAULT_MIN_VOXELS)
#  smooth_iter      – override global SMOOTH_ITERATIONS
#  decimate_fraction– override global DECIMATE_FRACTION (ignored if face_budget active)
#  face_budget      – absolute target triangle count for the label; for
#                     `all_cc` it is distributed across components.  When
#                     unset, the label's share of DEFAULT_FACE_BUDGET is used.
#  gaussian_sigma   – apply Gaussian blur to the *mask* before marching cubes;
#                     useful to obtain a smooth outer envelope (label 2).
# ──────────────────────────────────────────────────────────────────────────────

LABEL_CONFIG: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "background_cavities",
        "description": "Background – closed cavities not touching the image border",
        # Dark, slightly muted blue
        "color": (20, 50, 180),
        "opacity": 0.35,
        "processing": "interior_cc",
        "min_voxels": 200,
        "decimate_fraction": 0.1,
    },
    2: {
        "name": "aqueous_tissues",
        "description": "Aqueous tissues – outer envelope, ghost rendering",
        # Pale white-bluish grey
        "color": (210, 215, 228),
        "opacity": 0.01,
        "processing": "full_mask",
        "min_voxels": 0,
        # Blur the mask before meshing to get a clean, featureless envelope.
        "gaussian_sigma": 3.0,
        "smooth_iter": 120,
        "decimate_fraction": 0.05,
        # Envelope is already smoothed by the Gaussian blur — safely sub-sample
        # marching cubes to keep initial face count manageable on huge masks.
        "step_size": 2,
    },
    3: {
        "name": "iodine",
        "description": "Iodine / conductive structures – light cyan-green",
        # Cyan leaning slightly green
        "color": (30, 205, 195),
        "opacity": 1.0,
        "processing": "all_cc",
        "min_voxels": 200,
        "decimate_fraction": 0.1,
    },
    4: {
        "name": "scaffold",
        "description": "Scaffold / tuteur – muted ghost grey",
        "color": (105, 105, 98),
        "opacity": 0.01,
        "processing": "full_mask",
        "min_voxels": 0,
        "gaussian_sigma": 2.0,
        "decimate_fraction": 0.05,
    },
    5: {
        "name": "membranes",
        "description": "Membranes – many fragments, semi-transparent orange",
        # Vivid orange
        "color": (255, 138, 18),
        "opacity": 1.0,
        "processing": "all_cc",
        "min_voxels": 50,
        "decimate_fraction": 0.1,
    },
}
