#!/usr/bin/env python3
"""
Marvel Water-Conductance — pre-build & cache the cortex mesh.

Runs marching cubes on ``Wat_Norm_Cortex.tif`` once and writes the
resulting surface as a ``.vtk`` file, so subsequent launches of
``marvel-water-conductance`` (and ``marvel-water-movie``) can skip the
~slow re-meshing and just load the cached file.

Usage
-----
::

    marvel-water-conductance-build-meshes
    # → ./marvel_output/water_conductance/cortex.vtk

    # Force a rebuild (overwrite) with a different smoothing pass:
    marvel-water-conductance-build-meshes --force --smooth-iter 40

    # Custom output:
    marvel-water-conductance-build-meshes -o /tmp/cortex.vtk
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view.scripts.water_conductance import (  # noqa: E402
    DEFAULT_ALL_INPUT_PATH,
    DEFAULT_ALL_MESH_CACHE_PATH,
    DEFAULT_ARROW_STRIDE,
    DEFAULT_ARROWS_CACHE_PATH,
    DEFAULT_CENTRAL_AXIS_PATH,
    DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_CACHE,
    DEFAULT_CROWN_TRACKS_VTP_CACHE,
    DEFAULT_DENSITY_ALL_CACHE,
    DEFAULT_DENSITY_BRIDGES_CACHE,
    DEFAULT_DENSITY_RADIUS_PX,
    DEFAULT_DENSITY_SIGMA,
    DEFAULT_DENSITY_STEP_PX,
    DEFAULT_DENSITY_UPSAMPLE,
    DEFAULT_DILATATION_TIFF_PATH,
    DEFAULT_FINE_STRIDE,
    DEFAULT_GEODDIST_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_LEVEL,
    DEFAULT_LONG_AXIS_STRIDE,
    DEFAULT_MASK_CORTEX_PATH,
    DEFAULT_MEMBRANES_BG_DIST_PATH,
    DEFAULT_MEMBRANES_LABELS_CACHE,
    DEFAULT_MEMBRANES_META_CACHE,
    DEFAULT_MEMBRANES_VTP_CACHE,
    DEFAULT_MEMBRANES_ISO_CACHE_DIR,
    DEFAULT_LAMES_BG_DIST_PATH,
    DEFAULT_LAMES_LABELS_CACHE,
    DEFAULT_LAMES_META_CACHE,
    DEFAULT_LAMES_VTP_CACHE,
    DEFAULT_LAMES_ISO_CACHE_DIR,
    DEFAULT_LAME2_BG_DIST_PATH,
    DEFAULT_LAME2_LABELS_CACHE,
    DEFAULT_LAME2_META_CACHE,
    DEFAULT_LAME2_VTP_CACHE,
    DEFAULT_LAME2_ISO_CACHE_DIR,
    DEFAULT_MASK_ISO_LEVEL,
    DEFAULT_OUTSIDE_MASK_CACHE,
    DEFAULT_OUTSIDE_TIFF_PATH,
    DEFAULT_STELE_MASK_CACHE,
    DEFAULT_STELE_TIFF_PATH,
    DEFAULT_MESH_CACHE_PATH,
    DEFAULT_N_SOURCE_POINTS,
    DEFAULT_OVERLAY_CACHE_PATH,
    DEFAULT_OVERLAY_LEVEL,
    DEFAULT_OVERLAY_TIFF_PATH,
    DEFAULT_PATHS_DOMAIN_PATH,
    DEFAULT_PILLARS_CACHE_PATH,
    DEFAULT_PILLARS_LEVEL,
    DEFAULT_PILLARS_TIFF_PATH,
    DEFAULT_SMOOTH_ITER,
    DEFAULT_SOURCE_CROWN_PATH,
    DEFAULT_SPACING,
    DEFAULT_TARGET_CROWN_PATH,
    _build_arrow_field,
    _build_crown_dijkstra_tracks,
    _build_density_facet_scalars,
    _build_mesh,
    _write_tracks_arrows_vtp,
    _write_tracks_vtp,
)
from marvel_view.preprocessing import load_float_volume, mask_to_mesh  # noqa: E402
from marvel_view.preprocessing.water_membranes import (  # noqa: E402
    DEFAULT_BG_MIN_DIST as DEFAULT_MEMBRANE_BG_MIN_DIST,
    DEFAULT_LEVEL_STEP as DEFAULT_MEMBRANE_LEVEL_STEP,
    DEFAULT_N_SEEDS as DEFAULT_MEMBRANE_N_SEEDS,
    DEFAULT_PHASE_FACTOR as DEFAULT_MEMBRANE_PHASE_FACTOR,
    DEFAULT_SEED as DEFAULT_MEMBRANE_SEED,
    DEFAULT_TARGET_TRIS_PER_LEVEL as DEFAULT_MEMBRANE_TARGET_TRIS,
    build_water_membranes,
)
from marvel_view.preprocessing.water_lames import (  # noqa: E402
    DEFAULT_FADE_FRAMES        as DEFAULT_LAMES_FADE_FRAMES,
    DEFAULT_KMEANS_ITERS       as DEFAULT_LAMES_KMEANS_ITERS,
    DEFAULT_LAME_WIDTH         as DEFAULT_LAMES_LAME_WIDTH,
    DEFAULT_N_LEVELS_MAX       as DEFAULT_LAMES_N_LEVELS_MAX,
    DEFAULT_N_SEEDS            as DEFAULT_LAMES_N_SEEDS,
    DEFAULT_PHASE_FACTOR       as DEFAULT_LAMES_PHASE_FACTOR,
    DEFAULT_SEED               as DEFAULT_LAMES_SEED,
    DEFAULT_SMOOTH_ITER        as DEFAULT_LAMES_SMOOTH_ITER,
    DEFAULT_TARGET_TRIS_PER_SHELL as DEFAULT_LAMES_TARGET_TRIS,
    build_water_lames,
)
from marvel_view.preprocessing.water_lames2 import (  # noqa: E402
    DEFAULT_FADE_FRAMES        as DEFAULT_LAME2_FADE_FRAMES,
    DEFAULT_HOLD_RATIO         as DEFAULT_LAME2_HOLD_RATIO,
    DEFAULT_KMEANS_ITERS       as DEFAULT_LAME2_KMEANS_ITERS,
    DEFAULT_N_SEEDS            as DEFAULT_LAME2_N_SEEDS,
    DEFAULT_PHASE_FACTOR       as DEFAULT_LAME2_PHASE_FACTOR,
    DEFAULT_SEED               as DEFAULT_LAME2_SEED,
    DEFAULT_SMOOTH_ITER        as DEFAULT_LAME2_SMOOTH_ITER,
    DEFAULT_STEP_GROWTH        as DEFAULT_LAME2_STEP_GROWTH,
    DEFAULT_TARGET_TRIS_PER_SHELL as DEFAULT_LAME2_TARGET_TRIS,
    build_water_lame2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.water_conductance_build_meshes")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build & cache the Wat_Norm_Cortex marching-cubes mesh "
                    "as a .vtk file so the viewer / movie tools can load "
                    "it instantly afterwards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT_PATH),
                   help="Path to the Wat_Norm_Cortex.tif file.")
    p.add_argument("--output", "-o", default=str(DEFAULT_MESH_CACHE_PATH),
                   help="Where to write the cached .vtk mesh.")
    p.add_argument("--level", "-l", type=float, default=DEFAULT_LEVEL,
                   help="Marching-cubes iso-level on the float volume.")
    p.add_argument("--smooth-iter", type=int, default=DEFAULT_SMOOTH_ITER,
                   help="Laplacian smoothing iterations applied to the mesh.")
    p.add_argument("--force", "-f", action="store_true",
                   help="Overwrite the output file(s) if they already exist.")
    p.add_argument("--skip-mesh", action="store_true",
                   help="Don't build the cortex mesh (useful when only "
                        "refreshing the arrow-field cache).")
    p.add_argument("--skip-arrows", action="store_true",
                   help="Don't build the gradient-arrow cache.")
    p.add_argument("--skip-overlay", action="store_true",
                   help="Don't build the translucent overlay isosurface.")
    p.add_argument("--overlay-input", default=str(DEFAULT_OVERLAY_TIFF_PATH),
                   help="8-bit TIFF used to build the overlay isosurface.")
    p.add_argument("--overlay-output", default=str(DEFAULT_OVERLAY_CACHE_PATH),
                   help="Where to write the cached overlay .vtk mesh.")
    p.add_argument("--overlay-level", type=float, default=DEFAULT_OVERLAY_LEVEL,
                   help="Iso-value used to extract the overlay isosurface.")
    # Alternate 'All watered tissues' mesh.
    p.add_argument("--skip-all-mesh", action="store_true",
                   help="Don't build the 'All watered tissues' alternate mesh.")
    p.add_argument("--all-input", default=str(DEFAULT_ALL_INPUT_PATH),
                   help="TIFF (Wat_Norm_All.tif) used to build the alternate mesh.")
    p.add_argument("--all-output", default=str(DEFAULT_ALL_MESH_CACHE_PATH),
                   help="Where to write the cached 'All watered tissues' .vtk mesh.")
    # Glowing green Pillars overlay.
    p.add_argument("--skip-pillars", action="store_true",
                   help="Don't build the green-glow Pillars isosurface.")
    p.add_argument("--pillars-input", default=str(DEFAULT_PILLARS_TIFF_PATH),
                   help="8-bit TIFF (Pillars.tif) used to build the glow overlay.")
    p.add_argument("--pillars-output", default=str(DEFAULT_PILLARS_CACHE_PATH),
                   help="Where to write the cached Pillars .vtk mesh.")
    p.add_argument("--pillars-level", type=float, default=DEFAULT_PILLARS_LEVEL,
                   help="Iso-value used to extract the Pillars isosurface.")
    # Water "membranes" descending animation.
    p.add_argument("--membranes-only", action="store_true",
                   help="Skip every step except the membranes build "
                        "(equivalent to passing all --skip-* flags except "
                        "--skip-membranes).  Combine with --iso-only to run "
                        "only Phase 1a.")
    p.add_argument("--skip-membranes", action="store_true",
                   help="Don't build the water-membranes packed mesh.")
    p.add_argument("--membranes-output",
                   default=str(DEFAULT_MEMBRANES_VTP_CACHE),
                   help="Where to write the cached water-membranes .vtp.")
    p.add_argument("--membranes-meta-output",
                   default=str(DEFAULT_MEMBRANES_META_CACHE),
                   help="Where to write the JSON sidecar for the membranes.")
    p.add_argument("--membranes-labels-output",
                   default=str(DEFAULT_MEMBRANES_LABELS_CACHE),
                   help="Where to write the watershed-labels TIFF "
                        "(int32 per-voxel column id).")
    p.add_argument("--membranes-bg-dist",
                   default=str(DEFAULT_MEMBRANES_BG_DIST_PATH),
                   help="Float32 TIFF: distance-to-background inside the object "
                        "(Source_Target_Possible_Paths-dist.tif).")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_MEMBRANE_N_SEEDS,
                   help="Number of farthest-point seeds picked in the crown "
                        "(= number of water columns).")
    p.add_argument("--phase-factor", type=int,
                   default=DEFAULT_MEMBRANE_PHASE_FACTOR,
                   help="Animation length multiplier: total steps = "
                        "phase_factor × n_iso_levels.")
    p.add_argument("--level-step", type=float,
                   default=DEFAULT_MEMBRANE_LEVEL_STEP,
                   help="Step size between consecutive iso-levels of the "
                        "geodesic-distance field.")
    p.add_argument("--target-tris-per-level", type=int,
                   default=DEFAULT_MEMBRANE_TARGET_TRIS,
                   help="Decimation budget per iso-level (triangles).")
    p.add_argument("--membrane-bg-min-dist", type=float,
                   default=DEFAULT_MEMBRANE_BG_MIN_DIST,
                   help="Drop any triangle with a vertex closer than this to "
                        "the background.")
    p.add_argument("--membrane-seed", type=int,
                   default=DEFAULT_MEMBRANE_SEED,
                   help="RNG seed for reproducible seed sampling + phases.")
    p.add_argument("--speed-test-membrane", action="store_true",
                   help="Crop all 4 membrane input volumes to the first quarter "
                        "of their longest axis before computing.  Useful for a "
                        "quick sanity-check run.")
    p.add_argument("--iso-cache-dir",
                   default=str(DEFAULT_MEMBRANES_ISO_CACHE_DIR),
                   help="Directory for the per-level ISO-surface NPZ cache.  "
                        "Shared between runs; only rebuilt when level_step or "
                        "target-tris-per-level change.  "
                        f"Default: {DEFAULT_MEMBRANES_ISO_CACHE_DIR}")
    p.add_argument("--rebuild-iso-cache", action="store_true",
                   help="Force-rebuild the ISO-surface cache from scratch even "
                        "if it already exists.  Implies re-running marching "
                        "cubes for every iso-level.")
    p.add_argument("--iso-only", action="store_true",
                   help="Run only Phase 1a (marching cubes + decimation → ISO "
                        "cache) and exit.  Subsequent phases (filter, merge, "
                        "VTP write) are skipped.  Useful to pre-build the "
                        "cache on a high-RAM machine before the lighter pass.")
    p.add_argument("--membrane-workers", type=int, default=None,
                   help="Thread count for the per-level marching-cubes pass.  "
                        "Default: auto-detect (cap=4 locally, 8 on server).  "
                        "Override also via MARVEL_MAX_WORKERS env var.")
    # ── Water "lames" (V2) descending animation ──
    p.add_argument("--lames-only", action="store_true",
                   help="Skip every step except the lames (V2) build "
                        "(equivalent to all --skip-* flags except --skip-lames "
                        "+ --skip-membranes).")
    p.add_argument("--skip-lames", action="store_true",
                   help="Don't build the V2 water-lames packed mesh.")
    p.add_argument("--lames-output", default=str(DEFAULT_LAMES_VTP_CACHE),
                   help="Where to write the cached water-lames .vtp.")
    p.add_argument("--lames-meta-output",
                   default=str(DEFAULT_LAMES_META_CACHE),
                   help="Where to write the JSON sidecar for the lames.")
    p.add_argument("--lames-labels-output",
                   default=str(DEFAULT_LAMES_LABELS_CACHE),
                   help="Where to write the lames watershed-labels TIFF.")
    p.add_argument("--lames-bg-dist", default=str(DEFAULT_LAMES_BG_DIST_PATH),
                   help="Distance-to-background TIFF for the lames build "
                        "(same as the membranes one by default).")
    p.add_argument("--lames-iso-cache-dir",
                   default=str(DEFAULT_LAMES_ISO_CACHE_DIR),
                   help="Directory for the per-(label, level) ISO-shell NPZ "
                        "cache used by the lames pipeline.")
    p.add_argument("--rebuild-lames-iso-cache", action="store_true",
                   help="Force-rebuild the lames ISO cache from scratch.")
    p.add_argument("--lames-iso-only", action="store_true",
                   help="Run only Phase 1a of the lames build (per-shell "
                        "marching cubes → ISO cache) and exit.")
    p.add_argument("--n-lames-seeds", type=int, default=DEFAULT_LAMES_N_SEEDS,
                   help="Number of K-means columns (water columns).")
    p.add_argument("--lames-phase-factor", type=int,
                   default=DEFAULT_LAMES_PHASE_FACTOR,
                   help="Animation length multiplier: Nstep = phase × Nlevels.")
    p.add_argument("--lames-n-levels-max", type=int,
                   default=DEFAULT_LAMES_N_LEVELS_MAX,
                   help="Maximum number of iso-distance levels (NT).")
    p.add_argument("--lames-width", type=float,
                   default=DEFAULT_LAMES_LAME_WIDTH,
                   help="Base lame half-width before α-scaling.")
    p.add_argument("--lames-target-tris-per-shell", type=int,
                   default=DEFAULT_LAMES_TARGET_TRIS,
                   help="Decimation budget per (label, level) shell.")
    p.add_argument("--lames-smooth-iter", type=int,
                   default=DEFAULT_LAMES_SMOOTH_ITER,
                   help="Laplacian smoothing iterations per shell.")
    p.add_argument("--lames-fade-frames", type=int,
                   default=DEFAULT_LAMES_FADE_FRAMES,
                   help="Glow envelope ramp length (in animation steps).")
    p.add_argument("--lames-kmeans-iters", type=int,
                   default=DEFAULT_LAMES_KMEANS_ITERS,
                   help="Number of K-means (medoid) iterations.")
    p.add_argument("--lames-seed", type=int, default=DEFAULT_LAMES_SEED,
                   help="RNG seed for the lames build.")
    p.add_argument("--lames-workers", type=int, default=None,
                   help="Thread count for the per-label shell extraction.")
    p.add_argument("--lames-debug", action="store_true",
                   help="Debug mode: pick --lames-debug-n-cols random columns, "
                        "write binary shell-mask TIFFs and a distance-map TIF "
                        "to <lames-output-dir>/debug/. Does NOT build the full "
                        "lames VTP.")
    p.add_argument("--lames-debug-n-cols", type=int, default=3,
                   help="Number of random columns to inspect in --lames-debug "
                        "mode (default: 3).")
    # ── Water "lame2" (V3) descending animation ──
    p.add_argument("--lame2-only", action="store_true",
                   help="Skip every step except the lame2 (V3) build.")
    p.add_argument("--skip-lame2", action="store_true",
                   help="Don't build the V3 water-lame2 packed mesh.")
    p.add_argument("--lame2-output", default=str(DEFAULT_LAME2_VTP_CACHE),
                   help="Where to write the cached water-lame2 .vtp.")
    p.add_argument("--lame2-meta-output",
                   default=str(DEFAULT_LAME2_META_CACHE),
                   help="Where to write the JSON sidecar for the lame2.")
    p.add_argument("--lame2-labels-output",
                   default=str(DEFAULT_LAME2_LABELS_CACHE),
                   help="Where to write the lame2 watershed-labels TIFF.")
    p.add_argument("--lame2-bg-dist", default=str(DEFAULT_LAME2_BG_DIST_PATH),
                   help="Distance-to-background TIFF for the lame2 build.")
    p.add_argument("--lame2-iso-cache-dir",
                   default=str(DEFAULT_LAME2_ISO_CACHE_DIR),
                   help="Directory for the per-(label, frame) ISO-shell NPZ "
                        "cache used by the lame2 pipeline.")
    p.add_argument("--rebuild-lame2-iso-cache", action="store_true",
                   help="Force-rebuild the lame2 ISO cache from scratch.")
    p.add_argument("--lame2-iso-only", action="store_true",
                   help="Run only Phase 1a of the lame2 build.")
    p.add_argument("--n-lame2-seeds", type=int, default=DEFAULT_LAME2_N_SEEDS,
                   help="Number of K-means columns for lame2.")
    p.add_argument("--lame2-phase-factor", type=int,
                   default=DEFAULT_LAME2_PHASE_FACTOR,
                   help="Animation length multiplier: Nstep = phase × max-lifetime.")
    p.add_argument("--lame2-step-growth", type=float,
                   default=DEFAULT_LAME2_STEP_GROWTH,
                   help="Distance-unit increment between consecutive d1 (or "
                        "d0) values during the growth phases (default 2.0).")
    p.add_argument("--lame2-hold-ratio", type=float,
                   default=DEFAULT_LAME2_HOLD_RATIO,
                   help="Phase-B length as a fraction of (A + C) frame count.")
    p.add_argument("--lame2-target-tris-per-shell", type=int,
                   default=DEFAULT_LAME2_TARGET_TRIS,
                   help="Decimation budget per (label, frame) shell.")
    p.add_argument("--lame2-smooth-iter", type=int,
                   default=DEFAULT_LAME2_SMOOTH_ITER,
                   help="Laplacian smoothing iterations per shell.")
    p.add_argument("--lame2-fade-frames", type=int,
                   default=DEFAULT_LAME2_FADE_FRAMES,
                   help="Glow envelope ramp length (frames).")
    p.add_argument("--lame2-kmeans-iters", type=int,
                   default=DEFAULT_LAME2_KMEANS_ITERS,
                   help="Number of K-means (medoid) iterations for lame2.")
    p.add_argument("--lame2-seed", type=int, default=DEFAULT_LAME2_SEED,
                   help="RNG seed for the lame2 build.")
    p.add_argument("--lame2-workers", type=int, default=None,
                   help="Thread count for the per-label lame2 shell extraction.")
    # ── Stele / Outside mask isosurfaces (bundled with lame2) ──
    p.add_argument("--stele-tiff", default=str(DEFAULT_STELE_TIFF_PATH),
                   help="Stele_mask.tif: 8-bit binary mask for the stele.")
    p.add_argument("--stele-output", default=str(DEFAULT_STELE_MASK_CACHE),
                   help="Where to write the stele isosurface .vtk cache.")
    p.add_argument("--outside-tiff", default=str(DEFAULT_OUTSIDE_TIFF_PATH),
                   help="Outside_mask.tif: 8-bit binary mask for the outside.")
    p.add_argument("--outside-output", default=str(DEFAULT_OUTSIDE_MASK_CACHE),
                   help="Where to write the outside isosurface .vtk cache.")
    p.add_argument("--mask-iso-level", type=float, default=DEFAULT_MASK_ISO_LEVEL,
                   help="Iso-level for the stele/outside marching cubes "
                        "(default: 127.5).")
    p.add_argument("--skip-mask-overlays", action="store_true",
                   help="Don't build the Stele/Outside mask isosurface caches.")
    p.add_argument("--geoddist", default=str(DEFAULT_GEODDIST_PATH),
                   help="Path to the geodesic distance-to-exterior TIFF "
                        "used to compute the arrow field.")
    p.add_argument("--arrows-output", default=str(DEFAULT_ARROWS_CACHE_PATH),
                   help="Where to write the cached arrow field (.npz).")
    p.add_argument("--arrow-stride", type=int, default=DEFAULT_ARROW_STRIDE,
                   help="Sub-sampling stride on the two short axes.")
    p.add_argument("--long-stride", type=int, default=DEFAULT_LONG_AXIS_STRIDE,
                   help="Sub-sampling stride on the longest volume axis.")
    p.add_argument("--fine-stride", type=int, default=DEFAULT_FINE_STRIDE,
                   help="Finer stride used internally to evaluate the "
                        "gradient and its divergence (compression).")
    p.add_argument("--dilatation-output", default=str(DEFAULT_DILATATION_TIFF_PATH),
                   help="Path where the float32 dilatation (convergence) TIFF "
                        "will be written alongside the arrow field.")
    p.add_argument("--skip-dilatation", action="store_true",
                   help="Don't write the dilatation TIFF.")
    # Crown Dijkstra tracks.
    p.add_argument("--source-crown", default=str(DEFAULT_SOURCE_CROWN_PATH),
                   help="8-bit TIFF (Source_crown.tif) -- Dijkstra path sources.")
    p.add_argument("--target-crown", default=str(DEFAULT_TARGET_CROWN_PATH),
                   help="8-bit TIFF (Target_crown.tif) -- Dijkstra path targets.")
    p.add_argument("--paths-domain", default=str(DEFAULT_PATHS_DOMAIN_PATH),
                   help="8-bit TIFF (Source_Target_Possible_Paths.tif) -- traversable domain.")
    p.add_argument("--n-source-points", type=int, default=DEFAULT_N_SOURCE_POINTS,
                   help="Number of source voxels to trace.")
    p.add_argument("--tracks-output", default=str(DEFAULT_CROWN_TRACKS_CACHE),
                   help="Where to write the cached crown tracks (.npz).")
    p.add_argument("--tracks-vtp-output", default=str(DEFAULT_CROWN_TRACKS_VTP_CACHE),
                   help="Where to write the binary VTK PolyData (.vtp) of crown "
                        "track polylines.  Loaded at runtime instead of "
                        "reconstructing cells from the .npz.")
    p.add_argument("--tracks-arrows-vtp-output",
                   default=str(DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE),
                   help="Where to write the pre-computed arrow glyph centres / "
                        "tangents / arc-length fractions (.vtp).  When present "
                        "the viewer skips the per-path Python sampling loop.")
    p.add_argument("--tracks-vtp-from-cache", action="store_true",
                   help="Read the existing crown_tracks.npz and write both .vtp "
                        "files (polylines + arrows) WITHOUT re-running Dijkstra. "
                        "Use after a completed build to generate the .vtp files "
                        "from an already-existing .npz cache.")
    p.add_argument("--skip-tracks", action="store_true",
                   help="Don't build the crown Dijkstra tracks.")
    # Density colormap caches.
    p.add_argument("--skip-density", action="store_true",
                   help="Don't build the per-cell density colormap caches.")
    p.add_argument("--mask-cortex", default=str(DEFAULT_MASK_CORTEX_PATH),
                   help="Mask_cortex.tif (binary cortex domain).")
    p.add_argument("--central-axis", default=str(DEFAULT_CENTRAL_AXIS_PATH),
                   help="Coordinates_central_axis.txt -- X=, Y= image-space "
                        "coords of the principal-axis projection.")
    p.add_argument("--density-bridges-output",
                   default=str(DEFAULT_DENSITY_BRIDGES_CACHE),
                   help="Where to write the bridges-mesh per-cell density (.npy).")
    p.add_argument("--density-all-output",
                   default=str(DEFAULT_DENSITY_ALL_CACHE),
                   help="Where to write the all-mesh per-cell density (.npy).")
    p.add_argument("--density-radius", type=float,
                   default=DEFAULT_DENSITY_RADIUS_PX,
                   help="Object radius (px) used to size the θ bin count.")
    p.add_argument("--density-step", type=float,
                   default=DEFAULT_DENSITY_STEP_PX,
                   help="Target bin edge length on the surface (px).")
    p.add_argument("--density-upsample", type=int,
                   default=DEFAULT_DENSITY_UPSAMPLE,
                   help="Upsampling factor applied to the (L, θ) grid "
                        "before Gaussian smoothing.")
    p.add_argument("--density-sigma", type=float,
                   default=DEFAULT_DENSITY_SIGMA,
                   help="Gaussian sigma (upsampled-pixel units) for the "
                        "density-map smoothing (wrap on θ).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.membranes_only:
        args.skip_mesh       = True
        args.skip_all_mesh   = True
        args.skip_arrows     = True
        args.skip_overlay    = True
        args.skip_pillars    = True
        args.skip_dilatation = True
        args.skip_tracks     = True
        args.skip_density    = True
        args.skip_lames      = True
        args.skip_lame2      = True

    if args.lames_only:
        args.skip_mesh       = True
        args.skip_all_mesh   = True
        args.skip_arrows     = True
        args.skip_overlay    = True
        args.skip_pillars    = True
        args.skip_dilatation = True
        args.skip_tracks     = True
        args.skip_density    = True
        args.skip_membranes  = True
        args.skip_lame2      = True

    if args.lame2_only:
        args.skip_mesh       = True
        args.skip_all_mesh   = True
        args.skip_arrows     = True
        args.skip_overlay    = True
        args.skip_pillars    = True
        args.skip_dilatation = True
        args.skip_tracks     = True
        args.skip_density    = True
        args.skip_membranes  = True
        args.skip_lames      = True

    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    geoddist_path = Path(args.geoddist).expanduser().resolve()
    arrows_path = Path(args.arrows_output).expanduser().resolve()
    overlay_input = Path(args.overlay_input).expanduser().resolve()
    overlay_output = Path(args.overlay_output).expanduser().resolve()

    logger.info("═" * 64)
    logger.info("Marvel Water-Conductance · build & cache assets")
    logger.info("  Mesh input   : %s", input_path)
    logger.info("  Mesh output  : %s", out_path)
    logger.info("  Iso-level    : %.3f", args.level)
    logger.info("  Smooth       : %d iterations", args.smooth_iter)
    logger.info("  Geoddist TIFF: %s", geoddist_path)
    logger.info("  Arrows output: %s", arrows_path)
    logger.info("  Arrow stride : %d", args.arrow_stride)
    logger.info("═" * 64)

    # Mesh handles retained for the density-build step at the end.
    mesh = None
    all_mesh = None

    # ── Mesh ──────────────────────────────────────────────────────────────
    if not args.skip_mesh:
        if out_path.exists() and not args.force:
            logger.error(
                "Mesh output already exists: %s  (use --force to overwrite "
                "or --skip-mesh to keep it).", out_path,
            )
            return 2
        mesh = _build_mesh(
            input_path, level=args.level, smooth_iter=args.smooth_iter,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(str(out_path))
        logger.info("Mesh written to %s", out_path)
    else:
        logger.info("Skipping mesh build (--skip-mesh).")

    # ── Arrow field ───────────────────────────────────────────────────────
    if not args.skip_arrows:
        if arrows_path.exists() and not args.force:
            logger.error(
                "Arrows cache already exists: %s  (use --force to overwrite "
                "or --skip-arrows to keep it).", arrows_path,
            )
            return 2
        if not geoddist_path.exists():
            logger.warning(
                "Geodesic-distance TIFF not found: %s – skipping arrows.",
                geoddist_path,
            )
        else:
            import numpy as np
            volume = load_float_volume(geoddist_path)
            arrows = _build_arrow_field(
                volume, stride=args.arrow_stride,
                long_stride=args.long_stride,
                fine_stride=args.fine_stride,
                spacing=DEFAULT_SPACING,
            )
            arrows_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                arrows_path,
                pts=arrows["pts"],
                dirs=arrows["dirs"],
                scores=arrows["scores"],
                boost=arrows["boost"],
                stride=np.int32(arrows["stride"]),
            )
            logger.info(
                "Arrow field written to %s  (%d arrows)",
                arrows_path, len(arrows["pts"]),
            )

            # ── Dilatation TIFF ───────────────────────────────────────────
            if not args.skip_dilatation and "compr_volume" in arrows:
                dilatation_path = Path(
                    args.dilatation_output
                ).expanduser().resolve()
                try:
                    import tifffile
                    dilatation_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(
                        str(dilatation_path),
                        arrows["compr_volume"].astype(np.float32),
                    )
                    logger.info(
                        "Dilatation TIFF written to %s  shape=%s  "
                        "fine_stride=%d",
                        dilatation_path,
                        arrows["compr_volume"].shape,
                        arrows.get("fine_stride", args.fine_stride),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not write dilatation TIFF %s: %s",
                        dilatation_path, exc,
                    )
            elif args.skip_dilatation:
                logger.info("Skipping dilatation TIFF (--skip-dilatation).")
    else:
        logger.info("Skipping arrow-field build (--skip-arrows).")

    # ── Overlay ghost mesh ───────────────────────────────────────────────
    if not args.skip_overlay:
        if overlay_output.exists() and not args.force:
            logger.error(
                "Overlay cache already exists: %s  (use --force to overwrite "
                "or --skip-overlay to keep it).", overlay_output,
            )
            return 2
        if not overlay_input.exists():
            logger.warning(
                "Overlay input TIFF not found: %s – skipping overlay.",
                overlay_input,
            )
        else:
            logger.info("Building overlay isosurface from %s (iso=%.2f) …",
                        overlay_input, args.overlay_level)
            volume = load_float_volume(overlay_input)
            overlay_mesh = mask_to_mesh(
                volume, level=args.overlay_level, smooth_iter=0,
                spacing=DEFAULT_SPACING,
            )
            overlay_output.parent.mkdir(parents=True, exist_ok=True)
            overlay_mesh.write(str(overlay_output))
            logger.info("Overlay mesh written to %s", overlay_output)
    else:
        logger.info("Skipping overlay build (--skip-overlay).")

    # ── Alternate 'All watered tissues' mesh ────────────────────────────
    all_input = Path(args.all_input).expanduser().resolve()
    all_output = Path(args.all_output).expanduser().resolve()
    if not args.skip_all_mesh:
        if all_output.exists() and not args.force:
            logger.error(
                "All-mesh output already exists: %s  (use --force to overwrite "
                "or --skip-all-mesh to keep it).", all_output,
            )
            return 2
        if not all_input.exists():
            logger.warning(
                "All-mesh input TIFF not found: %s \u2013 skipping.", all_input,
            )
        else:
            logger.info(
                "Building 'All watered tissues' mesh from %s (iso=%.3f) \u2026",
                all_input, args.level,
            )
            all_mesh = _build_mesh(
                all_input, level=args.level, smooth_iter=args.smooth_iter,
            )
            all_output.parent.mkdir(parents=True, exist_ok=True)
            all_mesh.write(str(all_output))
            logger.info("All-mesh written to %s", all_output)
    else:
        logger.info("Skipping all-mesh build (--skip-all-mesh).")

    # ── Density colormap (per-cell scalars in [0, 1]) ───────────────────
    if not args.skip_density:
        import numpy as np
        # Re-load meshes from cache when their build was skipped, so the
        # density step can still produce up-to-date .npy files.
        if mesh is None and out_path.exists():
            try:
                import vedo as _vedo_d
                mesh = _vedo_d.Mesh(str(out_path))
                logger.info("Density: loaded bridges mesh from cache %s",
                            out_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Density: could not load bridges mesh from %s: %s",
                    out_path, exc,
                )
        if all_mesh is None and all_output.exists():
            try:
                import vedo as _vedo_d
                all_mesh = _vedo_d.Mesh(str(all_output))
                logger.info("Density: loaded all-mesh from cache %s",
                            all_output)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Density: could not load all-mesh from %s: %s",
                    all_output, exc,
                )

        if mesh is None and all_mesh is None:
            logger.warning(
                "Density: no meshes available -- skipping density build."
            )
        else:
            mask_p = Path(args.mask_cortex).expanduser().resolve()
            axis_p = Path(args.central_axis).expanduser().resolve()
            paths_p = Path(args.paths_domain).expanduser().resolve()
            missing = [str(p) for p in (mask_p, axis_p, paths_p)
                       if not p.exists()]
            if missing:
                logger.warning(
                    "Density: missing input files %s -- skipping.", missing,
                )
            else:
                logger.info(
                    "Density: building (mask=%s  paths=%s  axis=%s  "
                    "R=%.1f px  step=%.2f px)",
                    mask_p, paths_p, axis_p,
                    args.density_radius, args.density_step,
                )
                res = _build_density_facet_scalars(
                    mask_cortex_path=mask_p,
                    paths_domain_path=paths_p,
                    central_axis_path=axis_p,
                    bridges_mesh=mesh,
                    all_mesh=all_mesh,
                    radius_px=args.density_radius,
                    step_px=args.density_step,
                    upsample=args.density_upsample,
                    sigma_up=args.density_sigma,
                )
                for key, out_arg in (
                    ("bridges", args.density_bridges_output),
                    ("all",     args.density_all_output),
                ):
                    arr = res.get(key)
                    if arr is None:
                        continue
                    p = Path(out_arg).expanduser().resolve()
                    if p.exists() and not args.force:
                        logger.error(
                            "Density cache already exists: %s  (use --force "
                            "to overwrite or --skip-density to keep it).", p,
                        )
                        return 2
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, arr)
                    logger.info(
                        "Density (%s) written to %s  shape=%s  "
                        "range=[%.3f, %.3f]",
                        key, p, arr.shape,
                        float(arr.min()), float(arr.max()),
                    )
    else:
        logger.info("Skipping density build (--skip-density).")

    # ── Glowing green Pillars isosurface ────────────────────────────────
    pillars_input = Path(args.pillars_input).expanduser().resolve()
    pillars_output = Path(args.pillars_output).expanduser().resolve()
    if not args.skip_pillars:
        if pillars_output.exists() and not args.force:
            logger.error(
                "Pillars cache already exists: %s  (use --force to overwrite "
                "or --skip-pillars to keep it).", pillars_output,
            )
            return 2
        if not pillars_input.exists():
            logger.warning(
                "Pillars input TIFF not found: %s \u2013 skipping pillars.",
                pillars_input,
            )
        else:
            logger.info(
                "Building Pillars isosurface from %s (iso=%.2f) \u2026",
                pillars_input, args.pillars_level,
            )
            volume = load_float_volume(pillars_input)
            pillars_mesh = mask_to_mesh(
                volume, level=args.pillars_level, smooth_iter=0,
                spacing=DEFAULT_SPACING,
            )
            pillars_output.parent.mkdir(parents=True, exist_ok=True)
            pillars_mesh.write(str(pillars_output))
            logger.info("Pillars mesh written to %s", pillars_output)
    else:
        logger.info("Skipping pillars build (--skip-pillars).")

    # ── Water "membranes" descending animation ──────────────────────────
    if not args.skip_membranes:
        membranes_out      = Path(args.membranes_output).expanduser().resolve()
        membranes_meta_out = Path(args.membranes_meta_output).expanduser().resolve()
        membranes_labels_out = Path(args.membranes_labels_output).expanduser().resolve()
        bg_dist_path       = Path(args.membranes_bg_dist).expanduser().resolve()
        object_path        = Path(args.paths_domain).expanduser().resolve()
        crown_path         = Path(args.source_crown).expanduser().resolve()
        geoddist_for_membranes = Path(args.geoddist).expanduser().resolve()

        missing = [str(p) for p in (geoddist_for_membranes, bg_dist_path,
                                    object_path, crown_path) if not p.exists()]
        if missing:
            logger.warning(
                "Membranes: missing input TIFFs %s -- skipping.", missing,
            )
        else:
            build_water_membranes(
                geoddist_path=geoddist_for_membranes,
                bg_dist_path=bg_dist_path,
                object_path=object_path,
                crown_path=crown_path,
                output_vtp=membranes_out,
                output_meta=membranes_meta_out,
                output_labels=membranes_labels_out,
                iso_cache_dir=Path(args.iso_cache_dir),
                rebuild_iso_cache=args.rebuild_iso_cache,
                iso_only=args.iso_only,
                n_seeds=args.n_seeds,
                phase_factor=args.phase_factor,
                level_step=args.level_step,
                target_tris_per_level=args.target_tris_per_level,
                bg_min_dist=args.membrane_bg_min_dist,
                seed=args.membrane_seed,
                n_workers=args.membrane_workers,
                crop_fraction=0.25 if args.speed_test_membrane else None,
            )
    else:
        logger.info("Skipping membranes build (--skip-membranes).")

    # ── Water "lames" (V2) descending animation ─────────────────────────
    if not args.skip_lames:
        lames_out        = Path(args.lames_output).expanduser().resolve()
        lames_meta_out   = Path(args.lames_meta_output).expanduser().resolve()
        lames_labels_out = Path(args.lames_labels_output).expanduser().resolve()
        lames_bg_dist    = Path(args.lames_bg_dist).expanduser().resolve()
        object_path_l    = Path(args.paths_domain).expanduser().resolve()
        crown_path_l     = Path(args.source_crown).expanduser().resolve()
        geoddist_for_lames = Path(args.geoddist).expanduser().resolve()

        missing = [str(p) for p in (geoddist_for_lames, lames_bg_dist,
                                    object_path_l, crown_path_l)
                   if not p.exists()]
        if missing:
            logger.warning(
                "Lames (V2): missing input TIFFs %s -- skipping.", missing,
            )
        else:
            build_water_lames(
                geoddist_path=geoddist_for_lames,
                bg_dist_path=lames_bg_dist,
                object_path=object_path_l,
                crown_path=crown_path_l,
                output_vtp=lames_out,
                output_meta=lames_meta_out,
                output_labels=lames_labels_out,
                iso_cache_dir=Path(args.lames_iso_cache_dir),
                rebuild_iso_cache=args.rebuild_lames_iso_cache,
                iso_only=args.lames_iso_only,
                n_seeds=args.n_lames_seeds,
                phase_factor=args.lames_phase_factor,
                n_levels_max=args.lames_n_levels_max,
                lame_width=args.lames_width,
                target_tris_per_shell=args.lames_target_tris_per_shell,
                smooth_iter=args.lames_smooth_iter,
                fade_frames=args.lames_fade_frames,
                kmeans_iters=args.lames_kmeans_iters,
                seed=args.lames_seed,
                n_workers=args.lames_workers,
                debug_only=args.lames_debug,
                debug_n_cols=args.lames_debug_n_cols,
            )
    else:
        logger.info("Skipping lames build (--skip-lames).")

    # ── Water "lame2" (V3) descending animation ─────────────────────────
    if not args.skip_lame2:
        lame2_out        = Path(args.lame2_output).expanduser().resolve()
        lame2_meta_out   = Path(args.lame2_meta_output).expanduser().resolve()
        lame2_labels_out = Path(args.lame2_labels_output).expanduser().resolve()
        lame2_bg_dist    = Path(args.lame2_bg_dist).expanduser().resolve()
        object_path_l2   = Path(args.paths_domain).expanduser().resolve()
        crown_path_l2    = Path(args.source_crown).expanduser().resolve()
        geoddist_for_l2  = Path(args.geoddist).expanduser().resolve()

        missing = [str(p) for p in (geoddist_for_l2, lame2_bg_dist,
                                    object_path_l2, crown_path_l2)
                   if not p.exists()]
        if missing:
            logger.warning(
                "Lame2 (V3): missing input TIFFs %s -- skipping.", missing,
            )
        else:
            build_water_lame2(
                geoddist_path=geoddist_for_l2,
                bg_dist_path=lame2_bg_dist,
                object_path=object_path_l2,
                crown_path=crown_path_l2,
                output_vtp=lame2_out,
                output_meta=lame2_meta_out,
                output_labels=lame2_labels_out,
                iso_cache_dir=Path(args.lame2_iso_cache_dir),
                rebuild_iso_cache=args.rebuild_lame2_iso_cache,
                iso_only=args.lame2_iso_only,
                n_seeds=args.n_lame2_seeds,
                phase_factor=args.lame2_phase_factor,
                step_growth=args.lame2_step_growth,
                hold_ratio=args.lame2_hold_ratio,
                target_tris_per_shell=args.lame2_target_tris_per_shell,
                smooth_iter=args.lame2_smooth_iter,
                fade_frames=args.lame2_fade_frames,
                kmeans_iters=args.lame2_kmeans_iters,
                seed=args.lame2_seed,
                n_workers=args.lame2_workers,
            )
    else:
        logger.info("Skipping lame2 build (--skip-lame2).")

    # ── Stele / Outside mask isosurfaces ─────────────────────────────────
    if not args.skip_mask_overlays:
        _iso_level = float(args.mask_iso_level)
        for _tiff_arg, _out_arg, _name in (
            (args.stele_tiff,   args.stele_output,   "stele"),
            (args.outside_tiff, args.outside_output, "outside"),
        ):
            _tiff_p = Path(_tiff_arg).expanduser().resolve()
            _out_p  = Path(_out_arg).expanduser().resolve()
            if not _tiff_p.exists():
                logger.warning(
                    "Mask overlay '%s': TIFF not found at %s — skipping.",
                    _name, _tiff_p,
                )
                continue
            if _out_p.exists() and not args.force:
                logger.info(
                    "Mask overlay '%s' already built at %s  (--force to rebuild).",
                    _name, _out_p,
                )
                continue
            logger.info("Building '%s' isosurface (iso=%.1f) from %s …",
                        _name, _iso_level, _tiff_p)
            try:
                from marvel_view.preprocessing import load_float_volume, mask_to_mesh
                _vol  = load_float_volume(_tiff_p)
                _mesh = mask_to_mesh(
                    _vol,
                    level=_iso_level,
                    smooth_iter=DEFAULT_SMOOTH_ITER,
                    spacing=DEFAULT_SPACING,
                    step_size=1,
                )
                if _mesh is None:
                    logger.warning(
                        "mask_to_mesh returned None for '%s' — skipping.", _name)
                    continue
                _out_p.parent.mkdir(parents=True, exist_ok=True)
                _mesh.write(str(_out_p))
                logger.info("Mask overlay '%s' written to %s  (%d faces)",
                            _name, _out_p, _mesh.ncells)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not build mask overlay '%s': %s",
                               _name, exc)
    else:
        logger.info("Skipping mask-overlay build (--skip-mask-overlays).")
    if not args.skip_tracks and not args.tracks_vtp_from_cache:
        import numpy as np
        tracks_path = Path(args.tracks_output).expanduser().resolve()
        if tracks_path.exists() and not args.force:
            logger.error(
                "Crown tracks cache already exists: %s  (use --force to "
                "overwrite or --skip-tracks to keep it).", tracks_path,
            )
            return 2
        source_p = Path(args.source_crown).expanduser().resolve()
        target_p = Path(args.target_crown).expanduser().resolve()
        domain_p = Path(args.paths_domain).expanduser().resolve()
        missing = [str(p) for p in (source_p, target_p, domain_p)
                   if not p.exists()]
        if missing:
            logger.warning(
                "Crown tracks: missing TIFFs %s -- skipping.", missing,
            )
        else:
            source_vol = load_float_volume(source_p)
            target_vol = load_float_volume(target_p)
            domain_vol = load_float_volume(domain_p)
            # Reuse compr_volume from arrows build if it ran this session.
            _compr = (arrows.get("compr_volume")
                      if not args.skip_arrows and "arrows" in dir() else None)
            _fs = (arrows.get("fine_stride", args.fine_stride)
                   if not args.skip_arrows and "arrows" in dir() else args.fine_stride)
            tracks = _build_crown_dijkstra_tracks(
                source_vol, target_vol, domain_vol,
                spacing=DEFAULT_SPACING,
                n_source_points=args.n_source_points,
                compr_volume=_compr,
                fine_stride=_fs,
            )
            if tracks is not None:
                tracks_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    tracks_path,
                    segs_start=tracks["segs_start"],
                    segs_end=tracks["segs_end"],
                    scores=tracks["scores"],
                    boost=tracks["boost"],
                    dirs=tracks["dirs"],
                    path_lengths=tracks["path_lengths"],
                    stride=np.int32(tracks["stride"]),
                    format=np.bytes_(tracks["format"]),
                )
                logger.info(
                    "Crown tracks written to %s  (%d segments)",
                    tracks_path, len(tracks["segs_start"]),
                )

                # Write the binary .vtp alongside the .npz.  The viewer
                # loads this at runtime via vtkXMLPolyDataReader (no
                # Python-side cell construction loop).
                vtp_path = Path(args.tracks_vtp_output).expanduser().resolve()
                if vtp_path.exists() and not args.force:
                    logger.error(
                        "Crown tracks VTP already exists: %s  (use --force "
                        "to overwrite).", vtp_path,
                    )
                else:
                    _write_tracks_vtp(tracks, vtp_path)
                    # Write the pre-computed arrow glyphs VTP.
                    arrows_vtp_path = Path(
                        args.tracks_arrows_vtp_output
                    ).expanduser().resolve()
                    if arrows_vtp_path.exists() and not args.force:
                        logger.error(
                            "Crown tracks arrows VTP already exists: %s  "
                            "(use --force to overwrite).", arrows_vtp_path,
                        )
                    else:
                        _write_tracks_arrows_vtp(vtp_path, arrows_vtp_path)
    elif args.tracks_vtp_from_cache:
        # ── VTP-only rebuild from existing .npz (no Dijkstra re-run) ──
        import numpy as np
        tracks_path = Path(args.tracks_output).expanduser().resolve()
        if not tracks_path.exists():
            logger.error(
                "Crown tracks cache not found: %s  (run without "
                "--tracks-vtp-from-cache to build it first).", tracks_path,
            )
            return 2
        logger.info(
            "Rebuilding VTPs from existing .npz: %s", tracks_path,
        )
        data = np.load(tracks_path, allow_pickle=False)
        tracks_cache = {k: data[k] for k in data.files}
        vtp_path = Path(args.tracks_vtp_output).expanduser().resolve()
        if vtp_path.exists() and not args.force:
            logger.error(
                "Crown tracks VTP already exists: %s  (use --force to "
                "overwrite).", vtp_path,
            )
            return 2
        _write_tracks_vtp(tracks_cache, vtp_path)
        arrows_vtp_path = Path(
            args.tracks_arrows_vtp_output
        ).expanduser().resolve()
        if arrows_vtp_path.exists() and not args.force:
            logger.error(
                "Crown tracks arrows VTP already exists: %s  "
                "(use --force to overwrite).", arrows_vtp_path,
            )
            return 2
        _write_tracks_arrows_vtp(vtp_path, arrows_vtp_path)
        logger.info("VTP-from-cache rebuild complete.")
    else:
        logger.info("Skipping crown tracks build (--skip-tracks).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
