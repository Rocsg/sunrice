"""Command-line argument parser for ``marvel-water-conductance``.

Lifted verbatim from the historic monolithic script; behaviour and
flags are unchanged.
"""
from __future__ import annotations

import argparse

from .constants import (
    DEFAULT_ALL_INPUT_PATH,
    DEFAULT_ALL_MESH_CACHE_PATH,
    DEFAULT_ARROW_STRIDE,
    DEFAULT_ARROWS_CACHE_PATH,
    DEFAULT_CROWN_TRACKS_ALL_CROWN_ARROWS_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_ALL_CROWN_CACHE,
    DEFAULT_CROWN_TRACKS_ALL_CROWN_SPLINED_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_ALL_CROWN_VTP_CACHE,
    DEFAULT_DENSITY_ALL_CACHE,
    DEFAULT_DENSITY_BRIDGES_CACHE,
    DEFAULT_RADIAL_GRADIENT_ALL_CACHE,
    DEFAULT_RADIAL_GRADIENT_BRIDGES_CACHE,
    DEFAULT_FINE_STRIDE,
    DEFAULT_GEODDIST_PATH,
    DEFAULT_INPUT_PATH,
    DEFAULT_LAME2_META_CACHE,
    DEFAULT_LAME2_VTP_CACHE,
    DEFAULT_LAME2_NORMALS_CACHE_DIR,
    DEFAULT_LAMES_META_CACHE,
    DEFAULT_LAMES_VTP_CACHE,
    DEFAULT_LEVEL,
    DEFAULT_LONG_AXIS_STRIDE,
    DEFAULT_MEMBRANES_META_CACHE,
    DEFAULT_MEMBRANES_VTP_CACHE,
    DEFAULT_MESH_CACHE_PATH,
    DEFAULT_OUTSIDE_MASK_CACHE,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_OVERLAY_CACHE_PATH,
    DEFAULT_OVERLAY_LEVEL,
    DEFAULT_OVERLAY_TIFF_PATH,
    DEFAULT_PILLARS_ALPHA,
    DEFAULT_PILLARS_CACHE_PATH,
    DEFAULT_PILLARS_LEVEL,
    DEFAULT_PILLARS_TIFF_PATH,
    DEFAULT_RAW_PATH,
    DEFAULT_RAW_CORTEX_PATH,
    DEFAULT_RAW_CROWNS_PATH,
    DEFAULT_SMOOTH_ITER,
    DEFAULT_STELE_MASK_CACHE,
    DEFAULT_AIR_DUAL_ARROWS_CACHE,
    DEFAULT_WATER_DUAL_ARROWS_CACHE,
    DEFAULT_WIND_CH4_CACHE,
    DEFAULT_WIND_CH4_DISPLAY,
    DEFAULT_WIND_O2_CACHE,
    DEFAULT_WIND_O2_DISPLAY,
)


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
    p.add_argument("--raw-cortex", default=str(DEFAULT_RAW_CORTEX_PATH),
                   help="Raw volume masked to cortex only "
                        "(Raw_masked_with_only_cortex.tif). "
                        "Shown in the ortho panel when in cortex-mesh mode.")
    p.add_argument("--raw-crowns", default=str(DEFAULT_RAW_CROWNS_PATH),
                   help="Raw volume masked to crowns only "
                        "(Raw_masked_with_only_crowns.tif). "
                        "Shown in the ortho panel when in Dijkstra-path mode.")
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
    p.add_argument("--radial-gradient-bridges-cache",
                   default=str(DEFAULT_RADIAL_GRADIENT_BRIDGES_CACHE),
                   help="Per-cell radial-gradient scalar .npy for the "
                        "cortical-bridges mesh (built by "
                        "marvel-water-conductance-build-meshes).")
    p.add_argument("--radial-gradient-all-cache",
                   default=str(DEFAULT_RADIAL_GRADIENT_ALL_CACHE),
                   help="Per-cell radial-gradient scalar .npy for the "
                        "all-watered-tissues mesh.")
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
    # Water "lames" (V2) descending animation.
    p.add_argument("--lames-vtp-cache",
                   default=str(DEFAULT_LAMES_VTP_CACHE),
                   help="Cached binary .vtp of the V2 water-lames packed mesh.")
    p.add_argument("--lames-meta-cache",
                   default=str(DEFAULT_LAMES_META_CACHE),
                   help="Cached JSON sidecar for the V2 lames.")
    p.add_argument("--no-lames", action="store_true",
                   help="Don't load the V2 water-lames animation.  When BOTH "
                        "lames and membranes are available, lames take over "
                        "the Water ON/OFF toggle.")
    # Water "lame2" (V3) descending animation.
    p.add_argument("--lame2-vtp-cache",
                   default=str(DEFAULT_LAME2_VTP_CACHE),
                   help="Cached binary .vtp of the V3 water-lame2 packed "
                        "mesh.  When present it takes precedence over V2.")
    p.add_argument("--lame2-meta-cache",
                   default=str(DEFAULT_LAME2_META_CACHE),
                   help="Cached JSON sidecar for the V3 lame2.")
    p.add_argument("--no-lame2", action="store_true",
                   help="Don't load the V3 water-lame2 animation.  Falls "
                        "back to V2 lames when present.")
    p.add_argument("--lame2-normals-dir",
                   default=str(DEFAULT_LAME2_NORMALS_CACHE_DIR),
                   help="Directory of pre-baked per-step VTPs with point normals "
                        "(built by marvel-lame2-normals-build).  When present, "
                        "used instead of computing normals at startup.")
    # Stele / Outside mask overlays (displayed alongside the lames).
    p.add_argument("--stele-mask-cache",
                   default=str(DEFAULT_STELE_MASK_CACHE),
                   help="Cached .vtk isosurface of Stele_mask.tif (iso=127.5).")
    p.add_argument("--outside-mask-cache",
                   default=str(DEFAULT_OUTSIDE_MASK_CACHE),
                   help="Cached .vtk isosurface of Outside_mask.tif (iso=127.5).")
    p.add_argument("--no-mask-overlays", action="store_true",
                   help="Don't load the Stele / Outside mask overlay meshes.")
    # Wind / gas particles (built by marvel-wind-field-build).
    p.add_argument("--wind-o2-cache",  default=str(DEFAULT_WIND_O2_CACHE),
                   help="Cached .npz of O₂ particle trajectory templates "
                        "(produced by marvel-wind-field-build).")
    p.add_argument("--wind-ch4-cache", default=str(DEFAULT_WIND_CH4_CACHE),
                   help="Cached .npz of CH₄ particle trajectory templates "
                        "(produced by marvel-wind-field-build).")
    p.add_argument("--wind-o2-display",  type=int,
                   default=DEFAULT_WIND_O2_DISPLAY,
                   help="Number of O₂ particles displayed simultaneously "
                        "(templates are reused at random phase offsets).")
    p.add_argument("--wind-ch4-display", type=int,
                   default=DEFAULT_WIND_CH4_DISPLAY,
                   help="Number of CH₄ particles displayed simultaneously.")
    p.add_argument("--no-wind", action="store_true",
                   help="Don't load the wind / gas particle caches "
                        "(hides the O₂ and CH₄ toggle buttons).")
    # Crown geodesic tracks (Arrows view #2) — built by build-meshes, loaded here.
    p.add_argument("--tracks-cache", default=str(DEFAULT_CROWN_TRACKS_ALL_CROWN_CACHE),
                   help="Cached .npz of pre-computed Dijkstra track segments "
                        "(produced by marvel-water-conductance-build-meshes).")
    p.add_argument("--tracks-vtp-cache", default=str(DEFAULT_CROWN_TRACKS_ALL_CROWN_VTP_CACHE),
                   help="Cached binary .vtp of crown-track polylines "
                        "(produced alongside the .npz by build-meshes). "
                        "When present, loaded instead of reconstructing cells "
                        "from the .npz, giving near-instant actor construction.")
    p.add_argument("--tracks-arrows-vtp-cache",
                   default=str(DEFAULT_CROWN_TRACKS_ALL_CROWN_ARROWS_VTP_CACHE),
                   help="Cached binary .vtp of pre-computed arrow glyph centres, "
                        "tangents and arc-length fractions (produced by "
                        "build-meshes).  When present, skips the Python for-loop "
                        "in _make_tracks_actor_curves.")
    p.add_argument("--tracks-splined-vtp-cache",
                   default=str(DEFAULT_CROWN_TRACKS_ALL_CROWN_SPLINED_VTP_CACHE),
                   help="Cached binary .vtp of pre-splined crown-track polylines "
                        "(64 subdivisions, produced by build-meshes).  When "
                        "present, the viewer and movie skip vtkSplineFilter.")
    p.add_argument("--dual-water-cache",
                   default=str(DEFAULT_WATER_DUAL_ARROWS_CACHE),
                   help="Path to water_dual_arrows.npz (built by marvel-water-harmonic-build).")
    p.add_argument("--dual-air-cache",
                   default=str(DEFAULT_AIR_DUAL_ARROWS_CACHE),
                   help="Path to air_dual_arrows.npz (built by marvel-water-harmonic-build).")
    p.add_argument("--no-tracks", action="store_true",
                   help="Disable the crown-tracks view.")
    p.add_argument("--tracks-stride", type=int, default=4,
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
    p.add_argument("--keyboard", choices=["azerty", "qwerty"], default=None,
                   help="Force keyboard layout (overrides auto-detection). "
                        "Useful on headless servers where detection may be wrong.")
    return p.parse_args(argv)
