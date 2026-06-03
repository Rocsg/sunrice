#!/usr/bin/env python3
"""
Marvel Water-Conductance – Camera fly-through movie renderer.

Reads a ``positions/positions_*.json`` file (produced by the "Save pos"
button in :mod:`marvel_view.scripts.water_conductance`), spline-
interpolates the camera + mesh-actor states between control points, and
either renders an MP4 movie or replays the fly-through interactively
(``--preview``) so the trajectory can be verified before encoding.

A green "cable-car" line, slightly offset above the camera path, is
overlaid by default to make the trajectory visible in 3-D space.

* No UI / buttons / sliders – only the mesh + (optionally) the path.
* Output resolution: 1920×1080 (Full HD).
* Frame rate: 90 fps.
* 2 seconds (= 50 frames) between each consecutive control point.
* Progress is printed live: frame index, percentage, ETA.

Usage
-----
::

    marvel-water-movie                                  # latest positions, render MP4
    marvel-water-movie --preview                        # interactive replay, no MP4
    marvel-water-movie -p positions/positions_*.json -o flythrough.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view.scripts.water_conductance import (
    BACKGROUND,
    DEFAULT_ALL_MESH_CACHE_PATH,
    DEFAULT_ARROW_LENGTH,
    DEFAULT_ARROW_THICKNESS,
    DEFAULT_AIR_DUAL_ARROWS_CACHE,
    DEFAULT_ARROWS_CACHE_PATH,
    DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_SPLINED_VTP_CACHE,
    DEFAULT_CROWN_TRACKS_VTP_CACHE,
    DEFAULT_WATER_DUAL_ARROWS_CACHE,
    _load_or_build_dual_arrows,
    DEFAULT_INPUT_PATH,
    DEFAULT_LAME2_META_CACHE,
    DEFAULT_LAME2_MOVIE_META_CACHE,
    DEFAULT_LAME2_MOVIE_VTP_CACHE,
    DEFAULT_LAME2_SOURCE_FPS,
    DEFAULT_LAME2_VTP_CACHE,
    DEFAULT_LAME2_NORMALS_CACHE_DIR,
    DEFAULT_LAMES_META_CACHE,
    DEFAULT_LAMES_VTP_CACHE,
    DEFAULT_LEVEL,
    TORTUOSITY_CMAP_STOPS,
    TORTUOSITY_VMAX,
    TORTUOSITY_VMIN,
    TRACK_BIN_N,
    TRACK_BIN_RADIUS_FADE,
    TRACK_BIN_RADIUS_FULL,
    TRACK_LINE_WIDTH_MOVIE,
    TRACK_TUBE_RADIUS_MOVIE,
    TRACK_TUBE_SIDES_MOVIE,
    DEFAULT_MESH_CACHE_PATH,
    DEFAULT_MP4_DIR,
    DEFAULT_PILLARS_CACHE_PATH,
    DEFAULT_POSITIONS_DIR,
    DEFAULT_SMOOTH_ITER,
    _build_lames_step_polydatas,
    _load_lames,
    load_lame2_normals_cache,
    _load_or_build_mesh,
    _set_shading,
    _style_mesh,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.water_movie")


# ── defaults ─────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1920, 1080
FPS = 90
SECONDS_PER_SEGMENT = 3.0
HOLD_SECONDS = 10.0
EASE_SEGMENTS = 2     # number of trailing segments over which to ease-out
EASE_STRENGTH = 3.0   # power of the ease-out curve on the very last segment

# ── VR / stereo equirectangular defaults ────────────────────────────────────
# Side-by-side stereo equirectangular ("VR180" / "VR360") output.
# When ``--vr vr180`` (resp. ``vr360``) is selected, the canvas dimensions
# below override ``--width`` / ``--height`` unless the user passes them
# explicitly.  Per-eye image is (width/2, height); the two halves are
# concatenated horizontally (left|right) into the final frame.
VR180_SBS_W, VR180_SBS_H = 4096, 2048    # 2048×2048 per eye (1:1 angle 180°)
VR360_SBS_W, VR360_SBS_H = 8192, 2048    # 4096×2048 per eye (2:1 angle 360°)
VR_IPD_FRAC = 0.05                       # half-IPD = 2.5 % of cam→focal dist
VR_CUBE_RESOLUTION = 2048                # internal cube-face size (px) – 2048
                                         # gives noticeably sharper output
                                         # than 1024 with ~2× cost.

# Green "cable-car" trajectory line.
PATH_COLOR = (0, 220, 60)
PATH_LINE_WIDTH = 4                      # only used in mono / preview
# Tube radius as a fraction of the median camera-to-focal distance.
# A real 3D tube (vs. a screen-space Line) is needed for VR because
# vtkPanoramicProjectionPass remaps 6 cube faces to equirect, and a
# 4-px-wide Line shrinks to sub-pixel — invisible — after that remap.
PATH_TUBE_RADIUS_FRAC    = 0.0008   # flat / preview
PATH_TUBE_RADIUS_FRAC_VR = 0.00005  # VR (panoramic remap makes tube look larger)
# Fixed world-space offset above the camera path along the world-up axis.
PATH_VERTICAL_OFFSET = 5.0  # raised 1 m (2 vox at 0.5 m/vox) to clear billboard


# ──────────────────────────────────── CLI ────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render a fly-through MP4 (or interactive preview) from a saved "
            "positions JSON, with spline interpolation of camera + actor state."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--positions", "-p", default=None,
                   help="Path to a positions_*.json file "
                        "(default: latest under ./positions/).")
    p.add_argument("--input", "-i", default=str(DEFAULT_INPUT_PATH),
                   help="Path to the Wat_Norm_Cortex.tif file.")
    p.add_argument("--mesh-cache", default=str(DEFAULT_MESH_CACHE_PATH),
                   help="Path to a cached .vtk mesh (built once via "
                        "`marvel-water-conductance-build-meshes`).  Loaded "
                        "directly when present; otherwise marching cubes "
                        "runs from the TIFF input.")
    p.add_argument("--rebuild-mesh", action="store_true",
                   help="Ignore --mesh-cache and re-run marching cubes.")
    p.add_argument("--raw", default=None,
                   help="Path to the raw greyscale volume (Raw.tif) used "
                        "for the orthogonal-slice locator panel "
                        "(default: same folder as --input).")
    p.add_argument("--no-ortho-panel", action="store_true",
                   help="Disable the orthogonal-slice locator panel.")
    p.add_argument("--level", "-l", type=float, default=DEFAULT_LEVEL,
                   help="Marching-cubes iso-level.")
    p.add_argument("--smooth-iter", type=int, default=DEFAULT_SMOOTH_ITER,
                   help="Laplacian smoothing iterations applied to the mesh.")
    p.add_argument("--output", "-o", default=None,
                   help="Output MP4 path (default: alongside the input JSON).")
    p.add_argument("--prefix", "-P", default=None,
                   help="String prepended to the auto-generated MP4 / frames-dir names "
                        "(useful to run two renders in parallel without collisions).")
    p.add_argument("--fps", type=int, default=FPS,
                   help="Frame rate of the output movie.")
    p.add_argument("--seconds-per-segment", type=float,
                   default=SECONDS_PER_SEGMENT,
                   help="Duration between two consecutive control points "
                        "(seconds).")
    p.add_argument("--hold-seconds", type=float, default=HOLD_SECONDS,
                   help="Duration to freeze on the final frame (the midpoint "
                        "between the two last control points).")
    p.add_argument("--ease-segments", type=int, default=EASE_SEGMENTS,
                   help="Number of trailing segments over which the camera "
                        "decelerates (0 disables easing).")
    p.add_argument("--ease-strength", type=float, default=EASE_STRENGTH,
                   help="Strength of the ease-out (>1 = stronger braking).")
    p.add_argument("--width", type=int, default=WIDTH,
                   help="Output width (pixels).")
    p.add_argument("--height", type=int, default=HEIGHT,
                   help="Output height (pixels).")
    p.add_argument("--preview", action="store_true",
                   help="Open an interactive window and replay the "
                        "trajectory.  No MP4 is produced.")
    p.add_argument("--no-path", action="store_true",
                   help="Hide the green cable-car line.")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep individual PNG frames next to the MP4.")

    # ── stereo VR ─────────────────────────────────────────────────────────
    p.add_argument("--vr", action="store_true",
                   help="Render as stereo equirectangular side-by-side (SBS) "
                        "for VR headsets (full 360° sphere).  Each frame "
                        "renders twice (left + right eye) and roughly 6× "
                        "slower than mono per eye (panoramic cubemap "
                        "projection).")
    p.add_argument("--ipd-frac", type=float, default=VR_IPD_FRAC,
                   help="Interocular distance as a fraction of the camera→"
                        "focal-point distance (0.03 = subtle, 0.05 = "
                        "standard, 0.08 = strong relief).  Ignored when "
                        "--meters-per-voxel is set.")
    p.add_argument("--meters-per-voxel", type=float, default=0.5,
                   help="Physical scale: how many metres one voxel represents. "
                        "When >0, the IPD is fixed at --ipd-metres / "
                        "--meters-per-voxel voxels, independent of camera "
                        "distance.  Default 0.5 m/vox (= 200 m object for "
                        "400 vox).  Set to 0 to revert to the legacy "
                        "--ipd-frac × cam-distance formula.")
    p.add_argument("--ipd-metres", type=float, default=0.065,
                   help="Real human inter-pupillary distance in metres "
                        "(default 0.065 = 65 mm).  Only used together with "
                        "--meters-per-voxel.")
    p.add_argument("--vr-cube-resolution", type=int, default=VR_CUBE_RESOLUTION,
                   help="Internal cubemap face size (px) used by the "
                        "panoramic projection pass.  Higher = sharper but "
                        "slower; 1024 is a good compromise.")
    p.add_argument("--vr-no-path", action="store_true",
                   help="In VR mode, hide the green path line "
                        "(it can look weird inside a headset).")

    # ── codec / quality ───────────────────────────────────────────────────
    p.add_argument("--codec", choices=["h264", "h265", "auto"], default="auto",
                   help="Video codec for the MP4.  'auto' picks h265 (HEVC) "
                        "for VR (recommended at 4K-8K, supported natively by "
                        "Meta Quest & YouTube VR) and h264 for mono.")
    p.add_argument("--crf", type=int, default=None,
                   help="CRF quality (lower = better; 0 = lossless).  "
                        "Defaults: 20 for VR (h265), 18 for mono (h264).")
    p.add_argument("--preset", default=None,
                   help="x264/x265 encoding preset (ultrafast, superfast, "
                        "veryfast, faster, fast, medium, slow, slower, "
                        "veryslow).  Defaults: 'slow' for VR, 'medium' for "
                        "mono.  Slower = better compression at equal CRF.")

    p.add_argument("--stresstest", action="store_true",
                   help="Render only a short window of footage "
                        "(defined by --stresstest-start / --stresstest-end).  "
                        "Useful to validate a VR rendering pipeline before "
                        "committing to a multi-minute encode.")
    p.add_argument("--stresstest-start", type=float, default=0.0,
                   help="Start time (seconds) of the stresstest window.  "
                        "Default: 0.")
    p.add_argument("--stresstest-end", type=float, default=4.0,
                   help="End time (seconds) of the stresstest window.  "
                        "Default: 4.")
    p.add_argument("--print-debug", action="store_true",
                   help="Diagnostic mode: skip mesh loading, build the track,"
                        " apply all transformations (stresstest, VR lock, "
                        "smoothing), then print per-frame positions, "
                        "velocities, and an FFT analysis for the first 2 s "
                        "of the window.  Combine with --stresstest / "
                        "--stresstest-start / --stresstest-end to inspect any "
                        "time window without rendering a single frame.")
    p.add_argument("--fast", action="store_true",
                   help="Low-resolution quick-render mode.  Overrides "
                        "resolution to 960×540 (flat) or 2048×512 / "
                        "1024×512 (vr360 / vr180), sets cube-face to 1024 px, "
                        "and uses --preset ultrafast CRF 28 (unless those "
                        "are already specified).  Keeps the full duration; "
                        "combine with --stresstest for a ~4-second clip.")
    p.add_argument("--high", action="store_true",
                   help="High-quality render mode.  Sets flat resolution to "
                        "3840×2160 (4K), VR cube-face to 4096 px (keep SBS "
                        "canvas at default), CRF 16 and preset veryslow "
                        "(unless those are already specified).")

    # ── anti-aliasing ────────────────────────────────────────────────────
    p.add_argument("--msaa", type=int, default=8,
                   help="MSAA multi-sample count for the render window "
                        "(0 = off, 4 / 8 / 16 typical).  Note: silently "
                        "ignored when VR mode uses vtkPanoramicProjectionPass "
                        "(custom render passes bypass window-level MSAA); "
                        "in VR rely on FXAA + a larger --vr-cube-resolution.")
    p.add_argument("--no-fxaa", action="store_true",
                   help="Disable VTK's post-process FXAA pass.  FXAA is "
                        "cheap and dramatically reduces jaggies on mesh "
                        "silhouettes and text, including in VR.")
    p.add_argument("--fxaa-contrast", type=float, default=0.0833,
                   help="FXAA RelativeContrastThreshold (VTK default 0.0833; "
                        "lower = more aggressive smoothing).")
    p.add_argument("--fxaa-hard-contrast", type=float, default=0.0312,
                   help="FXAA HardContrastThreshold (VTK default 0.0625; "
                        "lower = more aggressive on high-contrast edges).")
    p.add_argument("--save-mid-png", action="store_true", default=True,
                   help="Also save the middle frame as a standalone PNG "
                        "next to the MP4 (uncompressed, pre-encoding).  "
                        "Useful to check whether residual aliasing comes "
                        "from rendering or from video encoding.")
    p.add_argument("--no-save-mid-png", dest="save_mid_png",
                   action="store_false",
                   help="Disable the mid-frame PNG dump.")
    p.add_argument("--ultra-high", action="store_true",
                   help="Ultra-high-quality render mode.  Superset of --high: "
                        "VR cube-face resolution bumped to 8192 px (4× SSAA "
                        "vs default 2048), flat output 5120×2880 (5K), "
                        "CRF 10 and preset veryslow.  Very slow (~4× --high). "
                        "Implies --high; explicit --crf / --preset still "
                        "take precedence.")
    p.add_argument("--compromise", action="store_true",
                   help="VR compromise mode for 90 fps playback on Meta Quest. "
                        "Reduces the SBS canvas from 8192\u00d72048 to 6144\u00d71536 "
                        "(3072\u00d71536 per eye, 75%% of normal pixels) to ease "
                        "decoder load, and bumps the cube-face from 2048 to "
                        "3072 px for sharper distant detail.  CRF 16.  "
                        "Override cube-face with --vr-cube-resolution or "
                        "canvas with --width/--height.  "
                        "Combine with --stresstest to test quickly.")
    p.add_argument("--compromise-better", action="store_true",
                   help="VR compromise-better mode: slightly larger canvas than "
                        "--compromise but still below normal.  "
                        "Sets SBS canvas to 7168\u00d71792 (3584\u00d71792 per eye, "
                        "~87%% of normal pixels), cube-face to 3584 px, CRF 14.  "
                        "A good middle ground when --compromise is too soft but "
                        "--normal is too heavy for the Quest decoder.")
    p.add_argument("--trick", type=int, default=None, metavar="N",
                   help="Keep only the first N control points before "
                        "building the track.  Handy to quickly render the "
                        "beginning of a trajectory without touching the "
                        "positions file.  E.g. --trick 12 stops at the "
                        "12th saved position.")

    p.add_argument("--debugdisplayindexframes", action="store_true",
                   help="Flat-mode only: burn an indicative keyframe index "
                        "near the bottom of every frame.  The index is a "
                        "float relative to the saved control points "
                        "(0 = first 'Save pos', N-1 = last).  Useful to "
                        "tell the assistant where in the trajectory to "
                        "trigger a specific behaviour.  Ignored when "
                        "--vr is not 'off'.")
    p.add_argument("--debug-tracks", action="store_true",
                   help="Print detailed diagnostics about ui_state/view_mode "
                        "replay and crown-track actor visibility while "
                        "rendering.  Useful to debug missing pathways.")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    # ── UI-state replay (matches the viewer's saved per-keyframe state)──
    p.add_argument("--alt-mesh-cache", default=str(DEFAULT_ALL_MESH_CACHE_PATH),
                   help="Path to a cached .vtk mesh for the alternate "
                        "('All watered tissues') view.  When provided AND "
                        "the positions JSON contains a 'mesh_all' view_mode "
                        "at some keyframe, the movie will swap to it. "
                        f"(default: {DEFAULT_ALL_MESH_CACHE_PATH})")
    p.add_argument("--pillars-cache", default=str(DEFAULT_PILLARS_CACHE_PATH),
                   help="Path to a cached .vtk overlay mesh ('Pillars').  "
                        "When provided AND any keyframe sets "
                        "pillars_visible=true, the overlay is shown / "
                        "styled / hue-rotated accordingly.  "
                        f"(default: {DEFAULT_PILLARS_CACHE_PATH})")
    p.add_argument("--membranes-vtp-cache", default=None,
                   help="Path to the cached water-membranes .vtp.  When "
                        "provided AND keyframes set membranes_visible=true, "
                        "the descending-water animation is replayed.")
    p.add_argument("--membranes-meta-cache", default=None,
                   help="JSON sidecar with the membranes n_steps / phases. "
                        "Required alongside --membranes-vtp-cache.")
    p.add_argument("--lames-vtp-cache",
                   default=None,
                   help="Path to the cached water-lames .vtp.  When "
                        "provided AND keyframes set lames_visible=true, "
                        "the descending-lames animation is replayed. "
                        "Defaults to auto-select based on --fps: "
                        "25 fps → lame2.vtp, other → lame2_<fps>fps.vtp.")
    p.add_argument("--lames-meta-cache",
                   default=None,
                   help="JSON sidecar with the lames n_steps. "
                        "Required alongside --lames-vtp-cache. "
                        "Defaults to auto-select based on --fps.")
    p.add_argument(
        "--lames-fps",
        type=int,
        default=None,
        help="Override source FPS for the lames animation. "
             "By default uses lames_meta.json['fps'] or 25.",
    )
    p.add_argument("--arrows-cache",
                   default=str(DEFAULT_ARROWS_CACHE_PATH),
                   help="Path to the cached geodesic-distance arrows .npz.  "
                        "When present, the 'Arrows grid' view-mode renders "
                        "the gradient field in the movie.  "
                        f"(default: {DEFAULT_ARROWS_CACHE_PATH})")
    p.add_argument("--dual-water-cache",
                   default=str(DEFAULT_WATER_DUAL_ARROWS_CACHE),
                   help="Path to the cached dual water arrows .npz.  "
                        "Used for the 'arrows_dual' view-mode.  "
                        f"(default: {DEFAULT_WATER_DUAL_ARROWS_CACHE})")
    p.add_argument("--dual-air-cache",
                   default=str(DEFAULT_AIR_DUAL_ARROWS_CACHE),
                   help="Path to the cached dual air arrows .npz.  "
                        "Used for the 'arrows_dual' view-mode.  "
                        f"(default: {DEFAULT_AIR_DUAL_ARROWS_CACHE})")
    p.add_argument("--crown-tracks-cache",
                   default=str(DEFAULT_CROWN_TRACKS_VTP_CACHE),
                   help="Path to the cached crown Dijkstra tracks .vtp.  "
                        "When present, the 'Conduction shorter paths' mode "
                        "shows splined path lines.  "
                        f"(default: {DEFAULT_CROWN_TRACKS_VTP_CACHE})")
    p.add_argument("--crown-tracks-arrows-cache",
                   default=str(DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE),
                   help="Path to the cached crown tracks glyph-arrow "
                        "centres .vtp (pre-built by "
                        "marvel-water-conductance-build-meshes).  "
                        f"(default: {DEFAULT_CROWN_TRACKS_ARROWS_VTP_CACHE})")
    p.add_argument("--crown-tracks-splined-cache",
                   default=str(DEFAULT_CROWN_TRACKS_SPLINED_VTP_CACHE),
                   help="Path to the pre-splined crown tracks .vtp (64 "
                        "subdivisions, pre-built by build-meshes).  When "
                        "present, skips vtkSplineFilter at load time.  "
                        f"(default: {DEFAULT_CROWN_TRACKS_SPLINED_VTP_CACHE})")
    p.add_argument("--no-tracks-small", action="store_true",
                   help="In VR mode, use the full splined tracks VTP instead "
                        "of the lighter half-density version "
                        "(crown_tracks_splined_small.vtp).  Ignored in flat "
                        "mode (small VTP is only auto-selected when --vr is "
                        "set and the small VTP exists).")
    p.add_argument("--ui-transition-seconds", type=float, default=0.5,
                   help="Duration of the smooth ramp at every keyframe "
                        "transition for UI-state values (opacity, hue, "
                        "etc.).  Set to 0 for instant step changes.  "
                        "Discrete fields (view_mode, fog_on, ssao_on) "
                        "always snap at the midpoint of the ramp.")
    p.add_argument("--ignore-ui-state", action="store_true",
                   help="Ignore any 'ui_state' block in the positions JSON "
                        "and render with a fixed configuration (legacy "
                        "behaviour).")
    p.add_argument("--egl", action="store_true",
                   help="Force VTK to use the EGL render window backend "
                        "(vtkEGLRenderWindow) instead of the default X11/GLX "
                        "window.  Required on headless servers where no X "
                        "display is available.  VTK must have been built with "
                        "EGL support (vtkEGLRenderWindow must exist).")
    # ── Depth / atmosphere ───────────────────────────────────────────
    p.add_argument("--fog", dest="fog", action="store_true", default=True,
                   help="Force depth-fog ON for the entire movie (default). "
                        "Applies a GLSL fragment fog to all mesh actors so "
                        "distant surfaces fade toward the background colour.")
    p.add_argument("--no-fog", dest="fog", action="store_false",
                   help="Disable depth fog.")
    p.add_argument("--fog-near", type=float, default=0.45, metavar="FRAC",
                   help="Start of fog ramp, as a fraction of the scene "
                        "bounding-box diagonal.  Below this distance surfaces "
                        "are unaffected.  Default: 0.45 (45%% of diagonal). "
                        "Use higher values to keep nearby cavern walls crisp.")
    p.add_argument("--fog-far", type=float, default=0.92, metavar="FRAC",
                   help="End of fog ramp (full background colour), as a "
                        "fraction of the scene diagonal.  Default: 0.92.")
    p.add_argument("--torch-tilt", type=float, default=10.0, metavar="DEG",
                   help="Tilt the VR torch light downward by this many "
                        "degrees from the travel direction.  Creates a "
                        "bright-near / dark-far floor gradient that helps "
                        "depth perception in enclosed cavern spaces.  "
                        "Only affects VR mode.  Default: 10.")
    # ── Parallel rendering ─────────────────────────────────────────────────
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Spawn N independent worker subprocesses that each "
                        "render 1/N of the frame sequence in parallel, then "
                        "encode once.  Each worker loads all VTK assets "
                        "independently; ensure enough RAM for N copies.  "
                        "Not compatible with --preview.")
    # ── Internal flags injected by --parallel; not intended for end users ──
    p.add_argument("--_worker", action="store_true", dest="_worker",
                   help=argparse.SUPPRESS)
    p.add_argument("--_chunk-start", type=int, default=None,
                   dest="_chunk_start", help=argparse.SUPPRESS)
    p.add_argument("--_chunk-end",   type=int, default=None,
                   dest="_chunk_end",   help=argparse.SUPPRESS)
    p.add_argument("--_frame-offset", type=int, default=0,
                   dest="_frame_offset", help=argparse.SUPPRESS)
    return p.parse_args(argv)


# ──────────────────────────────── helpers ────────────────────────────────────


def _pick_positions_file(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Positions file not found: {p}")
        return p
    candidates = sorted(DEFAULT_POSITIONS_DIR.glob("positions_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No positions_*.json found under {DEFAULT_POSITIONS_DIR}.  "
            "Open the viewer and click 'Save pos' a few times first."
        )
    return candidates[-1]


def _load_control_points(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        data = [data]
    if not data:
        raise ValueError(f"{json_path} contains no camera entries.")
    return data


def _stack_camera_field(points: Sequence[dict], key: str) -> np.ndarray:
    vals = [np.asarray(p["camera"][key], dtype=float) for p in points]
    return np.stack(vals, axis=0)


def _stack_actor_field(points: Sequence[dict], key: str, default) -> np.ndarray:
    """Stack an actor attribute; fall back to ``default`` if missing.

    Older JSON files (saved before actor capture was added) only contain
    the ``camera`` block; treat the mesh as having identity transform.
    """
    vals = []
    for p in points:
        actor = p.get("actor", {})
        v = actor.get(key, default)
        if v is None:
            v = default
        vals.append(np.asarray(v, dtype=float))
    return np.stack(vals, axis=0)


def _interp_vec(values: np.ndarray, t_out: np.ndarray, *,
                t_ctrl: np.ndarray | None = None) -> np.ndarray:
    """Interpolate (n_ctrl, d) → (len(t_out), d) via cubic spline.

    ``t_out`` is expected in [0, 1].  When ``t_ctrl`` is *None* the control
    points are placed uniformly on that interval; otherwise the caller
    supplies the exact knot positions (e.g. arc-length fractions).
    Falls back to linear when scipy is unavailable or fewer than 4 knots.
    """
    n_ctrl, d = values.shape
    n_frames = len(t_out)
    if n_ctrl == 1:
        return np.tile(values, (n_frames, 1))

    if t_ctrl is None:
        t_ctrl = np.linspace(0.0, 1.0, n_ctrl)

    if n_ctrl >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(t_ctrl, values, bc_type="natural", axis=0)
            return cs(t_out)
        except (ImportError, ValueError):
            # ValueError: non-strictly-increasing knots (e.g. idle keyframes
            # at the same position) — fall back to linear interpolation.
            pass

    out = np.empty((n_frames, d), dtype=float)
    for j in range(d):
        out[:, j] = np.interp(t_out, t_ctrl, values[:, j])
    return out


def _interp_scalar(values: np.ndarray, t_out: np.ndarray, *,
                   t_ctrl: np.ndarray | None = None) -> np.ndarray:
    return _interp_vec(values.reshape(-1, 1), t_out, t_ctrl=t_ctrl).ravel()


def _interp_orientation(euler_deg: np.ndarray, t_out: np.ndarray, *,
                        t_ctrl: np.ndarray | None = None) -> np.ndarray:
    """Slerp-interpolate orientations expressed as XYZ Euler angles (deg).

    ``t_out`` is in [0, 1].  When ``t_ctrl`` is *None* the control
    orientations are evenly spaced; otherwise the caller supplies knot
    positions (e.g. arc-length fractions).  Linearly interpolating Euler
    angles produces visibly wrong intermediate orientations as soon as
    more than one axis is non-zero, hence the detour through quaternions.
    """
    n_ctrl = len(euler_deg)
    n_frames = len(t_out)
    if n_ctrl == 1:
        return np.tile(euler_deg, (n_frames, 1))

    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError:
        return _interp_vec(euler_deg, t_out, t_ctrl=t_ctrl)

    if t_ctrl is None:
        t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    rots = Rotation.from_euler("xyz", euler_deg, degrees=True)
    slerp = Slerp(t_ctrl, rots)
    return slerp(t_out).as_euler("xyz", degrees=True)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rotate_yaw(v: np.ndarray, world_up: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate *v* around *world_up* by *angle_deg* (Rodrigues formula)."""
    axis = _normalize(world_up)
    theta = np.radians(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


# ───────────────────────────── camera + actor track ──────────────────────────

# ── Stage A: arc-length reparameterisation ────────────────────────────────────

def _arclength_spline(
    positions: np.ndarray,
    n_dense: int = 2048,
) -> tuple:
    """Build a chord-parameterised CubicSpline + arc-length inverse map.

    Returns ``(cs, t_ctrl_chord, s_ctrl, L_total, t_of_s)``:

    * ``cs``           – CubicSpline (t ∈ [0, 1] → 3-D position), or *None*
                         when scipy is unavailable.
    * ``t_ctrl_chord`` – spline-t values at each control point (chord fractions).
    * ``s_ctrl``       – cumulative arc-length at each control point.
    * ``L_total``      – total arc-length of the spline.
    * ``t_of_s``       – callable mapping arc-length s → spline parameter t.

    Chord-length parameterisation is used so that the spline parameter is
    proportional to physical distance rather than to control-point index.
    This eliminates the "fast on sparse segments, slow on dense segments"
    artefact that arises when waypoints are unevenly spaced.
    """
    n_ctrl = len(positions)
    if n_ctrl == 1:
        return (
            None,
            np.array([0.0]),
            np.array([0.0]),
            0.0,
            lambda s: np.zeros_like(np.asarray(s, dtype=float)),
        )

    chords    = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumchord  = np.concatenate([[0.0], np.cumsum(chords)])
    total_chord = float(cumchord[-1])
    t_ctrl_chord = (
        cumchord / total_chord
        if total_chord > 1e-12
        else np.linspace(0.0, 1.0, n_ctrl)
    )

    try:
        from scipy.interpolate import CubicSpline, interp1d
    except ImportError:
        # No scipy: fall back to chord-linear map.
        t_of_s_fb = (lambda s, _tc=t_ctrl_chord, _sc=cumchord:  # noqa: E731
                     np.interp(s, _sc, _tc))
        return None, t_ctrl_chord, cumchord.copy(), total_chord, t_of_s_fb

    cs = CubicSpline(t_ctrl_chord, positions, bc_type="not-a-knot", axis=0)

    # Densely sample the spline to compute arc-length numerically.
    t_dense   = np.linspace(0.0, 1.0, n_dense)
    pts_dense = cs(t_dense)                              # (n_dense, 3)
    seg_len   = np.linalg.norm(np.diff(pts_dense, axis=0), axis=1)
    s_dense   = np.concatenate([[0.0], np.cumsum(seg_len)])
    L_total   = float(s_dense[-1])

    # Arc-length at each control point (by interpolating s_dense at t_ctrl).
    s_ctrl = np.interp(t_ctrl_chord, t_dense, s_dense)

    # Inverse map: arc-length → spline parameter (for evaluating at any s).
    s_uniq, idx_uniq = np.unique(s_dense, return_index=True)
    t_uniq = t_dense[idx_uniq]
    t_of_s_fn = interp1d(
        s_uniq, t_uniq, kind="linear",
        bounds_error=False, fill_value=(t_uniq[0], t_uniq[-1]),
    )
    return cs, t_ctrl_chord, s_ctrl, L_total, t_of_s_fn


# ── Stage B: curvature-adaptive velocity profile ──────────────────────────────

def _velocity_profile_plan(
    cs,
    t_of_s,
    L_total: float,
    v_cruise: float,
    a_max: float,
    a_lat_max: float,
    n_dense: int = 2048,
) -> np.ndarray:
    """Return a smooth velocity profile v[i] (arc-length / frame) at
    ``n_dense`` evenly-spaced arc-length positions.

    Three constraints are enforced simultaneously:

    1. **Cruise limit** – v ≤ v_cruise everywhere.
    2. **Lateral acceleration** – v ≤ sqrt(a_lat_max / κ(s)) in curves so the
       camera never swings sideways faster than comfortable in VR.
    3. **Longitudinal acceleration** – |Δv / frame| ≤ a_max via a
       forward + backward trapezoidal pass, giving smooth ease-in from rest
       and ease-out back to rest (endpoints forced to v = 0).

    When *cs* is None (scipy unavailable) a simple trapezoidal profile is
    returned.
    """
    if L_total < 1e-9:
        return np.zeros(n_dense)

    s_arr = np.linspace(0.0, L_total, n_dense)
    ds    = float(L_total) / (n_dense - 1)

    if cs is None:
        v_max = np.full(n_dense, v_cruise)
    else:
        # Curvature κ(s) via central finite differences on dense arc-length samples.
        t_eval  = np.asarray(t_of_s(s_arr), dtype=float)
        pts     = cs(t_eval)                                   # (n_dense, 3)
        dp      = np.diff(pts, axis=0) / ds                    # (n-1, 3)
        d2p     = (pts[2:] - 2.0 * pts[1:-1] + pts[:-2]) / ds ** 2   # (n-2, 3)
        dp_norm = 0.5 * (np.linalg.norm(dp[:-1], axis=1)
                         + np.linalg.norm(dp[1:], axis=1))    # (n-2,)
        dp_norm = np.maximum(dp_norm, 1e-9)
        kappa_i = np.linalg.norm(d2p, axis=1) / dp_norm ** 2  # (n-2,)
        kappa   = np.concatenate([[kappa_i[0]], kappa_i, [kappa_i[-1]]])

        v_lat = np.sqrt(a_lat_max / np.maximum(kappa, 1e-12))
        v_max = np.minimum(v_cruise, v_lat)

    # Force rest at endpoints → drives the ease-in / ease-out ramps.
    v_max[0]  = 0.0
    v_max[-1] = 0.0

    # Forward pass: v[i+1] ≤ sqrt(v[i]² + 2·a_max·ds)
    v = v_max.copy()
    for i in range(1, n_dense):
        v[i] = min(v[i], np.sqrt(max(0.0, v[i - 1] ** 2 + 2.0 * a_max * ds)))

    # Backward pass: v[i] ≤ sqrt(v[i+1]² + 2·a_max·ds)
    for i in range(n_dense - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_max * ds)))

    return v


# ── Stage C: zero-phase position smoothing ────────────────────────────────────

def _smooth_track_positions(
    track: "List[dict]",
    fps: int,
    window_s: float = 0.8,
) -> None:
    """Apply zero-phase Gaussian smoothing to camera positions and focal points
    inside *track*, in place.

    Eliminates high-frequency jitter (spline oscillations, quantisation of
    control-point placement) from the ortho-panel red dot and info-billboard
    without introducing temporal offset.  Falls back silently when scipy is
    unavailable.

    Parameters
    ----------
    track:
        Per-frame state dicts as returned by :func:`build_track`.
    fps:
        Frame rate – used to convert *window_s* to a sample count.
    window_s:
        Smoothing half-width in seconds.  Recommended: 0.6 s for flat,
        1.0 s for VR.
    """
    if not track:
        return
    win = max(3, int(round(window_s * fps)))
    try:
        from scipy.signal import filtfilt, gaussian as _gauss
        kernel_len = win * 4 + 1
        kernel  = _gauss(kernel_len, std=float(win))
        kernel /= kernel.sum()
        pos = np.array([s["camera"]["position"]    for s in track], dtype=float)
        foc = np.array([s["camera"]["focal_point"] for s in track], dtype=float)
        # filtfilt requires signal length > padlen = 3*(kernel_len-1).
        # When the track is short (e.g. a stresstest slice), we pad with edge
        # values before filtering and strip the padding afterwards — this avoids
        # the edge-ringing that would otherwise produce 2-4 Hz camera jitter.
        min_len   = 3 * (kernel_len - 1) + 1
        n         = len(track)
        if n < min_len:
            pad_l = (min_len - n + 1) // 2
            pad_r = min_len - n - pad_l
            pos = np.pad(pos, ((pad_l, pad_r), (0, 0)), mode="edge")
            foc = np.pad(foc, ((pad_l, pad_r), (0, 0)), mode="edge")
            for ax in range(3):
                pos[:, ax] = filtfilt(kernel, [1.0], pos[:, ax])
                foc[:, ax] = filtfilt(kernel, [1.0], foc[:, ax])
            pos = pos[pad_l:pad_l + n]
            foc = foc[pad_l:pad_l + n]
        else:
            for ax in range(3):
                pos[:, ax] = filtfilt(kernel, [1.0], pos[:, ax])
                foc[:, ax] = filtfilt(kernel, [1.0], foc[:, ax])
        for i, state in enumerate(track):
            state["camera"]["position"]    = pos[i].tolist()
            state["camera"]["focal_point"] = foc[i].tolist()
    except Exception as exc:   # noqa: BLE001
        logger.debug("Stage-C position smoothing skipped: %s", exc)


def _build_time_axis(
    n_ctrl: int,
    frames_per_segment: int,
    *,
    hold_frames: int,
    ease_segments: int,
    ease_strength: float,
) -> np.ndarray:
    """Return the array ``t_out`` (in [0, 1]) at which to sample the splines.

    The output time line is built so that:

    * Each segment ``[i, i+1]`` between control points ``i`` and ``i+1``
      gets ``frames_per_segment`` frames (constant speed, except where
      eased — see below).
    * Over the **last ``ease_segments`` segments**, the camera decelerates
      using an ease-out power curve, so it arrives gently at the midpoint
      between the two last control points (which is the final frame
      before the hold).
    * The very last segment ``[n-2, n-1]`` is replaced by half a segment:
      the camera stops at the midpoint of the two last control points,
      i.e. at spline parameter ``(t_{n-2} + t_{n-1}) / 2``.
    * ``hold_frames`` frozen frames are appended at the end (same value).
    """
    if n_ctrl == 1:
        return np.zeros(max(frames_per_segment, 1))

    # Control points sit on uniform spline parameters in [0, 1].
    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    last_midpoint = 0.5 * (t_ctrl[-2] + t_ctrl[-1])

    segments: List[np.ndarray] = []

    # Number of segments between consecutive control points.
    n_seg = n_ctrl - 1

    # Build all segments except the last one normally; last segment stops
    # at the midpoint and may be eased.
    for s in range(n_seg):
        t0 = t_ctrl[s]
        t1 = t_ctrl[s + 1] if s < n_seg - 1 else last_midpoint
        # How many frames does this segment get?  The "real" full
        # segments use frames_per_segment; the truncated last segment
        # (half-length) gets half as many frames, rounded up, so its
        # average speed is the same as the others.
        if s < n_seg - 1:
            n = frames_per_segment
        else:
            n = max(1, int(round(frames_per_segment * (t1 - t0) /
                                 (t_ctrl[s + 1] - t0))))

        # Decide whether this segment is eased.  We ease the last
        # ``ease_segments`` segments (including the truncated one).
        is_eased = s >= n_seg - ease_segments

        # Normalised local time in [0, 1).  We exclude the right endpoint
        # because the next segment provides it; the final segment will add
        # the endpoint explicitly to make sure the hold starts at the
        # exact midpoint.
        u = np.linspace(0.0, 1.0, n + 1)[:-1]
        if is_eased:
            # Ease-out: heavier deceleration on the last segment, lighter
            # on the preceding ones.  We blend a power curve in by the
            # segment's distance to the very end (0 = first eased, 1 =
            # last eased).
            depth = (s - (n_seg - ease_segments)) / max(1, ease_segments - 1)
            depth = np.clip(depth, 0.0, 1.0)
            local_strength = 1.0 + depth * (ease_strength - 1.0)
            # Standard ease-out: 1 - (1 - u) ** k, k >= 1.
            u_eased = 1.0 - np.power(1.0 - u, local_strength)
            u = u_eased
        segments.append(t0 + (t1 - t0) * u)

    # Add the exact endpoint (= midpoint), so the camera stops there for
    # the hold.
    segments.append(np.array([last_midpoint]))

    # Hold: repeat the last value so the camera freezes for hold_frames
    # additional frames (minus 1 because the endpoint above already
    # contributes one frame).
    if hold_frames > 1:
        segments.append(np.full(hold_frames - 1, last_midpoint))

    return np.concatenate(segments)


def build_track(
    control_points: Sequence[dict],
    frames_per_segment: int,
    *,
    hold_frames: int = 0,
    ease_segments: int = 2,
    ease_strength: float = 3.0,
) -> "tuple[List[dict], np.ndarray]":
    """Return ``(track, t_out_ui)`` – one camera+actor state per frame plus a
    ``[0, 1]``-normalised fractional-control-point array for
    :func:`_build_ui_track`.

    Three-stage pipeline
    --------------------
    **A – Arc-length reparameterisation**: spline knots are placed at
    chord-length fractions so the parameter is proportional to physical
    distance, not to the number of saved control points.  This decouples
    spatial waypoint density from travel speed.

    **B – Curvature-adaptive velocity profile**: a forward + backward
    trapezoidal planner derives ``v(s)`` that (a) starts and ends at rest,
    (b) never exceeds the cruise speed derived from *frames_per_segment*,
    and (c) reduces speed in tight curves to bound lateral acceleration
    (VR comfort).

    **C** – Zero-phase positional smoothing is applied externally by
    :func:`_smooth_track_positions` (called by ``main()`` after build).

    Parameters
    ----------
    ease_segments:
        Controls the acceleration ramp:
        ``a_max = v_cruise / (ease_segments × frames_per_segment)``.
        Larger → gentler ease-in / ease-out.
    ease_strength:
        Retained for CLI compatibility; unused (ramp shape is now
        determined by the physics-based forward/backward pass).
    """
    n_ctrl = len(control_points)
    if n_ctrl < 1:
        raise ValueError("Need at least one control point.")

    # ── Extract per-control-point fields ──────────────────────────────────
    positions      = _stack_camera_field(control_points, "position")
    focals         = _stack_camera_field(control_points, "focal_point")
    view_ups       = _stack_camera_field(control_points, "view_up")
    view_angles    = np.array([p["camera"]["view_angle"]    for p in control_points])
    parallel_scale = np.array([p["camera"]["parallel_scale"] for p in control_points])
    act_pos   = _stack_actor_field(control_points, "position",    [0.0, 0.0, 0.0])
    act_scale = _stack_actor_field(control_points, "scale",       [1.0, 1.0, 1.0])
    act_orig  = _stack_actor_field(control_points, "origin",      [0.0, 0.0, 0.0])
    act_orie  = _stack_actor_field(control_points, "orientation", [0.0, 0.0, 0.0])

    # ── Degenerate: single control point ──────────────────────────────────
    if n_ctrl == 1:
        n_frames = max(frames_per_segment, 1) + max(hold_frames, 0)
        t_out_ui = np.zeros(n_frames)
        pos = positions[0]; foc = focals[0]; up = view_ups[0]
        look = _normalize(foc - pos)
        up   = up - look * float(np.dot(up, look))
        up   = _normalize(up)
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, 1.0, 0.0])
        frame = {
            "camera": {
                "position":       pos.tolist(),
                "focal_point":    foc.tolist(),
                "view_up":        up.tolist(),
                "view_angle":     float(view_angles[0]),
                "parallel_scale": float(parallel_scale[0]),
            },
            "actor": {
                "position":    act_pos[0].tolist(),
                "orientation": act_orie[0].tolist(),
                "scale":       act_scale[0].tolist(),
                "origin":      act_orig[0].tolist(),
            },
            "keyframe_index": 0.0,
        }
        return [frame] * n_frames, t_out_ui

    # ─────────────────────────── Stage A ──────────────────────────────────
    cs_pos, t_ctrl_chord, s_ctrl, L_total, t_of_s = _arclength_spline(positions)
    # Build additional splines for focal points and view-up on the *same*
    # chord knots so all camera components move consistently.
    try:
        from scipy.interpolate import CubicSpline as _CS
        cs_foc = _CS(t_ctrl_chord, focals,   bc_type="not-a-knot", axis=0)
        cs_up  = _CS(t_ctrl_chord, view_ups, bc_type="not-a-knot", axis=0)
    except ImportError:
        cs_foc = None
        cs_up  = None

    # ─────────────────────────── Stage B ──────────────────────────────────
    n_segs      = max(1, n_ctrl - 1)
    v_cruise    = L_total / (n_segs * max(1, frames_per_segment))
    ease_frames = max(1, ease_segments * max(1, frames_per_segment))
    a_max       = v_cruise / ease_frames       # arc-length / frame²
    a_lat_max   = 2.0 * a_max                  # lateral comfort margin (VR)
    _N_DENSE    = 2048
    v_dense = _velocity_profile_plan(
        cs_pos, t_of_s, L_total,
        v_cruise=v_cruise,
        a_max=a_max,
        a_lat_max=a_lat_max,
        n_dense=_N_DENSE,
    )

    # Integrate v(s) → s(t): convert velocity profile to cumulative time,
    # then resample at uniform frame times.
    s_arr    = np.linspace(0.0, L_total, _N_DENSE)
    ds_dense = float(L_total) / (_N_DENSE - 1)
    v_mid    = 0.5 * (v_dense[:-1] + v_dense[1:])         # midpoint velocities
    dt_steps = ds_dense / np.maximum(v_mid, 1e-15)         # frames per arc step
    t_cumul  = np.concatenate([[0.0], np.cumsum(dt_steps)])  # cumulative time
    T_total  = float(t_cumul[-1])                           # total motion (frames)

    n_motion     = max(2, int(np.ceil(T_total)) + 1)
    frame_times  = np.linspace(0.0, T_total, n_motion)
    s_out_motion = np.interp(frame_times, t_cumul, s_arr)   # arc-length per frame

    if hold_frames > 0:
        s_out = np.concatenate([s_out_motion, np.full(hold_frames, L_total)])
    else:
        s_out = s_out_motion

    total_frames = len(s_out)

    # ── Sample all splines at per-frame arc-length positions ──────────────
    t_eval = np.asarray(t_of_s(s_out), dtype=float)

    def _lin_fallback(vals: np.ndarray) -> np.ndarray:
        return np.array(
            [np.interp(t_eval, t_ctrl_chord, vals[:, ax]) for ax in range(3)],
            dtype=float,
        ).T

    pos_track = cs_pos(t_eval) if cs_pos is not None else _lin_fallback(positions)
    foc_track = cs_foc(t_eval) if cs_foc is not None else _lin_fallback(focals)
    up_track  = cs_up(t_eval)  if cs_up  is not None else _lin_fallback(view_ups)

    # Scalar / actor fields: spline parameterised by arc-length fraction so
    # segment transitions coincide with spatial, not click-index, boundaries.
    t_ctrl_norm = s_ctrl / max(L_total, 1e-12)          # control-point knots in [0, 1]
    kf_out      = np.interp(s_out, s_ctrl, np.arange(float(n_ctrl)))
    t_out_ui    = kf_out / max(1, n_ctrl - 1)           # → [0, 1] for _build_ui_track

    ang_track       = _interp_scalar(view_angles,    t_out_ui, t_ctrl=t_ctrl_norm)
    psc_track       = _interp_scalar(parallel_scale, t_out_ui, t_ctrl=t_ctrl_norm)
    act_pos_track   = _interp_vec(act_pos,   t_out_ui, t_ctrl=t_ctrl_norm)
    act_scale_track = _interp_vec(act_scale, t_out_ui, t_ctrl=t_ctrl_norm)
    act_orig_track  = _interp_vec(act_orig,  t_out_ui, t_ctrl=t_ctrl_norm)
    act_orie_track  = _interp_orientation(act_orie, t_out_ui, t_ctrl=t_ctrl_norm)

    # ── Assemble per-frame dicts ───────────────────────────────────────────
    track: List[dict] = []
    for i in range(total_frames):
        pos  = pos_track[i]
        foc  = foc_track[i]
        up   = up_track[i]
        # Re-orthonormalise view_up so the spline doesn't roll the camera.
        look = _normalize(foc - pos)
        up   = up - look * float(np.dot(up, look))
        up   = _normalize(up)
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, 1.0, 0.0])
        track.append({
            "camera": {
                "position":       pos.tolist(),
                "focal_point":    foc.tolist(),
                "view_up":        up.tolist(),
                "view_angle":     float(ang_track[i]),
                "parallel_scale": float(psc_track[i]),
            },
            "actor": {
                "position":    act_pos_track[i].tolist(),
                "orientation": act_orie_track[i].tolist(),
                "scale":       act_scale_track[i].tolist(),
                "origin":      act_orig_track[i].tolist(),
            },
            # Float keyframe index in [0, n_ctrl-1]: arc-length-aware
            # (e.g. 2.5 means halfway between saved positions 2 and 3).
            "keyframe_index": float(kf_out[i]),
        })
    return track, t_out_ui


def _compute_travel_dirs(
    track,
    world_up: np.ndarray,
    *,
    velocity_window: int = 8,
    alpha: float = 0.12,
    eps_frac: float = 0.005,
) -> np.ndarray:
    """Compute a smoothed per-frame travel-direction array from a keyframe track.

    Uses a central-difference velocity over a wide window, EMA smoothing, and
    freezes the direction when the camera is (nearly) stationary.  Near-vertical
    movement is ignored so the billboard stays stable even at the start/end of
    the path.

    Parameters
    ----------
    track:
        List of per-frame state dicts (each must contain ``state["camera"]["position"]``).
    world_up:
        Constant world-up unit vector.
    velocity_window:
        Half-window for central-difference velocity (frames).
    alpha:
        EMA weight for each new sample (0 = never update, 1 = no smoothing).
    eps_frac:
        Fraction of the median speed below which a frame is considered stationary.

    Returns
    -------
    np.ndarray
        ``(N, 3)`` array of unit travel-direction vectors.
    """
    N = len(track)
    positions = np.array([s["camera"]["position"] for s in track], dtype=float)

    # Central-difference velocity over the window.
    raw_vel = np.zeros_like(positions)
    for i in range(N):
        i0 = max(0, i - velocity_window)
        i1 = min(N - 1, i + velocity_window)
        raw_vel[i] = positions[i1] - positions[i0]

    speeds = np.linalg.norm(raw_vel, axis=1)
    nonzero = speeds[speeds > 0]
    median_speed = float(np.median(nonzero)) if nonzero.size else 1.0
    eps_abs = eps_frac * median_speed

    up = np.asarray(world_up, dtype=float)
    up_n = float(np.linalg.norm(up))
    up = up / up_n if up_n > 1e-9 else np.array([0.0, 1.0, 0.0])

    dirs = np.zeros((N, 3), dtype=float)
    current_dir: np.ndarray | None = None

    for i in range(N):
        if speeds[i] > eps_abs:
            candidate = raw_vel[i] / speeds[i]
            # Skip near-vertical movement (cross product with world_up near zero).
            if float(np.linalg.norm(np.cross(candidate, up))) > 0.05:
                if current_dir is None:
                    current_dir = candidate
                else:
                    blended = (1.0 - alpha) * current_dir + alpha * candidate
                    b_n = float(np.linalg.norm(blended))
                    current_dir = blended / b_n if b_n > 1e-9 else current_dir

        if current_dir is None:
            # Fallback: use focal → look direction from first frame.
            foc = np.asarray(track[i]["camera"]["focal_point"], dtype=float)
            pos = np.asarray(track[i]["camera"]["position"], dtype=float)
            d = foc - pos
            d_n = float(np.linalg.norm(d))
            current_dir = d / d_n if d_n > 1e-9 else np.array([1.0, 0.0, 0.0])

        dirs[i] = current_dir

    return dirs


def _apply_camera_state(cam, state: dict) -> None:
    cam.SetPosition(*state["position"])
    cam.SetFocalPoint(*state["focal_point"])
    cam.SetViewUp(*state["view_up"])
    cam.SetViewAngle(state["view_angle"])
    cam.SetParallelScale(state["parallel_scale"])


def _install_headlight(plt) -> list:
    """Install a single camera-attached headlight for flat-movie rendering.

    Matches the VTK interactor default exactly: one light at the camera
    position aimed at the focal point (the "torch" effect).  VTK
    transforms the camera-space coordinates to world space automatically
    before each render, so no per-frame update is needed.
    """
    import vtkmodules.vtkRenderingCore as _vtkrc
    created: list = []
    for renderer in plt.renderers:
        try:
            renderer.AutomaticLightCreationOff()
        except Exception:  # noqa: BLE001
            pass
        try:
            renderer.RemoveAllLights()
        except Exception:  # noqa: BLE001
            pass
        lt = _vtkrc.vtkLight()
        lt.SetLightTypeToCameraLight()
        # Camera-space coordinates: position directly behind the viewpoint,
        # aimed at the focal plane centre – the standard VTK headlight.
        lt.SetPosition(0.0, 0.0, 1.0)
        lt.SetFocalPoint(0.0, 0.0, 0.0)
        lt.SetColor(1.0, 1.0, 1.0)
        lt.SetIntensity(1.0)
        renderer.AddLight(lt)
        created.append(lt)
    return created


def _install_vr_headlight(plt) -> tuple:
    """Install world-space scene lights for VR rendering.

    Cannot use camera lights here: the panoramic pass renders 6 cube
    faces, each with a different camera orientation.  A camera-attached
    light would move between faces → different shading per face → visible
    seams in the equirectangular remap.

    Returns a 3-tuple ``(torch_lights, green_lights, red_lights)`` where
    each element is a list of vtkLight objects (one per renderer).

    Light roles
    -----------
    * torch  – white, intensity 1.1, follows the smoothed travel direction.
    * green  – faint green fill, intensity 0.12, fires toward back-right
               (travel direction yawed +135 ° around world-up).
    * red    – faint red fill, intensity 0.12, fires toward back-left
               (travel direction yawed -135 °).
    """
    import vtkmodules.vtkRenderingCore as _vtkrc
    torch_lights: list = []
    green_lights: list = []
    red_lights:   list = []
    for renderer in plt.renderers:
        try:
            renderer.AutomaticLightCreationOff()
        except Exception:  # noqa: BLE001
            pass
        try:
            renderer.RemoveAllLights()
        except Exception:  # noqa: BLE001
            pass
        # Main torch — white, slightly boosted.
        lt_torch = _vtkrc.vtkLight()
        lt_torch.SetLightTypeToSceneLight()
        lt_torch.SetPositional(False)
        lt_torch.SetColor(1.0, 1.0, 1.0)
        lt_torch.SetIntensity(1.1)
        renderer.AddLight(lt_torch)
        torch_lights.append(lt_torch)
        # Green fill — backs-right, very dim.
        lt_green = _vtkrc.vtkLight()
        lt_green.SetLightTypeToSceneLight()
        lt_green.SetPositional(False)
        lt_green.SetColor(0.78, 1.0, 0.78)
        lt_green.SetIntensity(0.12)
        renderer.AddLight(lt_green)
        green_lights.append(lt_green)
        # Red fill — back-left, very dim.
        lt_red = _vtkrc.vtkLight()
        lt_red.SetLightTypeToSceneLight()
        lt_red.SetPositional(False)
        lt_red.SetColor(1.0, 0.78, 0.78)
        lt_red.SetIntensity(0.12)
        renderer.AddLight(lt_red)
        red_lights.append(lt_red)
    return torch_lights, green_lights, red_lights


def _update_vr_headlight(
    lights: tuple,
    camera_state: dict,
    travel_dir: "np.ndarray | None" = None,
    world_up: "np.ndarray | None" = None,
    tilt_deg: float = 0.0,
) -> None:
    """Aim world-space VR lights for the current frame.

    *lights* is the 3-tuple returned by :func:`_install_vr_headlight`.
    When *travel_dir* is provided, the torch follows the smoothed travel
    direction; the green/red fills are yawed ±135 ° around *world_up*.
    Falls back to the camera focal direction when *travel_dir* is None.

    *tilt_deg* tilts the torch downward (toward -world_up) by that many
    degrees.  This creates a near-bright / far-dark floor gradient that
    enhances depth perception in enclosed cavern spaces.
    """
    import math as _math
    if not lights:
        return
    torch_lights, green_lights, red_lights = lights
    pos = np.asarray(camera_state["position"], dtype=float)
    fp  = np.asarray(camera_state["focal_point"], dtype=float)

    if travel_dir is not None and world_up is not None:
        forward = _normalize(np.asarray(travel_dir, dtype=float))
    else:
        d = fp - pos
        forward = _normalize(d) if np.linalg.norm(d) > 1e-9 else np.array([0.0, 0.0, -1.0])

    up = np.asarray(world_up, dtype=float) if world_up is not None else np.array([0.0, 1.0, 0.0])

    # Tilt the torch downward toward -world_up.
    if tilt_deg:
        alpha = _math.radians(tilt_deg)
        up_n = _normalize(up)
        torch_dir = _normalize(forward * _math.cos(alpha) + (-up_n) * _math.sin(alpha))
    else:
        torch_dir = forward

    green_dir = _normalize(_rotate_yaw(forward, up,  135.0))
    red_dir   = _normalize(_rotate_yaw(forward, up, -135.0))

    # Light position = camera; focal point = camera + direction (directional light).
    for lt in torch_lights:
        lt.SetPosition(*pos.tolist())
        lt.SetFocalPoint(*(pos + torch_dir).tolist())
    for lt in green_lights:
        lt.SetPosition(*pos.tolist())
        lt.SetFocalPoint(*(pos + green_dir).tolist())
    for lt in red_lights:
        lt.SetPosition(*pos.tolist())
        lt.SetFocalPoint(*(pos + red_dir).tolist())


def _lock_orientation_for_vr(track: List[dict]) -> None:
    """Mutate ``track`` so every camera keeps the **first frame's
    orientation** (look-direction + view-up).  Only the *translation*
    component of the spline is preserved.

    This is essential for VR comfort: any spline-driven rotation while
    the viewer is wearing a headset is interpreted as the world tilting
    around the user, which is a fast track to motion sickness.  We let
    the user rotate their head instead.
    """
    if not track:
        return
    first = track[0]["camera"]
    p0 = np.asarray(first["position"],    dtype=float)
    f0 = np.asarray(first["focal_point"], dtype=float)
    u0 = np.asarray(first["view_up"],     dtype=float)
    look = _normalize(f0 - p0)
    dist = float(np.linalg.norm(f0 - p0))
    if dist < 1e-9 or not np.isfinite(look).all():
        return  # degenerate; leave as-is
    # Re-orthonormalise view_up against the locked look direction so
    # SetViewUp() does not get rejected by VTK's internal sanitization.
    up = u0 - look * float(np.dot(u0, look))
    if np.linalg.norm(up) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
    up = _normalize(up)
    for entry in track:
        pos = np.asarray(entry["camera"]["position"], dtype=float)
        entry["camera"]["focal_point"] = (pos + look * dist).tolist()
        entry["camera"]["view_up"]     = up.tolist()


# ─────────────────── UI-state replay (viewer ↔ movie) ────────────────────
#
# The interactive viewer (``water_conductance``) saves a ``ui_state`` block
# alongside each ``Save pos`` click.  The movie consumes that block to
# reproduce the visual configuration *over time* — visibility swaps,
# pillars styling/hue, fog/SSAO toggles, etc.
#
# Two flavours of fields:
#   • **Continuous** (``pillars_hue_shift``, ``arrow_length``, …): linear
#     smoothstep interpolation over the user-controlled transition window
#     ``--ui-transition-seconds`` (default 0.5 s) starting at each saved
#     keyframe; then held until the next keyframe.
#   • **Discrete** (``view_mode``, ``pillars_style``, ``fog_on``,
#     ``ssao_on``, ``cmap_mode``): step at the *midpoint* of the
#     transition window, so the swap visually coincides with the smooth
#     ramp's halfway point.
#
# A transition of 0 s collapses both flavours to an instant snap at the
# keyframe boundary.

_UI_CONT_FIELDS = (
    "pillars_hue_shift",
    "arrow_length",
    "arrow_score_min",
    "arrow_score_max",
)
_UI_DISCRETE_FIELDS = (
    "view_mode",
    "pillars_style",
    "pillars_visible",
    "cmap_mode",
    "fog_on",
    "ssao_on",
    "membranes_visible",
    "lames_visible",
)


def _smoothstep(x: float) -> float:
    """Classic Hermite smoothstep clamped to [0, 1]."""
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * (3.0 - 2.0 * x)


def _build_ui_track(
    control_points: List[dict],
    t_out: np.ndarray,
    *,
    fps: int,
    frames_per_segment: int,
    transition_seconds: float,
) -> List[dict | None]:
    """Per-frame ``ui_state`` list aligned with ``t_out``.

    Returns a list of length ``len(t_out)``; entries are ``None`` when
    no ``ui_state`` was found on any control point (the caller should
    then short-circuit application entirely).
    """
    n_ctrl = len(control_points)
    ui_blocks: list = [
        (cp.get("ui_state") or {}) if isinstance(cp, dict) else {}
        for cp in control_points
    ]
    if not any(ui_blocks):
        return [None] * len(t_out)

    # Convert the per-keyframe-time transition window (in seconds) into
    # the same units as ``keyframe_index`` (which is normalized
    # [0, n_ctrl-1] over the segment graph).
    if transition_seconds <= 0.0 or frames_per_segment <= 0:
        r = 0.0
    else:
        # Per segment of ``frames_per_segment`` frames = 1 keyframe-index
        # unit; transition_seconds*fps frames within that segment.
        r = min(1.0, (transition_seconds * max(fps, 1)) / frames_per_segment)

    out: List[dict | None] = []
    max_kf = max(1, n_ctrl - 1)
    for t in t_out:
        kf = float(t) * max_kf
        seg = int(np.floor(kf))
        seg = max(0, min(n_ctrl - 2, seg)) if n_ctrl >= 2 else 0
        s = kf - seg  # within-segment fraction in [0, 1]
        a = ui_blocks[seg] if seg < n_ctrl else {}
        b = ui_blocks[min(seg + 1, n_ctrl - 1)]
        ramp = _smoothstep(s / r) if r > 0.0 else (1.0 if s > 0.0 else 0.0)

        frame_state: dict = {}
        # Continuous: linear interp with smoothstep, fall back gracefully
        # when the field is absent at one end.
        for k in _UI_CONT_FIELDS:
            va = a.get(k); vb = b.get(k)
            if va is None and vb is None:
                continue
            if va is None: va = vb
            if vb is None: vb = va
            try:
                frame_state[k] = float(va) * (1.0 - ramp) + float(vb) * ramp
            except (TypeError, ValueError):
                pass
        # Discrete: step at midpoint of the ramp.  When r == 0 (instant
        # mode), this is just "snap at segment boundary".
        switch = (ramp >= 0.5) if r > 0.0 else (s > 0.0)
        for k in _UI_DISCRETE_FIELDS:
            val = b.get(k) if switch else a.get(k)
            # If the active side doesn't set this field, carry forward the
            # value from the other side so discrete modes are "sticky" across
            # keyframes that don't explicitly override them.
            if val is None and switch:
                val = a.get(k)
            if val is not None:
                frame_state[k] = val
        # Carry forward previous values when a key is absent on both ends.
        # This mirrors the interactive viewer where toggles remain latched
        # until explicitly changed by the user.
        if out and out[-1] is not None:
            prev = out[-1]
            for k in _UI_DISCRETE_FIELDS:
                if k not in frame_state and k in prev:
                    frame_state[k] = prev[k]
            for k in _UI_CONT_FIELDS:
                if k not in frame_state and k in prev:
                    frame_state[k] = prev[k]
        out.append(frame_state)
    return out


class _UiReplayBundle:
    """Holds references to all actors / state required to apply a
    per-frame ``ui_state`` block during rendering."""

    def __init__(
        self,
        *,
        mesh,
        alt_mesh=None,
        pillars_mesh=None,
        fog_actors: list | None = None,
        plt=None,
        vr_mode: str = "off",
        membranes_actor=None,
        membranes_threshold=None,
        membranes_n_steps: int = 0,
        fog_near_frac: float = 0.45,
        fog_far_frac: float = 0.92,
        force_fog: bool = False,
        **kwargs,
    ) -> None:
        self.mesh = mesh
        self.alt_mesh = alt_mesh
        self.pillars_mesh = pillars_mesh
        self.fog_actors = fog_actors or [mesh]
        self.plt = plt
        # Water-membranes animation: actor is held in the scene (added
        # by ``main`` after construction) and toggled visible/invisible
        # in ``apply``; the threshold's lower/upper are advanced by one
        # step per frame while visible.
        self._vr_mode = vr_mode
        self.membranes_actor = membranes_actor
        self.membranes_threshold = membranes_threshold
        self.membranes_n_steps = int(membranes_n_steps)
        self._membranes_step = 0
        # Water lames (V2) animation.
        self.lames_actor = kwargs.get("lames_actor")
        self.lames_step_pds = kwargs.get("lames_step_pds") or []
        self.lames_n_steps = int(kwargs.get("lames_n_steps") or 0)
        self.lames_fps = float(kwargs.get("lames_fps") or DEFAULT_LAME2_SOURCE_FPS)
        self.movie_fps = float(kwargs.get("movie_fps") or FPS)
        self._lames_phase = 0.0
        # Gradient arrows (arrows_grid view-mode).
        self.arrows_actor = kwargs.get("arrows_actor")
        # Dual arrows for arrows_dual view-mode (water + air harmonic arrows).
        self.water_dual_actor = kwargs.get("water_dual_actor")
        self.air_dual_actor   = kwargs.get("air_dual_actor")
        # Crown Dijkstra tracks — list of bin-actors (lines + arrows).
        self.tracks_bin_actors = list(kwargs.get("tracks_bin_actors") or [])
        # Axis info for per-frame bin visibility update.
        self.tracks_axis_info = kwargs.get("tracks_axis_info")
        self.debug_tracks = bool(kwargs.get("debug_tracks", False))
        # Latched values so we don't re-apply identical state every frame.
        self._last: dict = {}
        # Fog / SSAO state mirrors viewer's ``_depth_state`` dict shape.
        self._depth = {
            "fog_on": False, "ssao_on": False,
            "ssao_keepalive": [], "prev_pass": {},
        }
        self._fog_near_frac: float = float(fog_near_frac)
        self._fog_far_frac:  float = float(fog_far_frac)
        self._force_fog:     bool  = bool(force_fog)
        # Pillars colour bases — kept in sync with viewer.
        self._pillars_base_glow  = (0.20, 0.95, 0.75)
        self._pillars_base_solid = (0.05, 0.50, 0.45)

    # --- pillars -----------------------------------------------------
    @staticmethod
    def _hue_shift(rgb_0_1, shift):
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(*rgb_0_1)
        h = (h + shift) % 1.0
        return colorsys.hsv_to_rgb(h, s, v)

    def _apply_pillars(self, *, visible: bool, style: str, hue: float):
        pm = self.pillars_mesh
        if pm is None:
            return
        # Visibility
        try:
            raw = getattr(pm, "actor", pm)
            raw.SetVisibility(1 if visible else 0)
        except Exception:  # noqa: BLE001
            pass
        if not visible:
            return
        # Colour + style
        base = (self._pillars_base_solid if style == "solid"
                else self._pillars_base_glow)
        col = self._hue_shift(base, float(hue or 0.0))
        try:
            pm.c(list(col))
        except Exception:  # noqa: BLE001
            pass
        try:
            if style == "solid":
                raw = getattr(pm, "actor", pm)
                raw.GetProperty().LightingOn()
                pm.alpha(1.0)
                try:
                    pm.lighting(ambient=0.30, diffuse=0.65,
                                specular=1.0, specular_power=60,
                                specular_color=(1.0, 1.0, 1.0))
                except TypeError:
                    pm.lighting(ambient=0.30, diffuse=0.65,
                                specular=1.0, specular_power=60)
            else:
                pm.alpha(0.12)
                pm.lighting("off")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pillars style apply failed: %s", exc)

    # --- fog / SSAO (mirrors viewer) ---------------------------------
    def _set_fog(self, on: bool):
        if on == self._depth["fog_on"]:
            return
        if on:
            # Approximate scene diagonal.
            try:
                bnd = self.mesh.bounds()
                diag = float(np.linalg.norm(
                    [bnd[1] - bnd[0], bnd[3] - bnd[2], bnd[5] - bnd[4]]
                ))
            except Exception:  # noqa: BLE001
                diag = 1000.0
            near, far = diag * self._fog_near_frac, diag * self._fog_far_frac
            bg = BACKGROUND
            if isinstance(bg, (tuple, list)) and len(bg) >= 3:
                fr, fg, fb = float(bg[0]), float(bg[1]), float(bg[2])
                if max(fr, fg, fb) > 1.5:
                    fr /= 255; fg /= 255; fb /= 255
            else:
                fr = fg = fb = 0.0
            snip = (
                "//VTK::Light::Impl\n"
                f"  float _fog_d = length(vertexVC.xyz);\n"
                f"  float _fog_f = clamp((_fog_d - {near:.3f}) / "
                f"({far:.3f} - {near:.3f}), 0.0, 1.0);\n"
                f"  gl_FragData[0].rgb = mix(gl_FragData[0].rgb, "
                f"vec3({fr:.3f}, {fg:.3f}, {fb:.3f}), _fog_f);\n"
            )
            for a in self.fog_actors:
                try:
                    sp = getattr(a, "actor", a).GetShaderProperty()
                    sp.AddFragmentShaderReplacement(
                        "//VTK::Light::Impl", True, snip, False)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Movie fog attach failed: %s", exc)
        else:
            for a in self.fog_actors:
                try:
                    sp = getattr(a, "actor", a).GetShaderProperty()
                    sp.ClearFragmentShaderReplacement(
                        "//VTK::Light::Impl", True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Movie fog detach failed: %s", exc)
        self._depth["fog_on"] = on

    def _set_ssao(self, on: bool):
        # SSAO is incompatible with the panoramic cubemap pass: it would
        # overwrite renderer.SetPass() and destroy VR rendering entirely.
        # Simply ignore the request when VR mode is active.
        if self._vr_mode != "off":
            return
        if on == self._depth["ssao_on"] or self.plt is None:
            return
        if on:
            try:
                from vtkmodules.vtkRenderingOpenGL2 import (
                    vtkCameraPass, vtkLightsPass, vtkOpaquePass,
                    vtkOverlayPass, vtkTranslucentPass,
                    vtkRenderPassCollection, vtkSequencePass, vtkSSAOPass,
                )
            except ImportError as exc:  # pragma: no cover
                logger.warning("Movie SSAO unavailable: %s", exc)
                return
            try:
                bnd = self.mesh.bounds()
                diag = float(np.linalg.norm(
                    [bnd[1] - bnd[0], bnd[3] - bnd[2], bnd[5] - bnd[4]]
                ))
            except Exception:  # noqa: BLE001
                diag = 1000.0
            keep = []
            for ren in self.plt.renderers:
                # Only touch layer-0 (3D scene) renderers.  Layer-1 is the
                # ortho-panel overlay renderer; applying SSAO there would
                # break the slice view.
                try:
                    if ren.GetLayer() != 0:
                        continue
                except Exception:  # noqa: BLE001
                    pass
                self._depth["prev_pass"][ren] = ren.GetPass()
                coll = vtkRenderPassCollection()
                coll.AddItem(vtkLightsPass())
                coll.AddItem(vtkOpaquePass())
                coll.AddItem(vtkTranslucentPass())
                # Keep vtkOverlayPass so the debug-index text (vtkActor2D)
                # and any 2D annotations remain visible.  There are no
                # interactive buttons in the movie renderer.
                coll.AddItem(vtkOverlayPass())
                seq = vtkSequencePass(); seq.SetPasses(coll)
                cam_pass = vtkCameraPass(); cam_pass.SetDelegatePass(seq)
                ssao = vtkSSAOPass()
                ssao.SetRadius(diag * 0.005)
                ssao.SetBias(diag * 0.0005)
                ssao.SetKernelSize(32); ssao.SetBlur(True)
                ssao.SetDelegatePass(cam_pass)
                ren.SetPass(ssao)
                keep.extend([ssao, cam_pass, seq, coll])
            self._depth["ssao_keepalive"] = keep
        else:
            for ren in self.plt.renderers:
                prev = self._depth["prev_pass"].get(ren)
                try:
                    ren.SetPass(prev)
                except Exception:  # noqa: BLE001
                    pass
            self._depth["prev_pass"].clear()
            self._depth["ssao_keepalive"] = []
        self._depth["ssao_on"] = on

    # --- view-mode actor swap ---------------------------------------
    def _set_view_mode(self, mode: str):
        if self.alt_mesh is None and mode == "mesh_all":
            mode = "mesh_bridges"
        try:
            show_main        = mode == "mesh_bridges"
            show_alt         = mode == "mesh_all"
            show_arrows_grid = mode == "arrows_grid"
            show_arrows_dual = mode == "arrows_dual"
            show_tracks      = mode == "arrows_tracks"
            getattr(self.mesh, "actor", self.mesh).SetVisibility(
                1 if show_main else 0)
            if self.alt_mesh is not None:
                getattr(self.alt_mesh, "actor", self.alt_mesh).SetVisibility(
                    1 if show_alt else 0)
            if self.arrows_actor is not None:
                getattr(self.arrows_actor, "actor",
                        self.arrows_actor).SetVisibility(
                    1 if show_arrows_grid else 0)
            for _dual_act in (self.water_dual_actor, self.air_dual_actor):
                if _dual_act is not None:
                    getattr(_dual_act, "actor", _dual_act).SetVisibility(
                        1 if show_arrows_dual else 0)
            if show_tracks and self.tracks_axis_info is None:
                # Fallback mode: no bin metadata available, show all tracks.
                for _ta in self.tracks_bin_actors:
                    try:
                        getattr(_ta, "actor", _ta).SetVisibility(1)
                        _alpha = 1.0 if self._vr_mode != "off" else float(
                            getattr(_ta, "_base_alpha", 0.40)
                        )
                        _ta.alpha(_alpha)
                    except Exception:  # noqa: BLE001
                        pass
            elif not show_tracks:
                for _ta in self.tracks_bin_actors:
                    try:
                        getattr(_ta, "actor", _ta).SetVisibility(0)
                    except Exception:  # noqa: BLE001
                        pass
            # When show_tracks=True, update_tracks_bins() manages per-bin
            # visibility every frame.
            if self.debug_tracks:
                print(
                    "      [debug-tracks] view_mode="
                    f"{mode} show_tracks={show_tracks} "
                    f"tracks_actors={len(self.tracks_bin_actors)} "
                    f"binning={'on' if self.tracks_axis_info is not None else 'off'}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("view_mode swap failed: %s", exc)

    # --- per-frame bin culling for arrow_tracks ----------------------------
    def update_tracks_bins(self, cam_pos: "np.ndarray") -> None:
        """Show/hide track bin actors based on camera position (mirrors viewer).

        Bins within ±BIN_FULL of the camera bin are shown at full opacity;
        bins between BIN_FULL and BIN_FADE fade linearly; beyond BIN_FADE
        are hidden.  No-op when view_mode != arrows_tracks.
        """
        if (not self.tracks_bin_actors
                or self._last.get("view_mode") != "arrows_tracks"):
            return
        if self.tracks_axis_info is None:
            # Fallback mode: no bin culling, keep all tracks visible.
            _vis_count = 0
            for _a in self.tracks_bin_actors:
                try:
                    getattr(_a, "actor", _a).SetVisibility(1)
                    _alpha = 1.0 if self._vr_mode != "off" else float(
                        getattr(_a, "_base_alpha", 0.40)
                    )
                    _a.alpha(_alpha)
                    _vis_count += 1
                except Exception:  # noqa: BLE001
                    pass
            if self.debug_tracks:
                print(
                    "      [debug-tracks] fallback visible actors="
                    f"{_vis_count}/{len(self.tracks_bin_actors)}"
                )
            return
        try:
            info      = self.tracks_axis_info
            axis_col  = int(info["axis_col"])
            axis_min  = float(info["axis_min"])
            axis_len  = float(info["axis_len"])
            n_bins    = int(info["n_bins"])
            bin_full  = int(info["bin_full"])
            bin_fade  = int(info["bin_fade"])
            fade_span = max(bin_fade - bin_full, 1)
            cam_a     = float(cam_pos[axis_col])
            cam_bin   = int(np.clip(
                int((cam_a - axis_min) / axis_len * n_bins),
                0, n_bins - 1,
            ))
            for _a in self.tracks_bin_actors:
                _d    = abs(_a._bin_idx - cam_bin)
                _mult = (1.0 if _d <= bin_full
                         else (1.0 - (_d - bin_full) / fade_span
                               if _d <= bin_fade else 0.0))
                # In VR, translucency is broken (panoramic pass blend-state).
                # Hide fading bins entirely rather than showing them opaque:
                # only bins within ±bin_full of the camera are visible.
                if self._vr_mode != "off":
                    _vis = 1 if _d <= bin_full else 0
                else:
                    _vis = 1 if _mult > 0.0 else 0
                try:
                    getattr(_a, "actor", _a).SetVisibility(_vis)
                except Exception:  # noqa: BLE001
                    pass
                if _vis:
                    try:
                        _alpha = (1.0 if self._vr_mode != "off"
                                  else _a._base_alpha * _mult)
                        _a.alpha(_alpha)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("tracks bin update failed: %s", exc)

    # --- entry point -------------------------------------------------
    def apply(self, ui_state: dict | None) -> None:
        if not ui_state:
            return
        # Save old view_mode BEFORE updating so the pillars block can detect the
        # transition (needed for the VR view-mode guard below).
        _old_vm = self._last.get("view_mode")
        vm = ui_state.get(
            "view_mode",
            _old_vm if _old_vm is not None else "mesh_bridges",
        )
        if vm != _old_vm:
            self._set_view_mode(vm)
        self._last["view_mode"] = vm
        # Pillars: re-apply on pillars state change, or in VR when view_mode
        # changes (because visibility depends on the current view_mode in VR).
        _vr_vm_changed = (
            self._vr_mode != "off"
            and vm != _old_vm
        )
        if (_vr_vm_changed
                or ui_state.get("pillars_visible")  != self._last.get("pillars_visible")
                or ui_state.get("pillars_style") != self._last.get("pillars_style")
                or ui_state.get("pillars_hue_shift") !=
                   self._last.get("pillars_hue_shift")):
            _vis = bool(ui_state.get("pillars_visible", False))
            _sty = str(ui_state.get("pillars_style") or "glow")
            if self._vr_mode != "off":
                # In VR translucency is broken; hide pillars in cortex modes
                # and force solid (opaque) when they are shown in arrow modes.
                _VR_ARROWS = {"arrows_grid", "arrows_tracks", "arrows_dual"}
                _cur_vm = vm or "mesh_bridges"
                _vis = _vis and (_cur_vm in _VR_ARROWS)
                if _sty == "glow":
                    _sty = "solid"
            self._apply_pillars(
                visible=_vis,
                style=_sty,
                hue=float(ui_state.get("pillars_hue_shift") or 0.0),
            )
            self._last["pillars_visible"]   = ui_state.get("pillars_visible")
            self._last["pillars_style"]     = ui_state.get("pillars_style")
            self._last["pillars_hue_shift"] = ui_state.get("pillars_hue_shift")
        _fog_want = True if self._force_fog else bool(ui_state.get("fog_on"))
        if _fog_want != self._last.get("fog_on"):
            self._set_fog(_fog_want)
            self._last["fog_on"] = _fog_want
        if ui_state.get("ssao_on") != self._last.get("ssao_on"):
            self._set_ssao(bool(ui_state.get("ssao_on")))
            self._last["ssao_on"] = ui_state.get("ssao_on")
        # ── Water "membranes" descending animation ────────────────────
        if self.membranes_actor is not None and self.membranes_n_steps > 0:
            want_vis = bool(ui_state.get("membranes_visible", False))
            if want_vis != bool(self._last.get("membranes_visible", False)):
                try:
                    self.membranes_actor.SetVisibility(1 if want_vis else 0)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("membranes visibility swap failed: %s", exc)
                self._last["membranes_visible"] = want_vis
            if want_vis:
                # Advance one step per applied frame; pre-sorted cells
                # → contiguous threshold slice each tick.
                self._membranes_step = (
                    self._membranes_step + 1
                ) % self.membranes_n_steps
                try:
                    s = float(self._membranes_step)
                    self.membranes_threshold.SetLowerThreshold(s)
                    self.membranes_threshold.SetUpperThreshold(s)
                    self.membranes_threshold.Modified()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("membranes threshold update failed: %s", exc)
        # ── Water lames (V2) descending animation ─────────────────────
        if self.lames_actor is not None and self.lames_n_steps > 0:
            want_vis = bool(ui_state.get("lames_visible", False))
            if want_vis != bool(self._last.get("lames_visible", False)):
                try:
                    self.lames_actor.SetVisibility(1 if want_vis else 0)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("lames visibility swap failed: %s", exc)
                self._last["lames_visible"] = want_vis
            if want_vis:
                _advance = self.lames_fps / max(self.movie_fps, 1e-6)
                self._lames_phase = (
                    self._lames_phase + _advance
                ) % self.lames_n_steps
                _lames_step = int(self._lames_phase)
                step_pd = (self.lames_step_pds[_lames_step]
                           if _lames_step < len(self.lames_step_pds)
                           else None)
                if step_pd is not None:
                    try:
                        self.lames_actor.GetMapper().SetInputData(step_pd)
                        self.lames_actor.GetMapper().Modified()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("lames step update failed: %s", exc)


def _apply_actor_state(actor, state: dict) -> None:
    actor.SetOrigin(*state["origin"])
    actor.SetPosition(*state["position"])
    actor.SetOrientation(*state["orientation"])
    actor.SetScale(*state["scale"])


def _world_up_from_track(track: Sequence[dict]) -> np.ndarray:
    """Pick a sensible world-up axis from the saved view_ups."""
    avg = np.mean([t["camera"]["view_up"] for t in track], axis=0)
    return _normalize(np.asarray(avg, dtype=float))


def _build_path_line(track: Sequence[dict], offset: float,
                     *, as_tube: bool = False, radius_frac: float = PATH_TUBE_RADIUS_FRAC):
    """Return a green trajectory above the camera path.

    When ``as_tube`` is true a real 3-D :class:`vedo.Tube` is returned
    instead of a :class:`vedo.Line`.  This is required for VR rendering
    because :class:`vtkPanoramicProjectionPass` remaps six cubemap faces
    into an equirectangular image, and a screen-space Line collapses to
    sub-pixel width during that remap (so it disappears).  A tube is
    real geometry and is sampled correctly by any projection.
    """
    import vedo
    cam_positions = np.array([t["camera"]["position"] for t in track])
    foc_positions = np.array([t["camera"]["focal_point"] for t in track])
    distances = np.linalg.norm(foc_positions - cam_positions, axis=1)
    median_dist = float(np.median(distances))
    up = _world_up_from_track(track)
    line_pts = cam_positions + up * offset
    color = [c / 255.0 for c in PATH_COLOR]
    if as_tube:
        radius = median_dist * radius_frac
        obj = vedo.Tube(line_pts, r=radius, c=color, res=16)
        # vedo.Tube stores parametric / vertex-index scalar data that VTK
        # maps through a rainbow LUT, overriding the solid colour.  Strip
        # all point- and cell-data arrays from the polydata, then turn off
        # scalar visibility on the mapper and re-assert the colour.
        try:
            pd = obj.dataset
        except AttributeError:
            pd = obj.polydata()   # vedo < 2.x
        for _attr in (pd.GetPointData(), pd.GetCellData()):
            for _i in range(_attr.GetNumberOfArrays() - 1, -1, -1):
                _attr.RemoveArray(_i)
        try:
            _mp = obj.mapper
            if callable(_mp):     # vedo 1.x
                _mp = _mp()
            if _mp is not None:
                _mp.ScalarVisibilityOff()
        except Exception:         # noqa: BLE001
            pass
        obj.color(color)
    else:
        obj = vedo.Line(line_pts, c=color, lw=PATH_LINE_WIDTH)
    try:
        obj.lighting("off")
    except Exception:  # noqa: BLE001
        pass
    obj.name = "cable_car_path"
    return obj


# ─────────────────────────── anti-aliasing helpers ───────────────────────────


def _setup_antialiasing(plt, *, msaa: int, fxaa: bool,
                        fxaa_contrast: float,
                        fxaa_hard_contrast: float,
                        vr_mode: str) -> None:
    """Enable hardware MSAA (mono only) and FXAA on every renderer.

    * **MSAA** is set on the OpenGL render window.  Must be called before
      the first render so VTK allocates a multi-sampled framebuffer.  In
      VR mode the panoramic projection pass uses its own off-screen FBOs
      for the 6 cube faces, which silently ignore the window-level MSAA
      flag — we still set it so any future code path benefits, but the
      real quality lever in VR is ``--vr-cube-resolution``.
    * **FXAA (mono only)**: in VR mode, ``renderer.SetUseFXAA`` is
      silently ignored because ``_setup_panoramic_pass`` later calls
      ``renderer.SetPass(pano)`` which replaces the whole pipeline.  For
      VR, FXAA is wired directly inside ``_setup_panoramic_pass`` as a
      ``vtkOpenGLFXAAPass`` wrapping the panoramic pass.
    """
    try:
        if msaa and msaa > 0:
            plt.window.SetMultiSamples(int(msaa))
    except Exception as exc:  # noqa: BLE001
        logger.debug("SetMultiSamples(%s) failed: %s", msaa, exc)

    if not fxaa or vr_mode != "off":
        # In VR, FXAA is handled inside _setup_panoramic_pass.
        if msaa and msaa > 0:
            suffix = "  (MSAA only; FXAA wired in pano pass)" if vr_mode != "off" else ""
            print(f"      AA               : MSAA×{msaa}{suffix}")
        return
    for renderer in plt.renderers:
        try:
            renderer.SetUseFXAA(True)
            opts = renderer.GetFXAAOptions()
            if opts is not None:
                if hasattr(opts, "SetRelativeContrastThreshold"):
                    opts.SetRelativeContrastThreshold(float(fxaa_contrast))
                if hasattr(opts, "SetHardContrastThreshold"):
                    opts.SetHardContrastThreshold(float(fxaa_hard_contrast))
                if hasattr(opts, "SetSubpixelBlendLimit"):
                    opts.SetSubpixelBlendLimit(0.75)
                if hasattr(opts, "SetSubpixelContrastThreshold"):
                    opts.SetSubpixelContrastThreshold(0.25)
                if hasattr(opts, "SetUseHighQualityEndpoints"):
                    opts.SetUseHighQualityEndpoints(True)
                if hasattr(opts, "SetEndpointSearchIterations"):
                    opts.SetEndpointSearchIterations(12)
        except Exception as exc:  # noqa: BLE001
            logger.debug("FXAA setup failed on %s: %s", renderer, exc)
    print(f"      AA               : MSAA×{msaa}"
          + f"  + FXAA(rel={fxaa_contrast}, hard={fxaa_hard_contrast})")


def _force_phong(*meshes) -> None:
    """Force VTK Phong (per-fragment) interpolation on every mesh.

    VTK's default is Gouraud (vertex-interpolated lighting); Phong
    yields noticeably smoother specular highlights on curved surfaces
    at a small cost.  Called once on every actor added to the movie
    plotter so the rendered look matches the interactive viewer's
    default shading.
    """
    for m in meshes:
        if m is None:
            continue
        try:
            _set_shading(m, 2)
        except Exception as exc:  # noqa: BLE001
            logger.debug("force_phong failed on %s: %s", m, exc)


# ─────────────────────────── stereo VR helpers ───────────────────────────────


def _setup_panoramic_pass(plt, angle_deg: float, cube_resolution: int,
                          *, fxaa: bool = False,
                          fxaa_contrast: float = 0.0833,
                          fxaa_hard_contrast: float = 0.0312):
    """Attach a :class:`vtkPanoramicProjectionPass` to every renderer of
    ``plt`` so each render produces an equirectangular image.

    The panoramic pass MUST be chained with a proper delegate pass
    sequence (Camera → Sequence(Lights, Opaque)).  Without that delegate,
    the pass appears to silently render nothing fresh — every output
    frame ends up identical to the very first one, which manifests as a
    tiny MP4 with absurdly low bitrate (every frame compresses to a
    near-empty P-frame).  See VTK's standard panoramic pass example.

    When *fxaa* is True, the panoramic pass is wrapped inside a
    :class:`vtkOpenGLFXAAPass` so FXAA is applied to the fully-remapped
    equirectangular output (the only reliable AA path in VR: window-level
    MSAA and ``renderer.SetUseFXAA`` are both silently ignored once a
    custom ``renderer.SetPass(pano)`` is in place).

    Returns a list of pass objects to keep alive (they must outlive
    the renderer, otherwise VTK frees them mid-render).
    """
    try:
        from vtkmodules.vtkRenderingOpenGL2 import (
            vtkCameraPass,
            vtkLightsPass,
            vtkOpaquePass,
            vtkOverlayPass,
            vtkPanoramicProjectionPass,
            vtkRenderPassCollection,
            vtkSequencePass,
            vtkTranslucentPass,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "vtkPanoramicProjectionPass and its delegate-pass dependencies "
            "are unavailable in this VTK build; stereo VR rendering requires "
            "VTK ≥ 9.0 with the OpenGL2 backend."
        ) from exc

    keepalive: list = []
    for renderer in plt.renderers:
        # Inner delegate: lights + opaque + translucent + overlay.
        #
        # NOTE on transparency in VR: we deliberately use plain
        # vtkTranslucentPass (not WBOIT / vtkOrderIndependentTranslucentPass).
        # WBOIT was tried to fix a "blend-state stale" issue across cube
        # faces, but it composites its result onto what it assumes is the
        # main window framebuffer — inside vtkPanoramicProjectionPass every
        # face is rendered to a private off-screen FBO, so WBOIT's composite
        # step lands on the wrong (empty) FBO and OVERWRITES the opaque
        # render, making all opaque actors appear transparent.  That is far
        # worse than the original translucency issue.
        #
        # To compensate for potential blend-state staleness across cube faces
        # we force a GL blend-state reset on the render window's OpenGL state
        # object right before each pass chain runs.  This is done by
        # vtkTranslucentPass itself in VTK ≥ 9.1 (it calls
        # GetState()->ResetGLBlendFunc() at entry), so in practice the stale-
        # cache problem only affects older VTK builds.
        lights = vtkLightsPass()
        opaque = vtkOpaquePass()
        translucent = vtkTranslucentPass()
        overlay = vtkOverlayPass()

        # ── VR blend-state fix ─────────────────────────────────────────
        # vtkPanoramicProjectionPass renders 6 cube faces by calling the
        # delegate chain once per face.  Between faces it issues raw GL
        # calls (FBO bind, equirect composite) that bypass VTK's OpenGL
        # state cache.  The cache then believes blend is still enabled /
        # correctly configured, so vtkTranslucentPass skips its
        # glEnable(GL_BLEND) / glBlendFuncSeparate calls on faces 2-6
        # → translucent actors (pillars, lames, membranes) appear opaque.
        # We insert a minimal Python pass that forces the toggle
        # disable→enable→setfunc through the cache before every
        # translucent pass invocation, ensuring the actual GL state and
        # the cache agree on every face.
        blend_reset = None
        try:
            from vtkmodules.vtkRenderingCore import (
                vtkRenderPass as _vtkRenderPassBase,
            )

            class _VRBlendStateResetPass(_vtkRenderPassBase):
                def Render(self, s):
                    try:
                        renderer = s.GetRenderer()
                        state = renderer.GetRenderWindow().GetState()
                        # Toggling disable→enable forces VTK to issue real
                        # glDisable/glEnable calls even if the cache says
                        # blend is already enabled.
                        state.vtkglDisable(0x0BE2)    # GL_BLEND
                        state.vtkglEnable(0x0BE2)     # GL_BLEND
                        state.vtkglBlendFuncSeparate(
                            0x0302, 0x0303,  # SRC_ALPHA, ONE_MINUS_SRC_ALPHA
                            0x0001, 0x0000,  # ONE, ZERO (alpha channel)
                        )
                    except Exception:          # noqa: BLE001 best-effort
                        pass

                def ReleaseGraphicsResources(self, w):
                    pass

            blend_reset = _VRBlendStateResetPass()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VR blend-state-reset pass unavailable (VTK Python "
                "subclassing not supported in this build?): %s  "
                "Translucent actors (pillars, lames) may appear opaque "
                "on some cube faces.", exc,
            )

        passes = vtkRenderPassCollection()
        passes.AddItem(lights)
        passes.AddItem(opaque)
        if blend_reset is not None:
            passes.AddItem(blend_reset)
            keepalive.append(blend_reset)
        passes.AddItem(translucent)
        passes.AddItem(overlay)
        seq = vtkSequencePass()
        seq.SetPasses(passes)
        cam_pass = vtkCameraPass()
        cam_pass.SetDelegatePass(seq)
        keepalive.extend([translucent, overlay])
        pano = vtkPanoramicProjectionPass()
        pano.SetCubeResolution(int(cube_resolution))
        if hasattr(pano, "SetAngle"):
            pano.SetAngle(float(angle_deg))
        if hasattr(pano, "SetProjectionType"):
            pano.SetProjectionType(0)  # 0 = equirectangular
        if hasattr(pano, "SetInterpolate"):
            pano.SetInterpolate(True)
        pano.SetDelegatePass(cam_pass)
        top_pass = pano  # may be wrapped by FXAA below
        if fxaa:
            try:
                from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLFXAAPass
                fxaa_pass = vtkOpenGLFXAAPass()
                fxaa_pass.SetDelegatePass(pano)
                opts = fxaa_pass.GetFXAAOptions()
                if opts is not None:
                    if hasattr(opts, "SetRelativeContrastThreshold"):
                        opts.SetRelativeContrastThreshold(float(fxaa_contrast))
                    if hasattr(opts, "SetHardContrastThreshold"):
                        opts.SetHardContrastThreshold(float(fxaa_hard_contrast))
                    if hasattr(opts, "SetSubpixelBlendLimit"):
                        opts.SetSubpixelBlendLimit(0.75)
                    if hasattr(opts, "SetSubpixelContrastThreshold"):
                        opts.SetSubpixelContrastThreshold(0.25)
                    if hasattr(opts, "SetUseHighQualityEndpoints"):
                        opts.SetUseHighQualityEndpoints(True)
                    if hasattr(opts, "SetEndpointSearchIterations"):
                        opts.SetEndpointSearchIterations(12)
                top_pass = fxaa_pass
                keepalive.append(fxaa_pass)
            except ImportError:
                logger.warning(
                    "vtkOpenGLFXAAPass not available in this VTK build; "
                    "FXAA disabled for VR panoramic pass."
                )
        renderer.SetPass(top_pass)
        keepalive.extend([pano, cam_pass, seq, passes, opaque, lights])
    return keepalive


def _eye_offset(state: dict, ipd_frac: float,
                ipd_abs: float = 0.0) -> np.ndarray:
    """Return the world-space half-IPD vector pointing to the camera's right.

    Both eyes shift their position AND focal point by ±this vector to
    yield true parallel stereo (no toe-in convergence), which is what
    VR equirectangular content expects.

    When *ipd_abs* > 0 it is used as the full IPD in voxels (independent
    of camera distance).  This is the physically-correct mode: set it to
    ``ipd_metres / metres_per_voxel`` so that a 65 mm human IPD maps to
    the right number of voxels.  When *ipd_abs* == 0 (default), the
    legacy *ipd_frac* × cam→focal distance formula is used instead.
    """
    pos = np.asarray(state["camera"]["position"], dtype=float)
    foc = np.asarray(state["camera"]["focal_point"], dtype=float)
    up  = np.asarray(state["camera"]["view_up"],     dtype=float)
    look = _normalize(foc - pos)
    right = _normalize(np.cross(look, up))
    if ipd_abs > 0.0:
        half_ipd = 0.5 * float(ipd_abs)
    else:
        dist = float(np.linalg.norm(foc - pos))
        half_ipd = 0.5 * float(ipd_frac) * dist
    return right * half_ipd


def _shift_camera_state(state: dict, offset: np.ndarray) -> dict:
    """Return a copy of ``state`` with camera position and focal point
    translated by ``offset`` (parallel stereo)."""
    cam = state["camera"]
    return {
        "camera": {
            "position":       (np.asarray(cam["position"], float) + offset).tolist(),
            "focal_point":    (np.asarray(cam["focal_point"], float) + offset).tolist(),
            "view_up":        list(cam["view_up"]),
            "view_angle":     float(cam["view_angle"]),
            "parallel_scale": float(cam["parallel_scale"]),
        },
        "actor": state["actor"],
    }


def _screenshot_array(plt) -> np.ndarray:
    """Return the current Plotter framebuffer as an (H, W, 3) uint8 array."""
    arr = plt.screenshot(asarray=True)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


# ─────────────────────────────────── main ────────────────────────────────────


def _run_track_debug(track: "List[dict]", fps: int, vr_mode: str) -> None:
    """Print diagnostic info about the camera track (--print-debug mode).

    Outputs a per-frame table (positions + speed + travel direction) for the
    first 2 seconds of the window, followed by acceleration statistics and an
    FFT of the longitudinal velocity.

    Interpretation guide
    --------------------
    * **Accel rms ≈ 0**  →  positions are smooth; jitter source is elsewhere
      (e.g. lighting direction, actor transform, VR reprojection).
    * **Accel rms large** or **FFT peaks at 2-4 Hz**  →  the camera-position
      data itself oscillates; fix is in the smoothing pipeline.
    """
    pos = np.array([s["camera"]["position"] for s in track], dtype=float)
    N = len(pos)

    # Per-frame speed (distance moved from previous frame).
    dpos = np.diff(pos, axis=0)                          # (N-1, 3)
    speed = np.linalg.norm(dpos, axis=1)                 # (N-1,)
    speed_all = np.concatenate([[speed[0] if N > 1 else 0.0], speed])  # (N,)

    # Acceleration = change in speed per frame.
    accel = np.diff(speed) if len(speed) > 1 else np.array([0.0])  # (N-2,)

    # Travel directions (same computation as main render loop).
    world_up_raw = np.asarray(
        track[0]["camera"].get("view_up", [0.0, 1.0, 0.0]), dtype=float)
    w_n = float(np.linalg.norm(world_up_raw))
    world_up = world_up_raw / w_n if w_n > 1e-9 else np.array([0.0, 1.0, 0.0])
    vel_win = max(1, fps // 4)
    travel_dirs = _compute_travel_dirs(
        track, world_up, velocity_window=vel_win, alpha=0.12)

    # ── Table ────────────────────────────────────────────────────────────────
    N_PRINT = min(N, fps * 2)
    hdr = (f"  {'fr':>5}  {'t(s)':>6}  {'pos.x':>9}  {'pos.y':>9}  "
           f"{'pos.z':>9}  {'speed':>8}  {'dir.x':>6}  {'dir.y':>6}  {'dir.z':>6}")
    sep = "  " + "  ".join(["-" * w for w in (5, 6, 9, 9, 9, 8, 6, 6, 6)])
    print(hdr)
    print(sep)
    for i in range(N_PRINT):
        t = i / fps
        p = pos[i]
        d = travel_dirs[i]
        print(f"  {i:>5}  {t:>6.3f}  {p[0]:>9.2f}  {p[1]:>9.2f}  {p[2]:>9.2f}"
              f"  {speed_all[i]:>8.4f}  {d[0]:>6.3f}  {d[1]:>6.3f}  {d[2]:>6.3f}")
    if N > N_PRINT:
        print(f"  … ({N - N_PRINT} more frames not shown)")

    # ── Statistics ───────────────────────────────────────────────────────────
    print(f"\n  ── Statistics over {N} frames ({'--vr' if vr_mode != 'off' else 'flat'}) ──")
    print(f"  Speed  : mean={np.mean(speed):.4f}  std={np.std(speed):.5f}  "
          f"max={np.max(speed):.4f}  vox/frame")
    accel_rms = float(np.sqrt(np.mean(accel ** 2)))
    accel_max = float(np.max(np.abs(accel)))
    print(f"  Accel  : rms={accel_rms:.5f}  max={accel_max:.5f}  vox/frame\u00b2")
    print(f"  \u2192 Accel rms near 0 = smooth positions; high = jitter in track data")

    # ── FFT of longitudinal velocity ─────────────────────────────────────────
    try:
        from numpy.fft import rfft, rfftfreq  # type: ignore[import]
        mean_dir = np.mean(travel_dirs, axis=0)
        mn = float(np.linalg.norm(mean_dir))
        if mn > 1e-9:
            mean_dir /= mn
        pos_proj = pos @ mean_dir          # forward-direction component (N,)
        vel_1d = np.diff(pos_proj)         # longitudinal speed (N-1,)
        if len(vel_1d) >= 8:
            freqs = rfftfreq(len(vel_1d), d=1.0 / fps)
            amps = np.abs(rfft(vel_1d)) / len(vel_1d)
            # Top 6 non-DC peaks.
            nondc = np.argsort(amps[1:])[::-1][:6] + 1
            print(f"\n  ── FFT of longitudinal velocity (top peaks) ──")
            print(f"  (peaks at 2-4 Hz = position jitter; flat spectrum = smooth)")
            for idx in nondc:
                if idx < len(freqs) and freqs[idx] > 0.01:
                    print(f"    {freqs[idx]:5.2f} Hz   amplitude {amps[idx]:.5f} vox/frame")
    except Exception as exc:  # noqa: BLE001
        print(f"  FFT skipped: {exc}")
    print()


def _print_progress(i: int, total: int, t_start: float) -> None:
    pct = 100.0 * (i + 1) / total
    elapsed = time.time() - t_start
    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
    eta = (total - i - 1) / rate if rate > 0 else 0.0
    msg = (
        f"  frame {i + 1:>5d}/{total}  "
        f"({pct:5.1f}%)  "
        f"elapsed {_format_eta(elapsed)}  "
        f"ETA {_format_eta(eta)}  "
        f"{rate:5.2f} fps"
    )
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()


def _summarise_control_points(control_points: List[dict]) -> None:
    """Print a compact recap of the saved control points, so the user
    can spot at a glance whether actor transforms were captured."""
    print(f"  Loaded {len(control_points)} control point(s):")
    for p in control_points:
        cam = p["camera"]
        act = p.get("actor", {})
        cp = tuple(round(v, 1) for v in cam["position"])
        cf = tuple(round(v, 1) for v in cam["focal_point"])
        ap = tuple(round(v, 1) for v in act.get("position", [0, 0, 0]))
        ao = tuple(round(v, 1) for v in act.get("orientation", [0, 0, 0]))
        flag = "" if "actor" in p else "  [legacy: no actor block]"
        print(f"   #{p.get('index', '?'):>2}  "
              f"cam.pos={cp}  cam.foc={cf}  "
              f"act.pos={ap}  act.ori={ao}{flag}")


def _run_parallel(
    args: "argparse.Namespace",
    control_points: list,
    frames_per_segment: int,
    hold_frames: int,
    vr_mode: str,
    frames_dir: "Path",
    out_path: "Path",
) -> int:
    """Orchestrate N parallel render workers, then encode once.

    The orchestrator calls ``build_track()`` (pure numpy, no VTK) to learn
    *total_frames*, splits that range into N even chunks, and spawns N
    independent subprocesses.  Each worker renders its slice directly into
    *frames_dir* with the correct global frame-number offset so that ffmpeg
    sees a contiguous ``frame_%05d.png`` sequence when all workers finish.
    """
    import subprocess

    N = args.parallel

    # ── Build track (numpy-only, no VTK) to learn total_frames ──────────
    track, _ = build_track(
        control_points,
        frames_per_segment,
        hold_frames=hold_frames,
        ease_segments=args.ease_segments,
        ease_strength=args.ease_strength,
    )
    total_frames = len(track)

    # Apply the user's --stresstest slicing so chunk boundaries reflect
    # the actual window that will be rendered by the workers.
    if args.stresstest:
        t_start_s = max(0.0, args.stresstest_start)
        t_end_s   = args.stresstest_end
        f_start   = min(total_frames - 1, int(round(t_start_s * args.fps)))
        f_end     = min(total_frames,     int(round(t_end_s   * args.fps)))
        total_frames = f_end - f_start

    if total_frames < N:
        print(f"  [parallel] WARNING: only {total_frames} frames for {N} workers; "
              f"reducing to {total_frames} worker(s).")
        N = max(1, total_frames)

    print(f"  [parallel] {total_frames} frames  →  {N} workers  "
          f"(~{total_frames // N} frames each)")

    # Prepare the shared frames directory.  Clean up once here so workers
    # don't wipe each other's output.
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    # ── Build worker argv: start from sys.argv, strip orchestrator flags ──
    _FLAGS_WITH_VALUE  = {"--parallel", "--_chunk-start", "--_chunk-end", "--_frame-offset"}
    _FLAGS_NO_VALUE    = {"--_worker"}
    base_argv: list[str] = []
    skip_next = False
    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in _FLAGS_WITH_VALUE:
            skip_next = True
            continue
        if tok in _FLAGS_NO_VALUE:
            continue
        if any(tok.startswith(f + "=") for f in _FLAGS_WITH_VALUE):
            continue
        base_argv.append(tok)

    # ── Spawn one subprocess per chunk ────────────────────────────────────
    procs:     list[subprocess.Popen] = []
    log_paths: list["Path"]             = []
    for k in range(N):
        chunk_start = k * total_frames // N
        chunk_end   = (k + 1) * total_frames // N
        worker_argv = base_argv + [
            "--_worker",
            "--_chunk-start", str(chunk_start),
            "--_chunk-end",   str(chunk_end),
            "--_frame-offset", str(chunk_start),
        ]
        cmd = [sys.executable, "-m", "marvel_view.scripts.water_movie"] + worker_argv
        log_path = frames_dir / f"_worker_{k}.log"
        print(f"  [parallel] spawning worker {k}:  "
              f"frames {chunk_start}\u2013{chunk_end - 1} "
              f"({chunk_end - chunk_start} frames)  log={log_path.name}")
        log_paths.append(log_path)
        procs.append(subprocess.Popen(cmd,
                                      stdout=open(log_path, "w"),  # noqa: SIM115
                                      stderr=subprocess.STDOUT))

    # ── Monitor progress (sliding-window FPS + ETA) ───────────────────────
    from collections import deque
    t_mon   = time.time()
    history: deque[tuple[float, int]] = deque()  # (timestamp, png_count)
    while any(p.poll() is None for p in procs):
        time.sleep(0.5)
        now  = time.time()
        done = len(list(frames_dir.glob("frame_*.png")))
        history.append((now, done))
        cutoff = now - 30.0
        while history and history[0][0] < cutoff:
            history.popleft()
        if len(history) >= 2:
            dt        = history[-1][0] - history[0][0]
            df        = history[-1][1] - history[0][1]
            fps_slide = df / dt if dt > 0 else 0.0
        else:
            elapsed_so_far = now - t_mon
            fps_slide = done / elapsed_so_far if elapsed_so_far > 0 else 0.0
        elapsed = now - t_mon
        eta     = (total_frames - done) / fps_slide if fps_slide > 0 else 0.0
        pct     = 100.0 * done / total_frames if total_frames > 0 else 0.0
        msg = (
            f"  frames {done:>5d}/{total_frames}  "
            f"({pct:5.1f}%)  "
            f"elapsed {_format_eta(elapsed)}  "
            f"ETA {_format_eta(eta)}  "
            f"{fps_slide:5.2f} fps"
        )
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
    sys.stdout.write("\n")

    exit_codes = [p.wait() for p in procs]
    # Close the log file handles that Popen opened.
    for p in procs:
        if p.stdout:
            p.stdout.close()
    failed = [k for k, ec in enumerate(exit_codes) if ec != 0]
    if failed:
        for k in failed:
            lp = frames_dir / f"_worker_{k}.log"
            if lp.exists():
                print(f"\n  ── worker {k} log (last 40 lines) ──────────────")
                lines = lp.read_text(errors="replace").splitlines()
                print("\n".join(lines[-40:]))
        print(f"ERROR: parallel worker(s) {failed} exited with non-zero status. "
              f"PNG frames are preserved in {frames_dir} for inspection.")
        return 1

    # ── Codec / CRF / preset (mirrors main()'s late resolution logic) ─────
    if args.codec == "auto":
        codec = "h265" if vr_mode != "off" else "h264"
    else:
        codec = args.codec
    crf    = args.crf    if args.crf    is not None else (20 if codec == "h265" else 18)
    preset = args.preset if args.preset is not None else ("slow" if codec == "h265" else "medium")

    # ── Encode ────────────────────────────────────────────────────────────
    print(f"[4/4] Encoding MP4 \u2192 {out_path}")
    ok = _encode_mp4(frames_dir, out_path, fps=args.fps,
                     codec=codec, crf=crf, preset=preset)
    if not ok:
        print("      ffmpeg not available \u2013 PNG frames left in place.")
        return 0

    if not args.keep_frames:
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass
        print("      cleaned up intermediate frames.")

    # ── VR spatial-media metadata ─────────────────────────────────────────
    if vr_mode != "off":
        print(f"[5/5] Tagging spatial-media metadata ({vr_mode}, SBS) \u2026")
        tagged_ok = _tag_spatial_media(out_path, vr_mode)
        if tagged_ok:
            print("      MP4 is now flagged as stereoscopic equirectangular "
                  "(left/right).  Ready for YouTube VR & Meta Quest.")
        else:
            print("      Could not inject metadata \u2013 the MP4 is still a "
                  "regular SBS file but won't auto-detect as VR.")

    _analyse_output_mp4(out_path, total_frames=total_frames,
                        fps=args.fps, vr_mode=vr_mode)
    print("Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Auto-select lames cache based on movie output fps ──────────────────
    # Explicit --lames-vtp-cache always wins; otherwise pick the variant that
    # matches the render fps (25 fps → lame2.vtp, 90 fps → lame2_90fps.vtp).
    if args.lames_vtp_cache is None or args.lames_meta_cache is None:
        _movie_fps = getattr(args, "fps", 25) or 25
        _vtk_dir = Path(DEFAULT_LAME2_VTP_CACHE).parent
        if _movie_fps == 25:
            _auto_lv = Path(DEFAULT_LAME2_VTP_CACHE)
            _auto_lm = Path(DEFAULT_LAME2_META_CACHE)
        else:
            _auto_lv = _vtk_dir / f"lame2_{_movie_fps}fps.vtp"
            _auto_lm = _vtk_dir / f"lame2_meta_{_movie_fps}fps.json"
            if not _auto_lv.exists():
                logger.warning(
                    "Lames cache for %d fps not found (%s), "
                    "falling back to 25 fps cache.", _movie_fps, _auto_lv
                )
                _auto_lv = Path(DEFAULT_LAME2_VTP_CACHE)
                _auto_lm = Path(DEFAULT_LAME2_META_CACHE)
        if args.lames_vtp_cache is None:
            args.lames_vtp_cache = str(_auto_lv)
        if args.lames_meta_cache is None:
            args.lames_meta_cache = str(_auto_lm)

    # ── Sanity check: refuse to write into the Trash ──────────────────────
    # Bash keeps the original cwd label even after the directory has been
    # moved to the Trash (e.g. deleted from a file manager), but Path.cwd()
    # follows the inode and lands in ~/.local/share/Trash/.  Writing there
    # silently dumps the MP4 into the Trash — very confusing.  Detect and
    # bail out early.
    cwd = Path.cwd()
    if "/Trash/" in str(cwd):
        logger.error(
            "Refusing to run: the current working directory is inside the "
            "system Trash (%s).  This typically happens when the folder you "
            "started bash in was moved to the Trash from a file manager — "
            "bash kept the old name but the inode now lives in the Trash.  "
            "Please `cd` to the real project folder (e.g. ~/Dev/Python/"
            "sunrice) and try again.", cwd,
        )
        return 2

    frames_per_segment = int(round(args.fps * args.seconds_per_segment))
    if frames_per_segment < 1:
        logger.error("fps × seconds_per_segment must yield at least 1 frame.")
        return 2
    hold_frames = max(0, int(round(args.fps * args.hold_seconds)))

    # ── VR mode: pick canvas dimensions & per-eye geometry ─────────────────
    vr_mode = "vr360" if args.vr else "off"
    vr_angle = 360.0 if vr_mode == "vr360" else None
    vr_user_set_size = any(f in (argv or sys.argv[1:])
                           for f in ("--width", "--height"))
    if vr_mode != "off" and not vr_user_set_size:
        args.width, args.height = VR360_SBS_W, VR360_SBS_H

    # ── Fast preview mode (--fast) ─────────────────────────────────────────
    # Downgrades resolution and codec settings for quick iteration.
    # Explicit --width / --height still take precedence over --fast.
    if args.fast:
        if not vr_user_set_size:
            if vr_mode == "vr360":
                args.width, args.height = 2048, 512
                args.vr_cube_resolution = 1024
            else:  # flat
                args.width, args.height = 960, 540
        if args.crf is None:
            args.crf = 28
        if args.preset is None:
            args.preset = "ultrafast"

    # ── High-quality mode (--high) ──────────────────────────────────────
    # Increases resolution and codec quality beyond the defaults.
    # Explicit --width / --height / --crf / --preset still take precedence.
    if args.high:
        if not vr_user_set_size:
            if vr_mode != "off":
                # SBS canvas stays at default (8192×2048 / 4096×2048);
                # cube-face resolution is the key quality lever for VR.
                args.vr_cube_resolution = 4096
            else:  # flat → 4K
                args.width, args.height = 3840, 2160
        if args.crf is None:
            args.crf = 12 if vr_mode != "off" else 18

    # ── Ultra-high-quality mode (--ultra-high) ──────────────────────────
    # Superset of --high: 8192-px cube faces (4× SSAA vs default 2048),
    # 5K flat, CRF 10.  Implies --high.
    if args.ultra_high:
        args.high = True
        if not vr_user_set_size:
            if vr_mode != "off":
                args.vr_cube_resolution = 8192
            else:  # flat → 5K
                args.width, args.height = 5120, 2880
        if args.crf is None:
            args.crf = 10
        if args.preset is None:
            args.preset = "veryslow"

    # ── Compromise mode (--compromise) ──────────────────────────────────
    # Reduced SBS canvas for 90 fps on Meta Quest: 75 % of normal pixels.
    # Cube-face bumped to 3072 px (between normal 2048 and high 4096) to
    # recover sharpness on distant objects without affecting decode load.
    # For an even sharper result at the cost of render time, override with
    # --vr-cube-resolution 4096.  For a slightly larger canvas (87% of
    # full, still lower decode load), use --compromise-better.
    if args.compromise:
        if not vr_user_set_size:
            if vr_mode != "off":
                args.width, args.height = 6144, 1536   # 3072×1536 per eye
        # Bump cube-face from default 2048 to 3072 unless the user explicitly
        # chose a different value with --vr-cube-resolution.
        if args.vr_cube_resolution == VR_CUBE_RESOLUTION:
            args.vr_cube_resolution = 3072
        if args.crf is None:
            args.crf = 16

    # ── Compromise-better mode (--compromise-better) ─────────────────────
    # Slightly larger canvas than --compromise: 87 % of normal pixels,
    # cube-face 3584 px (halfway between 3072 and 4096), CRF 14.
    if args.compromise_better:
        if not vr_user_set_size:
            if vr_mode != "off":
                args.width, args.height = 7168, 1792   # 3584×1792 per eye
        if args.vr_cube_resolution == VR_CUBE_RESOLUTION:
            args.vr_cube_resolution = 3584
        if args.crf is None:
            args.crf = 14

    if vr_mode != "off":
        if args.width % 2 != 0:
            logger.error("SBS canvas width must be even; got %d.", args.width)
            return 2
        eye_w = args.width // 2
        eye_h = args.height
    else:
        eye_w, eye_h = args.width, args.height

    positions_path = _pick_positions_file(args.positions)
    control_points = _load_control_points(positions_path)
    if args.trick is not None and args.trick > 0:
        control_points = control_points[: args.trick]
    n_ctrl = len(control_points)

    # Output convention:
    #   ./positions/positions_<stamp>.json   ← input control points
    #   ./mp4/positions_<stamp>.mp4          ← rendered movie
    # ``--output`` overrides the path entirely.
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        mp4_dir = DEFAULT_MP4_DIR
        if args.ultra_high:
            quality_prefix = "ULTRA_"
        elif args.high:
            quality_prefix = "HIGH_"
        elif args.compromise_better:
            quality_prefix = f"COMPB{args.width}_"
        elif args.compromise:
            # Include canvas width so different --width overrides don't collide.
            quality_prefix = f"COMP{args.width}_"
        elif args.fast:
            quality_prefix = "FAST_"
        else:
            quality_prefix = "NORMAL_"
        if args.stresstest:
            quality_prefix = "STRESS_" + quality_prefix
        if args.prefix:
            quality_prefix = args.prefix + "_" + quality_prefix
        suffix = f"_{vr_mode}" if vr_mode != "off" else ""
        out_path = mp4_dir / (quality_prefix + positions_path.stem + suffix + ".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path.parent / (out_path.stem + "_frames")

    # In VR, the cable-car path overlay is hidden by default (it can
    # look weird in a headset).  ``--no-path`` and ``--vr-no-path`` both
    # disable it; the latter only kicks in when VR is active.
    show_path = not args.no_path and not (vr_mode != "off" and args.vr_no_path)

    print("═" * 72)
    print("Marvel Water-Conductance · fly-through movie")
    print(f"  Input TIFF       : {Path(args.input).resolve()}")
    print(f"  Positions JSON   : {positions_path}")
    print(f"  Control points   : {n_ctrl}")
    print(f"  Mode             : "
          f"{'preview (interactive)' if args.preview else 'render MP4'}")
    if not args.preview:
        print(f"  Output MP4       : {out_path}")
    print(f"  Resolution       : {args.width}×{args.height}"
          + (f"  (SBS, per eye {eye_w}×{eye_h})" if vr_mode != "off" else ""))
    print(f"  Frame rate       : {args.fps} fps")
    print(f"  Sec / segment    : {args.seconds_per_segment}  "
          f"({frames_per_segment} frames)")
    print(f"  Hold at end      : {args.hold_seconds}s "
          f"({hold_frames} frames)  at midpoint of last 2 ctrl points")
    print(f"  Ease-out         : last {args.ease_segments} segment(s), "
          f"strength {args.ease_strength}")
    if vr_mode != "off":
        print(f"  Stereo VR        : {vr_mode}  "
              f"(equirectangular {vr_angle:.0f}°, parallel SBS)")
        if args.meters_per_voxel > 0.0:
            _ipd_vox_preview = args.ipd_metres / args.meters_per_voxel
            print(f"  IPD (physical)   : {args.ipd_metres * 1000:.1f} mm / "
                  f"{args.meters_per_voxel} m·vox⁻¹ "
                  f"= {_ipd_vox_preview:.4f} vox  [fixed]")
        else:
            print(f"  IPD fraction     : {args.ipd_frac}  "
                  f"(of cam→focal distance)  [legacy]")
        print(f"  Cube resolution  : {args.vr_cube_resolution} px / face")
    # Codec / quality is resolved later (depends on vr_mode); print a
    # one-liner here so the banner is still complete.
    _codec_pick = (args.codec if args.codec != "auto"
                   else ("h265" if vr_mode != "off" else "h264"))
    _crf_pick = (args.crf if args.crf is not None
                 else (20 if _codec_pick == "h265" else 18))
    _preset_pick = (args.preset if args.preset is not None
                    else ("slow" if _codec_pick == "h265" else "medium"))
    print(f"  Encoder          : {_codec_pick}  CRF {_crf_pick}  "
          f"preset {_preset_pick}")
    print(f"  Show path line   : {'yes (green)' if show_path else 'no'}")
    print("═" * 72)
    _summarise_control_points(control_points)
    print("═" * 72)

    # ── Parallel mode: spawn N workers BEFORE loading any VTK asset ────────
    # The orchestrator only runs build_track() (pure numpy) to learn
    # total_frames, then delegates all rendering to worker subprocesses.
    if args.parallel > 1 and not args._worker and not args.preview:
        return _run_parallel(
            args, control_points, frames_per_segment, hold_frames,
            vr_mode, frames_dir, out_path,
        )

    # ── --print-debug: skip heavy loading, go straight to track build ──────
    # All companion-actor and mesh variables are set to empty sentinels so
    # that [2/4] and [2bis] run normally; we exit right after smoothing.
    if args.print_debug:
        print("  [--print-debug] Skipping mesh + companion actor loading.")
        print("  Track will be built, transformed, smoothed, then analysed.")
        mesh = alt_mesh = pillars_mesh = None
        membranes_actor = membranes_threshold = None
        membranes_n_steps = lames_n_steps = 0
        lames_actor = None
        lames_step_pds: list = []
        arrows_actor = None
        tracks_bin_actors: list = []
        tracks_axis_info = None
    else:
        # ── 1. Build the mesh once ─────────────────────────────────────────
        print("[1/4] Building mesh from TIFF …")
        t0 = time.time()
        mesh = _load_or_build_mesh(
            Path(args.input).expanduser().resolve(),
            level=args.level,
            smooth_iter=args.smooth_iter,
            cache_path=Path(args.mesh_cache).expanduser().resolve()
                       if args.mesh_cache else None,
            rebuild=args.rebuild_mesh,
        )
        _style_mesh(mesh)
        print(f"      done in {time.time() - t0:.1f}s  "
              f"({mesh.npoints} vertices, {mesh.ncells} faces)")

    # ── 1bis. Optional companion actors (alt mesh + pillars overlay) ──
    # Loaded lazily from disk caches; used by the per-frame UI-state
    # replay to swap visibility / colour as the saved keyframes dictate.
    alt_mesh = None
    pillars_mesh = None
    if not args.ignore_ui_state:
        import vedo  # local import: keep top-level cost low for `--preview`
        if args.alt_mesh_cache:
            alt_path = Path(args.alt_mesh_cache).expanduser().resolve()
            if alt_path.exists():
                try:
                    alt_mesh = vedo.Mesh(str(alt_path))
                    _style_mesh(alt_mesh)
                    getattr(alt_mesh, "actor", alt_mesh).SetVisibility(0)
                    print(f"      [ui_state] alt mesh loaded "
                          f"({alt_mesh.npoints} vertices)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load --alt-mesh-cache %s: %s",
                                   alt_path, exc)
            else:
                logger.warning("--alt-mesh-cache %s does not exist", alt_path)
        if args.pillars_cache:
            pil_path = Path(args.pillars_cache).expanduser().resolve()
            if pil_path.exists():
                try:
                    pillars_mesh = vedo.Mesh(str(pil_path))
                    # Recompute normals so Phong shading is smooth across
                    # facets (same fix as in viewer's overlay load).
                    try:
                        pillars_mesh.compute_normals()
                    except Exception:  # noqa: BLE001
                        pass
                    # Start hidden — first ui_state apply will reveal it.
                    getattr(pillars_mesh, "actor", pillars_mesh).SetVisibility(0)
                    print(f"      [ui_state] pillars overlay loaded "
                          f"({pillars_mesh.npoints} vertices)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load --pillars-cache %s: %s",
                                   pil_path, exc)
            else:
                logger.warning("--pillars-cache %s does not exist", pil_path)

    # ── Water "membranes" descending animation (load-only) ────────────
    membranes_actor = None
    membranes_threshold = None
    membranes_n_steps = 0
    if (not args.ignore_ui_state
            and args.membranes_vtp_cache and args.membranes_meta_cache):
        mem_vtp  = Path(args.membranes_vtp_cache).expanduser().resolve()
        mem_meta = Path(args.membranes_meta_cache).expanduser().resolve()
        if mem_vtp.exists() and mem_meta.exists():
            try:
                import json as _json_m
                import vtk as _vtk_m
                _meta = _json_m.loads(mem_meta.read_text(encoding="utf-8"))
                membranes_n_steps = int(_meta.get("n_steps", 0))
                _rdr = _vtk_m.vtkXMLPolyDataReader()
                _rdr.SetFileName(str(mem_vtp))
                _rdr.Update()
                _pd = _rdr.GetOutput()
                if _pd is not None and _pd.GetNumberOfCells() > 0 and membranes_n_steps > 0:
                    _thr = _vtk_m.vtkThreshold()
                    _thr.SetInputData(_pd)
                    _thr.SetInputArrayToProcess(
                        0, 0, 0,
                        _vtk_m.vtkDataObject.FIELD_ASSOCIATION_CELLS,
                        "step_id",
                    )
                    try:
                        _thr.SetThresholdFunction(
                            _vtk_m.vtkThreshold.THRESHOLD_BETWEEN)
                    except AttributeError:
                        pass
                    _thr.SetLowerThreshold(0.0)
                    _thr.SetUpperThreshold(0.0)
                    _thr.Update()
                    _geo = _vtk_m.vtkGeometryFilter()
                    _geo.SetInputConnection(_thr.GetOutputPort())
                    _mp = _vtk_m.vtkPolyDataMapper()
                    _mp.SetInputConnection(_geo.GetOutputPort())
                    _mp.SetScalarModeToUseCellFieldData()
                    _mp.SelectColorArray("rgb")
                    _mp.SetColorModeToDirectScalars()
                    _mp.ScalarVisibilityOn()
                    membranes_actor = _vtk_m.vtkActor()
                    membranes_actor.SetMapper(_mp)
                    _prop = membranes_actor.GetProperty()
                    _prop.SetOpacity(0.25)
                    _prop.SetAmbient(0.85)
                    _prop.SetDiffuse(0.10)
                    _prop.SetSpecular(0.0)
                    if vr_mode != "off":
                        # Translucency broken in VR (blend-state stale); render
                        # membranes fully opaque with Phong shading instead.
                        _prop.SetOpacity(1.0)
                        _prop.SetAmbient(0.18)
                        _prop.SetDiffuse(0.78)
                        _prop.SetSpecular(0.55)
                        _prop.SetSpecularPower(35)
                        _prop.SetInterpolation(2)  # Phong
                    _prop.BackfaceCullingOff()
                    _prop.FrontfaceCullingOff()
                    membranes_actor.SetVisibility(0)
                    membranes_threshold = _thr
                    print(f"      [ui_state] water membranes loaded "
                          f"(n_steps={membranes_n_steps}, "
                          f"{_pd.GetNumberOfCells()} cells)")
                else:
                    logger.warning("Membranes cache empty or invalid: %s", mem_vtp)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load membranes caches %s / %s: %s",
                               mem_vtp, mem_meta, exc)
        else:
            logger.warning(
                "--membranes-vtp-cache / --membranes-meta-cache: file(s) missing "
                "(%s, %s)", mem_vtp, mem_meta,
            )

    # ── Water lames (V2) descending animation (load-only) ─────────────
    lames_actor = None
    lames_step_pds: list = []
    lames_n_steps = 0
    lames_fps = int(DEFAULT_LAME2_SOURCE_FPS)
    if not args.ignore_ui_state and args.lames_vtp_cache and args.lames_meta_cache:
        _requested_lv = Path(args.lames_vtp_cache).expanduser().resolve()
        _requested_lm = Path(args.lames_meta_cache).expanduser().resolve()

        for _lv, _lm in [(_requested_lv, _requested_lm)]:
            if not (_lv.exists() and _lm.exists()):
                logger.warning("Lames cache not found: %s / %s", _lv, _lm)
                break
            try:
                _lames_pd, _lames_meta = _load_lames(_lv, _lm)
                if _lames_pd is None or _lames_meta is None:
                    continue
                lames_n_steps = int(_lames_meta.get("n_steps", 0))
                if lames_n_steps <= 0:
                    continue
                lames_fps = int(_lames_meta.get("fps", DEFAULT_LAME2_SOURCE_FPS))
                if args.lames_fps is not None and int(args.lames_fps) > 0:
                    lames_fps = int(args.lames_fps)
                _normals_dir = DEFAULT_LAME2_NORMALS_CACHE_DIR
                _step_pds, _actor = load_lame2_normals_cache(
                    _normals_dir, lames_n_steps)
                if _step_pds is None:
                    _step_pds, _actor = _build_lames_step_polydatas(
                        _lames_pd, lames_n_steps)
                lames_step_pds = _step_pds
                lames_actor = _actor
                # Render lames fully opaque with Phong shading (flat and VR).
                # Translucency is also broken in VR, so this is doubly needed there.
                # Point normals available when cache loaded.
                _lp = _actor.GetProperty()
                _lp.SetOpacity(1.0)
                _lp.SetAmbient(0.18)
                _lp.SetDiffuse(0.78)
                _lp.SetSpecular(0.55)
                _lp.SetSpecularPower(35)
                _lp.SetInterpolation(2)  # Phong
                lames_actor.SetVisibility(0)
                print(f"      [ui_state] water lames loaded "
                      f"(n_steps={lames_n_steps}, fps={lames_fps}, "
                      f"{_lames_pd.GetNumberOfCells()} cells)")
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load lames caches %s / %s: %s",
                               _lv, _lm, exc)

    # ── Gradient arrows (arrows_grid view-mode) ────────────────────────────
    arrows_actor = None
    if not args.ignore_ui_state and args.arrows_cache:
        _arr_path = Path(args.arrows_cache).expanduser().resolve()
        if _arr_path.exists():
            try:
                import vedo as _vedo_arr
                _arr_data = np.load(_arr_path)
                _pts    = np.asarray(_arr_data["pts"],    dtype=np.float32)
                _dirs   = np.asarray(_arr_data["dirs"],   dtype=np.float32)
                _scores = np.asarray(_arr_data["scores"], dtype=np.float32)
                _boost  = (np.asarray(_arr_data["boost"], dtype=np.float32)
                           if "boost" in _arr_data
                           else np.zeros(len(_pts), dtype=np.float32))
                if len(_pts) > 0:
                    # Pick cmap_mode + arrow_length from the first keyframe
                    # where arrows_grid is active, or fall back to defaults.
                    _arr_cmap   = "angle"
                    _arr_length = DEFAULT_ARROW_LENGTH
                    for _cp in control_points:
                        _ui = (_cp.get("ui_state", {})
                               if isinstance(_cp, dict) else {})
                        if _ui.get("view_mode") == "arrows_grid":
                            _arr_cmap   = _ui.get("cmap_mode", "angle") or "angle"
                            _arr_length = float(
                                _ui.get("arrow_length") or DEFAULT_ARROW_LENGTH)
                            break
                    _ends = _pts + _dirs * _arr_length
                    # 5-stop perceptual ramp: blue → green → yellow → orange → red
                    _stops = np.array([
                        [0.45, 0.70, 1.00],
                        [0.20, 0.80, 0.30],
                        [0.95, 0.90, 0.15],
                        [1.00, 0.55, 0.10],
                        [0.90, 0.10, 0.10],
                    ], dtype=np.float32)
                    if _arr_cmap == "convergence" and len(_boost) > 1:
                        _b   = np.nan_to_num(_boost, nan=0.0, posinf=1.0, neginf=0.0)
                        _lo, _hi = np.percentile(_b, [5.0, 95.0])
                        _rng = float(_hi - _lo)
                        _t_c = (np.full(len(_b), 0.5, dtype=np.float32)
                                if _rng < 1e-6
                                else np.clip((_b - _lo) / _rng,
                                             0.0, 1.0).astype(np.float32))
                    else:
                        _sc  = np.nan_to_num(_scores, nan=0.0, posinf=1.0, neginf=0.0)
                        _t_c = np.clip(_sc, 0.0, 1.0).astype(np.float32)
                    _seg_c = _t_c * (len(_stops) - 1)
                    _i0_c  = np.clip(_seg_c.astype(int), 0, len(_stops) - 2)
                    _f_c   = (_seg_c - _i0_c)[:, None]
                    _rgb_c = _stops[_i0_c] * (1.0 - _f_c) + _stops[_i0_c + 1] * _f_c
                    _colors_c = (_rgb_c * 255).astype(np.uint8)
                    arrows_actor = _vedo_arr.Arrows(
                        _pts, _ends,
                        c=_colors_c,
                        thickness=DEFAULT_ARROW_THICKNESS,
                    )
                    try:
                        arrows_actor.lighting("off")
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        arrows_actor.linewidth(1).linecolor("black")
                    except Exception:  # noqa: BLE001
                        pass
                    getattr(arrows_actor, "actor", arrows_actor).SetVisibility(0)
                    print(f"      [ui_state] gradient arrows loaded "
                          f"({len(_pts)} arrows, cmap={_arr_cmap})")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load arrows cache %s: %s",
                               _arr_path, exc)
        else:
            logger.debug("--arrows-cache %s does not exist – "
                         "arrows_grid mode will be empty", _arr_path)

    # ── Dual arrows (arrows_dual view-mode: water + air harmonic arrows) ───
    water_dual_actor = None
    air_dual_actor   = None
    if not args.ignore_ui_state:
        _dw_path = (Path(args.dual_water_cache).expanduser().resolve()
                    if getattr(args, "dual_water_cache", None) else None)
        _da_path = (Path(args.dual_air_cache).expanduser().resolve()
                    if getattr(args, "dual_air_cache", None) else None)
        _dual_data = None
        try:
            _dual_data = _load_or_build_dual_arrows(
                None, None,          # harmonic/wind caches not needed for load-only
                dual_water_cache=_dw_path if (_dw_path and _dw_path.exists()) else None,
                dual_air_cache=_da_path   if (_da_path and _da_path.exists()) else None,
                rebuild=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load dual arrows: %s", exc)
        if _dual_data is not None:
            _dual_length = DEFAULT_ARROW_LENGTH * 1.3
            for _domain, _actor_var in (("water", "water_dual_actor"),
                                        ("air",   "air_dual_actor")):
                _dd = _dual_data.get(_domain)
                if _dd is None:
                    continue
                try:
                    import vedo as _vedo_d
                    _dpts   = np.asarray(_dd["pts"],    dtype=np.float32)
                    _ddirs  = np.asarray(_dd["dirs"],   dtype=np.float32)
                    _dscores = np.asarray(_dd["scores"], dtype=np.float32)
                    if len(_dpts) == 0:
                        continue
                    _dends = _dpts + _ddirs * _dual_length
                    if _domain == "water":
                        _dstops = np.array([
                            [0.10, 0.12, 0.72],
                            [0.25, 0.45, 1.00],
                            [0.70, 0.35, 0.90],
                            [0.90, 0.05, 0.05],
                        ], dtype=np.float32)
                        _dt = np.clip(
                            np.nan_to_num(_dscores, nan=0.0, posinf=1.0, neginf=0.0),
                            0.0, 1.0,
                        ).astype(np.float32)
                    else:  # air
                        _dstops = np.array([
                            [0.05, 0.38, 0.10],
                            [0.28, 0.68, 0.12],
                            [0.72, 0.88, 0.12],
                        ], dtype=np.float32)
                        _dnrm  = np.linalg.norm(_ddirs, axis=-1)
                        _dsafe = _dnrm > 1e-9
                        _du    = np.zeros_like(_ddirs)
                        _du[_dsafe] = _ddirs[_dsafe] / _dnrm[_dsafe, None]
                        _dt = np.clip(1.0 - np.abs(_du[:, 0]), 0.0, 1.0).astype(np.float32)
                    _dseg = _dt * (len(_dstops) - 1)
                    _di0  = np.clip(_dseg.astype(int), 0, len(_dstops) - 2)
                    _df   = (_dseg - _di0)[:, None]
                    _drgb = _dstops[_di0] * (1.0 - _df) + _dstops[_di0 + 1] * _df
                    _dact = _vedo_d.Arrows(
                        _dpts, _dends,
                        c=(_drgb * 255).astype(np.uint8),
                        thickness=DEFAULT_ARROW_THICKNESS,
                    )
                    try:
                        _dact.lighting("off")
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _dact.linewidth(1).linecolor("black")
                    except Exception:  # noqa: BLE001
                        pass
                    getattr(_dact, "actor", _dact).SetVisibility(0)
                    if _domain == "water":
                        water_dual_actor = _dact
                    else:
                        air_dual_actor = _dact
                    print(f"      [ui_state] dual {_domain} arrows loaded "
                          f"({len(_dpts)} arrows)")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not build dual %s arrows actor: %s",
                                   _domain, exc)

    # ── Crown Dijkstra tracks (arrows_tracks view-mode) ────────────────────
    # Instead of one monolithic actor, we build N_BINS sub-actors grouped by
    # position along the volume's longest axis — same system as the viewer.
    # _UiReplayBundle.update_tracks_bins() shows/hides bin actors around the
    # camera each frame, so only nearby paths are rendered at any one time.
    #
    # VR auto-select: when --vr is active and the half-density small VTP
    # exists, swap to it automatically (use --no-tracks-small to override).
    if args.vr and not getattr(args, "no_tracks_small", False):
        _small_default = str(DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE)
        _small_candidate = Path(
            getattr(args, "crown_tracks_splined_cache", _small_default)
            or _small_default
        ).parent / DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE.name
        if _small_candidate.exists():
            logger.info(
                "VR mode: auto-selecting small tracks VTP (%s).  "
                "Pass --no-tracks-small to use the full set.",
                _small_candidate,
            )
            args.crown_tracks_splined_cache = str(_small_candidate)
        elif DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE.exists():
            logger.info(
                "VR mode: auto-selecting small tracks VTP (%s).",
                DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE,
            )
            args.crown_tracks_splined_cache = str(
                DEFAULT_CROWN_TRACKS_SPLINED_SMALL_VTP_CACHE
            )
    tracks_bin_actors: list = []
    tracks_axis_info         = None
    if not args.ignore_ui_state and args.crown_tracks_cache:
        _trk_path     = Path(args.crown_tracks_cache).expanduser().resolve()
        _trk_arr_path = (
            Path(args.crown_tracks_arrows_cache).expanduser().resolve()
            if args.crown_tracks_arrows_cache else None
        )
        _trk_spl_path = (
            Path(args.crown_tracks_splined_cache).expanduser().resolve()
            if getattr(args, "crown_tracks_splined_cache", None) else None
        )
        if _trk_path.exists():
            try:
                _trk_stage = "imports"
                import vtk as _vtk_t
                from vtkmodules.util.numpy_support import (
                    vtk_to_numpy as _v2n,
                    numpy_to_vtk  as _n2v,
                )
                import vedo as _vedo_trk
                _trk_stage = "load-lines-vtp"
                _rdr_t = _vtk_t.vtkXMLPolyDataReader()
                _rdr_t.SetFileName(str(_trk_path))
                _rdr_t.Update()
                _pd_t = _rdr_t.GetOutput()
                if _pd_t.GetNumberOfCells() > 0:
                    # ── Spline (or load pre-splined cache) ───────────────
                    if (_trk_spl_path is not None
                            and _trk_spl_path.exists()):
                        _trk_stage = "load-splined-vtp"
                        _rdr_spl = _vtk_t.vtkXMLPolyDataReader()
                        _rdr_spl.SetFileName(str(_trk_spl_path))
                        _rdr_spl.Update()
                        _pd_sm = _rdr_spl.GetOutput()
                        print(f"      [ui_state] pre-splined tracks loaded: "
                              f"{_pd_sm.GetNumberOfCells()} polylines, "
                              f"{_pd_sm.GetNumberOfPoints()} pts")
                    else:
                        _spl = _vtk_t.vtkSplineFilter()
                        _spl.SetInputData(_pd_t)
                        _spl.SetSubdivideToSpecified()
                        _spl.SetNumberOfSubdivisions(64)
                        _spl.Update()
                        _pd_sm = _spl.GetOutput()
                    # ── Axis + bin parameters (mirrors viewer constants) ──
                    _N_BINS    = TRACK_BIN_N
                    _BIN_FULL  = TRACK_BIN_RADIUS_FULL
                    _BIN_FADE  = TRACK_BIN_RADIUS_FADE
                    _WATER_RGB = (0.72, 0.90, 1.00)
                    # ── Tortuosity LUT ───────────────────────────────────
                    _tort_lut_mv = None
                    _tort_arr_mv = _pd_sm.GetCellData().GetArray("tortuosity")
                    if _tort_arr_mv is not None:
                        import numpy as _np_tort_mv
                        _tort_stops_mv = _np_tort_mv.asarray(
                            TORTUOSITY_CMAP_STOPS, dtype=_np_tort_mv.float64
                        )
                        _tort_lut_mv = _vtk_t.vtkLookupTable()
                        _tort_lut_mv.SetNumberOfTableValues(256)
                        _tort_lut_mv.SetRange(TORTUOSITY_VMIN, TORTUOSITY_VMAX)
                        for _k in range(256):
                            _tk  = _k / 255.0
                            _sg  = _tk * (len(_tort_stops_mv) - 1)
                            _i0  = min(int(_sg), len(_tort_stops_mv) - 2)
                            _fk  = _sg - _i0
                            _rgb = (_tort_stops_mv[_i0] * (1.0 - _fk)
                                    + _tort_stops_mv[_i0 + 1] * _fk)
                            _tort_lut_mv.SetTableValue(
                                _k, float(_rgb[0]), float(_rgb[1]),
                                float(_rgb[2]), 1.0,
                            )
                        _tort_lut_mv.Build()
                        print("      [ui_state] tortuosity LUT built for tracks")
                    _b    = list(_pd_sm.GetBounds())
                    _ex   = [_b[1]-_b[0], _b[3]-_b[2], _b[5]-_b[4]]
                    _ac   = int(np.argmax(np.array(_ex, dtype=np.float64)))
                    _amin = float(_b[2 * _ac])
                    _alen = max(float(_b[2 * _ac + 1]) - _amin, 1.0)
                    # ── Per-cell bin assignment (splined polylines) ───────
                    _pts_sm = _v2n(_pd_sm.GetPoints().GetData())
                    _ca_sm = _pd_sm.GetLines()
                    _offs_arr = _ca_sm.GetOffsetsArray()
                    _conn_arr = _ca_sm.GetConnectivityArray()
                    if _offs_arr is not None and _conn_arr is not None:
                        _offs_sm = _v2n(_offs_arr).astype(np.int64)
                        _conn_sm = _v2n(_conn_arr).astype(np.int64)
                    else:
                        # VTK legacy cell-array layout fallback:
                        # [npts, id0, id1, ..., npts, ...]
                        _legacy = _vtk_t.vtkIdTypeArray()
                        _ca_sm.ExportLegacyFormat(_legacy)
                        _legacy_np = _v2n(_legacy).astype(np.int64)
                        _offs_list = [0]
                        _conn_list: list[int] = []
                        _idx = 0
                        _n_legacy = int(_legacy_np.size)
                        while _idx < _n_legacy:
                            _npt = int(_legacy_np[_idx])
                            _idx += 1
                            if _npt <= 0 or (_idx + _npt) > _n_legacy:
                                raise ValueError(
                                    "Malformed vtkCellArray legacy layout "
                                    f"(idx={_idx}, npts={_npt}, size={_n_legacy})"
                                )
                            _conn_list.extend(_legacy_np[_idx:_idx + _npt])
                            _idx += _npt
                            _offs_list.append(len(_conn_list))
                        _offs_sm = np.asarray(_offs_list, dtype=np.int64)
                        _conn_sm = np.asarray(_conn_list, dtype=np.int64)
                    _lba = []
                    _bins_geo = "lines"  # updated to "tubes" if TubeFilter succeeds
                    try:
                        _coord_pt = _pts_sm[_conn_sm, _ac].astype(np.float64)
                        _seg_sum  = np.add.reduceat(_coord_pt, _offs_sm[:-1])
                        _seg_cnt  = np.maximum(
                            np.diff(_offs_sm).astype(np.float64), 1.0)
                        _cell_bin = np.clip(
                            ((_seg_sum / _seg_cnt - _amin) / _alen
                             * _N_BINS).astype(int),
                            0, _N_BINS - 1,
                        )
                        # ── Line bin actors ───────────────────────────────
                        # Mirror the viewer's pure-numpy sub-polydata
                        # construction — avoids vtkExtractCells +
                        # vtkGeometryFilter which fail on polyline data
                        # in older VTK builds.
                        _cell_pt_starts = _offs_sm[:-1]
                        _cell_pt_counts = np.diff(_offs_sm)
                        _tort_arr_np_mv = None
                        if _tort_lut_mv is not None and _tort_arr_mv is not None:
                            _tort_arr_np_mv = _v2n(_tort_arr_mv).astype(np.float32)
                        for _bi in range(_N_BINS):
                            _ids = np.where(_cell_bin == _bi)[0]
                            if len(_ids) == 0:
                                continue
                            _sel_starts = _cell_pt_starts[_ids]
                            _sel_counts = _cell_pt_counts[_ids]
                            # Build contiguous point array for this bin.
                            _flat_idx = np.concatenate([
                                _conn_sm[int(_s):int(_s) + int(_c)]
                                for _s, _c in zip(_sel_starts, _sel_counts)
                            ])
                            _sub_pts_vtk = _vtk_t.vtkPoints()
                            _sub_pts_vtk.SetDataTypeToDouble()
                            _sub_pts_vtk.SetData(
                                _n2v(_pts_sm[_flat_idx].astype(np.float64),
                                     deep=True,
                                     array_type=_vtk_t.VTK_DOUBLE))
                            # Legacy cell-array: [n0, 0,1,..., n1, c,c+1,...]
                            _parts: list = []
                            _local_off = 0
                            for _nc in _sel_counts.tolist():
                                _parts.append(_nc)
                                _parts.extend(range(_local_off,
                                                    _local_off + _nc))
                                _local_off += _nc
                            _vtk_ids = _n2v(
                                np.array(_parts, dtype=np.int64),
                                deep=True,
                                array_type=_vtk_t.VTK_ID_TYPE)
                            _ca_sub = _vtk_t.vtkCellArray()
                            _ca_sub.SetCells(len(_ids), _vtk_ids)
                            _sub_pd = _vtk_t.vtkPolyData()
                            _sub_pd.SetPoints(_sub_pts_vtk)
                            _sub_pd.SetLines(_ca_sub)
                            if _tort_arr_np_mv is not None:
                                # CellData: one value per path
                                _tsub_cd = _n2v(
                                    _tort_arr_np_mv[_ids], deep=True,
                                    array_type=_vtk_t.VTK_FLOAT)
                                _tsub_cd.SetName("tortuosity")
                                _sub_pd.GetCellData().AddArray(_tsub_cd)
                                _sub_pd.GetCellData().SetActiveScalars(
                                    "tortuosity")
                                # PointData: repeat per-cell value for every
                                # point in that cell (np.repeat avoids
                                # vtkCellDataToPointData which is unreliable
                                # on legacy-format cell arrays).
                                _tsub_pt = _n2v(
                                    np.repeat(
                                        _tort_arr_np_mv[_ids],
                                        _sel_counts.astype(np.int64),
                                    ).astype(np.float32),
                                    deep=True,
                                    array_type=_vtk_t.VTK_FLOAT)
                                _tsub_pt.SetName("tortuosity")
                                _sub_pd.GetPointData().AddArray(_tsub_pt)
                                _sub_pd.GetPointData().SetActiveScalars(
                                    "tortuosity")
                            # ── TubeFilter: real 3D geometry, required in
                            # VR where screen-space lines vanish across
                            # panoramic cube-face seams. ──────────────────
                            _use_tubes  = False
                            _mapped_pd  = _sub_pd
                            try:
                                _tubef = _vtk_t.vtkTubeFilter()
                                _tubef.SetInputData(_sub_pd)
                                _tubef.SetRadius(TRACK_TUBE_RADIUS_MOVIE
                                    * (0.8 if vr_mode != "off" else 1.0))
                                _tubef.SetNumberOfSides(
                                    TRACK_TUBE_SIDES_MOVIE)
                                _tubef.CappingOff()
                                _tubef.Update()
                                _tube_out = _tubef.GetOutput()
                                if _tube_out.GetNumberOfCells() > 0:
                                    _mapped_pd = _tube_out
                                    _use_tubes = True
                                    _bins_geo  = "tubes"
                            except Exception:  # noqa: BLE001
                                pass
                            _mp = _vtk_t.vtkPolyDataMapper()
                            _mp.SetInputData(_mapped_pd)
                            if _tort_lut_mv is not None and _tort_arr_np_mv is not None:
                                if _use_tubes:
                                    _mp.SetScalarModeToUsePointData()
                                else:
                                    _mp.SetScalarModeToUseCellData()
                                _mp.SetLookupTable(_tort_lut_mv)
                                _mp.SetScalarRange(TORTUOSITY_VMIN,
                                                   TORTUOSITY_VMAX)
                                _mp.ScalarVisibilityOn()
                                _mp.SetColorModeToMapScalars()
                            else:
                                _mp.ScalarVisibilityOff()
                            _a = _vtk_t.vtkActor()
                            _a.SetMapper(_mp)
                            _prop = _a.GetProperty()
                            if _tort_lut_mv is None or _tort_arr_np_mv is None:
                                _prop.SetColor(*_WATER_RGB)
                            if not _use_tubes:
                                _prop.SetLineWidth(
                                    float(TRACK_LINE_WIDTH_MOVIE))
                            _prop.SetOpacity(0.40)
                            _prop.LightingOff()
                            _a._bin_idx    = _bi
                            _a._base_alpha = 0.40
                            getattr(_a, "actor", _a).SetVisibility(0)
                            _lba.append(_a)
                        tracks_axis_info  = {
                            "axis_col": _ac,
                            "axis_min": _amin,
                            "axis_len": _alen,
                            "n_bins":   _N_BINS,
                            "bin_full": _BIN_FULL,
                            "bin_fade": _BIN_FADE,
                        }
                    except Exception as _bin_exc:  # noqa: BLE001
                        logger.warning(
                            "Tracks binning failed in movie (%s); "
                            "falling back to single tracks actor.",
                            _bin_exc,
                        )
                        # Single-actor fallback: try TubeFilter first, then
                        # fall through to plain polylines.
                        _has_tort = (
                            _tort_lut_mv is not None
                            and _pd_sm.GetCellData().GetArray("tortuosity")
                               is not None
                        )
                        _fb_use_tubes = False
                        _fb_pd = _pd_sm
                        try:
                            _fb_c2p = _vtk_t.vtkCellDataToPointData()
                            _fb_c2p.SetInputData(_pd_sm)
                            _fb_c2p.Update()
                            _fb_tubef = _vtk_t.vtkTubeFilter()
                            _fb_tubef.SetInputData(_fb_c2p.GetOutput())
                            _fb_tubef.SetRadius(TRACK_TUBE_RADIUS_MOVIE
                                * (0.8 if vr_mode != "off" else 1.0))
                            _fb_tubef.SetNumberOfSides(TRACK_TUBE_SIDES_MOVIE)
                            _fb_tubef.CappingOff()
                            _fb_tubef.Update()
                            if _fb_tubef.GetOutput().GetNumberOfCells() > 0:
                                _fb_pd = _fb_tubef.GetOutput()
                                _fb_use_tubes = True
                        except Exception:  # noqa: BLE001
                            pass
                        _map_fallback = _vtk_t.vtkPolyDataMapper()
                        _map_fallback.SetInputData(_fb_pd)
                        if _has_tort:
                            if _fb_use_tubes:
                                _map_fallback.SetScalarModeToUsePointData()
                            else:
                                _pd_sm.GetCellData().SetActiveScalars(
                                    "tortuosity")
                                _map_fallback.SetScalarModeToUseCellData()
                            _map_fallback.SetLookupTable(_tort_lut_mv)
                            _map_fallback.SetScalarRange(
                                TORTUOSITY_VMIN, TORTUOSITY_VMAX)
                            _map_fallback.ScalarVisibilityOn()
                            _map_fallback.SetColorModeToMapScalars()
                        else:
                            _map_fallback.ScalarVisibilityOff()
                        _a = _vtk_t.vtkActor()
                        _a.SetMapper(_map_fallback)
                        if not _fb_use_tubes:
                            _a.GetProperty().SetLineWidth(
                                float(TRACK_LINE_WIDTH_MOVIE))
                        _a.GetProperty().SetOpacity(0.40)
                        _a.GetProperty().LightingOff()
                        if not _has_tort:
                            _a.GetProperty().SetColor(*_WATER_RGB)
                        _a._bin_idx = 0
                        _a._base_alpha = 0.40
                        getattr(_a, "actor", _a).SetVisibility(0)
                        _lba = [_a]
                        tracks_axis_info = None
                    _n_rendered = _pd_sm.GetNumberOfCells() if _pd_sm is not None else _pd_t.GetNumberOfCells()
                    print(f"      [ui_state] crown tracks ({_bins_geo}): "
                          f"{_n_rendered} paths → "
                          f"{len(_lba)} populated bins")
                    # Commit line actors immediately so that a failure in the
                    # arrow loading block below does not leave tracks empty.
                    tracks_bin_actors = list(_lba)
                    # ── Arrow bin actors ──────────────────────────────────
                    _aba = []
                    if (_trk_arr_path is not None
                            and _trk_arr_path.exists()):
                        try:
                            _trk_stage = "load-arrows-vtp"
                            _rdr_a = _vtk_t.vtkXMLPolyDataReader()
                            _rdr_a.SetFileName(str(_trk_arr_path))
                            _rdr_a.Update()
                            _pd_g = _rdr_a.GetOutput()
                            if _pd_g.GetNumberOfPoints() <= 0:
                                raise ValueError("tracks arrows VTP has no points")
                            _asrc = _vtk_t.vtkArrowSource()
                            _asrc.SetTipLength(0.35)
                            _asrc.SetTipRadius(0.12)
                            _asrc.SetShaftRadius(0.06)
                            _asrc.Update()
                            _gl = _vtk_t.vtkGlyph3D()
                            _gl.SetSourceConnection(_asrc.GetOutputPort())
                            _gl.SetInputData(_pd_g)
                            _gl.SetVectorModeToUseVector()
                            _gl.OrientOn()
                            try:
                                _gl.SetScaleModeToNoDataScaling()
                            except AttributeError:
                                _gl.SetScaleMode(3)  # VTK_DATA_SCALING_OFF
                            _gl.SetScaleFactor(DEFAULT_ARROW_LENGTH * 1.2)
                            _gl.SetColorModeToColorByScalar()
                            _gl.Update()
                            _a = _vedo_trk.Mesh(_gl.GetOutput())
                            _a._bin_idx = 0
                            _a._base_alpha = 1.0
                            try:
                                _a.lighting("off")
                            except Exception:  # noqa: BLE001
                                pass
                            getattr(_a, "actor", _a).SetVisibility(0)
                            _aba.append(_a)
                            print(f"      [ui_state] crown tracks arrows: "
                                  f"{_pd_g.GetNumberOfPoints()} glyph centres")
                        except Exception as _arr_exc:  # noqa: BLE001
                            logger.warning(
                                "Tracks arrows bins failed (%s): %s",
                                _trk_arr_path, _arr_exc,
                            )
                    if _aba:
                        # Arrow bin actors are intentionally NOT merged into
                        # tracks_bin_actors: without proper per-bin culling
                        # all 200k glyph arrows would render simultaneously,
                        # creating a visually overwhelming overlay.  They are
                        # only useful when binning is active (full viewer).
                        pass
                    print(f"      [ui_state] tracks culling: axis={_ac}, "
                          f"{len(tracks_bin_actors)} bin actors total "
                          f"(±{_BIN_FULL} full + ±{_BIN_FADE} fade)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load crown tracks from %s at stage '%s': %s",
                    _trk_path, locals().get("_trk_stage", "unknown"), exc,
                )
        else:
            logger.debug("--crown-tracks-cache %s does not exist – "
                         "arrows_tracks mode will be empty", _trk_path)

    # ── 2. Interpolate camera + actor track ───────────────────────────────
    print("[2/4] Interpolating camera + actor track …")
    track, t_out_ui = build_track(
        control_points,
        frames_per_segment,
        hold_frames=hold_frames,
        ease_segments=args.ease_segments,
        ease_strength=args.ease_strength,
    )
    total_frames = len(track)
    print(f"      {total_frames} frames "
          f"({total_frames / args.fps:.2f}s of footage)")

    if args.stresstest:
        t_start_s = max(0.0, args.stresstest_start)
        t_end_s   = args.stresstest_end
        f_start   = min(total_frames - 1, int(round(t_start_s * args.fps)))
        f_end     = min(total_frames,     int(round(t_end_s   * args.fps)))
        track     = track   [f_start:f_end]
        t_out_ui  = t_out_ui[f_start:f_end]
        total_frames = len(track)
        print(f"      [stresstest] frames {f_start}–{f_end}  "
              f"({t_start_s:.1f}s – {t_end_s:.1f}s)  "
              f"→ {total_frames} frames")

    # In VR mode, lock the orientation to the first keyframe so the
    # spline only carries the translation component.  Any orientation
    # change in a headset is interpreted as the world tilting around
    # the user → reliable motion-sickness trigger.  The user can still
    # look around freely with their head; we just keep the *virtual
    # camera basis* steady.
    # NOTE: must run on the full track BEFORE chunk slicing so that all
    # workers share the same locked orientation (track[0] = global frame 0).
    if vr_mode != "off":
        _lock_orientation_for_vr(track)
        print("      [VR] orientation locked to first keyframe "
              "(only translations are applied)")

    # ── Stage C: zero-phase position smoothing ──────────────────────────
    # Eliminates spline-residual jitter from the ortho-panel red dot and
    # info-billboard.  Wider window for VR (comfort-critical) than flat.
    # _smooth_track_positions handles short tracks (stresstest) by auto-
    # padding the signal before filtfilt — no edge-ringing artefacts.
    # NOTE: must run on the full track BEFORE chunk slicing; applying it
    # per-chunk would cause boundary discontinuities (each filtfilt run
    # has independent edge-padding → positional jumps at chunk joints).
    _smooth_window = 1.0 if vr_mode != "off" else 0.6
    _smooth_track_positions(track, args.fps, window_s=_smooth_window)
    logger.info("Stage-C smoothing applied (window=%.1f s, %d frames)",
                _smooth_window, len(track))

    # ── Worker chunk slicing (injected by --parallel; exact frame indices) ──
    # Kept AFTER VR-lock and Stage-C smoothing so those operate on the full
    # sequence and produce consistent results across all workers.
    # The full track is also saved for _build_path_line (cable-car line must
    # span the whole trajectory in every worker's rendered frame).
    full_track = track   # reference to the unsliced sequence
    if args._chunk_start is not None:
        track    = track   [args._chunk_start : args._chunk_end]
        t_out_ui = t_out_ui[args._chunk_start : args._chunk_end]
        total_frames = len(track)
        print(f"      [worker] chunk [{args._chunk_start}, {args._chunk_end})  "
              f"→ {total_frames} frames  (png_offset={args._frame_offset})")

    # ── --print-debug: analyse track, skip rendering ──────────────────
    if args.print_debug:
        print(f"\n[--print-debug] {len(track)} frames after slice/lock/smooth "
              f"({len(track)/args.fps:.2f}s)  vr_mode={vr_mode}")
        _run_track_debug(track, args.fps, vr_mode)
        return 0

    # ── 2bis. UI-state replay track ──────────────────────────────────
    if args.ignore_ui_state:
        ui_track: List[dict | None] = [None] * len(track)
    else:
        # t_out_ui comes directly from build_track and carries the
        # arc-length-aware fractional control-point position per frame.
        ui_track = _build_ui_track(
            control_points, t_out_ui,
            fps=args.fps,
            frames_per_segment=frames_per_segment,
            transition_seconds=float(args.ui_transition_seconds),
        )
        # Truncate to match the stresstest-cropped track.
        if len(ui_track) > len(track):
            ui_track = ui_track[:len(track)]
        n_with_ui = sum(1 for u in ui_track if u)
        if n_with_ui:
            print(f"      [ui_state] {n_with_ui}/{len(track)} frames have a "
                  f"UI state (transition={args.ui_transition_seconds:.2f}s)")
            if args.debug_tracks:
                _vm_counts: dict[str, int] = {}
                for _u in ui_track:
                    if not _u:
                        continue
                    _vm = str(_u.get("view_mode") or "<none>")
                    _vm_counts[_vm] = _vm_counts.get(_vm, 0) + 1
                print("      [debug-tracks] ui view_mode counts:", _vm_counts)
        else:
            print("      [ui_state] no UI state found in positions JSON "
                  "— skipping replay")
    # Attach the per-frame ui_state to track entries for convenient access.
    for _entry, _ui in zip(track, ui_track):
        if _ui is not None:
            _entry["ui_state"] = _ui

    path_line = None
    if show_path:
        # Build the cable-car line from the *full* track so the green path
        # covers the entire trajectory regardless of which chunk this worker
        # is rendering.  Using the sliced `track` here would produce a short
        # stub that changes shape at every chunk boundary in the final movie.
        path_line = _build_path_line(
            full_track, PATH_VERTICAL_OFFSET,
            as_tube=(vr_mode != "off"),
            radius_frac=PATH_TUBE_RADIUS_FRAC_VR if vr_mode != "off" else PATH_TUBE_RADIUS_FRAC,
        )
        kind = "tube" if vr_mode != "off" else "line"
        print(f"      built green path {kind} over {len(full_track)} samples")

    # ── 3a. Preview mode: interactive replay, no PNGs, no encoding ────────
    if args.preview:
        if vr_mode != "off":
            print("      [preview] stereo VR preview is not supported – "
                  "previewing the mono trajectory instead.")
        return _run_preview(
            mesh, path_line, track,
            width=min(args.width, 1920),
            height=min(args.height, 1080),
            fps=args.fps,
        )

    # ── 3b. Render frames off-screen ──────────────────────────────────────
    print(f"[3/4] Rendering frames to {frames_dir} …")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not args._worker:   # orchestrator already cleaned; workers share the dir
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()

    if args.egl:
        try:
            import vedo.vtkclasses as _vtki_egl
            import vtkmodules.vtkRenderingOpenGL2 as _vtkgl_egl
            if hasattr(_vtkgl_egl, "vtkEGLRenderWindow"):
                _vtki_egl.vtkRenderWindow = _vtkgl_egl.vtkEGLRenderWindow
                logger.info("EGL render window forced (--egl)")
            else:
                logger.warning("--egl requested but vtkEGLRenderWindow not "
                               "found in this VTK build; falling back to default")
        except Exception as _egl_exc:  # noqa: BLE001
            logger.warning("Could not force EGL backend: %s", _egl_exc)

    import vedo
    plt = vedo.Plotter(
        title="Marvel · Wat_Norm_Cortex (movie)",
        bg=BACKGROUND,
        axes=0,
        offscreen=True,
        size=(eye_w, eye_h),
    )
    # Enable MSAA on the render window BEFORE any actor is added or
    # rendered — VTK only allocates a multi-sampled framebuffer on the
    # first render.
    try:
        if args.msaa and args.msaa > 0:
            plt.window.SetMultiSamples(int(args.msaa))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Early SetMultiSamples failed: %s", exc)
    # Force Phong shading on every mesh actor.  VTK's default is
    # Gouraud (vertex-interpolated lighting) which produces visibly
    # facetted highlights on curved silhouettes.
    _force_phong(mesh, alt_mesh, pillars_mesh)
    plt.add(mesh)
    if alt_mesh is not None:
        plt.add(alt_mesh)
    if pillars_mesh is not None:
        plt.add(pillars_mesh)
    if arrows_actor is not None:
        plt.add(arrows_actor)
    if water_dual_actor is not None:
        plt.add(water_dual_actor)
    if air_dual_actor is not None:
        plt.add(air_dual_actor)
    for _tba in tracks_bin_actors:
        try:
            plt.add(_tba)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add track bin actor: %s", exc)
    if membranes_actor is not None:
        try:
            plt.renderer.AddActor(membranes_actor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add membranes actor: %s", exc)
    if lames_actor is not None:
        try:
            plt.renderer.AddActor(lames_actor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add lames actor: %s", exc)
    if path_line is not None:
        plt.add(path_line)
    plt.show(interactive=False, resetcam=True)

    # Enable FXAA on every renderer (post-process; works for both mono
    # and VR — applied after the panoramic remap when present).
    _setup_antialiasing(
        plt,
        msaa=args.msaa,
        fxaa=not args.no_fxaa,
        fxaa_contrast=args.fxaa_contrast,
        fxaa_hard_contrast=args.fxaa_hard_contrast,
        vr_mode=vr_mode,
    )

    # UI-state replay bundle (holds actors + caches diff state across
    # frames so we don't reapply unchanged values).
    fog_actors_bundle = [a for a in (
        mesh, alt_mesh, pillars_mesh, arrows_actor,
        membranes_actor, lames_actor,
    ) if a is not None]
    ui_bundle = _UiReplayBundle(
        mesh=mesh,
        fog_near_frac=args.fog_near,
        fog_far_frac=args.fog_far,
        force_fog=args.fog,
        alt_mesh=alt_mesh,
        pillars_mesh=pillars_mesh,
        fog_actors=fog_actors_bundle,
        plt=plt,
        vr_mode=vr_mode,
        membranes_actor=membranes_actor,
        membranes_threshold=membranes_threshold,
        membranes_n_steps=membranes_n_steps,
        lames_actor=lames_actor,
        lames_step_pds=lames_step_pds,
        lames_n_steps=lames_n_steps,
        lames_fps=lames_fps,
        movie_fps=args.fps,
        arrows_actor=arrows_actor,
        water_dual_actor=water_dual_actor,
        air_dual_actor=air_dual_actor,
        tracks_bin_actors=tracks_bin_actors,
        tracks_axis_info=tracks_axis_info,
        debug_tracks=args.debug_tracks,
    )

    # Install lighting that matches the interactive viewer.
    #
    # • Flat movie  → no custom lights; VTK's AutomaticLightCreation (ON by
    #                 default, same in offscreen mode) creates one headlight
    #                 at the camera position, exactly as in the interactor.
    # • VR movie    → world-space SceneLight whose position is synced to
    #                 the camera each frame (see _update_vr_headlight call
    #                 in the render loop below).  A true CameraLight / the
    #                 automatic headlight cannot be used here because the
    #                 panoramic pass renders 6 cube faces with different
    #                 orientations, and a camera-relative light shifts
    #                 between faces → visible seams.
    pano_passes = None
    vr_lights = None
    if not args.preview and vr_mode != "off":
        vr_lights = _install_vr_headlight(plt)
        _n_torch = len(vr_lights[0])
        print(f"      installed {_n_torch * 3} world-space lights "
              "(torch × 1 + green-fill × 1 + red-fill × 1, VR mode – updated per frame)")
    if vr_mode != "off":
        # Panoramic projection pass AFTER the lights are in place.
        # FXAA must be chained HERE (wrapping the pano pass) — calling
        # renderer.SetUseFXAA() before SetPass(pano) is silently ignored.
        pano_passes = _setup_panoramic_pass(
            plt, vr_angle, args.vr_cube_resolution,
            fxaa=not args.no_fxaa,
            fxaa_contrast=args.fxaa_contrast,
            fxaa_hard_contrast=args.fxaa_hard_contrast,
        )
        # Remove vtkFrustumCoverageCuller from every renderer.  The culler
        # uses the *main* camera frustum to zero-out props outside its FOV.
        # In VR360 the panoramic pass renders 6 cube faces each with a 90°
        # camera — a billboard at 30° left is inside the cube-face frustum
        # but outside the (narrow) main-camera FOV, so the culler drops it
        # from the prop array before any face is rendered.  Without the
        # culler all visible props are submitted to every cube face and the
        # GPU clips correctly per-face.
        for _ren in plt.renderers:
            _ren.GetCullers().RemoveAllItems()

    # World-up and per-frame travel directions — needed by both the VR
    # headlight update and the ortho billboard.  Computed once here so
    # they are available even when the ortho panel is disabled.
    vr_world_up: np.ndarray | None = None
    vr_travel_dirs: np.ndarray | None = None
    if vr_mode != "off" and track:
        vr_world_up = np.asarray(
            track[0]["camera"].get("view_up", [0.0, 1.0, 0.0]),
            dtype=float,
        )
        _up_n = float(np.linalg.norm(vr_world_up))
        vr_world_up = vr_world_up / _up_n if _up_n > 1e-9 else np.array([0.0, 1.0, 0.0])
        _vel_win = max(1, args.fps // 4)
        vr_travel_dirs = _compute_travel_dirs(
            track, vr_world_up, velocity_window=_vel_win, alpha=0.12,
        )

    # Optional orthogonal-slice locator panel — baked into every frame.
    ortho_overlay = None
    ortho_billboard = None
    _focal_dist = 100.0  # default; overwritten by the ortho-panel block below
    if not args.no_ortho_panel:
        raw_path = (
            Path(args.raw).expanduser().resolve()
            if args.raw
            else (Path(args.input).expanduser().resolve().parent / "Raw.tif")
        )
        if raw_path.exists():
            try:
                from marvel_view.visualization.ortho_panel import (
                    OrthoPanel3DBillboard, OrthoPanelOverlay, load_raw_volume,
                )
                raw_volume = load_raw_volume(raw_path)
                if vr_mode != "off":
                    # ── VR mode: 3-D billboard in layer-0 renderer ──────────────
                    # Estimate a representative focal distance from the track.
                    _foc_dists = np.linalg.norm(
                        np.array([s["camera"]["focal_point"] for s in track], dtype=float)
                        - np.array([s["camera"]["position"] for s in track], dtype=float),
                        axis=1,
                    )
                    _focal_dist = float(np.median(_foc_dists)) if _foc_dists.size else 100.0
                    # vr_world_up and vr_travel_dirs are already computed above.
                    ortho_billboard = OrthoPanel3DBillboard(
                        plt, raw_volume,
                        focal_dist=_focal_dist,
                        cell_pixels=256,
                        meters_per_voxel=args.meters_per_voxel,
                        angular_size_deg=16.8,    # 24 ° × 0.7
                        forward_metres=10.0,      # 10 m forward (↓ parallax)
                        left_metres=4.67,         # 10 × tan(25°) → 25 ° to the left
                        vert_metres=-0.87,        # 10 × tan(5°) → 5 ° lower
                    )
                    print(f"      ortho billboard attached  (Raw={raw_path.name}, "
                          f"shape={raw_volume.shape}, focal_dist={_focal_dist:.1f})")
                else:
                    # ── Flat mode: 2-D HUD anchored vertically centered ─────────
                    ortho_overlay = OrthoPanelOverlay(
                        plt, raw_volume,
                        viewport=(0.02, 0.15, 0.34, 0.69),
                        cell_pixels=round(92 * eye_h / 540),
                        center_vertically=True,
                    )
                    print(f"      ortho panel attached  (Raw={raw_path.name}, "
                          f"shape={raw_volume.shape})")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to attach ortho panel: %s", exc)
        else:
            logger.info("Raw volume not found (%s) – ortho panel disabled.",
                        raw_path)

    # ── Info panel (title + mode subtitle) ──────────────────────────────────
    info_overlay   = None
    info_billboard = None
    try:
        from marvel_view.visualization.ortho_panel import (
            InfoBillboard3D as _InfoBB, InfoPanelOverlay as _InfoPO,
        )
        from marvel_view.scripts.water_conductance.constants import (
            PANEL_TITLE         as _PANEL_TITLE,
            VIEW_MODE_SUBTITLES as _VIEW_MODE_SUBTITLES,
        )
        _init_mode = (track[0].get("ui_state") or {}).get("view_mode", "mesh_bridges")
        _init_sub  = _VIEW_MODE_SUBTITLES.get(_init_mode, "")
        if vr_mode != "off":
            info_billboard = _InfoBB(
                plt, _PANEL_TITLE, _init_sub,
                focal_dist=_focal_dist,
                meters_per_voxel=args.meters_per_voxel,
                angular_width_deg=40.32,  # 28.8 × 1.4
                tile_scale=2.38,          # 1.7 × 1.4
                forward_metres=10.0,      # 10 m forward (↓ parallax vs 3 m)
                left_metres=0.0,
                vert_metres=3.64,         # 10 × tan(20°) → 20° above
            )
            print("      info billboard attached")
        else:
            info_overlay = _InfoPO(plt, _PANEL_TITLE, _init_sub, opacity=0.72)
            print("      info overlay attached")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to attach info panel: %s", exc)

    # ── Colormap legend bars ─────────────────────────────────────────────────
    # Static legend overlays that match the per-face scalar colormaps applied
    # by the interactive viewer.  Visibility is toggled per frame based on the
    # ui_state["view_mode"] and ui_state["cmap_mesh_mode"] values in the track.
    import numpy as _np_cbar_mv
    _WATER_CMAP_STOPS_MV = _np_cbar_mv.array([
        [0.10, 0.12, 0.72], [0.25, 0.45, 1.00],
        [0.70, 0.35, 0.90], [0.90, 0.05, 0.05],
    ], dtype=_np_cbar_mv.float32)
    _AIR_CMAP_STOPS_MV = _np_cbar_mv.array([
        [0.05, 0.38, 0.10], [0.28, 0.68, 0.12], [0.72, 0.88, 0.12],
    ], dtype=_np_cbar_mv.float32)
    _cbar_mv_density  = None
    _cbar_mv_radial   = None
    _cbar_mv_water    = None
    _cbar_mv_air      = None
    _cbar_mv_tortuosity = None
    try:
        if vr_mode != "off":
            from marvel_view.visualization.colormap_bar import ColormapBar3DBillboard as _CB3D  # noqa: PLC0415
            _cbar_vr_font = round(11 * 1.8)  # 20 — bigger text for VR headset
            _cbar_mv_density = _CB3D(
                plt, cmap="viridis",  vmin=0.0, vmax=1.0, title="Density",
                focal_dist=_focal_dist, left_frac=-0.38, vert_frac=0.15,
                font_size=_cbar_vr_font,
            )
            _cbar_mv_radial = _CB3D(
                plt, cmap="coolwarm", vmin=-1.0, vmax=1.0, title="Slope of radial density",
                focal_dist=_focal_dist, left_frac=-0.38, vert_frac=-0.05,
                font_size=_cbar_vr_font,
            )
            _cbar_mv_water = _CB3D(
                plt, cmap=_WATER_CMAP_STOPS_MV, vmin=0.0, vmax=1.0, title="Water",
                focal_dist=_focal_dist, left_frac=-0.38, vert_frac=0.15,
                font_size=_cbar_vr_font,
            )
            _cbar_mv_air = _CB3D(
                plt, cmap=_AIR_CMAP_STOPS_MV, vmin=0.0, vmax=1.0, title="Air",
                focal_dist=_focal_dist, left_frac=-0.38, vert_frac=-0.05,
                font_size=_cbar_vr_font,
            )
            _cbar_mv_tortuosity = _CB3D(
                plt, cmap=TORTUOSITY_CMAP_STOPS,
                vmin=TORTUOSITY_VMIN, vmax=TORTUOSITY_VMAX,
                title="Tortuosity",
                focal_dist=_focal_dist, left_frac=-0.38, vert_frac=0.05,
                font_size=_cbar_vr_font,
            )
            print("      colormap billboards attached")
        else:
            from marvel_view.visualization.colormap_bar import ColormapBar2D as _CB2D  # noqa: PLC0415
            _cbar_mv_density = _CB2D(
                plt, cmap="viridis",  vmin=0.0, vmax=1.0, title="Density",
                pos=(0.87, 0.30),
            )
            _cbar_mv_radial = _CB2D(
                plt, cmap="coolwarm", vmin=-1.0, vmax=1.0, title="Slope of radial density",
                pos=(0.87, 0.30),
            )
            _cbar_mv_water = _CB2D(
                plt, cmap=_WATER_CMAP_STOPS_MV, vmin=0.0, vmax=1.0, title="Water bridge orientation",
                pos=(0.87, 0.52),
            )
            _cbar_mv_air = _CB2D(
                plt, cmap=_AIR_CMAP_STOPS_MV, vmin=0.0, vmax=1.0, title="Gas diffusion",
                pos=(0.87, 0.12),
            )
            _cbar_mv_tortuosity = _CB2D(
                plt, cmap=TORTUOSITY_CMAP_STOPS,
                vmin=TORTUOSITY_VMIN, vmax=TORTUOSITY_VMAX,
                title="Tortuosity",
                pos=(0.87, 0.30),
            )
            print("      colormap overlays attached")
        # Start all hidden; per-frame loop shows/hides as needed.
        for _b in (_cbar_mv_density, _cbar_mv_radial, _cbar_mv_water,
                   _cbar_mv_air, _cbar_mv_tortuosity):
            if _b is not None:
                _b.set_visible(False)
    except Exception as _cbar_mv_exc:  # noqa: BLE001
        logger.warning("Failed to attach colormap bars: %s", _cbar_mv_exc)

    actor = getattr(mesh, "actor", None) or mesh

    # Optional debug overlay: indicative keyframe index in [0, N-1].
    # Only added in flat (mono) mode — vtkPanoramicProjectionPass remaps
    # the cubemap, so 2-D HUD text becomes either invisible or doubled.
    debug_idx_text = None
    if args.debugdisplayindexframes and vr_mode == "off":
        import vedo as _vedo
        debug_idx_text = _vedo.Text2D(
            "keyframe 0.00",
            pos="bottom-center",
            c="white",
            bg="black",
            alpha=0.55,
            s=1.2,
        )
        try:
            plt.add(debug_idx_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not attach debug index overlay: %s", exc)
            debug_idx_text = None

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    # Temporary per-eye PNGs reused across frames (overwritten each
    # iteration).  We deliberately go through disk rather than
    # ``plt.screenshot(asarray=True)`` because the asarray path in vedo
    # interacts poorly with ``vtkPanoramicProjectionPass`` under
    # offscreen rendering: the framebuffer read appears to be cached
    # and every frame ends up identical to the first one (resulting in
    # an absurdly small MP4 — symptom: a 1-MB file for an 8K stereo
    # 47 s clip where every frame is bit-for-bit the same).
    _eye_suffix   = f"_{args._frame_offset}" if args._worker else ""
    left_eye_png  = frames_dir / f"_eye_left{_eye_suffix}.png"
    right_eye_png = frames_dir / f"_eye_right{_eye_suffix}.png"

    t_start = time.time()
    # Compute physical IPD once — used in the VR stereo offset per frame.
    # When --meters-per-voxel is set, the IPD is fixed in voxels regardless
    # of camera distance (physically correct).  Otherwise fall back to the
    # legacy ipd_frac × dist formula (ipd_abs=0 signals the fallback).
    _ipd_abs = 0.0
    if vr_mode != "off" and args.meters_per_voxel > 0.0:
        _ipd_abs = args.ipd_metres / args.meters_per_voxel
        print(f"      [VR] physical IPD  : {_ipd_abs:.4f} vox  "
              f"({args.ipd_metres * 1000:.1f} mm / "
              f"{args.meters_per_voxel} m·vox⁻¹)")

    # Pre-compute per-frame travel speed (µm/s) for the info panel speed line.
    _track_speeds_um_s: np.ndarray = np.zeros(len(track))
    if len(track) > 1 and (info_billboard is not None or info_overlay is not None):
        try:
            from marvel_view.scripts.water_conductance.constants import (
                VOXEL_SIZE_UM as _VOXEL_SIZE_UM,
            )
        except ImportError:
            _VOXEL_SIZE_UM = 6.71
        _tpos     = np.array([s["camera"]["position"] for s in track], dtype=float)
        _raw_spd  = np.linalg.norm(np.diff(_tpos, axis=0), axis=1) * args.fps * _VOXEL_SIZE_UM
        _raw_spd  = np.concatenate([_raw_spd, [_raw_spd[-1]]])
        _spd_win  = max(1, args.fps // 4)
        _kern     = np.ones(_spd_win) / _spd_win
        _track_speeds_um_s = np.convolve(_raw_spd, _kern, mode="same")

    for i, state in enumerate(track):
        _apply_actor_state(actor, state["actor"])
        # Per-frame UI-state replay (visibility swap, pillars, fog/SSAO).
        ui_bundle.apply(state.get("ui_state"))
        # Camera-based culling: slide clip planes to camera position so
        # only nearby tracks are rendered (avoids visual overload).
        ui_bundle.update_tracks_bins(
            np.asarray(state["camera"]["position"], dtype=float))
        if args.debug_tracks and i < 10:
            _vis = 0
            for _a in tracks_bin_actors:
                try:
                    _vis += int(getattr(_a, "actor", _a).GetVisibility())
                except Exception:  # noqa: BLE001
                    pass
            _vm_dbg = (state.get("ui_state") or {}).get("view_mode", "<none>")
            print(
                f"      [debug-tracks] frame={i:03d} vm={_vm_dbg} "
                f"tracks_visible={_vis}/{len(tracks_bin_actors)}"
            )

        if ortho_billboard is not None or ortho_overlay is not None:
            # Always use the un-shifted (mono) camera position — the red
            # dot represents the viewer's location, not the per-eye shift.
            cam_state = state["camera"]
            mono_pos = np.asarray(cam_state["position"], dtype=float)
            mono_foc = np.asarray(cam_state["focal_point"], dtype=float)
            mono_dir = mono_foc - mono_pos
            nrm = float(np.linalg.norm(mono_dir))
            if nrm > 1e-9:
                mono_dir = mono_dir / nrm
            else:
                mono_dir = np.array([0.0, 0.0, -1.0])
            if ortho_billboard is not None:
                ortho_billboard.update(mono_pos, mono_dir, vr_travel_dirs[i], vr_world_up)
            else:
                ortho_overlay.update(mono_pos, mono_dir)

        # ── Colormap bars: show/hide + pose ──────────────────────────────────
        if any(b is not None for b in (_cbar_mv_density, _cbar_mv_radial,
                                        _cbar_mv_water, _cbar_mv_air,
                                        _cbar_mv_tortuosity)):
            _ui_st_cb  = state.get("ui_state") or {}
            _vm_cb     = _ui_st_cb.get("view_mode", "")
            _cm_mode   = _ui_st_cb.get("cmap_mesh_mode", 0)
            _is_dual   = (_vm_cb == "arrows_dual")
            _is_mesh   = (_vm_cb in ("mesh_bridges", "mesh_all"))
            _is_tracks = (_vm_cb == "arrows_tracks")
            _show_den  = _is_mesh and _cm_mode == 1
            _show_rad  = _is_mesh and _cm_mode == 2
            for _b, _vis in (
                (_cbar_mv_density,    _show_den),
                (_cbar_mv_radial,     _show_rad),
                (_cbar_mv_water,      _is_dual),
                (_cbar_mv_air,        _is_dual),
                (_cbar_mv_tortuosity, _is_tracks),
            ):
                if _b is not None:
                    _b.set_visible(_vis)
            # VR mode: update billboard poses.
            if vr_mode != "off":
                _cb_pos = np.asarray(state["camera"]["position"], dtype=float)
                for _b, _vis in (
                    (_cbar_mv_density,    _show_den),
                    (_cbar_mv_radial,     _show_rad),
                    (_cbar_mv_water,      _is_dual),
                    (_cbar_mv_air,        _is_dual),
                    (_cbar_mv_tortuosity, _is_tracks),
                ):
                    if _b is not None and _vis:
                        _b.update_pose(_cb_pos, vr_travel_dirs[i], vr_world_up)

        # ── Info panel update (pose + subtitle) ─────────────────────────────
        if info_billboard is not None or info_overlay is not None:
            _ui_st    = state.get("ui_state") or {}
            _sub_text = _VIEW_MODE_SUBTITLES.get(_ui_st.get("view_mode", ""), "")
            _spd_text = f"Travelling speed:  {_track_speeds_um_s[i]:.1f} um/s"
            if info_billboard is not None:
                _ipos = np.asarray(state["camera"]["position"], dtype=float)
                info_billboard.update(_ipos, vr_travel_dirs[i], vr_world_up, _sub_text, _spd_text)
            elif info_overlay is not None:
                info_overlay.update(_sub_text, _spd_text)

        if vr_mode == "off":
            if debug_idx_text is not None:
                kf = float(state.get("keyframe_index", 0.0))
                try:
                    debug_idx_text.text(
                        f"keyframe {kf:6.2f}   frame {i + 1}/{total_frames}"
                    )
                except Exception:  # noqa: BLE001
                    pass
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, state["camera"])
                renderer.ResetCameraClippingRange()
                _nr, _fr = cam.GetClippingRange()
                cam.SetClippingRange(min(_nr, 2.0), _fr)
            plt.render()
            plt.screenshot(str(frames_dir / f"frame_{i + args._frame_offset:05d}.png"))
        else:
            offset = _eye_offset(state, args.ipd_frac, _ipd_abs)
            left_state  = _shift_camera_state(state, -offset)
            right_state = _shift_camera_state(state,  offset)

            # Sync the world-space lights to the mono camera once per frame
            # (same lights for both eyes — avoids any stereo shading discrepancy).
            _td = vr_travel_dirs[i] if vr_travel_dirs is not None else None
            _update_vr_headlight(vr_lights, state["camera"], _td, vr_world_up,
                                 tilt_deg=args.torch_tilt)

            # Left eye — render & write to disk (same code path as mono).
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, left_state["camera"])
                # Fixed range: panoramic pass calls ResetCameraClippingRange
                # independently per cube-face camera, but a tight fixed range
                # on the main camera avoids any legacy Reset on this renderer.
                cam.SetClippingRange(0.5, 10_000.0)
            plt.render()
            plt.screenshot(str(left_eye_png))

            # Right eye.
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, right_state["camera"])
                cam.SetClippingRange(0.5, 10_000.0)
            plt.render()
            plt.screenshot(str(right_eye_png))

            # Stitch side-by-side (left | right).
            left_img  = imageio.imread(str(left_eye_png))
            right_img = imageio.imread(str(right_eye_png))
            if left_img.ndim == 3 and left_img.shape[2] == 4:
                left_img  = left_img[..., :3]
            if right_img.ndim == 3 and right_img.shape[2] == 4:
                right_img = right_img[..., :3]
            if left_img.shape != right_img.shape:
                h = min(left_img.shape[0], right_img.shape[0])
                w = min(left_img.shape[1], right_img.shape[1])
                left_img = left_img[:h, :w]
                right_img = right_img[:h, :w]
            sbs = np.concatenate([left_img, right_img], axis=1)
            imageio.imwrite(str(frames_dir / f"frame_{i + args._frame_offset:05d}.png"), sbs)

        if not args._worker:
            _print_progress(i, total_frames, t_start)
    if not args._worker:
        sys.stdout.write("\n")

    # Clean up the per-eye scratch PNGs.
    for p in (left_eye_png, right_eye_png):
        if p.exists():
            p.unlink()

    plt.close()
    del pano_passes  # explicit – release the panoramic pass references
    print(f"      rendered {total_frames} frames in "
          f"{_format_eta(time.time() - t_start)}")

    # Workers: rendering done; encoding is handled by the orchestrator
    # once all workers have finished their respective frame slices.
    if args._worker:
        print("      [worker] chunk complete \u2014 returning to orchestrator.")
        return 0

    # Save the middle frame as a standalone, uncompressed PNG next to
    # the future MP4 so the user can verify whether residual aliasing
    # is a rendering artefact (visible in the PNG) or only a video-
    # encoding artefact (visible in the MP4 but not in the PNG).
    if args.save_mid_png and total_frames > 0:
        try:
            import shutil
            mid_idx = total_frames // 2
            src = frames_dir / f"frame_{mid_idx:05d}.png"
            if src.exists():
                dst = out_path.with_name(out_path.stem + "_midframe.png")
                shutil.copy2(src, dst)
                print(f"      mid-frame PNG    : {dst}  "
                      f"(frame {mid_idx + 1}/{total_frames}, pre-encoding)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save mid-frame PNG: %s", exc)

    # ── 4. Assemble MP4 with ffmpeg ───────────────────────────────────────
    # Resolve codec / CRF / preset defaults.  In VR mode we default to
    # h265 (HEVC) at CRF 20 / preset slow — that's the sweet spot for
    # 4K-8K stereo equirect content on Meta Quest and YouTube VR.  For
    # mono we keep the legacy h264 / CRF 18 / medium combo.
    if args.codec == "auto":
        codec = "h265" if vr_mode != "off" else "h264"
    else:
        codec = args.codec
    if args.crf is not None:
        crf = args.crf
    else:
        crf = 20 if codec == "h265" else 18
    if args.preset is not None:
        preset = args.preset
    else:
        preset = "slow" if codec == "h265" else "medium"

    print(f"[4/4] Encoding MP4 → {out_path}")
    ok = _encode_mp4(
        frames_dir, out_path,
        fps=args.fps, codec=codec, crf=crf, preset=preset,
    )
    if not ok:
        print("      ffmpeg not available – PNG frames left in place.")
        print(f"      Run manually, e.g.:")
        print(f"        ffmpeg -framerate {args.fps} -i "
              f"{frames_dir}/frame_%05d.png -c:v lib{codec.replace('h','x')} "
              f"-pix_fmt yuv420p -crf {crf} -preset {preset} {out_path}")
        return 0

    if not args.keep_frames:
        for f in frames_dir.glob("frame_*.png"):
            f.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass
        print("      cleaned up intermediate frames.")

    # ── 5. Spatial-media metadata injection (VR only) ─────────────────────
    if vr_mode != "off":
        print(f"[5/5] Tagging spatial-media metadata ({vr_mode}, SBS) …")
        tagged_ok = _tag_spatial_media(out_path, vr_mode)
        if tagged_ok:
            print("      MP4 is now flagged as stereoscopic equirectangular "
                  "(left/right).  Ready for YouTube VR & Meta Quest.")
        else:
            print("      Could not inject metadata – the MP4 is still a "
                  "regular SBS file but won't auto-detect as VR.")

    # ── Final report: size + integrity sanity check ───────────────────────
    _analyse_output_mp4(out_path, total_frames=total_frames,
                        fps=args.fps, vr_mode=vr_mode)

    print("Done.")
    return 0


def _run_preview(mesh, path_line, track, *, width, height, fps) -> int:
    """Open an interactive window and animate through ``track``.

    Closing the window stops playback.  Camera + actor are driven the
    same way as in the off-screen renderer, so a successful preview
    guarantees the recorded MP4 will match.
    """
    import vedo
    plt = vedo.Plotter(
        title="Marvel · fly-through preview",
        bg=BACKGROUND,
        axes=0,
        size=(width, height),
    )
    plt.add(mesh)
    if path_line is not None:
        plt.add(path_line)
    plt.show(interactive=False, resetcam=True)

    actor = getattr(mesh, "actor", None) or mesh
    total = len(track)
    period = 1.0 / max(fps, 1)

    print(f"[3/3] Playing {total} frames at {fps} fps in a window … "
          f"(close to stop)")
    t_start = time.time()
    try:
        for i, state in enumerate(track):
            tick = time.time()
            _apply_actor_state(actor, state["actor"])
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, state["camera"])
                renderer.ResetCameraClippingRange()
                _nr, _fr = cam.GetClippingRange()
                cam.SetClippingRange(min(_nr, 2.0), _fr)
            plt.render()
            _print_progress(i, total, t_start)
            dt = time.time() - tick
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        sys.stdout.write("\n  preview interrupted\n")
    sys.stdout.write("\n")

    print("Preview finished.  Window stays open; close it to exit.")
    plt.interactive()
    plt.close()
    return 0


def _analyse_output_mp4(
    mp4_path: Path, *, total_frames: int, fps: int, vr_mode: str,
) -> None:
    """Print a short post-encoding summary of the produced MP4.

    Flags suspicious outputs (e.g. a tiny file whose every frame compresses
    to a near-empty P-frame — a tell-tale sign of static content where a
    real animation was expected).
    """
    if not mp4_path.exists():
        print("      [analysis] MP4 file not found.")
        return

    size_bytes = mp4_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    expected_dur = total_frames / max(fps, 1)
    avg_bitrate_kbps = (size_bytes * 8 / 1000.0) / max(expected_dur, 0.001)

    print("─" * 72)
    print(f"  Output analysis  : {mp4_path}")
    print(f"  File size        : {size_bytes:,} bytes "
          f"({size_mb:.2f} MiB)")
    print(f"  Frames encoded   : {total_frames}  "
          f"({expected_dur:.2f}s at {fps} fps)")
    print(f"  Average bitrate  : {avg_bitrate_kbps:,.0f} kbps")

    # ffprobe (if available) for an authoritative readout.
    try:
        import shutil
        import subprocess
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            r = subprocess.run(
                [ffprobe, "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries",
                 "stream=width,height,nb_frames,r_frame_rate,bit_rate",
                 "-of", "default=noprint_wrappers=1",
                 str(mp4_path)],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0 and r.stdout.strip():
                print("  ffprobe          :")
                for line in r.stdout.strip().splitlines():
                    print(f"    {line}")
    except Exception:  # noqa: BLE001
        pass

    # Heuristic: an obviously-too-small file usually means every frame
    # compressed to a near-empty P-frame (i.e. all frames identical).
    threshold_kbps_per_frame = 50.0 if vr_mode != "off" else 20.0
    avg_kbps_per_frame = avg_bitrate_kbps / max(fps, 1)
    if avg_kbps_per_frame < threshold_kbps_per_frame:
        print("  ⚠  WARNING: bitrate is unusually low "
              f"({avg_kbps_per_frame:.1f} kbps/frame, "
              f"expected ≥ {threshold_kbps_per_frame:.0f}).")
        print("     This typically means every encoded frame is identical "
              "to the first one.")
        print("     Most likely cause: the panoramic render pass isn't "
              "actually refreshing between frames.  Re-run with "
              "--keep-frames and compare a few PNGs to confirm.")
    print("─" * 72)


def _tag_spatial_media(mp4_path: Path, vr_mode: str) -> bool:
    """Inject Google Spatial-Media v2 metadata into an MP4 in place.

    Adds the ``Spherical Video V2`` atoms so YouTube, Meta Quest, DeoVR &
    co. auto-detect the file as a VR video.

    * ``vr360`` → projection ``equirectangular``, stereo ``left-right``,
      no crop (full sphere).
    * ``vr180`` → projection ``equirectangular``, stereo ``left-right``,
      with ``bounds`` cropping to the forward hemisphere
      (top=0, bottom=0, left=25 %, right=25 %).  This is the convention
      used by VR180 SBS content packed into a 2:1 equirect canvas.
    """
    try:
        from spatialmedia import metadata_utils
    except ImportError:
        print("      spatialmedia not installed; skipping tagging.")
        print("      pip install 'git+https://github.com/google/spatial-media.git'")
        return False

    mp4_path = Path(mp4_path).resolve()
    tagged_path = mp4_path.with_suffix(".tagged.mp4")

    metadata = metadata_utils.Metadata()
    metadata.stereo_mode = "left-right"
    metadata.projection = metadata_utils.generate_spherical_xml(
        stereo="left-right",
    )

    try:
        metadata_utils.inject_metadata(
            str(mp4_path), str(tagged_path), metadata, print,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      spatial-media injection failed: {exc}")
        if tagged_path.exists():
            tagged_path.unlink()
        return False

    if not tagged_path.exists() or tagged_path.stat().st_size < 1024:
        print("      spatial-media produced no output.")
        if tagged_path.exists():
            tagged_path.unlink()
        return False

    tagged_path.replace(mp4_path)
    return True


def _encode_mp4(
    frames_dir: Path,
    out_path: Path,
    *,
    fps: int,
    codec: str = "h264",
    crf: int = 18,
    preset: str = "medium",
) -> bool:
    """Try to encode an MP4 from ``frame_*.png`` files.

    Parameters
    ----------
    codec:
        ``"h264"`` (libx264) or ``"h265"`` (libx265, a.k.a. HEVC).  HEVC
        is recommended for 4K+ VR content — it ships with the ``hvc1``
        tag so Meta Quest & QuickTime accept it.
    crf:
        Constant-rate-factor quality (lower = better, larger file).
        Sensible range: 0 (lossless) – 51.  Defaults: 18 for h264 mono,
        20 for h265 VR.
    preset:
        Encoding preset; slower presets yield smaller files at the same
        CRF.
    """
    import shutil
    import subprocess

    ffmpeg_bin: str | None = None
    try:
        import imageio_ffmpeg  # type: ignore
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_bin = shutil.which("ffmpeg")

    if not ffmpeg_bin:
        return False

    if codec == "h265":
        encoder = "libx265"
        extra = ["-tag:v", "hvc1"]  # required for Meta Quest / QuickTime
    else:
        encoder = "libx264"
        extra = []

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        *extra,
        str(out_path),
    ]
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    print(f"      codec={encoder}  crf={crf}  preset={preset}"
          + ("  (hvc1 tag)" if encoder == "libx265" else ""))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed (exit %d). Rerun with -v for details.",
                     exc.returncode)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
