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
* Frame rate: 25 fps.
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
    DEFAULT_INPUT_PATH,
    DEFAULT_LEVEL,
    DEFAULT_MESH_CACHE_PATH,
    DEFAULT_MP4_DIR,
    DEFAULT_POSITIONS_DIR,
    DEFAULT_SMOOTH_ITER,
    _load_or_build_mesh,
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
FPS = 25
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
PATH_TUBE_RADIUS_FRAC = 0.0035
# Fraction of the camera-to-focal distance used to offset the line upward
# along the world-up axis.  Just enough to make it visible without
# leaving the frame.
PATH_VERTICAL_OFFSET_FRAC = 0.03


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
    p.add_argument("--vr", choices=["off", "vr180", "vr360"], default="off",
                   help="Render as stereo equirectangular side-by-side (SBS) "
                        "for VR headsets.  'vr180' = forward hemisphere "
                        "(recommended for fly-throughs), 'vr360' = full "
                        "sphere.  Each frame renders twice (left + right "
                        "eye) and roughly 6× slower than mono per eye "
                        "(panoramic cubemap projection).")
    p.add_argument("--ipd-frac", type=float, default=VR_IPD_FRAC,
                   help="Interocular distance as a fraction of the camera→"
                        "focal-point distance (0.03 = subtle, 0.05 = "
                        "standard, 0.08 = strong relief).")
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
                   help="Render only the first ~4 seconds of footage "
                        "(truncates the track after fps×4 frames).  "
                        "Useful to validate a VR rendering pipeline before "
                        "committing to a multi-minute encode.")

    p.add_argument("--debugdisplayindexframes", action="store_true",
                   help="Flat-mode only: burn an indicative keyframe index "
                        "near the bottom of every frame.  The index is a "
                        "float relative to the saved control points "
                        "(0 = first 'Save pos', N-1 = last).  Useful to "
                        "tell the assistant where in the trajectory to "
                        "trigger a specific behaviour.  Ignored when "
                        "--vr is not 'off'.")

    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    # ── UI-state replay (matches the viewer's saved per-keyframe state)──
    p.add_argument("--alt-mesh-cache", default=None,
                   help="Path to a cached .vtk mesh for the alternate "
                        "('All watered tissues') view.  When provided AND "
                        "the positions JSON contains a 'mesh_all' view_mode "
                        "at some keyframe, the movie will swap to it.")
    p.add_argument("--pillars-cache", default=None,
                   help="Path to a cached .vtk overlay mesh ('Pillars').  "
                        "When provided AND any keyframe sets "
                        "pillars_visible=true, the overlay is shown / "
                        "styled / hue-rotated accordingly.")
    p.add_argument("--membranes-vtp-cache", default=None,
                   help="Path to the cached water-membranes .vtp.  When "
                        "provided AND keyframes set membranes_visible=true, "
                        "the descending-water animation is replayed.")
    p.add_argument("--membranes-meta-cache", default=None,
                   help="JSON sidecar with the membranes n_steps / phases. "
                        "Required alongside --membranes-vtp-cache.")
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


def _interp_vec(values: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Interpolate (n_ctrl, d) → (len(t_out), d) via cubic spline.

    ``t_out`` is expected in [0, 1] (control points are placed uniformly
    on that interval).  Falls back to linear if scipy is missing or fewer
    than 4 control points are available.
    """
    n_ctrl, d = values.shape
    n_frames = len(t_out)
    if n_ctrl == 1:
        return np.tile(values, (n_frames, 1))

    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)

    if n_ctrl >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(t_ctrl, values, bc_type="natural", axis=0)
            return cs(t_out)
        except ImportError:
            pass

    out = np.empty((n_frames, d), dtype=float)
    for j in range(d):
        out[:, j] = np.interp(t_out, t_ctrl, values[:, j])
    return out


def _interp_scalar(values: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    return _interp_vec(values.reshape(-1, 1), t_out).ravel()


def _interp_orientation(euler_deg: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Slerp-interpolate orientations expressed as XYZ Euler angles (deg).

    ``t_out`` is in [0, 1] (control orientations evenly spaced).
    Linearly interpolating Euler angles produces visibly wrong intermediate
    orientations as soon as more than one axis is non-zero, hence the
    detour through quaternions.
    """
    n_ctrl = len(euler_deg)
    n_frames = len(t_out)
    if n_ctrl == 1:
        return np.tile(euler_deg, (n_frames, 1))

    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError:
        return _interp_vec(euler_deg, t_out)

    t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
    rots = Rotation.from_euler("xyz", euler_deg, degrees=True)
    slerp = Slerp(t_ctrl, rots)
    return slerp(t_out).as_euler("xyz", degrees=True)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


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
) -> List[dict]:
    """Return one combined camera+actor state per frame.

    Parameters
    ----------
    control_points:
        Loaded JSON entries (one per "Save pos" click).
    frames_per_segment:
        Frames between two consecutive control points at constant speed.
    hold_frames:
        How many frozen frames to append at the end of the movie.
    ease_segments:
        Number of segments (counted from the end) over which to ease-out.
        Set to 0 to disable easing.
    ease_strength:
        Power used in the ease-out curve on the very last segment.  Values
        > 1 produce stronger deceleration.
    """
    n_ctrl = len(control_points)
    if n_ctrl < 1:
        raise ValueError("Need at least one control point.")

    t_out = _build_time_axis(
        n_ctrl, frames_per_segment,
        hold_frames=hold_frames,
        ease_segments=min(ease_segments, max(0, n_ctrl - 1)),
        ease_strength=ease_strength,
    )

    positions      = _stack_camera_field(control_points, "position")
    focals         = _stack_camera_field(control_points, "focal_point")
    view_ups       = _stack_camera_field(control_points, "view_up")
    view_angles    = np.array([p["camera"]["view_angle"] for p in control_points])
    parallel_scale = np.array([p["camera"]["parallel_scale"] for p in control_points])

    pos_track = _interp_vec(positions, t_out)
    foc_track = _interp_vec(focals,    t_out)
    up_track  = _interp_vec(view_ups,  t_out)
    ang_track = _interp_scalar(view_angles,    t_out)
    psc_track = _interp_scalar(parallel_scale, t_out)

    act_pos   = _stack_actor_field(control_points, "position",    [0.0, 0.0, 0.0])
    act_scale = _stack_actor_field(control_points, "scale",       [1.0, 1.0, 1.0])
    act_orig  = _stack_actor_field(control_points, "origin",      [0.0, 0.0, 0.0])
    act_orie  = _stack_actor_field(control_points, "orientation", [0.0, 0.0, 0.0])

    act_pos_track   = _interp_vec(act_pos,   t_out)
    act_scale_track = _interp_vec(act_scale, t_out)
    act_orig_track  = _interp_vec(act_orig,  t_out)
    act_orie_track  = _interp_orientation(act_orie, t_out)

    track: List[dict] = []
    for i in range(len(t_out)):
        pos = pos_track[i]
        foc = foc_track[i]
        up  = up_track[i]
        # Re-orthonormalise view_up so the spline doesn't roll the camera.
        look = _normalize(foc - pos)
        up = up - look * float(np.dot(up, look))
        up = _normalize(up)
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
            # Float "keyframe index" in [0, n_ctrl-1]: 0 = first saved
            # "Save pos" click, n_ctrl-1 = last.  Useful for debug overlays
            # so the user can tell us "trigger something at frame index 4.2".
            "keyframe_index": float(t_out[i]) * max(1, n_ctrl - 1),
        })
    return track


def _apply_camera_state(cam, state: dict) -> None:
    cam.SetPosition(*state["position"])
    cam.SetFocalPoint(*state["focal_point"])
    cam.SetViewUp(*state["view_up"])
    cam.SetViewAngle(state["view_angle"])
    cam.SetParallelScale(state["parallel_scale"])


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


def _install_scene_lights(plt) -> list:
    """Replace VTK's camera-following headlight by a small rig of
    scene-fixed omni lights placed around the volume bounds.

    Why this matters for VR cubemap rendering: the panoramic pass
    renders 6 cube faces from the same camera position but with 6
    different orientations.  A camera-attached headlight is *relative*
    to the camera basis, so its world position is recomputed per face
    -> each face sees slightly different lighting and the cube seams
    become visible after the equirectangular remap.

    Scene-fixed lights are evaluated in world coordinates, identical
    for every face, so the 6 renders share the exact same lighting
    field and seam together perfectly.  Returns the created
    :class:`vtkLight` objects (caller keeps them alive).
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
        # Compute scene bounds (fallback to a unit cube around the camera
        # focal point if the renderer has no actors yet).
        try:
            bounds = renderer.ComputeVisiblePropBounds()
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            diag = float(np.linalg.norm(
                [xmax - xmin, ymax - ymin, zmax - zmin]
            ))
            if not np.isfinite(diag) or diag <= 0:
                raise ValueError("bad bounds")
        except Exception:  # noqa: BLE001
            cx = cy = cz = 0.0
            diag = 1.0
        radius = max(diag, 1.0) * 1.5
        # 6 omni point-lights on the ±X, ±Y, ±Z faces of a bounding
        # cube (slightly different intensities for soft directionality
        # while keeping seams invisible).  Plus a faint pure ambient
        # via a 7th very-dim light at the centre.
        offsets = [
            (+1, 0, 0, 0.50),
            (-1, 0, 0, 0.45),
            (0, +1, 0, 0.55),
            (0, -1, 0, 0.50),
            (0, 0, +1, 0.50),
            (0, 0, -1, 0.45),
        ]
        for dx, dy, dz, intensity in offsets:
            lt = _vtkrc.vtkLight()
            lt.SetLightTypeToSceneLight()
            lt.SetPositional(False)  # directional/parallel -> identical on every cube face
            lt.SetPosition(cx + dx * radius,
                           cy + dy * radius,
                           cz + dz * radius)
            lt.SetFocalPoint(cx, cy, cz)
            lt.SetColor(1.0, 1.0, 1.0)
            lt.SetIntensity(intensity)
            renderer.AddLight(lt)
            created.append(lt)
        # Ambient fill (very low) — ensures concavities never go to pure
        # black, matching the cortex's "wet shiny" look across all faces.
        amb = _vtkrc.vtkLight()
        amb.SetLightTypeToSceneLight()
        amb.SetPositional(True)
        amb.SetPosition(cx, cy, cz)
        amb.SetFocalPoint(cx, cy + 1.0, cz)
        amb.SetColor(0.85, 0.90, 1.00)  # very slight cool tint
        amb.SetIntensity(0.10)
        renderer.AddLight(amb)
        created.append(amb)
    return created


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
            if val is not None:
                frame_state[k] = val
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
        membranes_actor=None,
        membranes_threshold=None,
        membranes_n_steps: int = 0,
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
        self.membranes_actor = membranes_actor
        self.membranes_threshold = membranes_threshold
        self.membranes_n_steps = int(membranes_n_steps)
        self._membranes_step = 0
        # Latched values so we don't re-apply identical state every frame.
        self._last: dict = {}
        # Fog / SSAO state mirrors viewer's ``_depth_state`` dict shape.
        self._depth = {
            "fog_on": False, "ssao_on": False,
            "ssao_keepalive": [], "prev_pass": {},
        }
        # Pillars colour bases — kept in sync with viewer.
        self._pillars_base_glow  = (0.20, 0.95, 0.75)
        self._pillars_base_solid = (0.05, 0.50, 0.45)

    # --- pillars -----------------------------------------------------
    @staticmethod
    def _hue_shift(rgb_0_1, shift):
        import colorsys
        h, l, s = colorsys.rgb_to_hls(*rgb_0_1)
        h = (h + shift) % 1.0
        return colorsys.hls_to_rgb(h, l, s)

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
            near, far = diag * 0.20, diag * 0.90
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
        # Movie cannot rebuild arrows on the fly (too expensive).  We map
        # arrows_* to mesh_bridges for visibility purposes.
        if self.alt_mesh is None and mode == "mesh_all":
            mode = "mesh_bridges"
        try:
            show_main = mode in ("mesh_bridges", "arrows_grid",
                                  "arrows_tracks")
            show_alt  = mode == "mesh_all"
            getattr(self.mesh, "actor", self.mesh).SetVisibility(
                1 if show_main else 0)
            if self.alt_mesh is not None:
                getattr(self.alt_mesh, "actor", self.alt_mesh).SetVisibility(
                    1 if show_alt else 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("view_mode swap failed: %s", exc)

    # --- entry point -------------------------------------------------
    def apply(self, ui_state: dict | None) -> None:
        if not ui_state:
            return
        if ui_state.get("view_mode") != self._last.get("view_mode"):
            vm = ui_state.get("view_mode") or "mesh_bridges"
            self._set_view_mode(vm)
            self._last["view_mode"] = ui_state.get("view_mode")
        # Pillars: any of the three triggers a re-apply.
        if (ui_state.get("pillars_visible")  != self._last.get("pillars_visible")
            or ui_state.get("pillars_style") != self._last.get("pillars_style")
            or ui_state.get("pillars_hue_shift") !=
               self._last.get("pillars_hue_shift")):
            self._apply_pillars(
                visible=bool(ui_state.get("pillars_visible", False)),
                style=str(ui_state.get("pillars_style") or "glow"),
                hue=float(ui_state.get("pillars_hue_shift") or 0.0),
            )
            self._last["pillars_visible"]   = ui_state.get("pillars_visible")
            self._last["pillars_style"]     = ui_state.get("pillars_style")
            self._last["pillars_hue_shift"] = ui_state.get("pillars_hue_shift")
        if ui_state.get("fog_on") != self._last.get("fog_on"):
            self._set_fog(bool(ui_state.get("fog_on")))
            self._last["fog_on"] = ui_state.get("fog_on")
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


def _apply_actor_state(actor, state: dict) -> None:
    actor.SetOrigin(*state["origin"])
    actor.SetPosition(*state["position"])
    actor.SetOrientation(*state["orientation"])
    actor.SetScale(*state["scale"])


def _world_up_from_track(track: Sequence[dict]) -> np.ndarray:
    """Pick a sensible world-up axis from the saved view_ups."""
    avg = np.mean([t["camera"]["view_up"] for t in track], axis=0)
    return _normalize(np.asarray(avg, dtype=float))


def _build_path_line(track: Sequence[dict], offset_frac: float,
                     *, as_tube: bool = False):
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
    offset = median_dist * offset_frac
    up = _world_up_from_track(track)
    line_pts = cam_positions + up * offset
    color = [c / 255.0 for c in PATH_COLOR]
    if as_tube:
        radius = median_dist * PATH_TUBE_RADIUS_FRAC
        obj = vedo.Tube(line_pts, r=radius, c=color, res=16)
    else:
        obj = vedo.Line(line_pts, c=color, lw=PATH_LINE_WIDTH)
    try:
        obj.lighting("off")
    except Exception:  # noqa: BLE001
        pass
    obj.name = "cable_car_path"
    return obj


# ─────────────────────────── stereo VR helpers ───────────────────────────────


def _setup_panoramic_pass(plt, angle_deg: float, cube_resolution: int):
    """Attach a :class:`vtkPanoramicProjectionPass` to every renderer of
    ``plt`` so each render produces an equirectangular image.

    The panoramic pass MUST be chained with a proper delegate pass
    sequence (Camera → Sequence(Lights, Opaque)).  Without that delegate,
    the pass appears to silently render nothing fresh — every output
    frame ends up identical to the very first one, which manifests as a
    tiny MP4 with absurdly low bitrate (every frame compresses to a
    near-empty P-frame).  See VTK's standard panoramic pass example.

    Returns a list of pass objects to keep alive (they must outlive
    the renderer, otherwise VTK frees them mid-render).
    """
    try:
        from vtkmodules.vtkRenderingOpenGL2 import (
            vtkCameraPass,
            vtkLightsPass,
            vtkOpaquePass,
            vtkPanoramicProjectionPass,
            vtkRenderPassCollection,
            vtkSequencePass,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "vtkPanoramicProjectionPass and its delegate-pass dependencies "
            "are unavailable in this VTK build; stereo VR rendering requires "
            "VTK ≥ 9.0 with the OpenGL2 backend."
        ) from exc

    keepalive: list = []
    for renderer in plt.renderers:
        # Inner delegate: lights + opaque geometry only (no translucent /
        # volumetric — we don't need them and they slow the cubemap
        # render down 6×).
        lights = vtkLightsPass()
        opaque = vtkOpaquePass()
        passes = vtkRenderPassCollection()
        passes.AddItem(lights)
        passes.AddItem(opaque)
        seq = vtkSequencePass()
        seq.SetPasses(passes)
        cam_pass = vtkCameraPass()
        cam_pass.SetDelegatePass(seq)
        pano = vtkPanoramicProjectionPass()
        pano.SetCubeResolution(int(cube_resolution))
        if hasattr(pano, "SetAngle"):
            pano.SetAngle(float(angle_deg))
        if hasattr(pano, "SetProjectionType"):
            pano.SetProjectionType(0)  # 0 = equirectangular
        if hasattr(pano, "SetInterpolate"):
            pano.SetInterpolate(True)
        pano.SetDelegatePass(cam_pass)
        renderer.SetPass(pano)
        keepalive.extend([pano, cam_pass, seq, passes, opaque, lights])
    return keepalive


def _eye_offset(state: dict, ipd_frac: float) -> np.ndarray:
    """Return the world-space half-IPD vector pointing to the camera's right.

    Both eyes shift their position AND focal point by ±this vector to
    yield true parallel stereo (no toe-in convergence), which is what
    VR equirectangular content expects.
    """
    pos = np.asarray(state["camera"]["position"], dtype=float)
    foc = np.asarray(state["camera"]["focal_point"], dtype=float)
    up  = np.asarray(state["camera"]["view_up"],     dtype=float)
    look = _normalize(foc - pos)
    right = _normalize(np.cross(look, up))
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
    vr_mode = args.vr
    vr_angle = {"vr180": 180.0, "vr360": 360.0}.get(vr_mode)
    vr_user_set_size = any(f in (argv or sys.argv[1:])
                           for f in ("--width", "--height"))
    if vr_mode != "off" and not vr_user_set_size:
        if vr_mode == "vr180":
            args.width, args.height = VR180_SBS_W, VR180_SBS_H
        else:  # vr360
            args.width, args.height = VR360_SBS_W, VR360_SBS_H

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
    n_ctrl = len(control_points)

    # Output convention:
    #   ./positions/positions_<stamp>.json   ← input control points
    #   ./mp4/positions_<stamp>.mp4          ← rendered movie
    # ``--output`` overrides the path entirely.
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        mp4_dir = DEFAULT_MP4_DIR
        suffix = f"_{vr_mode}" if vr_mode != "off" else ""
        out_path = mp4_dir / (positions_path.stem + suffix + ".mp4")
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
        print(f"  IPD fraction     : {args.ipd_frac}  "
              f"(of cam→focal distance)")
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

    # ── 1. Build the mesh once ─────────────────────────────────────────────
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

    # ── 2. Interpolate camera + actor track ───────────────────────────────
    print("[2/4] Interpolating camera + actor track …")
    track = build_track(
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
        keep = min(total_frames, args.fps * 4)
        track = track[:keep]
        total_frames = len(track)
        print(f"      [stresstest] truncated to first {total_frames} frames "
              f"({total_frames / args.fps:.2f}s)")

    # In VR mode, lock the orientation to the first keyframe so the
    # spline only carries the translation component.  Any orientation
    # change in a headset is interpreted as the world tilting around
    # the user → reliable motion-sickness trigger.  The user can still
    # look around freely with their head; we just keep the *virtual
    # camera basis* steady.
    if vr_mode != "off":
        _lock_orientation_for_vr(track)
        print("      [VR] orientation locked to first keyframe "
              "(only translations are applied)")

    # ── 2bis. UI-state replay track ──────────────────────────────────
    if args.ignore_ui_state:
        ui_track: List[dict | None] = [None] * len(track)
    else:
        # Re-derive t_out so we don't have to expose it from build_track.
        _t_out_for_ui = _build_time_axis(
            len(control_points), frames_per_segment,
            hold_frames=hold_frames,
            ease_segments=min(args.ease_segments,
                              max(0, len(control_points) - 1)),
            ease_strength=args.ease_strength,
        )
        ui_track = _build_ui_track(
            control_points, _t_out_for_ui,
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
        else:
            print("      [ui_state] no UI state found in positions JSON "
                  "— skipping replay")
    # Attach the per-frame ui_state to track entries for convenient access.
    for _entry, _ui in zip(track, ui_track):
        if _ui is not None:
            _entry["ui_state"] = _ui

    path_line = None
    if show_path:
        path_line = _build_path_line(
            track, PATH_VERTICAL_OFFSET_FRAC,
            as_tube=(vr_mode != "off"),
        )
        kind = "tube" if vr_mode != "off" else "line"
        print(f"      built green path {kind} over {total_frames} samples")

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
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    import vedo
    plt = vedo.Plotter(
        title="Marvel · Wat_Norm_Cortex (movie)",
        bg=BACKGROUND,
        axes=0,
        offscreen=True,
        size=(eye_w, eye_h),
    )
    plt.add(mesh)
    if alt_mesh is not None:
        plt.add(alt_mesh)
    if pillars_mesh is not None:
        plt.add(pillars_mesh)
    if membranes_actor is not None:
        try:
            plt.renderer.AddActor(membranes_actor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not add membranes actor: %s", exc)
    if path_line is not None:
        plt.add(path_line)
    plt.show(interactive=False, resetcam=True)

    # UI-state replay bundle (holds actors + caches diff state across
    # frames so we don't reapply unchanged values).
    fog_actors_bundle = [a for a in (mesh, alt_mesh, pillars_mesh)
                         if a is not None]
    ui_bundle = _UiReplayBundle(
        mesh=mesh,
        alt_mesh=alt_mesh,
        pillars_mesh=pillars_mesh,
        fog_actors=fog_actors_bundle,
        plt=plt,
        membranes_actor=membranes_actor,
        membranes_threshold=membranes_threshold,
        membranes_n_steps=membranes_n_steps,
    )

    # Attach the panoramic projection pass once (if VR).  Kept in a local
    # variable so the Python object isn't garbage-collected mid-render.
    pano_passes = None
    vr_lights = None
    if vr_mode != "off":
        # Scene-fixed lights BEFORE the panoramic pass: must be present
        # in the renderer at the time vtkLightsPass executes (which is
        # every cube-face render).
        vr_lights = _install_scene_lights(plt)
        print(f"      [VR] installed {len(vr_lights)} scene-fixed lights "
              "(headlight disabled)")
        pano_passes = _setup_panoramic_pass(
            plt, vr_angle, args.vr_cube_resolution,
        )

    # Optional orthogonal-slice locator panel — baked into every frame.
    ortho_overlay = None
    if not args.no_ortho_panel:
        raw_path = (
            Path(args.raw).expanduser().resolve()
            if args.raw
            else (Path(args.input).expanduser().resolve().parent / "Raw.tif")
        )
        if raw_path.exists():
            try:
                from marvel_view.visualization.ortho_panel import (
                    OrthoPanelOverlay, load_raw_volume,
                )
                raw_volume = load_raw_volume(raw_path)
                # In SBS VR the per-eye image is the full plotter canvas,
                # so the same normalized viewport works for both eyes.
                ortho_overlay = OrthoPanelOverlay(
                    plt, raw_volume,
                    viewport=(0.0, 0.5, 0.4, 1.0),
                    cell_pixels=320,
                )
                print(f"      ortho panel attached  (Raw={raw_path.name}, "
                      f"shape={raw_volume.shape})")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to attach ortho panel: %s", exc)
        else:
            logger.info("Raw volume not found (%s) – ortho panel disabled.",
                        raw_path)

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
    left_eye_png  = frames_dir / "_eye_left.png"
    right_eye_png = frames_dir / "_eye_right.png"

    t_start = time.time()
    for i, state in enumerate(track):
        _apply_actor_state(actor, state["actor"])
        # Per-frame UI-state replay (visibility swap, pillars, fog/SSAO).
        ui_bundle.apply(state.get("ui_state"))

        if ortho_overlay is not None:
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
            ortho_overlay.update(mono_pos, mono_dir)

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
            plt.render()
            plt.screenshot(str(frames_dir / f"frame_{i:05d}.png"))
        else:
            offset = _eye_offset(state, args.ipd_frac)
            left_state  = _shift_camera_state(state, -offset)
            right_state = _shift_camera_state(state,  offset)

            # Left eye — render & write to disk (same code path as mono).
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, left_state["camera"])
                renderer.ResetCameraClippingRange()
            plt.render()
            plt.screenshot(str(left_eye_png))

            # Right eye.
            for renderer in plt.renderers:
                cam = renderer.GetActiveCamera()
                _apply_camera_state(cam, right_state["camera"])
                renderer.ResetCameraClippingRange()
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
            imageio.imwrite(str(frames_dir / f"frame_{i:05d}.png"), sbs)

        _print_progress(i, total_frames, t_start)
    sys.stdout.write("\n")

    # Clean up the per-eye scratch PNGs.
    for p in (left_eye_png, right_eye_png):
        if p.exists():
            p.unlink()

    plt.close()
    del pano_passes  # explicit – release the panoramic pass references
    print(f"      rendered {total_frames} frames in "
          f"{_format_eta(time.time() - t_start)}")

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
    if vr_mode == "vr180":
        # Crop bounds expressed as 32-bit fractions of UINT32_MAX:
        # left/right each 25 % so the visible content spans 180° of the
        # 360°-wide equirect canvas (per eye).
        quarter = int(0.25 * 0xFFFFFFFF)
        metadata.bounds = [0, 0, quarter, quarter]  # top, bottom, left, right
        metadata.projection = metadata_utils.generate_spherical_xml(
            stereo="left-right",
            crop=None,
        )
    else:  # vr360
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
