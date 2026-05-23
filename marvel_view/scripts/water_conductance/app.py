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
# File now lives inside the ``water_conductance`` subpackage so the repo
# root is one level deeper than before.
_REPO_ROOT = Path(__file__).resolve().parents[3]
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

# ── extracted modules ────────────────────────────────────────────────────────
# Constants, CLI, pipeline (build/cache helpers) and styling helpers were
# split into sibling modules to keep this file focused on the viewer.
from .constants import *  # noqa: F401,F403  (re-exported for backward compat)
from .constants import _SHADING_MODES  # not exported by * (leading underscore)
from .cli import parse_args  # noqa: F401
from .pipeline import (  # noqa: F401
    _build_mesh,
    _parse_central_axis,
    _build_density_facet_scalars,
    _load_or_build_mesh,
    _build_arrow_field,
    _load_or_build_arrows,
    _build_crown_dijkstra_tracks,
    _build_tracks_polydata,
    _write_tracks_vtp,
    _write_tracks_arrows_vtp,
    _load_or_build_crown_tracks,
    _load_or_build_overlay_mesh,
    _load_or_build_membranes,
    _load_lames,
    _build_lames_step_polydatas,
)
from .styling import (  # noqa: F401
    _set_lighting,
    _hue_shifted_rgb,
    _style_mesh,
    _set_shading,
)

# Re-exported names also need to be available as module attributes used
# inside this file's _attach_controls / main bodies.  The ``from
# .constants import *`` above already takes care of that for the public
# DEFAULT_* / colour / status constants.

# (constants extracted to .constants; see `from .constants import *` above)

# (parse_args extracted to .cli; see `from .cli import parse_args` above)

# (pipeline / styling helpers extracted to .pipeline and .styling)

def _attach_controls(
    plt, mesh,
    arrows_data=None,
    ortho_overlay=None,
    alt_mesh=None,
    pillars_mesh=None,
    tracks_data=None,
    density_scalars=None,
    membranes_data=None,
    lames_data=None,
    mask_overlay_meshes=None,
    wind_data=None,
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
            "", pos=(0.52, 0.972), s=1.1, c="white", bg="black", alpha=0.65,
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
        now = _time.time()
        # Clear FPS display if no tick has fired for more than 2 s.
        if now - _fps_state["last_update"] > 2.0:
            _fps_t = _fps_state["text2d"]
            if _fps_t is not None:
                try:
                    _fps_t.text("")
                except Exception:  # noqa: BLE001
                    pass
        # Only clear status if no newer status message superseded this timer
        # and the expiry has actually passed.
        if now + 0.05 < status_state["expire_at"]:
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

    # ── FPS counter (top-left) — updated at most once per second ────────
    _fps_state = {
        "text2d":      None,
        "last_t":      0.0,
        "n_frames":    0,
        "last_update": 0.0,
    }
    try:
        _fps_state["text2d"] = _vedo_mod.Text2D(
            "", pos=(0.36, 0.972), s=1.0, c="yellow", bg="black", alpha=0.65,
        )
        plt.add(_fps_state["text2d"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create FPS Text2D: %s", exc)

    def _fps_tick() -> None:
        """Lightweight FPS counter; call once per animation tick."""
        _fps_state["n_frames"] += 1
        now = _time.time()
        dt = now - _fps_state["last_t"]
        if dt < 1.0:
            return
        fps = _fps_state["n_frames"] / dt
        _fps_state["n_frames"] = 0
        _fps_state["last_t"] = now
        _fps_state["last_update"] = now
        t = _fps_state["text2d"]
        if t is None:
            return
        try:
            t.text(f"▶ {fps:.1f} fps")
        except Exception:  # noqa: BLE001
            pass
        # Schedule auto-clear after 2.5 s of silence.
        _iren_fps = getattr(plt, "interactor", None)
        if _iren_fps is not None:
            try:
                _iren_fps.CreateOneShotTimer(2500)
            except Exception:  # noqa: BLE001
                pass

    # ── Render coalescer ───────────────────────────────────────────────
    # Several subsystems (membranes, lames, wind O2/CH4, plane, ship)
    # used to call ``plt.render()`` directly from their own repeating
    # TimerEvent observers.  When two or more were active in parallel,
    # multiple renders happened per ~40 ms slot:
    #
    #   * Reported FPS inflated (25 → 37, 50, ...)  — each render bumped
    #     the same shared FPS counter, regardless of which timer caused it.
    #   * CPU/GPU burned doing duplicate work.
    #   * The interactor's event loop was starved of time to process
    #     mouse / keyboard events → laggy controls.
    #
    # Now every subsystem tick only updates its state and calls
    # ``_request_render()``.  A single low-priority TimerEvent observer
    # flushes ONE ``plt.render()`` per ``_FRAME_INTERVAL_S`` (40 ms ≈
    # 25 fps), regardless of how many subsystem timers fire.
    _FRAME_INTERVAL_S: float = 0.016  # ~60 fps render cap; machine adapts
                                       # (slow renders naturally lower fps)

    _render_state = {
        "dirty":  False,
        "last_t": 0.0,
    }

    def _request_render() -> None:
        """Mark the scene dirty; the master tick will render at most once
        per :data:`_FRAME_INTERVAL_S`."""
        _render_state["dirty"] = True

    def _coalesce_render_tick() -> None:
        if not _render_state["dirty"]:
            return
        now = _time.time()
        if now - _render_state["last_t"] < _FRAME_INTERVAL_S:
            return
        _render_state["dirty"] = False
        _render_state["last_t"] = now
        _fps_tick()
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    # ── Master scheduler ───────────────────────────────────────────────
    # Single repeating timer (16 ms) drives every periodic subsystem
    # (plane input + integration, ship integration, lames / membranes /
    # wind frame swaps, render flush).  Subsystems are plain Python
    # callables registered via ``_master_register``; their cadence is
    # honoured via per-sub accumulators.
    #
    # Why one timer instead of N:
    #   * Eliminates cross-subsystem TimerEvent priority races (every
    #     observer used to fire on every other timer's events, hence
    #     the per-tick ``GetTimerId()`` guards).
    #   * Makes ordering deterministic (input → sim → render every tick).
    #   * Lets the plane subsystem poll keyboard state at a fixed
    #     cadence regardless of which other subsystem is also active.
    #   * Mirrors the standard game-loop architecture.
    #
    # Why on-demand (start/stop) rather than permanent:
    #   * ``vtkInteractorStyleTrackballCamera`` reapplies the last
    #     mouse delta on every TimerEvent, producing a visible
    #     "inertia" feel during click-drag.  Stopping the timer when
    #     no subsystem needs it restores the vanilla trackball feel
    #     in Spaceship / Custom modes.
    _MASTER_TICK_MS: int = 16
    _master = {
        "iren":     None,
        "timer_id": None,
        "subs":     [],   # list[dict]
    }

    def _master_register(name: str, fn, period_ms: int,
                         enabled: bool = False) -> dict:
        sub = {
            "name":     str(name),
            "fn":       fn,
            "period_s": max(0.001, float(period_ms) / 1000.0),
            "accum_s":  0.0,
            "enabled":  bool(enabled),
            "last_t":   None,
        }
        _master["subs"].append(sub)
        return sub

    def _master_refresh_timer() -> None:
        """Start the master timer if any sub is enabled; stop it when all
        are idle (preserves trackball feel in non-Avion modes)."""
        iren = _master["iren"]
        if iren is None:
            return
        any_on = any(s["enabled"] for s in _master["subs"])
        if any_on and _master["timer_id"] is None:
            try:
                _master["timer_id"] = iren.CreateRepeatingTimer(
                    int(_MASTER_TICK_MS)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Master timer could not start: %s", exc)
        elif (not any_on) and _master["timer_id"] is not None:
            try:
                iren.DestroyTimer(_master["timer_id"])
            except Exception:  # noqa: BLE001
                pass
            _master["timer_id"] = None

    def _master_set_enabled(sub: dict, on: bool) -> None:
        was = sub["enabled"]
        sub["enabled"] = bool(on)
        if on:
            sub["last_t"] = None
            sub["accum_s"] = 0.0
        if was != sub["enabled"]:
            _master_refresh_timer()

    def _master_tick(obj=None, _ev=None) -> None:
        # Only react to *our* timer (other observers — status hide,
        # FPS one-shot — share the iren's TimerEvent stream).
        _exp = _master["timer_id"]
        if _exp is not None and obj is not None:
            try:
                if obj.GetTimerId() != _exp:
                    return
            except Exception:  # noqa: BLE001
                pass
        now = _time.time()
        for sub in _master["subs"]:
            if not sub["enabled"]:
                sub["last_t"] = None
                sub["accum_s"] = 0.0
                continue
            if sub["last_t"] is None:
                sub["last_t"] = now
                sub["accum_s"] = 0.0
                # Run once on (re-)enable so the first frame is reactive.
                try:
                    sub["fn"]()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Master sub '%s' first-call failed: %s",
                                   sub["name"], exc)
                continue
            dt = now - sub["last_t"]
            sub["last_t"] = now
            if dt < 0.0 or dt > 1.0:
                sub["accum_s"] = 0.0
                continue
            sub["accum_s"] += dt
            if sub["accum_s"] + 1e-9 >= sub["period_s"]:
                # Cap credit at one period so a long stall doesn't
                # cause a burst of catch-up calls.
                sub["accum_s"] = min(
                    sub["accum_s"] - sub["period_s"], sub["period_s"]
                )
                try:
                    sub["fn"]()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Master sub '%s' failed: %s",
                                   sub["name"], exc)
        # Render coalesce always at end (cheap if !dirty).
        _coalesce_render_tick()

    _master["iren"] = getattr(plt, "interactor", None)
    if _master["iren"] is not None:
        try:
            _master["iren"].AddObserver("TimerEvent", _master_tick)
            logger.info(
                "Master scheduler attached (%d ms tick, on demand).",
                _MASTER_TICK_MS,
            )
        except Exception as _exc:  # noqa: BLE001
            logger.warning("Master scheduler observer could not attach: %s",
                           _exc)

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
            title_size=0.4,
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
        ("Trackball Cam",        vtk.vtkInteractorStyleTrackballCamera),
        ("Navigation (buttons)", vtk.vtkInteractorStyleUser),
    ]
    style_state = {
        "idx": 1,   # default: Navigation (buttons) — Avion mode
        "instances": [cls() for _, cls in _INTERACTION_STYLES],
        "custom_idx": len(_INTERACTION_STYLES) - 1,
    }

    # ── Navigation sub-mode: 0=Avion (default), 1=Spaceship, 2=Custom ───
    _nav_mode: list = [0]
    _avion_buttons: list = []         # airplane mode buttons (left panel)
    _ship_buttons: list = []          # spaceship mode buttons (left panel)
    _nav_mode_btn_ref: list = [None]  # forward ref; filled after spaceship section
    _restart_nav_btn_ref: list = [None]
    _keyboard_btn_ref: list = [None]  # "Keys ▸ off/on" toggle (avion keys)
    _keyboard_on: list = [False]      # is keyboard navigation armed? (armed after first render)

    def _apply_style(i: int) -> None:
        iren = getattr(plt, "interactor", None)
        if iren is None:
            return
        iren.SetInteractorStyle(style_state["instances"][i])

    _apply_style(1)   # start in Navigation (buttons) mode

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
            logger.debug(
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
        is_nav = (style_state["idx"] == style_state["custom_idx"])
        mode = _nav_mode[0]  # 0=Avion, 1=Spaceship, 2=Custom

        def _set_vis(btn_list, visible: bool) -> None:
            for b in btn_list:
                actor = getattr(b, "actor", None)
                if actor is None:
                    continue
                try:
                    actor.SetVisibility(1 if visible else 0)
                except Exception:  # noqa: BLE001
                    pass

        _set_vis(_nav_buttons,   is_nav and mode == 2)  # Custom panel (right)
        _set_vis(_ship_buttons,  is_nav and mode == 1)  # Spaceship  (left)
        _set_vis(_avion_buttons, is_nav and mode == 0)  # Airplane   (left)

        # Nav mode switch + Restart + Keys: always visible when in
        # Navigation mode.
        for _ref in (_nav_mode_btn_ref, _restart_nav_btn_ref,
                     _keyboard_btn_ref):
            _b = _ref[0]
            if _b is None:
                continue
            actor = getattr(_b, "actor", None)
            if actor is not None:
                try:
                    actor.SetVisibility(1 if is_nav else 0)
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
    # Whether the bottom gauge sliders are visible.  Toggled by the
    # "Gauges" button placed next to the keyboard-navigation toggle.
    _gauge_state = {"on": True}

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

    def _membrane_tick() -> None:
        if not membranes_state["visible"] or membranes_state["n_steps"] <= 0:
            return
        membranes_state["step"] = (
            (int(membranes_state["step"]) + 1) % int(membranes_state["n_steps"])
        )
        _membrane_apply_step()
        _request_render()

    _membrane_sub = _master_register(
        "membranes", _membrane_tick, int(MEMBRANE_TICK_MS), enabled=False,
    )

    def _membrane_start_timer() -> None:
        if membranes_state["actor"] is None:
            return
        _master_set_enabled(_membrane_sub, True)
        # Keep the legacy slot so other code paths can still detect
        # "timer running" via a non-None value.
        membranes_state["timer_id"][0] = 1

    def _membrane_stop_timer() -> None:
        _master_set_enabled(_membrane_sub, False)
        membranes_state["timer_id"][0] = None

    _membranes_btn: list = [None]

    def _toggle_membranes(*_args, **_kwargs) -> None:
        if membranes_state["actor"] is None:
            logger.warning("_toggle_membranes: actor is None — ignoring click")
            return
        want = not membranes_state["visible"]
        if want:
            try:
                plt.renderer.AddActor(membranes_state["actor"])
                membranes_state["visible"] = True
                _membrane_apply_step()
                _membrane_start_timer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Show membranes failed: %s", exc)
        else:
            try:
                _membrane_stop_timer()
                plt.renderer.RemoveActor(membranes_state["actor"])
                membranes_state["visible"] = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hide membranes failed: %s", exc)
        if _membranes_btn[0] is not None:
            try:
                _membranes_btn[0].switch()
            except Exception:  # noqa: BLE001
                pass
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    if membranes_state["actor"] is not None:
        _membranes_btn[0] = plt.add_button(
            _toggle_membranes,
            states=["Water OFF", "Water ON"],
            c=["white", "white"],
            bc=["#22336e", "#3aa0ff"],
            pos=(0.02, 0.10),
            size=14,
            bold=True,
        )

    _ui_capturers.append(lambda: {
        "membranes_visible": bool(membranes_state["visible"]),
        "membranes_step":    int(membranes_state["step"]),
    })

    # ─── Water "lames" (V2) descending animation ────────────────────────
    # Storage: ONE packed vtkPolyData with per-cell ``step_id`` and
    # ``rgb`` arrays.  At load we split it into Nstep static per-step
    # actors (each with its own PolyData) all initially hidden.  The
    # Single actor whose PolyData is swapped at each tick — O(1) render.
    lames_state = {
        "visible":  False,
        "step":     0,
        "n_steps":  0,
        "step_pds": None,           # list[vtkPolyData | None]
        "actor":    None,           # single vtkActor
        "added":    False,
        "timer_id": [None],
    }
    _lames_pd, _lames_meta = (None, None)
    if lames_data is not None:
        _lames_pd, _lames_meta = lames_data

    if _lames_pd is not None and _lames_meta is not None:
        try:
            n_steps_l = int(_lames_meta.get("n_steps", 0))
            step_pds_l, actor_l = _build_lames_step_polydatas(
                _lames_pd, n_steps_l)
            lames_state["n_steps"]  = n_steps_l
            lames_state["step_pds"] = step_pds_l
            lames_state["actor"]    = actor_l
            # Pre-add to the renderer right now (actor is already hidden:
            # _build_lames_step_polydatas sets SetVisibility(False)).
            # This avoids a cold-start VTK pipeline stall on first toggle-ON.
            try:
                plt.renderer.AddActor(actor_l)
                lames_state["added"] = True
            except Exception as _exc_pre:  # noqa: BLE001
                logger.warning("Could not pre-add lames actor: %s", _exc_pre)
            logger.info(
                "Water lames (V2) ready: n_steps=%d  (toggle via 'Lames' button)",
                n_steps_l,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise water lames (V2): %s", exc)

    def _lames_apply_step() -> None:
        step_pds = lames_state["step_pds"]
        actor    = lames_state["actor"]
        if actor is None or not step_pds or lames_state["n_steps"] <= 0:
            return
        cur = int(lames_state["step"]) % int(lames_state["n_steps"])
        pd  = step_pds[cur]
        if pd is not None:
            actor.GetMapper().SetInputData(pd)
            actor.SetVisibility(True)
        else:
            actor.SetVisibility(False)

    def _lames_tick() -> None:
        if not lames_state["visible"] or lames_state["n_steps"] <= 0:
            return
        lames_state["step"] = (
            (int(lames_state["step"]) + 1) % int(lames_state["n_steps"])
        )
        _lames_apply_step()
        _request_render()

    _lames_sub = _master_register(
        "lames", _lames_tick, int(LAMES_TICK_MS), enabled=False,
    )

    def _lames_start_timer() -> None:
        if lames_state["actor"] is None:
            return
        _master_set_enabled(_lames_sub, True)
        lames_state["timer_id"][0] = 1

    def _lames_stop_timer() -> None:
        _master_set_enabled(_lames_sub, False)
        lames_state["timer_id"][0] = None

    # Normalise mask_overlay_meshes to a list of vtkActor-compatible objects.
    _mask_overlays: list = list(mask_overlay_meshes) if mask_overlay_meshes else []
    _mask_overlay_added: list[bool] = [False] * len(_mask_overlays)
    # Pre-add mask overlays hidden so first toggle is instant (no pipeline stall).
    for _i_pre, _ov_pre in enumerate(_mask_overlays):
        try:
            _vtk_ov_pre = getattr(_ov_pre, "actor", _ov_pre)
            plt.renderer.AddActor(_vtk_ov_pre)
            _vtk_ov_pre.SetVisibility(False)
            _mask_overlay_added[_i_pre] = True
        except Exception as _exc_pre:  # noqa: BLE001
            logger.warning("Could not pre-add mask overlay %d: %s", _i_pre, _exc_pre)

    _lames_btn: list = [None]

    def _toggle_lames(*_args, **_kwargs) -> None:
        actor = lames_state["actor"]
        if actor is None:
            logger.warning("_toggle_lames: actor unavailable — ignoring click")
            return
        want = not lames_state["visible"]
        logger.info("_toggle_lames: want=%s", "ON" if want else "OFF")
        if want:
            try:
                if not lames_state["added"]:
                    plt.renderer.AddActor(actor)
                    lames_state["added"] = True
                lames_state["visible"] = True
                _lames_apply_step()
                _lames_start_timer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Show lames failed: %s", exc)
            # Show mask overlays alongside lames.
            for _i, _ov in enumerate(_mask_overlays):
                try:
                    _vtk_actor = getattr(_ov, "actor", _ov)
                    if not _mask_overlay_added[_i]:
                        plt.renderer.AddActor(_vtk_actor)
                        _mask_overlay_added[_i] = True
                    _vtk_actor.SetVisibility(True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Show mask overlay %d failed: %s", _i, exc)
        else:
            try:
                _lames_stop_timer()
                actor.SetVisibility(False)
                lames_state["visible"] = False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hide lames failed: %s", exc)
            # Hide mask overlays.
            for _i, _ov in enumerate(_mask_overlays):
                try:
                    getattr(_ov, "actor", _ov).SetVisibility(False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hide mask overlay %d failed: %s", _i, exc)
        if _lames_btn[0] is not None:
            try:
                _lames_btn[0].switch()
            except Exception:  # noqa: BLE001
                pass
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    if lames_state["actor"] is not None:
        _lames_btn[0] = plt.add_button(
            _toggle_lames,
            states=["Lames OFF", "Lames ON"],
            c=["white", "white"],
            bc=["#1a4f7a", "#4cc2ff"],
            pos=(0.02, 0.04),
            size=14,
            bold=True,
        )

    _ui_capturers.append(lambda: {
        "lames_visible": bool(lames_state["visible"]),
        "lames_step":    int(lames_state["step"]),
    })

    # ── Wind / gas particles (O₂ + CH₄) ─────────────────────────────────
    # Two independent toggle buttons placed to the right of "Lames".  Each
    # species owns:
    #   • a vedo.Points actor (created lazily on first toggle-ON),
    #   • a state dict with timer / animation cursor / per-particle phase,
    #   • a TimerEvent observer that no-ops while the species is hidden.
    _wind_data: dict = dict(wind_data) if wind_data else {}
    wind_states: dict[str, dict] = {}

    def _wind_make_actor(species: str) -> "object | None":  # noqa: F821
        """Build the vedo.Points actor for ``species`` on first toggle-ON.

        Picks ``display`` random (template, phase) pairs from the cached
        trajectory pool, seeds the actor with the corresponding frame-0
        positions, and registers an alpha LUT so each particle fades in
        at birth and out at death (triangular ramp on its lifespan)."""
        cfg = _wind_data.get(species)
        if cfg is None:
            return None
        try:
            import numpy as np
            import vedo as _vedo_w
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wind %s: vedo/numpy import failed: %s",
                           species.upper(), exc)
            return None

        tpl        = cfg["tpl"]
        positions  = np.ascontiguousarray(tpl["positions"], dtype=np.float32)
        alive      = np.ascontiguousarray(tpl["alive"], dtype=bool)
        n_tpl      = int(tpl["n_templates"])
        n_frames   = int(tpl["n_frames"])
        life_frm   = int(tpl["life_frames"])
        n_disp     = int(cfg["display"])
        if n_disp <= 0 or n_tpl <= 0 or n_frames <= 0:
            return None

        rng = np.random.default_rng(0xC0DE if species == "o2" else 0xBEEF)
        template_idx = rng.integers(0, n_tpl, size=n_disp, dtype=np.int64)
        phase        = rng.integers(0, n_frames, size=n_disp, dtype=np.int64)

        # Birth offsets — each particle's birth frame inside its template
        # is the start of its trajectory (frame 0 of that template), but
        # the *display* phase is randomised so deaths/births don't pulse.
        # Triangular life ramp peaks at lifespan/2.
        if life_frm > 0:
            life_t = np.arange(life_frm, dtype=np.float32) / float(life_frm)
            life_alpha = (1.0 - np.abs(2.0 * life_t - 1.0)).astype(np.float32)
        else:
            life_alpha = np.ones(1, dtype=np.float32)

        # Seed actor with frame-0 positions of the chosen templates.
        pts0 = positions[template_idx, 0, :].copy()
        col = cfg["color"]
        col01 = (col[0] / 255.0, col[1] / 255.0, col[2] / 255.0)
        try:
            actor = _vedo_w.Points(pts0, r=float(cfg["point_size"]), c=col01)
            actor.lighting("off")
            actor.name = f"wind_{species}_particles"
            # Configure the VTK mapper for direct RGBA so per-point alpha works.
            # actor.alpha() must NOT be called with 0 — Property.Opacity is a
            # global multiplier that would zero out all per-point alpha values.
            _vtk_m = actor.actor.GetMapper()
            _vtk_m.SetColorModeToDirectScalars()
            _vtk_m.SetScalarModeToUsePointData()
            _vtk_m.ScalarVisibilityOn()
            actor.actor.GetProperty().SetOpacity(1.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wind %s: Points actor creation failed: %s",
                           species.upper(), exc)
            return None

        # Per-frame RGBA buffer (uint8) for fast point-color uploads.
        rgba = np.empty((n_disp, 4), dtype=np.uint8)
        rgba[:, 0] = col[0]
        rgba[:, 1] = col[1]
        rgba[:, 2] = col[2]
        rgba[:, 3] = 0

        wind_states[species]["positions"]    = positions
        wind_states[species]["alive"]        = alive
        wind_states[species]["template_idx"] = template_idx
        wind_states[species]["phase"]        = phase
        wind_states[species]["life_alpha"]   = life_alpha
        wind_states[species]["rgba"]         = rgba
        wind_states[species]["n_frames"]     = n_frames
        wind_states[species]["life_frames"]  = max(life_frm, 1)
        wind_states[species]["n_display"]    = n_disp
        wind_states[species]["actor"]        = actor
        logger.info(
            "Wind %s actor built: %d particles, %d templates, "
            "%d frames, life=%d frm, point_r=%.1f, color=(%d,%d,%d)",
            species.upper(), n_disp, n_tpl, n_frames, max(life_frm, 1),
            float(cfg["point_size"]), col[0], col[1], col[2],
        )
        try:
            flat = positions.reshape(-1, 3)
            bb_min = flat.min(axis=0)
            bb_max = flat.max(axis=0)
            vol = float(np.prod(bb_max - bb_min))
            density = n_tpl / vol if vol > 1.0 else float("inf")
            displ = np.linalg.norm(np.diff(positions, axis=1), axis=2)
            mean_spd_frm = float(displ.mean())
            fps_tpl = int(tpl.get("fps", 25))
            mean_spd_s = mean_spd_frm * fps_tpl
            alive_frac = 100.0 * float(alive.mean())
            logger.info(
                "Wind %s stats: bbox X=[%.1f,%.1f] Y=[%.1f,%.1f] Z=[%.1f,%.1f]  "
                "vol=%.0f vox\u00b3  density=%.2e tpl/vox\u00b3  "
                "speed=%.3f vox/frm = %.2f vox/s  "
                "point_r=%.1f px  alive=%.1f%%",
                species.upper(),
                bb_min[0], bb_max[0], bb_min[1], bb_max[1], bb_min[2], bb_max[2],
                vol, density,
                mean_spd_frm, mean_spd_s,
                float(cfg["point_size"]),
                alive_frac,
            )
        except Exception as _se:  # noqa: BLE001
            logger.warning("Wind %s: stats failed: %s", species.upper(), _se)
        return actor

    def _wind_apply_frame(species: str) -> None:
        st = wind_states.get(species)
        if st is None or st.get("actor") is None or not st["visible"]:
            return
        try:
            import numpy as np
            positions    = st["positions"]
            alive        = st["alive"]
            template_idx = st["template_idx"]
            phase        = st["phase"]
            life_alpha   = st["life_alpha"]
            rgba         = st["rgba"]
            n_frames     = st["n_frames"]
            life_frames  = st["life_frames"]
            frame        = int(st["frame"])

            # Frame index along each particle's own trajectory.
            tframe = (frame + phase) % n_frames
            pts  = positions[template_idx, tframe, :]
            aliv = alive[template_idx, tframe]
            # Alpha = triangular life ramp evaluated at (tframe % life_frames).
            life_idx = (tframe % life_frames).astype(np.int64, copy=False)
            a = (life_alpha[life_idx] * aliv.astype(np.float32) * 255.0)
            rgba[:, 3] = np.clip(a, 0, 255).astype(np.uint8)

            actor = st["actor"]
            actor.points = pts
            try:
                actor.pointcolors = rgba
            except Exception:
                # VTK direct fallback: write RGBA as point scalars.
                try:
                    from vtk.util.numpy_support import numpy_to_vtk as _n2v
                    _arr = _n2v(rgba, deep=False, array_type=3)  # VTK_UNSIGNED_CHAR
                    _arr.SetName("RGBA")
                    _arr.SetNumberOfComponents(4)
                    _vtk_pd = actor.actor.GetMapper().GetInput()
                    _vtk_pd.GetPointData().SetScalars(_arr)
                    _vtk_pd.GetPointData().Modified()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wind %s frame apply failed: %s",
                           species.upper(), exc)

    def _wind_tick_factory(species: str):
        def _tick() -> None:
            st = wind_states.get(species)
            if st is None or not st["visible"]:
                return
            st["frame"] = (int(st["frame"]) + 1) % int(st["n_frames"])
            _wind_apply_frame(species)
            _request_render()
        return _tick

    _iren_w = getattr(plt, "interactor", None)

    def _wind_start_timer(species: str) -> None:
        st = wind_states.get(species)
        if st is None:
            return
        sub = st.get("master_sub")
        if sub is None:
            return
        _master_set_enabled(sub, True)
        st["timer_id"][0] = 1

    def _wind_stop_timer(species: str) -> None:
        st = wind_states.get(species)
        if st is None:
            return
        sub = st.get("master_sub")
        if sub is not None:
            _master_set_enabled(sub, False)
        st["timer_id"][0] = None

    def _wind_toggle_factory(species: str):
        def _toggle(*_a, **_kw) -> None:
            st = wind_states[species]
            want = not st["visible"]
            logger.info("_wind_toggle[%s]: want=%s",
                        species.upper(), "ON" if want else "OFF")
            if want:
                actor = st.get("actor")
                if actor is None:
                    actor = _wind_make_actor(species)
                    if actor is None:
                        logger.warning("Wind %s: actor unavailable.",
                                       species.upper())
                        return
                try:
                    _vtk_actor = getattr(actor, "actor", actor)
                    if not st["added"]:
                        plt.renderer.AddActor(_vtk_actor)
                        st["added"] = True
                    _vtk_actor.SetVisibility(True)
                    st["visible"] = True
                    _wind_apply_frame(species)
                    _wind_start_timer(species)
                    logger.info(
                        "Wind %s ON: %d particles visible, timer_id=%s",
                        species.upper(), st.get("n_display", 0),
                        st["timer_id"][0],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Show wind %s failed: %s",
                                   species.upper(), exc)
            else:
                try:
                    _wind_stop_timer(species)
                    st["visible"] = False
                    actor = st.get("actor")
                    if actor is not None:
                        getattr(actor, "actor", actor).SetVisibility(False)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hide wind %s failed: %s",
                                   species.upper(), exc)
            if st.get("btn") and st["btn"][0] is not None:
                try:
                    st["btn"][0].switch()
                except Exception:  # noqa: BLE001
                    pass
            try:
                plt.render()
            except Exception:  # noqa: BLE001
                pass
        return _toggle

    # Initialise per-species state and register one TimerEvent observer
    # per loaded species (no-ops while the species is hidden).
    _wind_button_layout = {
        "o2":  {"pos": (0.12, 0.04), "labels": ["O₂ OFF",  "O₂ ON"],
                "bc":  ["#1a4f7a", "#e8f4ff"]},
        "ch4": {"pos": (0.22, 0.04), "labels": ["CH₄ OFF", "CH₄ ON"],
                "bc":  ["#4f3a1a", "#ffd470"]},
    }
    for _sp in ("o2", "ch4"):
        if _sp not in _wind_data:
            continue
        wind_states[_sp] = {
            "actor":    None,
            "visible":  False,
            "added":    False,
            "frame":    0,
            "timer_id": [None],
            "btn":      [None],
        }
        _tick_cb = _wind_tick_factory(_sp)
        wind_states[_sp]["master_sub"] = _master_register(
            f"wind_{_sp}", _tick_cb, int(WIND_TICK_MS), enabled=False,
        )
        _layout = _wind_button_layout[_sp]
        wind_states[_sp]["btn"][0] = plt.add_button(
            _wind_toggle_factory(_sp),
            states=_layout["labels"],
            c=["white", "black"],
            bc=_layout["bc"],
            pos=_layout["pos"],
            size=14,
            bold=True,
        )

    _ui_capturers.append(lambda: {
        f"wind_{_sp}_visible": bool(wind_states.get(_sp, {}).get("visible", False))
        for _sp in ("o2", "ch4") if _sp in wind_states
    })

    # Pre-build and pre-add wind particle actors hidden so the first toggle-ON
    # is instant.  Building happens here (startup) rather than inside a
    # button callback, so the numpy work and VTK actor construction never
    # block the event loop.
    for _sp_pre in list(wind_states.keys()):
        _act_pre = _wind_make_actor(_sp_pre)
        if _act_pre is not None:
            _vtk_pre = getattr(_act_pre, "actor", _act_pre)
            try:
                plt.renderer.AddActor(_vtk_pre)
                _vtk_pre.SetVisibility(False)
                wind_states[_sp_pre]["added"] = True
            except Exception as _exc_pre:  # noqa: BLE001
                logger.warning("Could not pre-add wind %s actor: %s",
                               _sp_pre.upper(), _exc_pre)

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
                title_size=0.4,
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
                title_size=0.4,
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
                _set_widget_visible(s, not in_arrows and _gauge_state["on"])
            for s in arrow_sliders:
                _set_widget_visible(s, in_arrows and _gauge_state["on"])
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
            "arrows_tracks": "Conduction ▸ shorter paths",
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

    def _ship_tick() -> None:
        # Ignore ticks when not in Spaceship sub-mode.
        if _nav_mode[0] != 1:
            return
        if not _ship_any_active():
            _ship["last_t"] = None
            # Disable the master sub when idle so the master timer
            # itself can stop — needed to restore the vanilla,
            # inertia-free trackball feel during click-and-drag.
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
            _request_render()

    _iren_ship = getattr(plt, "interactor", None)
    _ship_timer_id: list = [None]
    _ship_sub = _master_register("ship", _ship_tick, _MASTER_TICK_MS,
                                 enabled=False)

    def _ensure_ship_timer() -> None:
        """Enable the ship subsystem in the master scheduler.

        We *only* keep the subsystem (and therefore the master timer)
        active while the ship is actually moving, because VTK's
        ``vtkInteractorStyleTrackballCamera`` responds to every
        ``TimerEvent`` by re-applying the last rotate/pan/zoom delta
        when a mouse button is held — which makes the trackball feel
        like it has inertia.  Disabling the sub when v ≡ 0 lets the
        master timer stop and restores the vanilla trackball behaviour.
        """
        if _iren_ship is None:
            return
        _master_set_enabled(_ship_sub, True)
        _ship_timer_id[0] = 1

    def _stop_ship_timer() -> None:
        if _iren_ship is None:
            return
        _master_set_enabled(_ship_sub, False)
        _ship_timer_id[0] = None

    if _iren_ship is not None:
        logger.info(
            "Navigating-observer-thrust sub registered "
            "(master scheduler, on demand)."
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
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 2, +1), states=["▲"],
        pos=(_SHIP_X_C, _t_rows[0]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 1, -1), states=["◀"],
        pos=(_SHIP_X_L, _t_rows[1]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 1, +1), states=["▶"],
        pos=(_SHIP_X_R, _t_rows[1]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 2, -1), states=["▼"],
        pos=(_SHIP_X_C, _t_rows[2]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 0, +1), states=["FWD"],
        pos=(_SHIP_X_L, _t_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_stop("t"), states=["STOP"],
        pos=(_SHIP_X_C, _t_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_STOP_FG], bc=[_SHIP_STOP_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("t", 0, -1), states=["BCK"],
        pos=(_SHIP_X_R, _t_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_T_FG], bc=[_SHIP_T_BG]))

    # Rotation panel (lower).
    #       [ P↑ ]
    # [ Y← ]       [ Y→ ]
    #       [ P↓ ]
    # [R⟲ ][STOP ][ R⟳]
    _r_rows = [0.20, 0.16, 0.12, 0.08]
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 1, +1), states=["P▲"],
        pos=(_SHIP_X_C, _r_rows[0]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))
    # Yaw signs flipped so that the arrow direction matches the
    # actual rotation of the view (cam.Yaw(+deg) swings view left).
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 0, +1), states=["Y◀"],
        pos=(_SHIP_X_L, _r_rows[1]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 0, -1), states=["Y▶"],
        pos=(_SHIP_X_R, _r_rows[1]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 1, -1), states=["P▼"],
        pos=(_SHIP_X_C, _r_rows[2]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 2, -1), states=["R↺"],
        pos=(_SHIP_X_L, _r_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_stop("r"), states=["STOP"],
        pos=(_SHIP_X_C, _r_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_STOP_FG], bc=[_SHIP_STOP_BG]))
    _ship_buttons.append(plt.add_button(
        _mk_bump("r", 2, +1), states=["R↻"],
        pos=(_SHIP_X_R, _r_rows[3]),
        size=_SHIP_SIZE, bold=True,
        c=[_SHIP_R_FG], bc=[_SHIP_R_BG]))

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
        # Reset airplane mode state (intents + velocities + timer).
        try:
            for _k in list(_plane_held.keys()):
                _plane_held[_k] = False
                _plane_release_at[_k] = None
                _plane_kbd_held[_k] = False
                _plane_intent_until[_k] = 0.0
            _plane["fwd_v"] = 0.0
            _plane["fwd_a"] = 0.0
            _plane["v_yaw"] = 0.0
            _plane["v_pitch"] = 0.0
            _plane["v_roll"] = 0.0
            _plane["last_t"] = None
            _stop_plane_timer()
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

    _restart_nav_btn_ref[0] = plt.add_button(
        _restart_nav_cb,
        states=["Restart ↺"],
        c=["white"],
        bc=["#4527a0"],   # deep indigo, distinct from the STOP red
        pos=(_SHIP_X_C, 0.44),   # just above the movement panels
        size=14,
        bold=True,
    )

    # ─── Nav mode switch (Avion / Spaceship / Custom) ───────────────────
    # Single button in the left panel to cycle between the three
    # navigation sub-modes.  Visible whenever the interaction style is
    # "Navigation (buttons)".
    _NAV_MODE_NAMES = ["Avion", "Spaceship", "Custom"]

    def _cycle_nav_mode_cb(*_a, **_kw):
        old_mode = _nav_mode[0]
        new_mode = (old_mode + 1) % len(_NAV_MODE_NAMES)
        _nav_mode[0] = new_mode
        # Stop any ongoing motion from the previous mode.
        if old_mode == 0:   # leaving Avion
            try:
                for _k in list(_plane_held.keys()):
                    _plane_held[_k] = False
                    _plane_release_at[_k] = None
                    _plane_kbd_held[_k] = False
                    _plane_intent_until[_k] = 0.0
                _plane["fwd_v"] = 0.0
                _plane["fwd_a"] = 0.0
                _plane["v_yaw"] = 0.0
                _plane["v_pitch"] = 0.0
                _plane["v_roll"] = 0.0
                _plane["last_t"] = None
                _stop_plane_timer()
            except Exception:  # noqa: BLE001
                pass
        elif old_mode == 1:  # leaving Spaceship
            try:
                _ship["v_world"] = _np_ship.zeros(3, dtype=_np_ship.float64)
                for _i in range(3):
                    _ship["v_r"][_i] = 0.0
                _ship["last_t"] = None
                _stop_ship_timer()
            except Exception:  # noqa: BLE001
                pass
        _nav_mode_btn_ref[0].switch()
        _refresh_nav_panel_visibility()
        _show_status(f"Navigation mode: {_NAV_MODE_NAMES[new_mode]}")
        plt.render()

    _nav_mode_btn_ref[0] = plt.add_button(
        _cycle_nav_mode_cb,
        states=[f"Nav ▸ {n}" for n in _NAV_MODE_NAMES],
        c=["white"] * len(_NAV_MODE_NAMES),
        bc=["#1565c0", "#0d47a1", "#455a64"],
        pos=(_SHIP_X_C, 0.48),
        size=14,
        bold=True,
    )

    # ─── Airplane mode physics ───────────────────────────────────────────
    # See the "Airplane control model (state-based)" block below for the
    # full description.  In short: held key / clicked button refreshes
    # an intent; per-tick integrator drives forward and per-axis
    # angular velocities toward the corresponding target asymptotes.
    try:
        _xmn2, _xmx2, _ymn2, _ymx2, _zmn2, _zmx2 = mesh.bounds()
        _scene_diag_plane = float(
            ((_xmx2 - _xmn2) ** 2 + (_ymx2 - _ymn2) ** 2 + (_zmx2 - _zmn2) ** 2) ** 0.5
        )
        _scene_max_side_plane = float(
            max(_xmx2 - _xmn2, _ymx2 - _ymn2, _zmx2 - _zmn2)
        )
    except Exception:  # noqa: BLE001
        _scene_diag_plane = _scene_diag_ship
        _scene_max_side_plane = _scene_diag_ship

    # ─── Airplane control model (state-based) ───────────────────────────
    # Replaces the previous Hamming-window queue approach with a
    # continuous-state integrator inspired by classic action-game flight
    # controls:
    #
    #   * Per-axis angular velocity (deg/s) and a scalar forward
    #     velocity (units/s) are stored in ``_plane``.  Each tick they
    #     exponentially approach a target value derived from the
    #     "intents" currently active.
    #   * An "intent" is a (timestamped) request to move in some
    #     direction.  A KeyPress refreshes the intent for that
    #     direction; subsequent KeyRelease (filtered via a short
    #     grace period to ignore X11 autorepeat fake-releases) clears
    #     it.  An on-screen button click refreshes the same intent
    #     for ``_BUTTON_HOLD_S`` seconds, so a single click feels like
    #     a brief tap on the equivalent key.
    #
    # Consequences:
    #   * Tap a key  → velocity rises a little then decays back to 0.
    #   * Hold a key → velocity exponentially approaches the asymptote
    #     (MAX_FWD or ±MAX_DPS).
    #   * Release    → velocity decays to 0 over ~tau_release seconds.
    #   * Tap + hold give the same "logarithmic" throttle feel: short
    #     presses for micro-tissue precision, longer holds for fast
    #     scene traversal, with a hard cap at MAX_FWD = one X-axis
    #     length per second.
    import math as _math_plane

    # Caps.
    _PLANE_MAX_FWD = float(_scene_max_side_plane)   # units/s (≈ X-length / s)
    _PLANE_MAX_YAW_PITCH_DPS = 80.0   # deg/s — yaw & pitch
    _PLANE_MAX_ROLL_DPS      = 180.0   # deg/s — roll (tunable independently)
    _PLANE_ROT_DEG = 2.0  # button-click micro-impulse magnitude (legacy)

    # ── Forward: acceleration-control model ──────────────────────────────
    # The pilot controls acceleration, not velocity directly.
    #   * PgUp held  → fwd_a ramps toward MAX_ACCEL (tau_a_atk).
    #   * PgUp released → fwd_a decays quickly to 0 (tau_a_rel);
    #     fwd_v is NOT touched — achieved speed is maintained.
    #   * PgDown held → hard brake: fwd_a zeroed, fwd_v decays toward 0
    #     at tau_brake rate.  Braking is intentionally faster than accel.
    #   * PgDown released → velocity holds at its current (lower) value.
    # This gives the "logarithmic" range: a tap adds a tiny velocity
    # increment (great for micro-tissue), a long hold lets speed grow
    # toward MAX_FWD (good for scene traversal).
    _PLANE_MAX_ACCEL    = _PLANE_MAX_FWD / 3.0  # reach MAX_FWD in ≈3 s continuous hold
    _PLANE_TAU_FWD_A_ATK = 1.80   # s — accel ramps up while z held
    _PLANE_TAU_FWD_A_REL = 0.08   # s — accel snaps to 0 on release (vel holds)
    _PLANE_TAU_BRAKE     = 0.22   # s — velocity kill while s held

    # ── Rotation ──────────────────────────────────────────────────────────
    _PLANE_TAU_ROT_ATK = 0.55   # s — held → ramp up to ±MAX_DPS
    _PLANE_TAU_ROT_REL = 0.06   # s — released → snap to 0 quickly

    # ── Press / release tracking (replaces the previous timeout model) ─
    # The old model relied on OS keyboard autorepeat to *refresh* an
    # intent every ~30-50ms.  That assumption is wrong on every modern
    # desktop OS: the keyboard autorepeat *initial delay* is ~500ms,
    # not ~50ms.  Symptom: a key press caused a tiny initial motion,
    # then ~500ms of stillness, then continuous motion once autorepeat
    # kicked in.  In addition, when other heavy timers (lames, wind,
    # membranes) stalled the event loop, queued autorepeat KeyPress
    # events kept refreshing the intent long after the user released
    # the key — producing seconds-long "phantom" motion.
    #
    # The current model tracks press/release explicitly:
    #   * KeyPress  → set held=True (cancel any pending release).
    #   * KeyRelease → schedule release after _KEY_RELEASE_GRACE_S to
    #                   filter X11's fake releases between autorepeat
    #                   presses (X11 sends a Release immediately
    #                   followed by a Press when autorepeating).
    #   * Button click → set held=True with a short auto-release
    #                   timer (acts like a brief tap).
    # An on-screen button click acts like a brief tap via a scheduled
    # auto-release at now + _BUTTON_HOLD_S.
    _KEY_RELEASE_GRACE_S = 0.030  # 30ms: filters X11 fake-releases (gap ≪ 5ms); 1 tick   # ≥ typical X11 autorepeat interval (~30-50 ms)
    _BUTTON_HOLD_S       = 0.20

    # Threshold below which velocity / accel are snapped to zero.
    _PLANE_V_EPS = _PLANE_MAX_FWD / 5000.0
    _PLANE_A_EPS = _PLANE_MAX_ACCEL / 1000.0

    _plane = {
        "fwd_v":   0.0,   # forward velocity (units/s)  — persists on release
        "fwd_a":   0.0,   # forward acceleration (units/s²) — pilot's control lever
        "v_yaw":   0.0,   # angular velocities (deg/s)
        "v_pitch": 0.0,
        "v_roll":  0.0,
        "last_t":  None,
    }

    # Per-intent "currently held" flag (True while the user is actively
    # commanding that direction).  Updated by press / release callbacks
    # and consumed by ``_plane_tick``.
    _plane_held: dict[str, bool] = {
        "fwd": False, "bck": False,
        "yaw_l": False, "yaw_r": False,
        "pitch_u": False, "pitch_d": False,
        "roll_l": False, "roll_r": False,
    }
    # Per-intent scheduled release timestamp (None = no pending release).
    # ``_plane_tick`` flips ``_plane_held[k]`` to False once now ≥ this
    # value.  Used both for the X11 grace period after KeyRelease and
    # for the auto-release timer of button taps.
    _plane_release_at: dict[str, float | None] = {k: None for k in _plane_held}
    # Fallback "KeyPress without matching KeyRelease yet" flags. Only
    # consulted when X11 polling is unavailable.
    _plane_kbd_held: dict[str, bool] = {k: False for k in _plane_held}
    # Per-intent button-tap auto-release deadline (0.0 = inactive).
    _plane_intent_until: dict[str, float] = {k: 0.0 for k in _plane_held}

    _iren_plane = getattr(plt, "interactor", None)

    # ── X11 keyboard polling (primary input source) ──────────────────
    # Polling avoids OS autorepeat latency (~500ms initial delay) and
    # dropped-release races: we ask the X server every tick which keys
    # are physically down right now.
    _AVION_KEYCODES: dict[str, list[int]] = {}
    try:
        from Xlib import display as _x_display_mod, XK as _x_XK_mod
        _x_disp_plane = _x_display_mod.Display()
        logger.info("X11 keyboard polling available for Avion mode.")
    except Exception as _exc:  # noqa: BLE001
        _x_disp_plane = None
        _x_XK_mod = None
        logger.info(
            "X11 keyboard polling unavailable (%s); "
            "falling back to event-driven press/release.",
            _exc,
        )

    def _x11_resolve_keycodes(keysym_names) -> list[int]:
        """Resolve a set of X11 keysym names to a list of keycodes."""
        if _x_disp_plane is None or _x_XK_mod is None:
            return []
        out: list[int] = []
        for name in keysym_names:
            try:
                ks = _x_XK_mod.string_to_keysym(name)
                if not ks:
                    continue
                kc = _x_disp_plane.keysym_to_keycode(ks)
                if kc:
                    out.append(int(kc))
            except Exception:  # noqa: BLE001
                continue
        return out

    def _poll_plane_keys():
        """Return {intent: is_physically_down} or None if X11 polling
        is unavailable."""
        if _x_disp_plane is None:
            return None
        try:
            keymap = _x_disp_plane.query_keymap()
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, bool] = {}
        for intent, kcs in _AVION_KEYCODES.items():
            down = False
            for kc in kcs:
                if 0 <= kc < 256 and (keymap[kc >> 3] >> (kc & 7)) & 1:
                    down = True
                    break
            out[intent] = down
        return out

    # Master-scheduler subscription; populated after _plane_tick is
    # defined.  Legacy slot kept for code that checks "is running".
    _plane_sub_ref: list = [None]
    _plane_timer_id: list = [None]

    def _ensure_plane_timer() -> None:
        sub = _plane_sub_ref[0]
        if sub is not None:
            _master_set_enabled(sub, True)
            _plane_timer_id[0] = 1

    def _stop_plane_timer() -> None:
        sub = _plane_sub_ref[0]
        if sub is not None:
            _master_set_enabled(sub, False)
            _plane_timer_id[0] = None

    def _plane_key_press(intent: str) -> None:
        """Keyboard press fallback (used when X11 polling is
        unavailable; ignored when polling is active)."""
        _plane_kbd_held[intent] = True
        _plane_release_at[intent] = None
        _ensure_plane_timer()

    def _plane_key_release(intent: str) -> None:
        """Keyboard release fallback: schedule a release after a short
        grace period to filter X11 fake-releases from autorepeat."""
        _plane_release_at[intent] = (
            _time_ship.perf_counter() + _KEY_RELEASE_GRACE_S
        )
        _ensure_plane_timer()

    def _plane_button_tap(intent: str) -> None:
        """On-screen button click: set an auto-release deadline so the
        click behaves like a brief key tap."""
        _plane_intent_until[intent] = (
            _time_ship.perf_counter() + _BUTTON_HOLD_S
        )
        _ensure_plane_timer()

    def _plane_tick() -> None:
        """Integrate plane velocities toward the held-intent targets and
        translate / rotate the camera accordingly.

        Dispatched by the master scheduler (no obj/event params)."""
        if _nav_mode[0] != 0:
            return

        now = _time_ship.perf_counter()

        # Primary input: poll the X server for physically-held keys.
        polled = _poll_plane_keys()

        # Apply pending fallback releases (X11 grace + button auto-release).
        for _k, _rel_t in list(_plane_release_at.items()):
            if _rel_t is not None and now >= _rel_t:
                _plane_kbd_held[_k] = False
                _plane_release_at[_k] = None

        # When X11 polling is active and confirms a key is NOT physically
        # held, immediately cancel any stale kbd grace-period for that key.
        # This prevents render-frame-queued events from leaving a zombie
        # release timer that would show up as kbd_down=True on the next tick.
        if polled is not None:
            for _k, _down in polled.items():
                if not _down:
                    _plane_kbd_held[_k] = False
                    _plane_release_at[_k] = None

        # Compose final per-intent held state from all sources.
        # When X11 polling is available, use it as the *sole* source of
        # physical key state and ignore the event-driven kbd fallback.
        # Rationale: during a long render frame (>16 ms), OS events queue
        # up and are flushed all at once afterwards.  A quick press+release
        # inside that render appears as a back-to-back KeyPress / KeyRelease
        # pair *after* the render, so poll_down is already False by then.
        # Keeping kbd_down True for the grace period (30 ms) would cause a
        # phantom "tap" → unintended motion.  X11 keymap polling is the
        # authoritative physical state and is immune to event-queue lag.
        for _k in _plane_held:
            poll_down = bool(polled.get(_k, False)) if polled is not None else False
            kbd_down = _plane_kbd_held[_k] if polled is None else False
            btn_down = _plane_intent_until[_k] > now
            _plane_held[_k] = poll_down or kbd_down or btn_down


        def _held(name: str) -> bool:
            return _plane_held[name]

        # ── 1. Resolve targets from current intents ─────────────────
        fwd_h, bck_h = _held("fwd"), _held("bck")
        any_fwd_intent = fwd_h or bck_h

        yl, yr = _held("yaw_l"), _held("yaw_r")
        target_yaw = (+_PLANE_MAX_YAW_PITCH_DPS if yl and not yr
                      else -_PLANE_MAX_YAW_PITCH_DPS if yr and not yl
                      else 0.0)
        any_yaw_intent = yl or yr

        pu, pd = _held("pitch_u"), _held("pitch_d")
        target_pitch = (+_PLANE_MAX_YAW_PITCH_DPS if pu and not pd
                        else -_PLANE_MAX_YAW_PITCH_DPS if pd and not pu
                        else 0.0)
        any_pitch_intent = pu or pd

        rl, rr = _held("roll_l"), _held("roll_r")
        target_roll = (+_PLANE_MAX_ROLL_DPS if rl and not rr
                       else -_PLANE_MAX_ROLL_DPS if rr and not rl
                       else 0.0)
        any_roll_intent = rl or rr

        # ── 2. Compute dt (skip the very first tick) ─────────────────
        if _plane["last_t"] is None:
            _plane["last_t"] = now
            return
        dt = now - _plane["last_t"]
        _plane["last_t"] = now
        if dt <= 0.0 or dt > 1.0:
            return
        # Cap dt so a delayed tick (blocked by heavy lames / wind tick)
        # doesn’t cause a large camera lurch.
        dt = min(dt, 0.100)

        # ── 3. Integrate forward acceleration → velocity ─────────────
        def _expd(cur: float, target: float, tau: float) -> float:
            return cur + (target - cur) * (1.0 - _math_plane.exp(-dt / tau))

        if bck_h:
            # Hard brake: kill acceleration, exponentially decay velocity.
            _plane["fwd_a"] = 0.0
            _plane["fwd_v"] = _expd(_plane["fwd_v"], 0.0, _PLANE_TAU_BRAKE)
        else:
            # Acceleration lever: ramps up while fwd held, snaps to 0 on
            # release — velocity is *not* touched, it holds at current value.
            target_a = _PLANE_MAX_ACCEL if fwd_h else 0.0
            tau_a    = _PLANE_TAU_FWD_A_ATK if fwd_h else _PLANE_TAU_FWD_A_REL
            _plane["fwd_a"] = _expd(_plane["fwd_a"], target_a, tau_a)
            _plane["fwd_v"] = max(
                0.0,
                min(_plane["fwd_v"] + _plane["fwd_a"] * dt, _PLANE_MAX_FWD),
            )

        # ── 4. Integrate rotation velocities ─────────────────────────
        _plane["v_yaw"]   = _expd(_plane["v_yaw"],   target_yaw,
                                   _PLANE_TAU_ROT_ATK if any_yaw_intent
                                   else _PLANE_TAU_ROT_REL)
        _plane["v_pitch"] = _expd(_plane["v_pitch"], target_pitch,
                                   _PLANE_TAU_ROT_ATK if any_pitch_intent
                                   else _PLANE_TAU_ROT_REL)
        _plane["v_roll"]  = _expd(_plane["v_roll"],  target_roll,
                                   _PLANE_TAU_ROT_ATK if any_roll_intent
                                   else _PLANE_TAU_ROT_REL)

        # Snap tiny residuals to zero so the timer can sleep.
        if abs(_plane["fwd_a"]) < _PLANE_A_EPS and not fwd_h:
            _plane["fwd_a"] = 0.0
        if abs(_plane["fwd_v"]) < _PLANE_V_EPS and not any_fwd_intent:
            _plane["fwd_v"] = 0.0
        for k, intent_active in (("v_yaw",   any_yaw_intent),
                                  ("v_pitch", any_pitch_intent),
                                  ("v_roll",  any_roll_intent)):
            if abs(_plane[k]) < 1e-3 and not intent_active:
                _plane[k] = 0.0

        # ── 5. Idle?  Stop the timer until the next intent. ──────────
        any_motion = (
            _plane["fwd_v"] != 0.0
            or _plane["fwd_a"] != 0.0
            or _plane["v_yaw"] != 0.0
            or _plane["v_pitch"] != 0.0
            or _plane["v_roll"] != 0.0
        )
        any_intent = (any_fwd_intent or any_yaw_intent
                      or any_pitch_intent or any_roll_intent)
        any_pending_release = any(
            t is not None for t in _plane_release_at.values()
        )
        if not any_motion and not any_intent and not any_pending_release:
            _plane["last_t"] = None
            _stop_plane_timer()
            return

        # ── 6. Apply camera motion ───────────────────────────────────
        # Clamp per-frame rotation to MAX_DPS * dt so step-size stays
        # geometrically reasonable even if dt happens to be large.
        rot_deg = {
            "yaw":   _plane["v_yaw"]   * dt,
            "pitch": _plane["v_pitch"] * dt,
            "roll":  _plane["v_roll"]  * dt,
        }
        has_rot = any(abs(v) > 1e-9 for v in rot_deg.values())

        cam = plt.camera
        moved = False
        try:
            if has_rot:
                fp  = np.asarray(cam.GetFocalPoint(), dtype=float)
                pos = np.asarray(cam.GetPosition(),   dtype=float)
                vu  = np.asarray(cam.GetViewUp(),     dtype=float)
                fwd = fp - pos
                dist = float(np.linalg.norm(fwd))
                if dist < 1e-12:
                    fwd = np.array([0., 0., -1.]); dist = 1.0
                else:
                    fwd /= dist
                right = np.cross(fwd, vu)
                rn = float(np.linalg.norm(right))
                right = right / rn if rn > 1e-12 else np.array([1., 0., 0.])
                up = np.cross(right, fwd)
                un = float(np.linalg.norm(up))
                up = up / un if un > 1e-12 else vu / max(float(np.linalg.norm(vu)), 1e-12)

                def _rod(v, ax, rad):
                    c, s = _math_plane.cos(rad), _math_plane.sin(rad)
                    return v * c + np.cross(ax, v) * s + ax * float(np.dot(ax, v)) * (1.0 - c)

                for _ax_name, _ax_vec in [("yaw", up), ("pitch", right), ("roll", fwd)]:
                    _deg = rot_deg[_ax_name]
                    if abs(_deg) > 1e-12:
                        _rad = _math_plane.radians(_deg)
                        fwd   = _rod(fwd,  _ax_vec, _rad)
                        up    = _rod(up,   _ax_vec, _rad)
                        right = np.cross(fwd, up)
                        rn = float(np.linalg.norm(right))
                        if rn > 1e-12:
                            right /= rn
                        up = np.cross(right, fwd)
                        un = float(np.linalg.norm(up))
                        if un > 1e-12:
                            up /= un

                cam.SetFocalPoint(*(pos + fwd * dist))
                cam.SetViewUp(*up)
                moved = True
                _refresh_ortho()

            if abs(_plane["fwd_v"]) > 0.0:
                fwd_now = np.asarray(cam.GetDirectionOfProjection(),
                                     dtype=float)
                fn = float(np.linalg.norm(fwd_now))
                if fn > 1e-12:
                    fwd_now /= fn
                    delta = fwd_now * _plane["fwd_v"] * dt
                    cam.SetPosition(*(np.asarray(cam.GetPosition(),
                                                 dtype=float) + delta))
                    cam.SetFocalPoint(*(np.asarray(cam.GetFocalPoint(),
                                                   dtype=float) + delta))
                    moved = True

            if moved:
                try:
                    for _ren in plt.renderers:
                        _ren.ResetCameraClippingRange()
                except Exception:  # noqa: BLE001
                    try:
                        plt.renderer.ResetCameraClippingRange()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            return

        if moved:
            _request_render()

    # Register the plane tick with the master scheduler.  The master
    # timer is created on-demand the first time any subsystem enables
    # its sub, so plane motion never pays for a permanently-running
    # timer (which would also feed vtkInteractorStyleTrackballCamera
    # inertia during mouse drags).
    _plane_sub_ref[0] = _master_register(
        "plane", _plane_tick, _MASTER_TICK_MS, enabled=False,
    )

    # Resolve X11 keycodes now that the _AVION_KEYS_* sets exist below;
    # we populate _AVION_KEYCODES after their definition.

    # ── Button callbacks (kept under the same names so the existing
    #    button wiring below works untouched). Each click registers a
    #    brief virtual press on the matching intent; the integrator
    #    handles the actual ramp / decay.
    def _plane_fwd(*_a, **_kw):
        _plane_button_tap("fwd")
        _show_status("Avion: ↑ avant")

    def _plane_bck(*_a, **_kw):
        _plane_button_tap("bck")
        _show_status("Avion: ↓ arrière")

    def _plane_stop(*_a, **_kw):
        # Hard stop: zero all velocities + acceleration, clear intents, kill timer.
        for k in list(_plane_held.keys()):
            _plane_held[k] = False
            _plane_release_at[k] = None
            _plane_kbd_held[k] = False
            _plane_intent_until[k] = 0.0
        _plane["fwd_v"] = 0.0
        _plane["fwd_a"] = 0.0
        _plane["v_yaw"] = 0.0
        _plane["v_pitch"] = 0.0
        _plane["v_roll"] = 0.0
        _plane["last_t"] = None
        _stop_plane_timer()
        _show_status("Avion: STOP")

    # Maps a button-click ``(axis, sign)`` to the intent name the new
    # state-based controller understands.
    _PLANE_ROT_INTENT = {
        ("yaw",   +1.0): "yaw_l",   ("yaw",   -1.0): "yaw_r",
        ("pitch", +1.0): "pitch_u", ("pitch", -1.0): "pitch_d",
        ("roll",  +1.0): "roll_l",  ("roll",  -1.0): "roll_r",
    }

    def _mk_plane_rot(axis: str, deg: float):
        intent = _PLANE_ROT_INTENT[(axis, 1.0 if deg > 0 else -1.0)]

        def _cb(*_a, **_kw):
            _plane_button_tap(intent)
        return _cb

    # ─── Airplane buttons ────────────────────────────────────────────────
    # Left panel, same column x-positions as the spaceship panel.
    # Layout:
    #        [  FWD  ]    y=0.40  (big)
    #        [  STOP ]    y=0.35
    #        [  BCK  ]    y=0.30  (big)
    #           [P▲]      y=0.24  pitch up   (centre)
    # [Y◀]             [Y▶]  y=0.20  yaw left / right
    #           [P▼]      y=0.16  pitch down (centre)
    # [R↺]             [R↻]  y=0.12  roll (below yaw, outer)
    _AVION_FG       = "white"
    _AVION_FWD_BG   = "#1b5e20"  # dark green
    _AVION_STOP_BG  = "#b71c1c"  # red
    _AVION_YAW_BG   = "#0d47a1"  # deep blue
    _AVION_PITCH_BG = "#4a148c"  # deep purple
    _AVION_ROLL_BG  = "#bf360c"  # deep orange
    _AVION_BIG  = 22             # bigger font for FWD / BCK
    _AVION_SIZE = 18

    _avion_buttons.append(plt.add_button(
        _plane_fwd, states=["FWD"],
        c=[_AVION_FG], bc=[_AVION_FWD_BG],
        pos=(_SHIP_X_C, 0.40), size=_AVION_BIG, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _plane_stop, states=["STOP"],
        c=[_AVION_FG], bc=[_AVION_STOP_BG],
        pos=(_SHIP_X_C, 0.35), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _plane_bck, states=["BCK"],
        c=[_AVION_FG], bc=[_AVION_FWD_BG],
        pos=(_SHIP_X_C, 0.30), size=_AVION_BIG, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("pitch", +_PLANE_ROT_DEG), states=["P▲"],
        c=[_AVION_FG], bc=[_AVION_PITCH_BG],
        pos=(_SHIP_X_C, 0.24), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("yaw", +_PLANE_ROT_DEG), states=["Y◀"],
        c=[_AVION_FG], bc=[_AVION_YAW_BG],
        pos=(_SHIP_X_L, 0.20), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("yaw", -_PLANE_ROT_DEG), states=["Y▶"],
        c=[_AVION_FG], bc=[_AVION_YAW_BG],
        pos=(_SHIP_X_R, 0.20), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("pitch", -_PLANE_ROT_DEG), states=["P▼"],
        c=[_AVION_FG], bc=[_AVION_PITCH_BG],
        pos=(_SHIP_X_C, 0.16), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("roll", +_PLANE_ROT_DEG), states=["R↺"],
        c=[_AVION_FG], bc=[_AVION_ROLL_BG],
        pos=(_SHIP_X_L, 0.12), size=_AVION_SIZE, bold=True,
    ))
    _avion_buttons.append(plt.add_button(
        _mk_plane_rot("roll", -_PLANE_ROT_DEG), states=["R↻"],
        c=[_AVION_FG], bc=[_AVION_ROLL_BG],
        pos=(_SHIP_X_R, 0.12), size=_AVION_SIZE, bold=True,
    ))

    # ─── Keyboard navigation toggle (Avion) ───────────────────
    # Inactive by default. When ON *and* the current sub-mode is Avion,
    # arrow keys / PageUp-Down / numpad 0 and . feed the same impulse
    # queues as the on-screen buttons — so the existing timer-driven
    # render path is reused (no extra render in the key callback, no
    # flicker, no clipping-plane fight).
    def _toggle_keyboard_cb(*_a, **_kw):
        _keyboard_on[0] = not _keyboard_on[0]
        try:
            _keyboard_btn_ref[0].switch()
        except Exception:  # noqa: BLE001
            pass
        _show_status(
            "Keyboard navigation: "
            + ("ON (Avion: ←→ yaw, ↑↓ pitch, PgUp/Dn speed, KP7/8 roll)"
               if _keyboard_on[0] else "off")
        )
        try:
            plt.render()
        except Exception:  # noqa: BLE001
            pass

    _keyboard_btn_ref[0] = plt.add_button(
        _toggle_keyboard_cb,
        states=["Keys ▸ off", "Keys ▸ on"],
        c=["#9e9e9e", "white"],
        bc=["#263238", "#00695c"],
        pos=(0.22, 0.48),
        size=14,
        bold=True,
    )

    # One-shot RenderEvent: activate keyboard after the first complete render
    # (= window shown, all actors uploaded to GPU, panels painted).
    _kbd_first_tag: list = [None]

    def _kbd_activate_once(_obj, _event):
        """Flip keyboard ON exactly once after the viewer is fully drawn."""
        # Remove ourselves first to avoid re-entrancy if render() fires again.
        try:
            _iren_k = getattr(plt, "interactor", None)
            if _iren_k is not None and _kbd_first_tag[0] is not None:
                _iren_k.RemoveObserver(_kbd_first_tag[0])
        except Exception:  # noqa: BLE001
            pass
        _kbd_first_tag[0] = None
        if _keyboard_on[0]:
            return  # already toggled manually before first render
        _keyboard_on[0] = True
        try:
            _keyboard_btn_ref[0].switch()
        except Exception:  # noqa: BLE001
            pass
        _show_status(
            "Keyboard navigation: ON "
            "(Avion: \u2190\u2192 yaw, \u2191\u2193 pitch, PgUp/Dn speed, KP7/8 roll)"
        )

    _iren_kbd = getattr(plt, "interactor", None)
    if _iren_kbd is not None:
        try:
            _kbd_first_tag[0] = _iren_kbd.AddObserver("RenderEvent", _kbd_activate_once)
        except Exception as _exc:  # noqa: BLE001
            logger.warning("Could not attach keyboard-ready observer: %s", _exc)
            # Fallback: arm immediately.
            _keyboard_on[0] = True

    # ─── Toggle gauges (bottom sliders) ─────────────────────────────────
    # Hides / shows all bottom sliders (mesh lighting + arrow length) in
    # every view.  Useful when the sliders obstruct the lower part of the
    # scene.  Always visible, independent of the navigation sub-mode.
    def _toggle_gauge_cb(*_a, **_kw):
        _gauge_state["on"] = not _gauge_state["on"]
        on = _gauge_state["on"]
        in_arrows = view_mode.get("name") in ("arrows_grid", "arrows_tracks")
        for s in mesh_sliders:
            _set_widget_visible(s, on and not in_arrows)
        for s in arrow_sliders:
            _set_widget_visible(s, on and in_arrows)
        _gauge_btn.switch()
        plt.render()

    _gauge_btn = plt.add_button(
        _toggle_gauge_cb,
        states=["Gauges ▸ on", "Gauges ▸ off"],
        c=["white", "#9e9e9e"],
        bc=["#00695c", "#263238"],
        pos=(0.35, 0.48),
        size=14,
        bold=True,
    )

    # KeySym sets. We accept both NumLock-on and NumLock-off variants of
    # the numpad keys so the roll bindings work either way.
    _AVION_KEYS_YAW_LEFT   = {"Left"}
    _AVION_KEYS_YAW_RIGHT  = {"Right"}
    _AVION_KEYS_PITCH_UP   = {"Up"}    # flèche haut = pitch vers le haut
    _AVION_KEYS_PITCH_DOWN = {"Down"}  # flèche bas = pitch vers le bas
    _AVION_KEYS_ACCEL      = {"z", "Z"}
    _AVION_KEYS_DECEL      = {"s", "S"}
    _AVION_KEYS_ROLL_LEFT  = {"q", "Q"}
    _AVION_KEYS_ROLL_RIGHT = {"d", "D"}

    # Resolve keysym names to X11 keycodes for fast polling.
    for _intent, _keysyms in (
        ("yaw_l",   _AVION_KEYS_YAW_LEFT),
        ("yaw_r",   _AVION_KEYS_YAW_RIGHT),
        ("pitch_u", _AVION_KEYS_PITCH_UP),
        ("pitch_d", _AVION_KEYS_PITCH_DOWN),
        ("roll_l",  _AVION_KEYS_ROLL_LEFT),
        ("roll_r",  _AVION_KEYS_ROLL_RIGHT),
        ("fwd",     _AVION_KEYS_ACCEL),
        ("bck",     _AVION_KEYS_DECEL),
    ):
        _AVION_KEYCODES[_intent] = _x11_resolve_keycodes(_keysyms)
    if _x_disp_plane is not None:
        logger.info("Avion keycodes resolved: %s",
                    {k: len(v) for k, v in _AVION_KEYCODES.items()})

    def _avion_keypress_cb(obj, _event):
        # Bail out silently in every situation that is not strictly
        # "Avion + keyboard armed + Navigation interaction style".
        # That keeps VTK / vedo default key handling and all other
        # interaction modes completely unaffected.
        if not _keyboard_on[0]:
            return
        if style_state["idx"] != style_state["custom_idx"]:
            return
        if _nav_mode[0] != 0:
            return
        try:
            key = obj.GetKeySym()
        except Exception:  # noqa: BLE001
            return
        if not key:
            return
        if key in _AVION_KEYS_YAW_LEFT:
            _plane_key_press("yaw_l")
        elif key in _AVION_KEYS_YAW_RIGHT:
            _plane_key_press("yaw_r")
        elif key in _AVION_KEYS_PITCH_UP:
            _plane_key_press("pitch_u")
        elif key in _AVION_KEYS_PITCH_DOWN:
            _plane_key_press("pitch_d")
        elif key in _AVION_KEYS_ROLL_LEFT:
            _plane_key_press("roll_l")
        elif key in _AVION_KEYS_ROLL_RIGHT:
            _plane_key_press("roll_r")
        elif key in _AVION_KEYS_ACCEL:
            _plane_key_press("fwd")
        elif key in _AVION_KEYS_DECEL:
            _plane_key_press("bck")
        else:
            return
        # Intent refreshed; the integrator-driven timer is already armed
        # by _plane_key_press().  No render here — the tick will request one.
        logger.debug("KBD-PRESS:   key=%-10s  held=%s", key,
                     {k: v for k, v in _plane_held.items() if v})
        _consume_key(obj)

    def _avion_keyrelease_cb(obj, _event):
        # Mirror of _avion_keypress_cb: schedule a deferred release for
        # the matching intent.  The grace period in _plane_key_release
        # filters X11's autorepeat fake-releases.
        if not _keyboard_on[0]:
            return
        if style_state["idx"] != style_state["custom_idx"]:
            return
        if _nav_mode[0] != 0:
            return
        try:
            key = obj.GetKeySym()
        except Exception:  # noqa: BLE001
            return
        if not key:
            return
        if key in _AVION_KEYS_YAW_LEFT:
            _plane_key_release("yaw_l")
        elif key in _AVION_KEYS_YAW_RIGHT:
            _plane_key_release("yaw_r")
        elif key in _AVION_KEYS_PITCH_UP:
            _plane_key_release("pitch_u")
        elif key in _AVION_KEYS_PITCH_DOWN:
            _plane_key_release("pitch_d")
        elif key in _AVION_KEYS_ROLL_LEFT:
            _plane_key_release("roll_l")
        elif key in _AVION_KEYS_ROLL_RIGHT:
            _plane_key_release("roll_r")
        elif key in _AVION_KEYS_ACCEL:
            _plane_key_release("fwd")
        elif key in _AVION_KEYS_DECEL:
            _plane_key_release("bck")
        else:
            return
        logger.debug("KBD-RELEASE: key=%-10s  release_at=%s",
                     key,
                     {k: f"{t:.3f}" for k, t in _plane_release_at.items()
                      if t is not None})

    def _consume_key(iren_obj, tag_ref=None) -> None:
        """Prevent vedo / VTK default key handlers from also seeing this
        keypress. Vedo binds `,` / `.` to opacity adjustments, `b` to
        background colour, etc.; without consuming, our numpad `.` /
        arrow keys would also trigger those side effects.

        Two complementary mechanisms are used:
        * Abort the current event invocation via the command's
          AbortFlag, which short-circuits VTK's observer loop (our
          observer is registered at priority 1.0, so it runs first).
        * Clear the KeySym / KeyCode so any handler that survives the
          abort sees no key at all.

        ``tag_ref`` selects which observer's command to abort
        (defaults to the KeyPress observer tag).
        """
        try:
            tag = (tag_ref or _avion_kp_tag)[0]
            if tag is not None:
                cmd = iren_obj.GetCommand(tag)
                if cmd is not None:
                    cmd.AbortFlagOn()
        except Exception:  # noqa: BLE001
            pass
        try:
            iren_obj.SetKeySym("")
        except Exception:  # noqa: BLE001
            pass
        try:
            iren_obj.SetKeyCode("\0")
        except Exception:  # noqa: BLE001
            pass

    _avion_kp_tag: list = [None]
    _avion_kr_tag: list = [None]
    if _iren_plane is not None:
        try:
            # Priority > 0 so our handler runs before vedo's default
            # keypress handler; combined with AbortFlagOn() in
            # _consume_key, vedo never sees the consumed keys.
            _avion_kp_tag[0] = _iren_plane.AddObserver(
                "KeyPressEvent", _avion_keypress_cb, 10.0,
            )
            _avion_kr_tag[0] = _iren_plane.AddObserver(
                "KeyReleaseEvent", _avion_keyrelease_cb, 10.0,
            )
            logger.info(
                "Avion keyboard observers (press + release) attached "
                "(off by default)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Avion keyboard observers could not attach: %s", exc,
            )

    # ─── Guard: re-assert interactor style on any mouse click ───────────
    # vedo can silently re-install its own observers or reset the
    # interactor style when the user clicks in the scene (picker,
    # lazy observer registration, etc.).  A high-priority mouse-press
    # observer detects the drift and corrects it immediately — this
    # prevents vedo from stealing key bindings mid-navigation.
    def _reassert_style_on_click(_obj, _event):
        if style_state["idx"] != style_state["custom_idx"]:
            return  # trackball mode: vedo is free to do its thing
        iren = getattr(plt, "interactor", None)
        if iren is None:
            return
        expected = style_state["instances"][style_state["idx"]]
        try:
            if iren.GetInteractorStyle() is not expected:
                iren.SetInteractorStyle(expected)
        except Exception:  # noqa: BLE001
            pass

    _iren_click_guard = getattr(plt, "interactor", None)
    if _iren_click_guard is not None:
        for _click_ev in (
            "LeftButtonPressEvent",
            "RightButtonPressEvent",
            "MiddleButtonPressEvent",
        ):
            try:
                _iren_click_guard.AddObserver(
                    _click_ev, _reassert_style_on_click, 10.0
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Style-guard observer (%s) could not attach: %s",
                    _click_ev, exc,
                )
        logger.info("Interactor style guard (click) attached.")

    # ─── Enter key → Restart navigation (respawn) ───────────────────────
    def _enter_keypress_cb(obj, _event):
        try:
            key = obj.GetKeySym()
        except Exception:  # noqa: BLE001
            return
        if key not in ("Return", "KP_Enter"):
            return
        _restart_nav_cb()
        # Consume so vedo's default Enter handler doesn't also fire.
        try:
            cmd = obj.GetCommand(_enter_kp_tag[0])
            if cmd is not None:
                cmd.AbortFlagOn()
        except Exception:  # noqa: BLE001
            pass
        try:
            obj.SetKeySym("")
        except Exception:  # noqa: BLE001
            pass

    _enter_kp_tag: list = [None]
    _iren_enter = getattr(plt, "interactor", None)
    if _iren_enter is not None:
        try:
            _enter_kp_tag[0] = _iren_enter.AddObserver(
                "KeyPressEvent", _enter_keypress_cb, 1.0,
            )
            logger.info("Enter-key respawn observer attached.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enter-key respawn observer could not attach: %s", exc)

    # Refresh visibility now that all button lists are populated:
    # Avion is the default → avion buttons visible, ship + custom hidden.
    _refresh_nav_panel_visibility()

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

    # ── Shader warmup ───────────────────────────────────────────────────
    # All lazily-created actors (lames, mask overlays, wind particles) are
    # now pre-added to the renderer but hidden (SetVisibility False).
    # VTK only compiles GLSL shader programs for *visible* actors, so the
    # first toggle-ON would still stall the event loop for 1–3 s while
    # shaders are compiled.
    # Fix: temporarily make each actor visible with opacity=0 (rendered
    # but invisible), call plt.render() once to trigger compilation, then
    # restore their hidden state.  Total cost: one extra render at startup.
    try:
        _wu_actors: list = []
        _wu_lames = lames_state.get("actor")
        if _wu_lames is not None and lames_state.get("added"):
            _wu_actors.append(_wu_lames)
        for _wu_ov in _mask_overlays:
            _wu_actors.append(getattr(_wu_ov, "actor", _wu_ov))
        for _wu_st in wind_states.values():
            _wu_a = _wu_st.get("actor")
            if _wu_a is not None and _wu_st.get("added"):
                _wu_actors.append(getattr(_wu_a, "actor", _wu_a))
        if _wu_actors:
            _wu_saved: list = []
            for _wu in _wu_actors:
                try:
                    _wu_saved.append(_wu.GetProperty().GetOpacity())
                    _wu.SetVisibility(True)
                    _wu.GetProperty().SetOpacity(0.0)
                except Exception:  # noqa: BLE001
                    _wu_saved.append(None)
            try:
                plt.render()
            except Exception as _wu_e:  # noqa: BLE001
                logger.warning("Shader warmup render failed: %s", _wu_e)
            for _wu, _op in zip(_wu_actors, _wu_saved):
                try:
                    _wu.SetVisibility(False)
                    if _op is not None:
                        _wu.GetProperty().SetOpacity(_op)
                except Exception:  # noqa: BLE001
                    pass
            logger.info(
                "Shader warmup: %d actor(s) compiled at startup.",
                len(_wu_actors),
            )
    except Exception as _wu_exc:  # noqa: BLE001
        logger.warning("Shader warmup failed (non-fatal): %s", _wu_exc)


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

    # ── Water "membranes" V1 removed (no longer loaded) ─────────────────
    membranes_data = None

    # ── Water "lames" (V2 / V3) descending animation (load-only) ──────
    # Prefer V3 (lame2.vtp) when present; fall back to V2 (lames.vtp).
    lames_data = None
    loaded_v3 = False
    if not getattr(args, "no_lame2", False):
        try:
            lame2_loaded = _load_lames(
                Path(args.lame2_vtp_cache).expanduser().resolve(),
                Path(args.lame2_meta_cache).expanduser().resolve(),
            )
            if lame2_loaded != (None, None):
                lames_data = lame2_loaded
                loaded_v3 = True
                # Slight specular bump for the V3 (lame2) actor since the
                # progression spends more time on screen and a touch of
                # highlight reads nicely even on translucent surfaces.
                try:
                    _, _actor = lames_data
                    prop = _actor.GetProperty()
                    prop.SetSpecular(0.25)
                    prop.SetSpecularPower(20)
                    prop.SetSpecularColor(1.0, 1.0, 1.0)
                except Exception:  # noqa: BLE001
                    pass
                logger.info("Water-lame2 (V3) cache loaded — taking over.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load lame2 cache: %s", exc)
            lames_data = None

    if lames_data is None and not args.no_lames:
        try:
            lames_data = _load_lames(
                Path(args.lames_vtp_cache).expanduser().resolve(),
                Path(args.lames_meta_cache).expanduser().resolve(),
            )
            if lames_data == (None, None):
                lames_data = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load lames cache: %s", exc)
            lames_data = None

    # ── Stele / Outside mask overlays ───────────────────────────────────
    mask_overlay_meshes: list = []
    if not getattr(args, "no_mask_overlays", False):
        import vedo as _vedo_masks
        for _cache_arg, _label in (
            (args.stele_mask_cache,   "stele"),
            (args.outside_mask_cache, "outside"),
        ):
            _cpath = Path(_cache_arg).expanduser().resolve()
            if _cpath.exists():
                try:
                    _m = _vedo_masks.Mesh(str(_cpath))
                    _m.alpha(DEFAULT_LAMES_ALPHA)
                    _m.color([c / 255.0 for c in DEFAULT_LAMES_COLOR])
                    _m.lighting("off")
                    _m.name = f"{_label}_mask_overlay"
                    mask_overlay_meshes.append(_m)
                    logger.info("Mask overlay '%s' loaded from %s", _label, _cpath)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load mask overlay %s: %s",
                                   _cpath, exc)
            else:
                logger.info("Mask overlay '%s' cache not found: %s — skipped.",
                            _label, _cpath)

    # ── Wind / gas particles (O₂ + CH₄) ─────────────────────────────────
    wind_data: dict = {}
    if not getattr(args, "no_wind", False):
        try:
            from marvel_view.preprocessing.wind_field import (
                load_particle_templates as _load_wind_tpl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("wind_field module unavailable: %s — wind disabled.", exc)
            _load_wind_tpl = None  # type: ignore[assignment]

        if _load_wind_tpl is not None:
            for _species, _arg_cache, _display, _color, _psize in (
                ("o2",  args.wind_o2_cache,  int(args.wind_o2_display),
                 DEFAULT_WIND_O2_COLOR,  DEFAULT_WIND_O2_POINT_SIZE),
                ("ch4", args.wind_ch4_cache, int(args.wind_ch4_display),
                 DEFAULT_WIND_CH4_COLOR, DEFAULT_WIND_CH4_POINT_SIZE),
            ):
                _path = Path(_arg_cache).expanduser().resolve()
                if not _path.exists():
                    logger.info("Wind %s cache not found: %s — skipped.",
                                _species.upper(), _path)
                    continue
                try:
                    _tpl = _load_wind_tpl(_path)
                    wind_data[_species] = {
                        "tpl":        _tpl,
                        "display":    _display,
                        "color":      _color,
                        "point_size": _psize,
                    }
                    logger.info(
                        "Wind %s templates loaded: %d × %d frames @ %d fps "
                        "(display=%d, size=%.1fpx)",
                        _species.upper(),
                        _tpl["n_templates"], _tpl["n_frames"], _tpl["fps"],
                        _display, _psize,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not load wind %s cache (%s): %s",
                                   _species.upper(), _path, exc)

    _attach_controls(
        plt, mesh,
        arrows_data=arrows_data,
        ortho_overlay=ortho_overlay,
        alt_mesh=alt_mesh,
        pillars_mesh=pillars_mesh,
        tracks_data=tracks_data,
        density_scalars=density_scalars,
        membranes_data=membranes_data,
        lames_data=lames_data,
        mask_overlay_meshes=mask_overlay_meshes,
        wind_data=wind_data,
    )
    plt.interactive()
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

