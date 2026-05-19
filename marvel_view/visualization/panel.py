"""
Interactive control panel for the vedo viewer.

Adds two widgets to the plotter:

* **Visibility column (top-left):** one toggle button per label.  Clicking a
  button hides/shows every actor belonging to that label.  Button colour
  matches the label's configured colour.

* **Tuning row (bottom):** a "Tune" selector button that cycles through the
  labels, plus four sliders wired to the currently-selected label:
  ``opacity``, ``ambient``, ``diffuse``, ``specular``.  Each label keeps
  its own independent set of values, so switching the selector simply
  rebinds the sliders and updates their displayed handle position.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from .actors import _normalise_color

logger = logging.getLogger(__name__)


# ──────────────────────────────── helpers ─────────────────────────────────────


def _contrast_text(rgb: Tuple[int, int, int]) -> str:
    """Return ``'black'`` or ``'white'`` for best contrast on ``rgb``."""
    r, g, b = rgb
    # Rec. 601 luma
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "white"


def _apply_visibility(actors: list, visible: bool) -> None:
    for m in actors:
        # vedo.Mesh is a vtkActor subclass → SetVisibility is available.
        try:
            m.SetVisibility(bool(visible))
        except AttributeError:
            # Fallback: fade to zero / restore a cached alpha.
            if visible:
                m.alpha(getattr(m, "_saved_alpha", 1.0))
            else:
                m._saved_alpha = m.alpha()
                m.alpha(0.0)


def _apply_lighting(actors: list, ambient: float, diffuse: float, specular: float) -> None:
    for m in actors:
        m.lighting(ambient=ambient, diffuse=diffuse, specular=specular)


def _apply_opacity(actors: list, opacity: float) -> None:
    for m in actors:
        m.alpha(float(opacity))


# VTK interpolation modes (vtkProperty::SetInterpolation)
#   0 = Flat, 1 = Gouraud, 2 = Phong, 3 = PBR
_SHADING_MODES: List[Tuple[str, int]] = [
    ("Phong",   2),
    ("Gouraud", 1),
    ("Flat",    0),
    ("PBR",     3),
]


def _apply_shading(actors_by_label: Dict[int, list], mode: int) -> None:
    for group in actors_by_label.values():
        for m in group:
            prop = getattr(m, "properties", None)
            if prop is None:
                # Fallback for older vedo / raw vtkActor.
                try:
                    prop = m.GetProperty()
                except AttributeError:
                    continue
            prop.SetInterpolation(int(mode))


# ──────────────────────────────── main entry point ───────────────────────────


def attach_panel(
    plt,
    actors_by_label: Dict[int, list],
    label_config: Dict[int, dict],
) -> None:
    """Attach visibility buttons + per-label tuning sliders to ``plt``.

    Parameters
    ----------
    plt:
        A :class:`vedo.Plotter` instance, already populated with meshes
        via :meth:`vedo.Plotter.show`.
    actors_by_label:
        Mapping ``label_id -> [vedo.Mesh, …]``.  Each list groups every
        mesh that belongs to the same label.
    label_config:
        Slice of :data:`marvel_view.config.LABEL_CONFIG` containing only
        the labels that are actually rendered (same keys as
        ``actors_by_label``).
    """
    # Deterministic ordering so button positions don't jitter between runs.
    label_ids: List[int] = sorted(actors_by_label.keys())
    if not label_ids:
        return

    # Per-label state: current values of the four sliders.  We seed these
    # with the values used when the meshes were first styled.
    state: Dict[int, Dict[str, float]] = {}
    visible: Dict[int, bool] = {}
    for lid in label_ids:
        cfg = label_config.get(lid, {})
        first = actors_by_label[lid][0] if actors_by_label[lid] else None
        state[lid] = {
            "opacity":  float(cfg.get("opacity", 1.0)),
            "ambient":  float(cfg.get("ambient",  0.1)),
            "diffuse":  float(cfg.get("diffuse",  0.8)),
            "specular": float(cfg.get("specular", 0.25)),
        }
        visible[lid] = True
        del first  # only used for documentation clarity

    # ────────────── visibility buttons (top-left, vertical) ───────────────
    for i, lid in enumerate(label_ids):
        cfg = label_config.get(lid, {})
        name = cfg.get("name", f"label{lid}")
        col  = cfg.get("color", (200, 200, 200))
        rgb  = col if isinstance(col, (list, tuple)) else (200, 200, 200)
        hex_col = "#%02x%02x%02x" % tuple(int(c) for c in rgb)
        txt_col = _contrast_text(rgb)
        y = 0.95 - i * 0.055

        def _make_vis_cb(label_id: int, actors: list):
            def _cb(*_a, **_kw):
                visible[label_id] = not visible[label_id]
                _apply_visibility(actors, visible[label_id])
                plt.render()
            return _cb

        plt.add_button(
            _make_vis_cb(lid, actors_by_label[lid]),
            states=[f"◉ {lid}  {name}", f"○ {lid}  {name}"],
            c=[txt_col, txt_col],
            bc=[hex_col, "#444444"],
            pos=(0.12, y),
            size=18,
            bold=True,
        )

    # ────────────── tuning selector + sliders (bottom) ────────────────────
    selected = [label_ids[0]]   # mutable box so nested callbacks can write to it

    def _refresh_sliders() -> None:
        """Push the active label's state back to the slider widgets."""
        lid = selected[0]
        s = state[lid]
        for slider, key in _slider_registry:
            try:
                rep = slider.GetRepresentation()
                rep.SetValue(s[key])
                rep.BuildRepresentation()
            except Exception:  # noqa: BLE001
                pass
        plt.render()

    def _make_slider_cb(key: str):
        def _cb(widget, _event):
            lid = selected[0]
            val = float(widget.value)
            state[lid][key] = val
            if key == "opacity":
                _apply_opacity(actors_by_label[lid], val)
            else:
                s = state[lid]
                _apply_lighting(
                    actors_by_label[lid],
                    ambient=s["ambient"],
                    diffuse=s["diffuse"],
                    specular=s["specular"],
                )
            plt.render()
        return _cb

    _slider_registry: List[Tuple[object, str]] = []

    # 4 horizontal sliders stacked at the bottom, spanning the middle
    # 55% of the window width.
    x1, x2 = 0.30, 0.85
    slider_defs = [
        ("opacity",  0.0, 1.0, 0.04),
        ("ambient",  0.0, 1.0, 0.09),
        ("diffuse",  0.0, 1.0, 0.14),
        ("specular", 0.0, 1.0, 0.19),
    ]
    init = state[selected[0]]
    for key, vmin, vmax, y in slider_defs:
        sl = plt.add_slider(
            _make_slider_cb(key),
            vmin, vmax, value=init[key],
            pos=((x1, y), (x2, y)),
            title=key,
            title_size=0.7,
            show_value=True,
        )
        _slider_registry.append((sl, key))

    # ────────────── selector button: cycles the active label ──────────────
    def _cycle_cb(*_a, **_kw):
        i = label_ids.index(selected[0])
        selected[0] = label_ids[(i + 1) % len(label_ids)]
        _selector_btn.switch()  # rotate the button's displayed state
        _refresh_sliders()

    _states = [
        f"Tune  ▸  {lid}  {label_config.get(lid, {}).get('name', '')}"
        for lid in label_ids
    ]
    _colors_bg = [
        "#%02x%02x%02x" % tuple(
            int(c) for c in label_config.get(lid, {}).get("color", (80, 80, 80))
        )
        for lid in label_ids
    ]
    _colors_fg = [
        _contrast_text(label_config.get(lid, {}).get("color", (80, 80, 80)))
        for lid in label_ids
    ]
    _selector_btn = plt.add_button(
        _cycle_cb,
        states=_states,
        c=_colors_fg,
        bc=_colors_bg,
        pos=(0.15, 0.24),
        size=16,
        bold=True,
    )

    # ────────────── shading mode selector (top-right) ─────────────────────
    # Cycles Phong → Gouraud → Flat → PBR on every mesh.  Applied globally
    # because per-label shading mode would be visually confusing (different
    # interpolation models side-by-side rarely helps interpretation).
    shading_idx = [0]  # start on Phong
    _apply_shading(actors_by_label, _SHADING_MODES[0][1])

    def _shading_cb(*_a, **_kw):
        shading_idx[0] = (shading_idx[0] + 1) % len(_SHADING_MODES)
        _apply_shading(actors_by_label, _SHADING_MODES[shading_idx[0]][1])
        _shading_btn.switch()
        plt.render()

    _shading_btn = plt.add_button(
        _shading_cb,
        states=[f"Shading ▸ {name}" for name, _ in _SHADING_MODES],
        c=["white"] * len(_SHADING_MODES),
        bc=["#3b5b8c", "#6b8e23", "#8c6b3b", "#703b8c"],
        pos=(0.88, 0.95),
        size=16,
        bold=True,
    )

    # ────────────── interaction style + walk keys ─────────────────────────
    # Stick to TrackballCamera (the only style that's actually useful here)
    # and bind w/s to a real "walk forward / backward" along the view
    # direction.  The default VTK mouse wheel only dollies toward a fixed
    # focal point, which feels like a zoom when exploring inside a long
    # object; w/s translate camera and focal point together so you really
    # move through the scene.
    import vtk

    _trackball_style = vtk.vtkInteractorStyleTrackballCamera()

    iren = getattr(plt, "interactor", None)
    if iren is not None:
        iren.SetInteractorStyle(_trackball_style)
    else:
        logger.warning("No interactor available – cannot set TrackballCamera style.")

    def _walk_camera(step_frac: float) -> None:
        cam = plt.camera
        pos = list(cam.GetPosition())
        fp = list(cam.GetFocalPoint())
        direction = cam.GetDirectionOfProjection()
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
        try:
            key = obj.GetKeySym()
        except Exception:  # noqa: BLE001
            return
        if key in ("w", "W"):
            _walk_camera(+0.05)
        elif key in ("s", "S"):
            _walk_camera(-0.05)

    if iren is not None:
        iren.AddObserver("KeyPressEvent", _keypress_cb)

    # ────────────── save current camera position (top-right) ──────────────
    # Each click appends the current camera state to a JSON file in
    # ./positions/.  All saves of a single session go to the same file
    # so they can later be replayed as a fly-through / movie.
    positions_dir = (
        Path(os.environ.get("MARVEL_DATA_DIR", "/home/rfernandez/Data/Arize/Hollow_test"))
        / "positions"
    )
    positions_dir.mkdir(parents=True, exist_ok=True)
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    positions_file = positions_dir / f"positions_{session_stamp}.json"
    save_counter = [0]

    def _camera_state() -> dict:
        cam = plt.camera
        return {
            "position":       list(cam.GetPosition()),
            "focal_point":    list(cam.GetFocalPoint()),
            "view_up":        list(cam.GetViewUp()),
            "view_angle":     float(cam.GetViewAngle()),
            "clipping_range": list(cam.GetClippingRange()),
            "parallel_scale": float(cam.GetParallelScale()),
            "distance":       float(cam.GetDistance()),
        }

    def _save_cb(*_a, **_kw):
        entry = {
            "index":     save_counter[0],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera":    _camera_state(),
        }
        if positions_file.exists():
            try:
                data = json.loads(positions_file.read_text())
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        data.append(entry)
        positions_file.write_text(json.dumps(data, indent=2))
        save_counter[0] += 1
        logger.info("Saved camera position #%d → %s", entry["index"], positions_file)
        plt.render()

    _save_btn = plt.add_button(
        _save_cb,
        states=["📷 Save position"],
        c=["white"],
        bc=["#444444"],
        pos=(0.88, 0.83),
        size=16,
        bold=True,
    )
