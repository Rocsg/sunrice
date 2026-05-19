"""
Roots dataset interactive viewer.

Three view modes accessible via a toolbar button:

* ``mesh``   – preprocessed VTK meshes (one renderer, full window).
* ``volume`` – direct volumetric rendering of the 8-bit source TIFF
                with two threshold sliders + an opacity slider.
* ``split``  – two renderers side by side (meshes left, volume right),
                shared camera so they stay in sync while orbiting.

Lighting is set up as a single camera-attached light (slightly offset)
to produce a "torchlight in a tunnel" feel.

A small panel offers:

* per-label visibility toggles + tuning sliders (opacity / lighting),
* volume threshold + opacity sliders (active in volume / split modes),
* **Save** button → writes the current setup to a JSON sidecar next to
  the data (no path is asked),
* **Reset** button → restores the initial state captured at startup,
* **Mode** button → cycles mesh → volume → split → mesh (rebuilds the
  plotter; the sidecar is reloaded so settings persist across modes).

The whole application is a single :class:`RootsViewer` instance; it owns
the plotter, all actors, the volume, and the settings dict.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .actors import _normalise_color, style_mesh

logger = logging.getLogger(__name__)


# ──────────────────────────────── helpers ─────────────────────────────────────


def _contrast_text(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "white"


def _hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def _set_actor_visibility(mesh, visible: bool) -> None:
    """Toggle visibility of a vedo mesh, robust across vedo versions.

    In recent vedo releases, ``Mesh`` is no longer a ``vtkActor`` subclass
    — the underlying actor lives on ``mesh.actor``.  Fall back to
    ``mesh.SetVisibility`` for older versions.
    """
    actor = getattr(mesh, "actor", None)
    if actor is not None and hasattr(actor, "SetVisibility"):
        actor.SetVisibility(bool(visible))
        return
    if hasattr(mesh, "SetVisibility"):
        mesh.SetVisibility(bool(visible))
        return
    # Last-resort fallback: fade alpha to 0.
    if visible:
        mesh.alpha(getattr(mesh, "_saved_alpha", 1.0))
    else:
        mesh._saved_alpha = mesh.alpha()
        mesh.alpha(0.0)


# ──────────────────────────────── settings I/O ────────────────────────────────


_VIEW_MODES = ("mesh", "volume", "split")

# Volume colormaps cycled by the LUT button.  Names are vedo / matplotlib
# colormap identifiers; all are tone-mapped grayscale-friendly choices that
# work well on 8-bit microscopy data.
_VOLUME_CMAPS: Tuple[str, ...] = (
    "bone", "gray", "hot", "viridis", "magma", "plasma",
    "inferno", "cividis", "coolwarm", "jet",
)

# Mesh shading modes (vtkProperty::SetInterpolation): name → VTK constant.
_SHADING_MODES: Tuple[Tuple[str, int], ...] = (
    ("Phong",   2),
    ("Gouraud", 1),
    ("Flat",    0),
    ("PBR",     3),
)


def _default_settings(rcfg) -> Dict[str, Any]:
    """Build the default settings dict from the roots_config module."""
    labels: Dict[str, Dict[str, float]] = {}
    for lid, cfg in rcfg.LABEL_CONFIG.items():
        labels[str(lid)] = {
            "visible":  True,
            "opacity":  float(cfg.get("opacity", 1.0)),
            "ambient":  float(cfg.get("ambient", 0.15)),
            "diffuse":  float(cfg.get("diffuse", 0.85)),
            "specular": float(cfg.get("specular", 0.20)),
        }
    return {
        "view_mode":   rcfg.VIEW_MODE_DEFAULT,
        "background":  rcfg.BACKGROUND_DEFAULT,
        "shading_mode": 2,  # Phong
        "labels":      labels,
        "volume": {
            "thr_low":  int(rcfg.VOLUME_THR_LOW_DEFAULT),
            "thr_high": int(rcfg.VOLUME_THR_HIGH_DEFAULT),
            "opacity":  float(rcfg.VOLUME_OPACITY_DEFAULT),
            "cmap":     "bone",
        },
        "camera": None,  # filled at first render
    }


def _load_settings(path: Path, defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        logger.info("No settings file at %s — starting from defaults.", path)
        return copy.deepcopy(defaults)
    try:
        with path.open("r") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s (%s) — using defaults.", path, exc)
        return copy.deepcopy(defaults)

    # Merge defaults <- loaded so missing keys still resolve.
    out = copy.deepcopy(defaults)
    for k, v in loaded.items():
        if k == "labels" and isinstance(v, dict):
            for lid, vals in v.items():
                if lid in out["labels"] and isinstance(vals, dict):
                    out["labels"][lid].update(vals)
        elif k == "volume" and isinstance(v, dict):
            out["volume"].update(v)
        else:
            out[k] = v
    if out.get("view_mode") not in _VIEW_MODES:
        out["view_mode"] = defaults["view_mode"]
    logger.info("Loaded settings from %s", path)
    return out


def _save_settings(path: Path, settings: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved to %s", path)
    except OSError as exc:
        logger.warning("Failed to save settings to %s: %s", path, exc)


# ──────────────────────────────── volume helpers ──────────────────────────────


def _load_source_volume(path: Path) -> np.ndarray:
    import tifffile
    logger.info("Loading source volume %s …", path)
    t = time.perf_counter()
    arr = tifffile.imread(str(path))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D source volume, got shape {arr.shape}")
    logger.info(
        "  └─ loaded  shape=%s  dtype=%s  (%.1f s)",
        arr.shape, arr.dtype, time.perf_counter() - t,
    )
    if arr.dtype != np.uint8:
        # Rescale to 0-255 for a stable threshold range across datasets.
        a = arr.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            a = (a - lo) * (255.0 / (hi - lo))
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def _build_volume(arr: np.ndarray, spacing: Tuple[float, float, float]):
    import vedo
    vol = vedo.Volume(arr, spacing=spacing)
    vol.mode(0)  # composite
    # "smart" mapper auto-falls-back to CPU when GPU volume rendering is
    # unavailable / unstable.  Crucial on large (>500 MB) uint8 volumes
    # where the default GPU mapper has been observed to segfault.
    try:
        vol.mapper("smart")
    except Exception as exc:  # noqa: BLE001
        logger.warning("vol.mapper('smart') failed (%s) — using default mapper.", exc)
    # Enable shading so the scene lights (incl. our camera-attached
    # torchlight) actually contribute — otherwise the volume is rendered
    # as pure absorption and looks very dark.
    try:
        prop = vol.properties
        prop.ShadeOn()
        prop.SetAmbient(0.30)
        prop.SetDiffuse(0.80)
        prop.SetSpecular(0.20)
        prop.SetSpecularPower(20.0)
        # Linear interpolation gives a smoother volume than nearest-neighbour.
        prop.SetInterpolationTypeToLinear()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Volume property setup failed: %s", exc)
    return vol


def _apply_volume_transfer(
    vol,
    thr_low: int,
    thr_high: int,
    opacity: float,
    cmap: str = "bone",
) -> None:
    """Apply a band-pass alpha + colormap to a vedo.Volume."""
    lo = max(0, min(255, int(thr_low)))
    hi = max(lo + 1, min(255, int(thr_high)))
    op = max(0.0, min(1.0, float(opacity)))

    # Sharp band: 0 outside [lo, hi], op inside.
    alpha = [(0, 0.0),
             (max(0, lo - 1), 0.0),
             (lo, op),
             (hi, op),
             (min(255, hi + 1), 0.0),
             (255, 0.0)]
    vol.alpha(alpha)
    try:
        vol.cmap(cmap)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vol.cmap(%r) failed (%s) — keeping previous LUT.", cmap, exc)


# ──────────────────────────────── lighting ────────────────────────────────────


def _setup_torchlight(plt, intensity: float = 1.2, offset: float = 0.15) -> None:
    """Replace renderer lights with a single camera-attached light.

    Slight off-axis position relative to the camera gives the
    "torchlight in a tunnel" feel the user asked for.
    """
    import vtk
    for renderer in _iter_renderers(plt):
        renderer.AutomaticLightCreationOff()
        renderer.RemoveAllLights()
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        # In camera coords: +Z is along the view direction "out of" the
        # camera; small X/Y offsets shift the light slightly off-axis.
        light.SetPosition(offset, offset, 1.0)
        light.SetFocalPoint(0.0, 0.0, 0.0)
        light.SetIntensity(intensity)
        light.SetColor(1.0, 0.97, 0.92)  # warm-white
        renderer.AddLight(light)


def _iter_renderers(plt):
    # vedo's Plotter exposes ``renderers`` for multi-view layouts.
    rs = getattr(plt, "renderers", None)
    if rs:
        yield from rs
    else:
        yield plt.renderer


# ──────────────────────────────── main viewer class ───────────────────────────


class RootsViewer:
    """Interactive viewer for the roots dataset.

    A single instance is reused across view-mode switches: when the user
    cycles ``mesh → volume → split``, we tear down the current plotter
    and build a fresh one, but keep the underlying actors / volume in
    memory and reapply the settings dict.
    """

    def __init__(
        self,
        rcfg,
        vtk_dir: Path,
        source_path: Optional[Path] = None,
        settings_path: Optional[Path] = None,
    ) -> None:
        self.rcfg = rcfg
        self.vtk_dir = Path(vtk_dir)
        self.source_path = Path(source_path) if source_path else None
        self.settings_path = Path(settings_path) if settings_path else None

        # Captured at first build, used by the Reset button.
        self.defaults: Dict[str, Any] = _default_settings(rcfg)
        # Settings on disk override defaults at startup.
        if self.settings_path and self.settings_path.exists():
            self.settings = _load_settings(self.settings_path, self.defaults)
        else:
            self.settings = copy.deepcopy(self.defaults)

        # Built lazily.
        self._actors_by_label: Dict[int, list] = {}
        self._meshes_loaded: bool = False
        self._volume = None
        self._volume_array: Optional[np.ndarray] = None
        self._plt = None
        # Slider widgets, indexed by key, so Reset can push values back.
        self._sliders: List[Tuple[object, str, Optional[int]]] = []

        # ── position-recording state ────────────────────────────────────
        # Each click on "Save pos" appends to one JSON file per session,
        # under ./positions/.  Created lazily on first save.
        self._positions_dir: Path = Path(
            os.environ.get(
                "MARVEL_POSITIONS_DIR",
                "/home/rfernandez/Data/Arize/Hollow_test/positions",
            )
        )
        self._positions_file: Optional[Path] = None
        self._positions_count: int = 0

    # ── persistence ──────────────────────────────────────────────────────

    def _capture_camera(self) -> None:
        if self._plt is None:
            return
        try:
            cam = self._plt.camera
            self.settings["camera"] = {
                "pos":        list(cam.GetPosition()),
                "focal":      list(cam.GetFocalPoint()),
                "up":         list(cam.GetViewUp()),
                "view_angle": float(cam.GetViewAngle()),
            }
        except Exception:  # noqa: BLE001
            pass

    def _restore_camera(self) -> None:
        if self._plt is None:
            return
        cam_settings = self.settings.get("camera")
        if not cam_settings:
            return
        try:
            for renderer in _iter_renderers(self._plt):
                cam = renderer.GetActiveCamera()
                cam.SetPosition(*cam_settings["pos"])
                cam.SetFocalPoint(*cam_settings["focal"])
                cam.SetViewUp(*cam_settings["up"])
                cam.SetViewAngle(cam_settings["view_angle"])
                renderer.ResetCameraClippingRange()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Restore camera failed: %s", exc)

    # ── mesh / volume loading (idempotent) ───────────────────────────────

    def _load_meshes(self) -> None:
        if self._meshes_loaded:
            return
        import vedo
        for lid, cfg in self.rcfg.LABEL_CONFIG.items():
            label_dir = self.vtk_dir / f"label{lid}_{cfg['name']}"
            if not label_dir.exists():
                logger.warning(
                    "No directory for label %d (%s) under %s — skipping.",
                    lid, cfg["name"], self.vtk_dir,
                )
                continue
            files = sorted(label_dir.glob("*.vtk"))
            if not files:
                logger.warning("No VTK files in %s — skipping.", label_dir)
                continue
            group: list = []
            for f in files:
                m = vedo.Mesh(str(f))
                style_mesh(
                    m,
                    color=cfg["color"],
                    opacity=cfg.get("opacity", 1.0),
                    name=f"{cfg['name']}_{f.stem}",
                    ambient=cfg.get("ambient", 0.15),
                    diffuse=cfg.get("diffuse", 0.85),
                    specular=cfg.get("specular", 0.20),
                )
                group.append(m)
                logger.info("  [label %d] loaded %s  (%d faces)",
                            lid, f.name, m.ncells)
            if group:
                self._actors_by_label[lid] = group
        self._meshes_loaded = True

    def _load_volume(self) -> None:
        if self._volume is not None:
            return
        if self.source_path is None or not self.source_path.exists():
            logger.warning(
                "Source volume not available at %s — volume mode disabled.",
                self.source_path,
            )
            return
        arr = _load_source_volume(self.source_path)
        self._volume_array = arr
        self._volume = _build_volume(arr, self.rcfg.DEFAULT_SPACING)
        v = self.settings["volume"]
        _apply_volume_transfer(
            self._volume, v["thr_low"], v["thr_high"], v["opacity"],
            cmap=v.get("cmap", "bone"),
        )

    # ── apply settings to actors ─────────────────────────────────────────

    def _apply_label_settings(self) -> None:
        for lid, group in self._actors_by_label.items():
            s = self.settings["labels"].get(str(lid))
            if not s:
                continue
            for m in group:
                m.alpha(float(s["opacity"]))
                m.lighting(
                    ambient=float(s["ambient"]),
                    diffuse=float(s["diffuse"]),
                    specular=float(s["specular"]),
                )
                _set_actor_visibility(m, bool(s["visible"]))

    def _apply_volume_settings(self) -> None:
        if self._volume is None:
            return
        v = self.settings["volume"]
        _apply_volume_transfer(
            self._volume, v["thr_low"], v["thr_high"], v["opacity"],
            cmap=v.get("cmap", "bone"),
        )

    def _apply_shading(self) -> None:
        """Push current shading_mode to every mesh's vtkProperty."""
        mode = int(self.settings.get("shading_mode", 2))
        for group in self._actors_by_label.values():
            for m in group:
                prop = getattr(m, "properties", None)
                if prop is None:
                    try:
                        prop = m.GetProperty()
                    except AttributeError:
                        continue
                try:
                    prop.SetInterpolation(mode)
                except Exception:  # noqa: BLE001
                    pass

    # ── plotter (re)build ────────────────────────────────────────────────

    def _build_plotter(self) -> None:
        import vedo
        mode = self.settings["view_mode"]
        bg = self.settings.get("background", "black")
        title = f"Marvel View – Roots  ({mode})"

        if mode == "split":
            self._plt = vedo.Plotter(
                shape=(1, 2), sharecam=True, bg=bg, title=title, axes=1,
            )
        else:
            self._plt = vedo.Plotter(bg=bg, title=title, axes=1)

        # Populate renderers.
        if mode == "mesh":
            self._load_meshes()
            self._apply_label_settings()
            for group in self._actors_by_label.values():
                self._plt.add(*group)
        elif mode == "volume":
            self._load_volume()
            if self._volume is not None:
                self._plt.add(self._volume)
        elif mode == "split":
            self._load_meshes()
            self._apply_label_settings()
            for group in self._actors_by_label.values():
                for m in group:
                    self._plt.at(0).add(m)
            self._load_volume()
            if self._volume is not None:
                self._plt.at(1).add(self._volume)

        _setup_torchlight(self._plt)
        self._apply_shading()
        self._sliders = []
        self._attach_panel()

    # ── panel widgets ────────────────────────────────────────────────────

    def _attach_panel(self) -> None:
        """Attach buttons + sliders to the current plotter."""
        cfg_map = self.rcfg.LABEL_CONFIG
        label_ids = sorted(self._actors_by_label.keys()) or sorted(cfg_map.keys())

        # ─── visibility toggles (top-left) ──────────────────────────────
        for i, lid in enumerate(label_ids):
            cfg = cfg_map.get(lid, {})
            name = cfg.get("name", f"label{lid}")
            rgb = cfg.get("color", (200, 200, 200))
            hex_col = _hex(rgb)
            txt_col = _contrast_text(rgb)
            y = 0.95 - i * 0.055

            def _make_vis_cb(label_id: int):
                def _cb(*_a, **_kw):
                    s = self.settings["labels"].setdefault(
                        str(label_id), copy.deepcopy(self.defaults["labels"][str(label_id)])
                    )
                    s["visible"] = not s["visible"]
                    for m in self._actors_by_label.get(label_id, []):
                        _set_actor_visibility(m, bool(s["visible"]))
                    self._plt.render()
                return _cb

            self._plt.at(0).add_button(
                _make_vis_cb(lid),
                states=[f"◉ {lid}  {name}", f"○ {lid}  {name}"],
                c=[txt_col, txt_col],
                bc=[hex_col, "#444444"],
                pos=(0.12, y),
                size=16,
                bold=True,
            )

        # ─── per-label tuning sliders (bottom) ──────────────────────────
        if label_ids:
            self._selected_lid = [label_ids[0]]
            slider_defs = [
                ("opacity",  0.0, 1.0, 0.04),
                ("ambient",  0.0, 1.0, 0.09),
                ("diffuse",  0.0, 1.0, 0.14),
                ("specular", 0.0, 1.0, 0.19),
            ]

            def _make_label_slider_cb(key: str):
                def _cb(widget, _event):
                    lid = self._selected_lid[0]
                    val = float(widget.value)
                    s = self.settings["labels"].setdefault(
                        str(lid), copy.deepcopy(self.defaults["labels"][str(lid)])
                    )
                    s[key] = val
                    for m in self._actors_by_label.get(lid, []):
                        if key == "opacity":
                            m.alpha(val)
                        else:
                            m.lighting(
                                ambient=float(s["ambient"]),
                                diffuse=float(s["diffuse"]),
                                specular=float(s["specular"]),
                            )
                    self._plt.render()
                return _cb

            init = self.settings["labels"][str(label_ids[0])]
            for key, vmin, vmax, y in slider_defs:
                sl = self._plt.at(0).add_slider(
                    _make_label_slider_cb(key),
                    vmin, vmax, value=init[key],
                    pos=((0.05, y), (0.32, y)),
                    title=key, title_size=0.7, show_value=True,
                )
                self._sliders.append((sl, key, None))  # None = label slider

            # Selector button cycles which label the sliders act on.
            def _cycle_cb(*_a, **_kw):
                ids = label_ids
                i = ids.index(self._selected_lid[0])
                self._selected_lid[0] = ids[(i + 1) % len(ids)]
                self._refresh_label_sliders()
                _selector_btn.switch()

            _states = [
                f"Tune ▸ {lid}  {cfg_map.get(lid, {}).get('name', '')}"
                for lid in label_ids
            ]
            _bc = [_hex(cfg_map.get(lid, {}).get("color", (80, 80, 80)))
                   for lid in label_ids]
            _fg = [_contrast_text(cfg_map.get(lid, {}).get("color", (80, 80, 80)))
                   for lid in label_ids]
            _selector_btn = self._plt.at(0).add_button(
                _cycle_cb, states=_states, c=_fg, bc=_bc,
                pos=(0.15, 0.24), size=14, bold=True,
            )

        # ─── volume sliders (right side, only when volume is loaded) ────
        # In split mode we attach them to renderer 1; in volume mode to
        # renderer 0; not attached in mesh mode.
        mode = self.settings["view_mode"]
        if mode in ("volume", "split") and self._volume is not None:
            target = 1 if mode == "split" else 0
            v = self.settings["volume"]

            def _make_vol_cb(key: str):
                def _cb(widget, _event):
                    val = float(widget.value)
                    self.settings["volume"][key] = (
                        int(val) if key in ("thr_low", "thr_high") else val
                    )
                    self._apply_volume_settings()
                    self._plt.render()
                return _cb

            for key, vmin, vmax, y, init in [
                ("thr_low",  0,   255, 0.04, v["thr_low"]),
                ("thr_high", 0,   255, 0.09, v["thr_high"]),
                ("opacity",  0.0, 1.0, 0.14, v["opacity"]),
            ]:
                sl = self._plt.at(target).add_slider(
                    _make_vol_cb(key),
                    vmin, vmax, value=init,
                    pos=((0.05, y), (0.32, y)),
                    title=f"vol.{key}", title_size=0.7, show_value=True,
                )
                self._sliders.append((sl, key, target))

            # ─── LUT cycle button (above the volume sliders) ────────────
            cmaps = list(_VOLUME_CMAPS)
            current_cmap = self.settings["volume"].get("cmap", "bone")
            try:
                cmap_idx = [cmaps.index(current_cmap)]
            except ValueError:
                cmap_idx = [0]

            def _cycle_lut_cb(*_a, **_kw):
                cmap_idx[0] = (cmap_idx[0] + 1) % len(cmaps)
                self.settings["volume"]["cmap"] = cmaps[cmap_idx[0]]
                self._apply_volume_settings()
                _lut_btn.switch()
                self._plt.render()

            _lut_btn = self._plt.at(target).add_button(
                _cycle_lut_cb,
                states=[f"LUT ▸ {n}" for n in cmaps],
                c=["white"] * len(cmaps),
                bc=["#444"] * len(cmaps),
                pos=(0.18, 0.22), size=14, bold=True,
            )
            # Sync displayed state with current cmap.
            for _ in range(cmap_idx[0]):
                _lut_btn.switch()

        # ─── mode / save / reset / quit buttons (top-right) ─────────────
        def _cycle_mode_cb(*_a, **_kw):
            self._capture_camera()
            i = _VIEW_MODES.index(self.settings["view_mode"])
            self.settings["view_mode"] = _VIEW_MODES[(i + 1) % len(_VIEW_MODES)]
            # Auto-save so the next launch reflects the cycle.
            if self.settings_path:
                _save_settings(self.settings_path, self.settings)
            # Closing the plotter from inside a button callback (while the
            # VTK interactor is processing the click) is unsafe and was
            # observed to segfault on large volumes.  We just ask the
            # interactor to exit its event loop; the actual close+rebuild
            # is then handled cleanly in ``run()``.
            self._needs_rebuild = True
            try:
                self._plt.interactor.ExitCallback()
            except Exception:  # noqa: BLE001
                try:
                    self._plt.interactor.TerminateApp()
                except Exception:  # noqa: BLE001
                    pass

        self._plt.at(0).add_button(
            _cycle_mode_cb,
            states=[f"Mode ▸ {m}" for m in _VIEW_MODES],
            c=["white"] * len(_VIEW_MODES),
            bc=["#3b5b8c", "#8c6b3b", "#6b3b8c"],
            pos=(0.85, 0.95), size=14, bold=True,
        )

        def _save_cb(*_a, **_kw):
            self._capture_camera()
            if self.settings_path:
                _save_settings(self.settings_path, self.settings)

        self._plt.at(0).add_button(
            _save_cb, states=["Save"], c=["white"], bc=["#2e7d32"],
            pos=(0.85, 0.89), size=14, bold=True,
        )

        def _reset_cb(*_a, **_kw):
            keep_mode = self.settings["view_mode"]
            self.settings = copy.deepcopy(self.defaults)
            self.settings["view_mode"] = keep_mode  # don't re-trigger rebuild
            self._apply_label_settings()
            self._apply_volume_settings()
            self._refresh_all_sliders()
            self._plt.render()

        self._plt.at(0).add_button(
            _reset_cb, states=["Reset"], c=["white"], bc=["#c62828"],
            pos=(0.85, 0.83), size=14, bold=True,
        )

        # ─── Shading mode button (only useful when meshes are shown) ────
        if mode in ("mesh", "split") and self._actors_by_label:
            shading_idx = [0]
            current_mode = int(self.settings.get("shading_mode", 2))
            for i, (_name, val) in enumerate(_SHADING_MODES):
                if val == current_mode:
                    shading_idx[0] = i
                    break

            def _cycle_shading_cb(*_a, **_kw):
                shading_idx[0] = (shading_idx[0] + 1) % len(_SHADING_MODES)
                self.settings["shading_mode"] = int(_SHADING_MODES[shading_idx[0]][1])
                self._apply_shading()
                _shading_btn.switch()
                self._plt.render()

            _shading_btn = self._plt.at(0).add_button(
                _cycle_shading_cb,
                states=[f"Shading ▸ {n}" for n, _ in _SHADING_MODES],
                c=["white"] * len(_SHADING_MODES),
                bc=["#3b5b8c", "#6b8e23", "#8c6b3b", "#703b8c"],
                pos=(0.85, 0.77), size=14, bold=True,
            )
            for _ in range(shading_idx[0]):
                _shading_btn.switch()

        # ─── interaction style + walk keys ──────────────────────────────
        # Lock to TrackballCamera (the only mode that's actually useful
        # for exploring a root) and bind w/s to a real "walk forward /
        # backward" along the view direction.  VTK's mouse wheel only
        # dollies toward a fixed focal point, which feels like a zoom
        # when navigating inside a long cylindrical object; w/s translate
        # camera AND focal point together so you truly move through it.
        import vtk
        if not hasattr(self, "_trackball_style") or self._trackball_style is None:
            self._trackball_style = vtk.vtkInteractorStyleTrackballCamera()

        iren = getattr(self._plt, "interactor", None)
        if iren is not None:
            try:
                iren.SetInteractorStyle(self._trackball_style)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not set TrackballCamera style: %s", exc)
        else:
            logger.debug("Interactor not ready yet; style will apply on show().")

        def _walk_camera(step_frac: float) -> None:
            cam = self._plt.camera
            pos = list(cam.GetPosition())
            fp = list(cam.GetFocalPoint())
            direction = cam.GetDirectionOfProjection()
            step = float(cam.GetDistance()) * step_frac
            new_pos = [pos[i] + step * direction[i] for i in range(3)]
            new_fp = [fp[i] + step * direction[i] for i in range(3)]
            cam.SetPosition(*new_pos)
            cam.SetFocalPoint(*new_fp)
            try:
                self._plt.renderer.ResetCameraClippingRange()
            except Exception:  # noqa: BLE001
                pass
            self._plt.render()

        def _keypress_cb(obj, _event):
            try:
                key = obj.GetKeySym()
            except Exception:  # noqa: BLE001
                return
            if key in ("w", "W"):
                _walk_camera(+0.05)
            elif key in ("s", "S"):
                _walk_camera(-0.05)

        # Install the observer only once across view-mode rebuilds.
        if iren is not None and not getattr(self, "_walk_keys_installed", False):
            iren.AddObserver("KeyPressEvent", _keypress_cb)
            self._walk_keys_installed = True

        # ─── save current camera position to ./positions/ ───────────────
        # Each click appends one entry to a per-session JSON file so the
        # sequence can later be replayed as a fly-through.
        def _save_position_cb(*_a, **_kw):
            try:
                self._positions_dir.mkdir(exist_ok=True)
                if self._positions_file is None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._positions_file = self._positions_dir / f"positions_{stamp}.json"
                cam = self._plt.camera
                entry = {
                    "index":     self._positions_count,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "view_mode": self.settings.get("view_mode"),
                    "camera": {
                        "position":       list(cam.GetPosition()),
                        "focal_point":    list(cam.GetFocalPoint()),
                        "view_up":        list(cam.GetViewUp()),
                        "view_angle":     float(cam.GetViewAngle()),
                        "clipping_range": list(cam.GetClippingRange()),
                        "parallel_scale": float(cam.GetParallelScale()),
                        "distance":       float(cam.GetDistance()),
                    },
                }
                if self._positions_file.exists():
                    try:
                        data = json.loads(self._positions_file.read_text())
                        if not isinstance(data, list):
                            data = [data]
                    except json.JSONDecodeError:
                        data = []
                else:
                    data = []
                data.append(entry)
                self._positions_file.write_text(json.dumps(data, indent=2))
                self._positions_count += 1
                logger.info(
                    "Saved camera position #%d → %s",
                    entry["index"], self._positions_file,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Save position failed: %s", exc)

        self._plt.at(0).add_button(
            _save_position_cb,
            states=["Save pos"],
            c=["white"],
            bc=["#00838f"],
            pos=(0.85, 0.65), size=14, bold=True,
        )

    def _refresh_label_sliders(self) -> None:
        lid = self._selected_lid[0]
        s = self.settings["labels"][str(lid)]
        for sl, key, vol_at in self._sliders:
            if vol_at is not None:
                continue  # volume slider, not affected
            try:
                rep = sl.GetRepresentation()
                rep.SetValue(s[key])
                rep.BuildRepresentation()
            except Exception:  # noqa: BLE001
                pass

    def _refresh_all_sliders(self) -> None:
        # Label sliders
        self._refresh_label_sliders()
        # Volume sliders
        v = self.settings["volume"]
        for sl, key, vol_at in self._sliders:
            if vol_at is None:
                continue
            try:
                rep = sl.GetRepresentation()
                rep.SetValue(v[key])
                rep.BuildRepresentation()
            except Exception:  # noqa: BLE001
                pass

    # ── main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the viewer; loops to handle view-mode rebuilds."""
        while True:
            self._needs_rebuild = False
            self._build_plotter()
            self._restore_camera()
            self._plt.show(interactive=True, resetcam=(self.settings.get("camera") is None))
            # On exit, save final state.
            self._capture_camera()
            if self.settings_path:
                _save_settings(self.settings_path, self.settings)
            if not self._needs_rebuild:
                self._plt.close()
                break
            try:
                self._plt.close()
            except Exception:  # noqa: BLE001
                pass
            self._plt = None
