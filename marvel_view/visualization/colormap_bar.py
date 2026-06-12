"""Colormap bar overlays for the water-conductance viewer.

Two implementations share the same PIL rendering logic:

* :class:`ColormapBar2D`  — HUD overlay (``vtkActor2D`` on a dedicated
  layer-1 renderer), suitable for the flat interactor and flat movie.
* :class:`ColormapBar3DBillboard` — textured ``vtkPlaneSource`` in the
  main scene (layer 0), suitable for VR/stereo movies.  The panel is
  positioned to the left of the camera line of sight and updated every
  frame via :meth:`ColormapBar3DBillboard.update_pose`.

Colormaps can be specified either as a matplotlib colormap name (str)
or as a ``(N, 3)`` float32 NumPy array of pre-sampled RGB colours in
[0, 1].
"""
from __future__ import annotations

import logging
from typing import Sequence, Union

import numpy as np
import vtk

logger = logging.getLogger("marvel_view.water_conductance")

# ── Type alias ─────────────────────────────────────────────────────────────────
CmapSpec = Union[str, np.ndarray]  # matplotlib name or (N,3) float32 RGB stops

# Bar image parameters (pixels)
_DEFAULT_BAR_W: int = 28       # width of the gradient strip
_DEFAULT_BAR_H: int = 200      # height of the gradient strip
_DEFAULT_LABEL_W: int = 58     # extra width to the right for tick labels
_DEFAULT_TITLE_H: int = 28     # extra height at the top for the title
_DEFAULT_PAD: int = 6          # inner padding around the bar strip


# ── Shared PIL rendering ────────────────────────────────────────────────────────

def _sample_cmap(cmap: CmapSpec, n: int = 256) -> np.ndarray:
    """Return an ``(n, 3)`` uint8 array of RGB colours for *cmap*."""
    if isinstance(cmap, np.ndarray):
        arr = np.asarray(cmap, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"cmap array must be (N, 3), got {arr.shape}")
        # Re-sample to n entries using linear interpolation.
        x_src = np.linspace(0.0, 1.0, arr.shape[0])
        x_dst = np.linspace(0.0, 1.0, n)
        rgb = np.stack(
            [np.interp(x_dst, x_src, arr[:, c]) for c in range(3)], axis=1
        )
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    # matplotlib name
    try:
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
        cm = _cm.get_cmap(cmap, n)
        rgba = cm(np.linspace(0.0, 1.0, n))   # (n, 4) float
        return np.clip(rgba[:, :3] * 255.0, 0, 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        # Fallback: grayscale gradient
        g = np.linspace(0, 255, n, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)


def _render_bar_image(
    cmap: CmapSpec,
    vmin: float,
    vmax: float,
    title: str,
    *,
    bar_w: int = _DEFAULT_BAR_W,
    bar_h: int = _DEFAULT_BAR_H,
    label_w: int = _DEFAULT_LABEL_W,
    title_h: int = _DEFAULT_TITLE_H,
    pad: int = _DEFAULT_PAD,
    bg_color: tuple = (15, 15, 20),
    text_color: tuple = (220, 220, 220),
    border_color: tuple = (80, 80, 90),
    font_size: int = 11,
) -> np.ndarray:
    """Render a vertical colormap bar as an RGB uint8 ``(H, W, 3)`` array."""
    total_w = pad + bar_w + pad + label_w
    total_h = title_h + pad + bar_h + pad
    img = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    # ── colour gradient ─────────────────────────────────────────────────
    colors = _sample_cmap(cmap, bar_h)   # (bar_h, 3) uint8
    bar_x0 = pad
    bar_x1 = pad + bar_w
    bar_y0 = title_h + pad
    bar_y1 = title_h + pad + bar_h
    # Row 0 of img = top = vmax; row bar_h-1 = bottom = vmin.
    for row in range(bar_h):
        # colors[0] = vmin → bottom; colors[-1] = vmax → top
        c = colors[bar_h - 1 - row]
        img[bar_y0 + row, bar_x0:bar_x1] = c

    # ── border around gradient ──────────────────────────────────────────
    img[bar_y0, bar_x0:bar_x1] = border_color
    img[bar_y1 - 1, bar_x0:bar_x1] = border_color
    img[bar_y0:bar_y1, bar_x0] = border_color
    img[bar_y0:bar_y1, bar_x1 - 1] = border_color

    try:
        from PIL import Image as _I, ImageDraw as _D, ImageFont as _F  # noqa: PLC0415
        pil = _I.fromarray(img, "RGB")
        draw = _D.Draw(pil)
        try:
            font = _F.load_default(size=font_size)
            font_sm = _F.load_default(size=max(8, font_size - 2))
        except TypeError:   # Pillow < 9.2
            font = _F.load_default()
            font_sm = font

        # ── title ───────────────────────────────────────────────────────
        try:
            tb = draw.textbbox((0, 0), title, font=font_sm)
            tw = tb[2] - tb[0]
        except AttributeError:
            tw = len(title) * (font_size - 2)
        tx = max(pad, (total_w - tw) // 2)
        ty = (title_h - font_size) // 2
        draw.text((tx, ty), title, fill=text_color, font=font_sm)

        # ── tick labels: vmax (top), mid, vmin (bottom) ─────────────────
        mid = (vmax + vmin) / 2.0
        # Format numbers compactly.
        def _fmt(v: float) -> str:
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0.0):
                return f"{v:.2e}"
            if v == int(v):
                return f"{int(v)}"
            return f"{v:.2f}"

        label_x = bar_x1 + 4   # 4 px gap after bar
        # vmax → top of bar
        draw.text((label_x, bar_y0 - 1), _fmt(vmax), fill=text_color, font=font_sm)
        # mid → middle of bar
        mid_y = bar_y0 + bar_h // 2 - font_size // 2
        draw.text((label_x, mid_y), _fmt(mid), fill=text_color, font=font_sm)
        # vmin → bottom of bar
        draw.text((label_x, bar_y1 - font_size - 1), _fmt(vmin),
                  fill=text_color, font=font_sm)

        # Tick marks (small horizontal lines)
        tick_x0, tick_x1 = bar_x1, bar_x1 + 3
        draw.line([(tick_x0, bar_y0), (tick_x1, bar_y0)], fill=text_color, width=1)
        draw.line([(tick_x0, bar_y0 + bar_h // 2),
                   (tick_x1, bar_y0 + bar_h // 2)], fill=text_color, width=1)
        draw.line([(tick_x0, bar_y1 - 1), (tick_x1, bar_y1 - 1)],
                  fill=text_color, width=1)

        img = np.array(pil, dtype=np.uint8)
    except Exception:  # noqa: BLE001  (PIL not available)
        pass
    return img


def _render_bar_image_h(
    cmap: CmapSpec,
    vmin: float,
    vmax: float,
    title: str,
    *,
    bar_w: int = _DEFAULT_BAR_W,
    bar_h: int = _DEFAULT_BAR_H,
    label_w: int = _DEFAULT_LABEL_W,
    title_h: int = _DEFAULT_TITLE_H,
    pad: int = _DEFAULT_PAD,
    bg_color: tuple = (15, 15, 20),
    text_color: tuple = (220, 220, 220),
    border_color: tuple = (80, 80, 90),
    font_size: int = 11,
    left_label: str | None = None,
    right_label: str | None = None,
    reverse: bool = False,
) -> np.ndarray:
    """Render a *horizontal* colormap bar as an RGB uint8 ``(H, W, 3)`` array.

    The gradient runs left (vmin) → right (vmax) unless *reverse* is True,
    in which case it runs left (vmax) → right (vmin).
    *left_label* / *right_label* override the auto-formatted numeric tick labels
    at the left and right ends of the bar respectively.
    ``bar_h`` controls the length (width) of the strip; ``bar_w`` controls the
    strip height.
    """
    total_w = 2 * pad + bar_h
    label_row_h = font_size + 4          # space below the strip for tick labels
    total_h = title_h + pad + bar_w + pad + label_row_h
    img = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    # ── colour gradient (left = vmin, right = vmax, or reversed) ───────
    colors = _sample_cmap(cmap, bar_h)   # (bar_h, 3) uint8
    if reverse:
        colors = colors[::-1]
    bar_x0 = pad
    bar_x1 = pad + bar_h
    bar_y0 = title_h + pad
    bar_y1 = title_h + pad + bar_w
    for col in range(bar_h):
        img[bar_y0:bar_y1, bar_x0 + col] = colors[col]

    # ── border around gradient ──────────────────────────────────────────
    img[bar_y0, bar_x0:bar_x1] = border_color
    img[bar_y1 - 1, bar_x0:bar_x1] = border_color
    img[bar_y0:bar_y1, bar_x0] = border_color
    img[bar_y0:bar_y1, bar_x1 - 1] = border_color

    try:
        from PIL import Image as _I, ImageDraw as _D, ImageFont as _F  # noqa: PLC0415
        pil = _I.fromarray(img, "RGB")
        draw = _D.Draw(pil)
        try:
            font = _F.load_default(size=font_size)
            font_sm = _F.load_default(size=max(8, font_size - 2))
        except TypeError:   # Pillow < 9.2
            font = _F.load_default()
            font_sm = font

        # ── title (centred above the bar) ───────────────────────────────
        try:
            tb = draw.textbbox((0, 0), title, font=font_sm)
            tw = tb[2] - tb[0]
        except AttributeError:
            tw = len(title) * (font_size - 2)
        tx = max(pad, (total_w - tw) // 2)
        ty = (title_h - font_size) // 2
        draw.text((tx, ty), title, fill=text_color, font=font_sm)

        # ── tick labels: vmin (left), mid (centre), vmax (right) ────────
        mid = (vmax + vmin) / 2.0

        def _fmt(v: float) -> str:
            if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0.0):
                return f"{v:.2e}"
            if v == int(v):
                return f"{int(v)}"
            return f"{v:.2f}"

        # If custom labels provided, use them for left/right ends; suppress mid tick.
        _left_str  = left_label  if left_label  is not None else (_fmt(vmax) if reverse else _fmt(vmin))
        _right_str = right_label if right_label is not None else (_fmt(vmin) if reverse else _fmt(vmax))
        _show_mid  = (left_label is None and right_label is None)

        label_y = bar_y1 + 3
        # left end label
        draw.text((bar_x0, label_y), _left_str, fill=text_color, font=font_sm)
        # mid – centred (only when no custom labels)
        if _show_mid:
            mid_s = _fmt(mid)
            try:
                tb2 = draw.textbbox((0, 0), mid_s, font=font_sm)
                tw2 = tb2[2] - tb2[0]
            except AttributeError:
                tw2 = len(mid_s) * (font_size - 2)
            draw.text((bar_x0 + bar_h // 2 - tw2 // 2, label_y),
                      mid_s, fill=text_color, font=font_sm)
        # right end label – right-aligned
        try:
            tb3 = draw.textbbox((0, 0), _right_str, font=font_sm)
            tw3 = tb3[2] - tb3[0]
        except AttributeError:
            tw3 = len(_right_str) * (font_size - 2)
        draw.text((bar_x1 - tw3, label_y), _right_str, fill=text_color, font=font_sm)

        # ── tick marks (short vertical lines below the strip) ───────────
        for _tick_x in (bar_x0, bar_x0 + bar_h // 2, bar_x1 - 1):
            draw.line([(_tick_x, bar_y1), (_tick_x, bar_y1 + 3)],
                      fill=text_color, width=1)

        img = np.array(pil, dtype=np.uint8)
    except Exception:  # noqa: BLE001  (PIL not available)
        pass
    return img


# ── 2-D HUD overlay ────────────────────────────────────────────────────────────

class ColormapBar2D:
    """Colormap legend rendered as a 2-D HUD overlay via ``vtkActor2D``.

    Two positioning modes are available:

    * **Rectangle mode** (recommended, resize-robust): pass ``rect=(x0, y0,
      x1, y1)`` in normalised window coordinates.  The image is
      stretched/fitted to the rectangle via ``RenderToRectangleOn``, so
      position and size are always correct regardless of when the bar is
      created relative to window initialisation.
    * **Anchor mode** (legacy): pass ``pos=(x, y)`` only.  The
      bottom-left corner is pinned in *display* (pixel) coordinates
      computed at construction time; the image renders at its natural
      pixel size.  This can mis-position the bar if the window is not
      yet at its final size when the bar is created.

    Parameters
    ----------
    plotter:
        A vedo ``Plotter`` (or any object exposing ``.window`` as the
        ``vtkRenderWindow``).
    rect:
        ``(x0, y0, x1, y1)`` normalised window coordinates of the
        **bottom-left** and **top-right** corners.  Preferred over
        ``pos`` for resize-robustness.
    pos:
        ``(x, y)`` normalised window coordinates of the **bottom-left**
        corner (legacy / anchor mode, used when ``rect`` is not given).
    """

    def __init__(
        self,
        plotter,
        *,
        cmap: CmapSpec,
        vmin: float,
        vmax: float,
        title: str,
        rect: tuple[float, float, float, float] | None = None,
        pos: tuple[float, float] = (0.87, 0.55),
        bar_w: int = _DEFAULT_BAR_W,
        bar_h: int = _DEFAULT_BAR_H,
        label_w: int = _DEFAULT_LABEL_W,
        title_h: int = _DEFAULT_TITLE_H,
        pad: int = _DEFAULT_PAD,
        font_size: int = 11,
        horizontal: bool = False,
        reverse: bool = False,
        left_label: str | None = None,
        right_label: str | None = None,
    ) -> None:
        self._plotter = plotter
        self._cmap = cmap
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._title = title
        self._pos = pos
        self._rect = rect
        self._bar_w = bar_w
        self._bar_h = bar_h
        self._label_w = label_w
        self._title_h = title_h
        self._pad = pad
        self._horizontal = bool(horizontal)

        if horizontal:
            self._img = _render_bar_image_h(
                cmap, vmin, vmax, title,
                bar_w=bar_w, bar_h=bar_h,
                label_w=label_w, title_h=title_h, pad=pad,
                font_size=font_size,
                reverse=reverse,
                left_label=left_label,
                right_label=right_label,
            )
        else:
            self._img = _render_bar_image(
                cmap, vmin, vmax, title,
                bar_w=bar_w, bar_h=bar_h,
                label_w=label_w, title_h=title_h, pad=pad,
                font_size=font_size,
            )
        total_h, total_w = self._img.shape[:2]

        # ── VTK image pipeline ───────────────────────────────────────────
        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, total_w - 1, 0, total_h - 1, 0, 0)
        self._importer.SetDataExtent(0, total_w - 1, 0, total_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._cached_bytes: bytes = b""
        self._push_pixels(self._img)

        self._mapper = vtk.vtkImageMapper()
        self._mapper.SetInputConnection(self._importer.GetOutputPort())
        self._mapper.SetColorWindow(255)
        self._mapper.SetColorLevel(127.5)

        self.image_actor = vtk.vtkActor2D()
        self.image_actor.SetMapper(self._mapper)

        if rect is not None:
            # ── Rectangle mode: normalised coords, resize-robust ─────────
            # Identical to the controls-image positioning strategy.
            self._mapper.RenderToRectangleOn()
            self.image_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            self.image_actor.GetPositionCoordinate().SetValue(rect[0], rect[1])
            self.image_actor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            self.image_actor.GetPosition2Coordinate().SetValue(rect[2], rect[3])
        else:
            # ── Anchor mode: pixel coords (legacy) ───────────────────────
            renwin = plotter.window
            win_w, win_h = renwin.GetSize()
            px = int(round(self._pos[0] * win_w))
            py = int(round(self._pos[1] * win_h))
            self.image_actor.SetPosition(px, py)
            self.image_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()

        # ── Dedicated layer-1 renderer ───────────────────────────────────
        n_layers = renwin.GetNumberOfLayers()
        renwin.SetNumberOfLayers(max(n_layers, 2))
        self._ren = vtk.vtkRenderer()
        self._ren.SetLayer(1)
        self._ren.InteractiveOff()
        self._ren.SetViewport(0.0, 0.0, 1.0, 1.0)
        self._ren.AddActor2D(self.image_actor)
        renwin.AddRenderer(self._ren)

        self._visible = True

    # ------------------------------------------------------------------

    def _push_pixels(self, rgb: np.ndarray) -> None:
        """Upload ``(H, W, 3)`` uint8 array into the VTK image pipeline."""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        # VTK y=0 is bottom; NumPy row-0 is top → flip vertically.
        flipped = np.ascontiguousarray(rgb[::-1, :, :])
        self._cached_bytes = flipped.tobytes()
        self._importer.CopyImportVoidPointer(self._cached_bytes, len(self._cached_bytes))
        self._importer.Modified()
        try:
            self._importer.Update()
        except Exception:  # noqa: BLE001
            pass

    def set_visible(self, visible: bool) -> None:
        """Show or hide the colormap bar."""
        self._visible = bool(visible)
        try:
            self.image_actor.SetVisibility(1 if visible else 0)
        except Exception:  # noqa: BLE001
            pass

    def remove(self) -> None:
        """Remove the overlay renderer from the render window."""
        try:
            self._plotter.window.RemoveRenderer(self._ren)
        except Exception:  # noqa: BLE001
            pass


# ── 3-D billboard for VR / stereo movies ──────────────────────────────────────

class ColormapBar3DBillboard:
    """Colormap legend as a textured 3-D billboard for VR/stereo rendering.

    The panel is placed to the **left** of the camera's forward direction
    at the distance ``focal_dist × forward_frac`` ahead of the camera and
    ``focal_dist × left_frac`` to the left.  Call :meth:`update_pose` once
    per frame.

    The angular size in the scene is controlled by ``angular_size_deg``
    (how many degrees the *width* of the bar subtends at the camera).
    """

    def __init__(
        self,
        plotter,
        *,
        cmap: CmapSpec,
        vmin: float,
        vmax: float,
        title: str,
        focal_dist: float = 500.0,
        forward_frac: float = 0.6,
        left_frac: float = 0.45,
        vert_frac: float = 0.0,
        angular_size_deg: float = 6.0,
        meters_per_voxel: float = 0.0,
        forward_metres: float = 0.0,
        bar_w: int = _DEFAULT_BAR_W,
        bar_h: int = _DEFAULT_BAR_H,
        label_w: int = _DEFAULT_LABEL_W,
        title_h: int = _DEFAULT_TITLE_H,
        pad: int = _DEFAULT_PAD,
        font_size: int = 11,
        horizontal: bool = False,
        reverse: bool = False,
        left_label: str | None = None,
        right_label: str | None = None,
    ) -> None:
        import math
        self._cmap = cmap
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._title = title
        self.focal_dist = float(focal_dist)
        self.forward_frac = float(forward_frac)
        self.left_frac = float(left_frac)
        self.vert_frac = float(vert_frac)
        self._meters_per_voxel = float(meters_per_voxel)
        self._forward_metres   = float(forward_metres)
        self._angular_size_deg = float(angular_size_deg)
        # half-angle tangent for the bar *width*
        self._panel_tan_w = math.tan(math.radians(angular_size_deg / 2.0))

        if horizontal:
            self._img = _render_bar_image_h(
                cmap, vmin, vmax, title,
                bar_w=bar_w, bar_h=bar_h,
                label_w=label_w, title_h=title_h, pad=pad,
                font_size=font_size,
                reverse=reverse,
                left_label=left_label,
                right_label=right_label,
            )
        else:
            self._img = _render_bar_image(
                cmap, vmin, vmax, title,
                bar_w=bar_w, bar_h=bar_h,
                label_w=label_w, title_h=title_h, pad=pad,
                font_size=font_size,
            )
        total_h, total_w = self._img.shape[:2]
        self._aspect = float(total_h) / float(total_w) if total_w > 0 else 1.0

        # ── VTK image pipeline ───────────────────────────────────────────
        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, total_w - 1, 0, total_h - 1, 0, 0)
        self._importer.SetDataExtent(0, total_w - 1, 0, total_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._cached_bytes: bytes = b""
        self._push_pixels(self._img)

        self._texture = vtk.vtkTexture()
        self._texture.SetInputConnection(self._importer.GetOutputPort())
        self._texture.InterpolateOn()
        self._texture.RepeatOff()
        try:
            self._texture.EdgeClampOn()
        except AttributeError:
            pass

        self._plane = vtk.vtkPlaneSource()
        self._plane.SetResolution(1, 1)
        self._poly_mapper = vtk.vtkPolyDataMapper()
        self._poly_mapper.SetInputConnection(self._plane.GetOutputPort())
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._poly_mapper)
        self._actor.SetTexture(self._texture)
        self._actor.GetProperty().SetOpacity(1.0)
        try:
            self._actor.GetProperty().LightingOff()
        except AttributeError:
            pass
        self._actor.GetProperty().BackfaceCullingOff()
        # Always drawn on top (HUD-like in VR).
        try:
            _sp = self._actor.GetShaderProperty()
            _sp.AddFragmentShaderReplacement(
                "//VTK::Depth::Impl", True,
                "gl_FragDepth = 0.0001;\n",
                False,
            )
            _sp.AddVertexShaderReplacement(
                "//VTK::PositionVC::Impl", True,
                "//VTK::PositionVC::Impl\n  gl_Position.z = -0.999 * gl_Position.w;\n",
                False,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "ColormapBar3DBillboard depth override failed (%s); "
                "billboard may be occluded by scene geometry.",
                _e,
            )

        # Add to layer-0 renderer (captured by the panoramic pass).
        main_ren = (
            plotter.renderers[0]
            if hasattr(plotter, "renderers") and plotter.renderers
            else plotter.renderer
        )
        main_ren.AddActor(self._actor)
        self._renderer = main_ren
        # Start hidden; caller must explicitly call set_visible(True).
        # Avoids the actor appearing at the default vtkPlaneSource origin
        # (0,0,0)→(1,0,0) before the first update_pose call.
        self._actor.SetVisibility(0)
        self._visible = False

    # ------------------------------------------------------------------

    def _push_pixels(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        flipped = np.ascontiguousarray(rgb[::-1, :, :])
        self._cached_bytes = flipped.tobytes()
        self._importer.CopyImportVoidPointer(self._cached_bytes, len(self._cached_bytes))
        self._importer.Modified()
        try:
            self._importer.Update()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._texture.Modified()
        except Exception:  # noqa: BLE001
            pass

    def _place_plane(
        self,
        panel_center: np.ndarray,
        cam_pos: np.ndarray,
        world_up: np.ndarray,
    ) -> None:
        """Orient and size the plane so it faces ``cam_pos``."""
        normal = np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        n_len = float(np.linalg.norm(normal))
        if n_len < 1e-9:
            return
        normal /= n_len

        up = np.asarray(world_up, dtype=float).copy()
        up -= np.dot(up, normal) * normal
        up_n = float(np.linalg.norm(up))
        if up_n < 1e-6:
            fallback = (np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
                        else np.array([0.0, 1.0, 0.0]))
            up = fallback - np.dot(fallback, normal) * normal
            up_n = float(np.linalg.norm(up))
            if up_n < 1e-9:
                return
        up /= up_n

        panel_right = np.cross(up, normal)
        pr_n = float(np.linalg.norm(panel_right))
        if pr_n < 1e-9:
            return
        panel_right /= pr_n

        d_to_panel = float(np.linalg.norm(
            np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        ))
        half_w = d_to_panel * self._panel_tan_w
        half_h = half_w * self._aspect   # preserve bar's aspect ratio

        c = np.asarray(panel_center, dtype=float)
        origin = c - panel_right * half_w - up * half_h
        point1 = c + panel_right * half_w - up * half_h
        point2 = c - panel_right * half_w + up * half_h
        self._plane.SetOrigin(*origin.tolist())
        self._plane.SetPoint1(*point1.tolist())
        self._plane.SetPoint2(*point2.tolist())
        self._plane.Modified()

    def update_pose(
        self,
        cam_pos: np.ndarray,
        travel_dir: np.ndarray,
        world_up: np.ndarray,
    ) -> None:
        """Reposition the billboard for the current frame.

        Parameters
        ----------
        cam_pos:
            Camera position in VTK world coordinates.
        travel_dir:
            Pre-smoothed unit travel-direction vector.
        world_up:
            Constant world-up unit vector (view_up of first keyframe).
        """
        travel = np.asarray(travel_dir, dtype=float).copy()
        t_n = float(np.linalg.norm(travel))
        if t_n < 1e-9:
            return
        travel /= t_n

        up = np.asarray(world_up, dtype=float).copy()
        up_n = float(np.linalg.norm(up))
        up = up / up_n if up_n > 1e-9 else np.array([0.0, 1.0, 0.0])

        right = np.cross(travel, up)
        r_n = float(np.linalg.norm(right))
        if r_n < 1e-6:
            return
        right /= r_n

        D = self.focal_dist
        if self._meters_per_voxel > 0.0:
            fwd_vox  = self._forward_metres / self._meters_per_voxel
            left_vox = self.left_frac * fwd_vox   # left_frac = tan(horiz angle)
            vert_vox = self.vert_frac * fwd_vox   # vert_frac = tan(vert angle)
        else:
            fwd_vox  = D * self.forward_frac
            left_vox = D * self.left_frac
            vert_vox = D * self.vert_frac
        panel_center = (
            np.asarray(cam_pos, dtype=float)
            + travel   * fwd_vox
            + (-right) * left_vox   # "left" = negative right
            + up       * vert_vox
        )
        self._place_plane(panel_center, cam_pos, up)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the billboard."""
        self._visible = bool(visible)
        try:
            self._actor.SetVisibility(1 if visible else 0)
        except Exception:  # noqa: BLE001
            pass

    def remove(self) -> None:
        """Remove the actor from its renderer."""
        try:
            self._renderer.RemoveActor(self._actor)
        except Exception:  # noqa: BLE001
            pass
