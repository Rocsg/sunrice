"""HUD-style orthogonal-slice overlay for marvel-water viewers.

Three square ortho views are stacked horizontally at the top of the
window:

* **XY** (slice at the camera's Z) — full aspect-fit.
* **YZ** rotated 90° clockwise (slice at X), square-cropped around the
  red dot so it has the same on-screen width as XY.
* **XZ** (slice at Y), square-cropped around the red dot likewise.

Each view shows a red dot at the camera's position and a small,
semi-transparent pink cone indicating the projection of the camera's
look direction onto that plane.  A thin gray frame surrounds each
view.

Axis mapping camera→voxel follows the project convention: meshes are
built by :func:`marvel_view.preprocessing.meshing.mask_to_mesh` which
feeds ``skimage.measure.marching_cubes`` verts (z, y, x order) straight
into ``vedo.Mesh``.  So VTK world (X, Y, Z) maps to voxel indices
(z, y, x).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import math

import numpy as np
import tifffile
import vtk

logger = logging.getLogger(__name__)

# ── Alpha LUT for transparency compositing ────────────────────────────────────
# Maps uint8 gray value → float32 opacity:
#   0..30  → 0.30  (background/air stays mostly transparent)
#   30..70 → linear ramp  0.30 → 0.90
#   70+    → 0.90  (dense tissue, nearly opaque)
_xs = np.arange(256, dtype=np.float32)
_ALPHA_LUT: np.ndarray = np.clip(
    0.30 + (_xs - 30.0) / (70.0 - 30.0) * 0.60,
    0.30, 0.90,
).astype(np.float32)
del _xs


def _premultiply_tile(tile: np.ndarray) -> None:
    """Pre-multiply a ``(H, W, 3)`` uint8 RGB tile against black.

    Alpha is derived from ``_ALPHA_LUT`` using each pixel's gray value
    (channel 0, equal to channels 1-2 at the time this is called — before
    :func:`_draw_cone`, :func:`_draw_dot`, and :func:`_draw_cell_border`).
    Those primitives are drawn afterwards and are unaffected, so the cone,
    dot, and cell border remain at full intended intensity.
    """
    alpha = _ALPHA_LUT[tile[:, :, 0]]           # (H, W) float32
    tile[:] = (tile.astype(np.float32) * alpha[:, :, np.newaxis]).astype(np.uint8)


def _rescale_tile_3ch(tile: np.ndarray, new_s: int) -> np.ndarray:
    """Nearest-neighbour rescale a (H, W, 3) uint8 tile to (new_s × new_s)."""
    if tile.ndim != 3 or tile.shape[2] != 3 or tile.shape[0] < 1 or tile.shape[1] < 1:
        return np.zeros((new_s, new_s, 3), dtype=np.uint8)
    h, w = tile.shape[:2]
    ys = np.linspace(0, h - 1, new_s).astype(np.int32)
    xs = np.linspace(0, w - 1, new_s).astype(np.int32)
    return tile[ys[:, None], xs[None, :], :]


def load_raw_volume(path: Path) -> np.ndarray:
    """Load ``Raw.tif`` (or any 3-D TIFF) as ``uint8`` ``(Z, Y, X)``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw volume not found: {path}")
    logger.info("Loading raw volume from %s …", path)
    vol = tifffile.imread(str(path))
    vol = np.asarray(vol).squeeze()
    if vol.ndim != 3:
        raise ValueError(
            f"Expected a 3-D raw volume, got shape {vol.shape} from {path}."
        )
    lo = float(np.percentile(vol, 0.5))
    hi = float(np.percentile(vol, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = (vol.astype(np.float32) - lo) * (255.0 / (hi - lo))
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


class OrthoPanelOverlay:
    """Three-ortho-view HUD overlay attached to a vedo Plotter."""

    def __init__(
        self,
        plotter,
        raw_volume: np.ndarray,
        *,
        viewport: Sequence[float] = (0.0, 0.5, 0.4, 1.0),
        cell_pixels: int = 320,
        dot_radius: int = 7,
        dot_color: Tuple[int, int, int] = (255, 50, 50),
        cell_border_color: Tuple[int, int, int] = (140, 140, 140),
        cone_color: Tuple[int, int, int] = (255, 130, 190),
        cone_alpha: float = 0.45,
        cone_length: int = 48,
        cone_half_width: int = 18,
        center_vertically: bool = False,
    ) -> None:
        if raw_volume.ndim != 3:
            raise ValueError("raw_volume must be a 3-D array (Z, Y, X).")
        self._center_vertically = bool(center_vertically)
        self._cell_pixels = int(cell_pixels)   # user-supplied upper limit
        self._last_cam_world: tuple = (0.0, 0.0, 0.0)
        self._last_cam_dir: tuple   = (0.0, 0.0, -1.0)
        self.plotter = plotter
        self.volume = raw_volume
        # Adapt the panel raster to the current window size so the 2×2
        # grid fills the requested viewport rectangle.  ``vtkImageMapper``
        # paints pixels 1:1 in display coordinates.
        try:
            win_w, win_h = plotter.window.GetSize()
        except Exception:  # noqa: BLE001
            win_w, win_h = 1600, 900
        x0, y0, x1, y1 = viewport
        target_w = max(64, int(round((x1 - x0) * win_w)))
        target_h = max(64, int(round((y1 - y0) * win_h)))
        # Layout is 2×2 (XY top-left | YZ top-right // XZ bottom-left | empty).
        cell = max(32, min(int(cell_pixels), target_h // 2, target_w // 2))
        self.cell = int(cell)
        self.panel_w = self.cell * 2
        self.panel_h = self.cell * 2
        self.dot_radius = int(dot_radius)
        self.dot_color = tuple(int(c) for c in dot_color)
        self.cell_border_color = tuple(int(c) for c in cell_border_color)
        self.cone_color = tuple(int(c) for c in cone_color)
        self.cone_alpha = float(cone_alpha)
        self.cone_length = float(cone_length)
        self.cone_half_width = float(cone_half_width)

        # Reused canvas + strong byte-buffer ref for vtkImageImport.
        self._canvas = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)

        # Bottom-right: compact text label — no full cell border.
        # A small near-black rectangle placed at the top of the 4th quadrant
        # so it sits close to the three slice tiles.
        self._br_label_tile: np.ndarray = np.zeros(
            (self.cell, self.cell, 3), dtype=np.uint8
        )
        try:
            from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font  # noqa: PLC0415
            _text = "Current position in the\nX-ray original volume"
            _fg = (155, 155, 155)
            _bg = (12, 12, 12)              # near-black ‘0.6 opacity’ rectangle
            _lm = max(3, self.cell // 32)   # left/right margin
            _vm = max(2, self.cell // 48)   # top/bottom margin
            _box_h = self.cell // 3         # compact height (~1/3 of quadrant)
            _box_w = self.cell - 2 * _lm
            _vy = (self.cell - _box_h) // 2  # vertically centred in the tile
            # Font size scales with cell (~11 px at cell=180, the normal-mode ref).
            _fsize = max(8, round(self.cell / 16))
            try:
                _font = _PIL_Font.load_default(size=_fsize)
            except TypeError:   # Pillow < 9.2: load_default has no 'size' arg
                _font = _PIL_Font.load_default()
            _pil = _PIL_Image.new("RGB", (self.cell, self.cell), (0, 0, 0))
            _d = _PIL_Draw.Draw(_pil)
            _d.rectangle([_lm, _vy, _lm + _box_w, _vy + _box_h], fill=_bg)
            try:
                _bbox = _d.multiline_textbbox((0, 0), _text, font=_font, align="center")
                _tw = _bbox[2] - _bbox[0]
                _th = _bbox[3] - _bbox[1]
                _tx = _lm + (_box_w - _tw) // 2
                _ty = _vy + (_box_h - _th) // 2
                _d.multiline_text((_tx, _ty), _text, font=_font, fill=_fg, align="center")
            except Exception:  # noqa: BLE001
                _d.multiline_text((_lm + 2, _vy + 2), _text, fill=_fg, align="center")
            self._br_label_tile = np.array(_pil, dtype=np.uint8)
        except Exception:  # noqa: BLE001  (PIL not available)
            pass
        self._cached_bytes: bytes = b""

        renwin = plotter.window
        n_layers_before = renwin.GetNumberOfLayers()
        renwin.SetNumberOfLayers(max(n_layers_before, 2))
        n_layers_after = renwin.GetNumberOfLayers()
        # Make sure every pre-existing renderer is on layer 0 so our
        # layer-1 overlay is unambiguously the topmost.
        existing_renderers = []
        rens = renwin.GetRenderers()
        rens.InitTraversal()
        while True:
            r = rens.GetNextItem()
            if r is None:
                break
            existing_renderers.append(r)
            try:
                r.SetLayer(0)
            except Exception:  # noqa: BLE001
                pass
        logger.info(
            "OrthoPanelOverlay START  viewport=%s  panel=%dx%d  "
            "renwin.size=%s  layers: %d→%d  pre-existing renderers=%d",
            tuple(viewport), self.panel_w, self.panel_h,
            tuple(renwin.GetSize()), n_layers_before, n_layers_after,
            len(existing_renderers),
        )

        self.ren = vtk.vtkRenderer()
        # Overlay renderer covers the WHOLE window — we place the image
        # ourselves in display coordinates via a vtkActor2D so we don't
        # have to fight with parallel-projection cameras or viewport
        # aspect ratios.
        self.ren.SetViewport(0.0, 0.0, 1.0, 1.0)
        self.ren.SetLayer(1)
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.InteractiveOff()
        self.ren.SetUseDepthPeeling(False)
        # Don't clear the layer-0 framebuffer.
        try:
            self.ren.SetEraseOff() if hasattr(self.ren, "SetEraseOff") \
                else self.ren.EraseOff()
        except Exception:  # noqa: BLE001
            pass
        renwin.AddRenderer(self.ren)
        # Keep target viewport for resize-aware placement.
        self._target_viewport = tuple(viewport)
        logger.info(
            "OrthoPanelOverlay renderer added.  total renderers in window=%d  "
            "layer=%d",
            renwin.GetRenderers().GetNumberOfItems(), self.ren.GetLayer(),
        )

        # ── image pipeline via vtkImageImport (canonical numpy → VTK) ─────
        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, self.panel_w - 1,
                                      0, self.panel_h - 1, 0, 0)
        self._importer.SetDataExtent(0, self.panel_w - 1,
                                     0, self.panel_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._push_pixels(self._canvas)

        # 2-D image mapper / actor: positions in display (pixel) coords.
        self._mapper = vtk.vtkImageMapper()
        self._mapper.SetInputConnection(self._importer.GetOutputPort())
        self._mapper.SetColorWindow(255.0)
        self._mapper.SetColorLevel(127.5)

        self.image_actor = vtk.vtkActor2D()
        self.image_actor.SetMapper(self._mapper)
        # Global alpha-blend against layer-0 scene content.
        # Combined with the per-pixel pre-multiply (dark areas → black → fade
        # into the background), this gives a genuine semi-transparent panel
        # rather than a fully opaque rectangle.
        self.image_actor.GetProperty().SetOpacity(0.72)
        # We position the actor in normalized-display coords so the
        # panel tracks window resizes.
        pos_coord = self.image_actor.GetPositionCoordinate()
        pos_coord.SetCoordinateSystemToNormalizedDisplay()
        self.ren.AddActor2D(self.image_actor)

        # Initial placement using current window size:
        self._reposition()
        # Re-place on window resize / render so the overlay stays anchored
        # to the requested viewport rectangle even if the user resizes.
        try:
            # ConfigureEvent fires only on real window resizes, not on every
            # internal render / vtkWindowToImageFilter tile pass.
            renwin.AddObserver("ConfigureEvent", lambda *a, **k: self._reposition())
        except Exception:  # noqa: BLE001
            pass

        # Initial paint with cam at volume center.
        Z, Y, X = self.volume.shape
        self.update((Z * 0.5, Y * 0.5, X * 0.5), (0.0, 0.0, -1.0))
        logger.info(
            "OrthoPanelOverlay READY.  volume shape=(Z=%d, Y=%d, X=%d).  "
            "Forcing first render…",
            Z, Y, X,
        )
        try:
            renwin.Render()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OrthoPanelOverlay initial Render() failed: %s", exc)

    # ─── pixel injection ──────────────────────────────────────────────────

    def _push_pixels(self, rgb: np.ndarray) -> None:
        """Push an ``(H, W, 3)`` uint8 canvas to VTK (flipped for VTK row order)."""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        flipped = np.ascontiguousarray(rgb[::-1, :, :])
        self._cached_bytes = flipped.tobytes()
        self._importer.CopyImportVoidPointer(
            self._cached_bytes, len(self._cached_bytes)
        )
        self._importer.Modified()
        # Force the pipeline to execute now so the mapper observes the
        # new pixels on its next Render() pass.
        try:
            self._importer.Update()
        except Exception:  # noqa: BLE001
            pass
        # Touch the mapper too — vtkImageMapper has been observed to
        # cache when only the upstream importer was marked dirty.
        try:
            self._mapper.Modified()
        except Exception:  # noqa: BLE001
            pass

    def _reposition(self) -> None:
        """Resize the panel canvas to fill the target viewport at the current
        window size, then anchor the actor's UPPER-LEFT corner at (x0, y1).

        Called once at init and again on every ``ConfigureEvent`` (user
        window resize).  ``vtkActor2D`` anchors its lower-left corner at
        the position we set, so we subtract panel height in normalised units.
        """
        try:
            w, h = self.plotter.window.GetSize()
        except Exception:  # noqa: BLE001
            return
        if w < 1 or h < 1:
            return
        x0, y0, x1, y1 = self._target_viewport

        # ── resize canvas if the window dimensions changed ─────────────────
        target_w = max(64, int(round((x1 - x0) * w)))
        target_h = max(64, int(round((y1 - y0) * h)))
        new_cell = max(32, min(self._cell_pixels, target_h // 2, target_w // 2))
        if new_cell != self.cell:
            self.cell    = new_cell
            self.panel_w = new_cell * 2
            self.panel_h = new_cell * 2
            self._canvas = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
            self._br_label_tile = _rescale_tile_3ch(self._br_label_tile, new_cell)
            self._importer.SetWholeExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
            self._importer.SetDataExtent( 0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
            try:
                rgb = self.compose(self._last_cam_world, self._last_cam_dir)
                self._push_pixels(rgb)
            except Exception:  # noqa: BLE001
                pass

        # ── reposition actor (lower-left in normalised display coords) ─────
        panel_h_norm = self.panel_h / float(h)
        nx = float(x0)
        if self._center_vertically:
            ny = 0.5 - panel_h_norm / 2.0
        else:
            ny = float(y1) - panel_h_norm
        pos_coord = self.image_actor.GetPositionCoordinate()
        pos_coord.SetValue(nx, ny)
        logger.debug(
            "OrthoPanelOverlay reposition  win=%dx%d  cell=%d  anchor=(%.3f, %.3f)",
            w, h, self.cell, nx, ny,
        )

    # ─── primitives ───────────────────────────────────────────────────────

    @staticmethod
    def _nn_resize(arr2d: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        h, w = arr2d.shape
        if w == 0 or h == 0 or new_w <= 0 or new_h <= 0:
            return np.zeros((max(0, new_h), max(0, new_w)), dtype=arr2d.dtype)
        ys = np.linspace(0, h - 1, new_h).astype(np.int32)
        xs = np.linspace(0, w - 1, new_w).astype(np.int32)
        return arr2d[ys[:, None], xs[None, :]]

    def _draw_cell_border(self, tile: np.ndarray) -> None:
        pass  # borders removed

    def _draw_dot(self, tile: np.ndarray, cx: float, cy: float) -> None:
        H, W = tile.shape[:2]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r = self.dot_radius
        y0, y1 = max(0, cy_i - r), min(H, cy_i + r + 1)
        x0, x1 = max(0, cx_i - r), min(W, cx_i + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx_i) ** 2 + (yy - cy_i) ** 2 <= r * r
        for c in range(3):
            tile[y0:y1, x0:x1, c] = np.where(
                mask, self.dot_color[c], tile[y0:y1, x0:x1, c]
            )

    def _draw_cone(
        self, tile: np.ndarray, apex: Tuple[float, float],
        direction: Tuple[float, float],
    ) -> None:
        """Filled, semi-transparent pink triangle pointing along ``direction``."""
        d = np.asarray(direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        d /= n
        perp = np.array([-d[1], d[0]], dtype=np.float32)
        a = np.asarray(apex, dtype=np.float32)
        b = a + self.cone_length * d + self.cone_half_width * perp
        c = a + self.cone_length * d - self.cone_half_width * perp
        pts = np.stack([a, b, c])
        H, W = tile.shape[:2]
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        x1 = min(W, int(np.ceil(pts[:, 0].max())) + 1)
        y1 = min(H, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-6:
            return
        px = xx - pts[0, 0]
        py = yy - pts[0, 1]
        u = (px * v1[1] - v1[0] * py) / den
        v = (v0[0] * py - px * v0[1]) / den
        mask = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if not mask.any():
            return
        alpha = self.cone_alpha
        pink = np.array(self.cone_color, dtype=np.float32)
        sub = tile[y0:y1, x0:x1].astype(np.float32)
        sub[mask] = sub[mask] * (1.0 - alpha) + pink * alpha
        tile[y0:y1, x0:x1] = sub.astype(np.uint8)

    # ─── cell builders ────────────────────────────────────────────────────

    def _fit_aspect(
        self, gray: np.ndarray, dot_rc: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Aspect-preserving fit into a (cell, cell) RGB tile.

        Returns the tile and the dot position in tile pixels (x, y).
        """
        h, w = gray.shape
        S = self.cell
        if w == 0 or h == 0:
            return np.zeros((S, S, 3), dtype=np.uint8), (S / 2.0, S / 2.0)
        scale = min(S / w, S / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = self._nn_resize(gray, nw, nh)
        tile = np.zeros((S, S, 3), dtype=np.uint8)
        ox = (S - nw) // 2
        oy = (S - nh) // 2
        tile[oy:oy + nh, ox:ox + nw, 0] = resized
        tile[oy:oy + nh, ox:ox + nw, 1] = resized
        tile[oy:oy + nh, ox:ox + nw, 2] = resized
        dot_row, dot_col = dot_rc
        return tile, (ox + dot_col * scale, oy + dot_row * scale)

    def _square_crop(
        self, gray: np.ndarray, dot_rc: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Crop the smaller-dim-sized square centred on the dot, fit to cell.

        Pads with zeros if the window straddles the array edge.
        """
        h, w = gray.shape
        S = self.cell
        if w == 0 or h == 0:
            return np.zeros((S, S, 3), dtype=np.uint8), (S / 2.0, S / 2.0)
        win = int(min(h, w))
        dr, dc = dot_rc
        # Choose top-left corner of the window so the dot is roughly
        # centred but the window stays within the array (allow padding
        # if the array is smaller than the window in one dim, which
        # shouldn't happen here but is safe).
        r0_ideal = dr - win // 2
        c0_ideal = dc - win // 2
        r0 = int(np.clip(r0_ideal, 0, max(0, h - win)))
        c0 = int(np.clip(c0_ideal, 0, max(0, w - win)))
        crop = np.zeros((win, win), dtype=gray.dtype)
        r1 = min(h, r0 + win)
        c1 = min(w, c0 + win)
        crop[: r1 - r0, : c1 - c0] = gray[r0:r1, c0:c1]
        # Dot in crop coords:
        dot_in_crop = (dr - r0, dc - c0)
        # Resize crop (square) to cell.
        scale = S / win
        resized = self._nn_resize(crop, S, S)
        tile = np.zeros((S, S, 3), dtype=np.uint8)
        tile[..., 0] = resized
        tile[..., 1] = resized
        tile[..., 2] = resized
        dx = dot_in_crop[1] * scale
        dy = dot_in_crop[0] * scale
        return tile, (dx, dy)

    # ─── main composer ────────────────────────────────────────────────────

    def compose(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float],
    ) -> np.ndarray:
        Z, Y, X = self.volume.shape
        # World (X, Y, Z) ↦ voxel (z, y, x).
        z_idx = int(np.clip(round(cam_world[0]), 0, Z - 1))
        y_idx = int(np.clip(round(cam_world[1]), 0, Y - 1))
        x_idx = int(np.clip(round(cam_world[2]), 0, X - 1))

        # World direction → voxel direction (same mapping).
        d = np.asarray(cam_dir, dtype=np.float32)
        if np.linalg.norm(d) < 1e-6:
            d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d_z, d_y, d_x = float(d[0]), float(d[1]), float(d[2])

        canvas = self._canvas
        canvas.fill(0)
        S = self.cell

        # 2×2 grid:  XY top-left  |  YZ top-right
        #            XZ bottom-left | empty bottom-right
        # (numpy canvas has row 0 at the TOP; _push_pixels flips for VTK.)

        # ── XY view (Y, X), dot row=y_idx, col=x_idx — TOP-LEFT cell ────
        xy = self.volume[z_idx, :, :]
        tile, (dx, dy) = self._fit_aspect(xy, (y_idx, x_idx))
        _premultiply_tile(tile)
        scale_xy = min(S / xy.shape[1], S / xy.shape[0])
        self._draw_cone(tile, (dx, dy), (d_x * scale_xy, d_y * scale_xy))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[0:S, 0:S] = tile

        # ── YZ view, rotated 90° CW: original (Z, Y) → (Y, Z) — TOP-RIGHT cell ─
        yz = self.volume[:, :, x_idx]
        yz_rot = np.ascontiguousarray(np.rot90(yz, k=-1))   # (Y, Z)
        # rot90(k=-1) maps (z, y) → (y, Z-1-z).
        dot_rot = (y_idx, (Z - 1) - z_idx)
        tile, (dx, dy) = self._square_crop(yz_rot, dot_rot)
        _premultiply_tile(tile)
        scale_yz = S / float(min(yz_rot.shape))
        # In the rotated image, screen x = (Z-1 - z_world), so the
        # x-direction on screen is -d_z; screen y = y_world, so +d_y.
        self._draw_cone(tile, (dx, dy), (-d_z * scale_yz, d_y * scale_yz))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[0:S, S:2 * S] = tile

        # ── XZ view (Z, X), dot row=z_idx, col=x_idx — BOTTOM-LEFT cell ─
        xz = self.volume[:, y_idx, :]
        tile, (dx, dy) = self._square_crop(xz, (z_idx, x_idx))
        _premultiply_tile(tile)
        scale_xz = S / float(min(xz.shape))
        # Screen x = x_world (+d_x), screen y = z_world (+d_z).
        self._draw_cone(tile, (dx, dy), (d_x * scale_xz, d_z * scale_xz))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[S:2 * S, 0:S] = tile

        # Bottom-right: label tile ("Position on / 3D orthoslice").
        canvas[S:2 * S, S:2 * S] = self._br_label_tile
        return canvas

    def update(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        """Recompose the panel for the given camera position and direction."""
        self._last_cam_world = tuple(cam_world)
        self._last_cam_dir   = tuple(cam_dir)
        rgb = self.compose(cam_world, cam_dir)
        self._push_pixels(rgb)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the ortho-panel overlay."""
        self.image_actor.SetVisibility(1 if visible else 0)


# ─── helper ─────────────────────────────────────────────────────────────────


def _cam_direction(plotter) -> Tuple[float, float, float]:
    try:
        cam = plotter.camera
        pos = np.asarray(cam.GetPosition(), dtype=np.float64)
        foc = np.asarray(cam.GetFocalPoint(), dtype=np.float64)
        d = foc - pos
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return (0.0, 0.0, -1.0)
        d /= n
        return (float(d[0]), float(d[1]), float(d[2]))
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, -1.0)


def attach_to_plotter(
    plotter,
    raw_volume: np.ndarray,
    *,
    viewport: Sequence[float] = (0.0, 0.5, 0.4, 1.0),
    cell_pixels: int = 320,
) -> OrthoPanelOverlay:
    """Create an :class:`OrthoPanelOverlay` and refresh it on every
    camera-affecting interaction event."""
    overlay = OrthoPanelOverlay(
        plotter, raw_volume,
        viewport=viewport,
        cell_pixels=cell_pixels,
    )

    def _cb(*_a, **_kw):
        try:
            cam = plotter.camera
            overlay.update(cam.GetPosition(), _cam_direction(plotter))
        except Exception:  # noqa: BLE001
            pass

    iren = getattr(plotter, "interactor", None)
    if iren is not None:
        iren.AddObserver("EndInteractionEvent", _cb)
        iren.AddObserver("KeyPressEvent", _cb)
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# 3-D billboard variant for VR rendering
# ─────────────────────────────────────────────────────────────────────────────

class OrthoPanel3DBillboard:
    """Three-ortho-view panel rendered as a 3-D billboard in the main scene.

    Unlike :class:`OrthoPanelOverlay` (a screen-space 2-D HUD), this class
    inserts a texture-mapped ``vtkPlaneSource`` directly into the layer-0
    renderer so it is captured by the VR panoramic-projection pass and
    rendered with correct stereo depth.

    The panel is positioned relative to the camera's *travel direction*
    (the direction the camera is moving, not where it is looking).
    ``world_up`` is a **constant** vector — the viewer never lies down.

    The panoramic pass uses ``vtkOpaquePass`` exclusively, so the actor
    is kept fully opaque (``opacity = 1.0``) to guarantee it appears in
    all six cubemap faces.

    Per-frame usage::

        billboard.update(cam_pos, cam_dir, travel_dir, world_up)

    Placement geometry::

        panel_center = cam_pos
                     + travel_dir  * (focal_dist * forward_frac)
                     + left        * (focal_dist * left_frac)
                     + world_up    * (focal_dist * vert_frac)

    where ``left = −cross(travel_dir, world_up)``.
    """

    @staticmethod
    def _nn_resize(arr2d: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        h, w = arr2d.shape
        if w == 0 or h == 0 or new_w <= 0 or new_h <= 0:
            return np.zeros((max(0, new_h), max(0, new_w)), dtype=arr2d.dtype)
        ys = np.linspace(0, h - 1, new_h).astype(np.int32)
        xs = np.linspace(0, w - 1, new_w).astype(np.int32)
        return arr2d[ys[:, None], xs[None, :]]

    def __init__(
        self,
        plotter,
        raw_volume: np.ndarray,
        *,
        focal_dist: float = 100.0,
        forward_frac: float = 0.50,
        left_frac: float = 0.38,
        vert_frac: float = 0.00,
        angular_size_deg: float = 14.0,
        cell_pixels: int = 256,
        dot_radius: int = 6,
        dot_color: Tuple[int, int, int] = (255, 50, 50),
        cell_border_color: Tuple[int, int, int] = (140, 140, 140),
        cone_color: Tuple[int, int, int] = (255, 130, 190),
        cone_alpha: float = 0.45,
        cone_length: int = 40,
        cone_half_width: int = 16,
        meters_per_voxel: float = 0.0,
        forward_metres: float = 2.0,
        left_metres: float = 0.5,
        vert_metres: float = 0.0,
    ) -> None:
        if raw_volume.ndim != 3:
            raise ValueError("raw_volume must be a 3-D array (Z, Y, X).")
        self.plotter = plotter
        self.volume = raw_volume
        self.focal_dist = float(focal_dist)
        self.forward_frac = float(forward_frac)
        self.left_frac = float(left_frac)
        self.vert_frac = float(vert_frac)
        self._meters_per_voxel = float(meters_per_voxel)
        self._forward_metres   = float(forward_metres)
        self._left_metres      = float(left_metres)
        self._vert_metres      = float(vert_metres)
        self._angular_size_deg = float(angular_size_deg)
        self._panel_tan = math.tan(math.radians(self._angular_size_deg / 2.0))
        self.cell = int(cell_pixels)
        self.panel_w = self.cell * 2
        self.panel_h = self.cell * 2
        self.dot_radius = int(dot_radius)
        self.dot_color = tuple(int(c) for c in dot_color)
        self.cell_border_color = tuple(int(c) for c in cell_border_color)
        self.cone_color = tuple(int(c) for c in cone_color)
        self.cone_alpha = float(cone_alpha)
        self.cone_length = float(cone_length)
        self.cone_half_width = float(cone_half_width)
        self._canvas = np.zeros((self.panel_h, self.panel_w, 3), dtype=np.uint8)
        self._cached_bytes: bytes = b""

        # ── VTK image pipeline ───────────────────────────────────────────
        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._push_pixels(self._canvas)

        # ── Texture ──────────────────────────────────────────────────────
        self._texture = vtk.vtkTexture()
        self._texture.SetInputConnection(self._importer.GetOutputPort())
        self._texture.InterpolateOn()
        self._texture.RepeatOff()
        try:
            self._texture.EdgeClampOn()
        except AttributeError:
            pass

        # ── Plane source + actor ─────────────────────────────────────────
        self._plane = vtk.vtkPlaneSource()
        self._plane.SetResolution(1, 1)
        self._poly_mapper = vtk.vtkPolyDataMapper()
        self._poly_mapper.SetInputConnection(self._plane.GetOutputPort())
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._poly_mapper)
        self._actor.SetTexture(self._texture)
        # Must stay fully opaque: panoramic pass uses vtkOpaquePass only.
        self._actor.GetProperty().SetOpacity(1.0)
        try:
            self._actor.GetProperty().LightingOff()
        except AttributeError:
            pass
        self._actor.GetProperty().BackfaceCullingOff()

        # Add to layer-0 renderer (passes through the panoramic pass).
        main_ren = (
            plotter.renderers[0]
            if hasattr(plotter, "renderers") and plotter.renderers
            else plotter.renderer
        )
        main_ren.AddActor(self._actor)
        self._renderer = main_ren
        # Remove any mapper-level clipping planes (e.g. added by vedo/VTK
        # internals) so the billboard is never clipped by them in VR.
        self._poly_mapper.RemoveAllClippingPlanes()

        # ── Bottom-right label tile ──────────────────────────────────────
        self._br_speed_text: str = ""
        self._br_label_tile: np.ndarray = np.zeros(
            (self.cell, self.cell, 3), dtype=np.uint8
        )
        self._br_label_tile = self._make_br_label_tile("")

        # Initial placement (overwritten on first update_pose call).
        Z, Y, X = self.volume.shape
        self._place_plane(
            panel_center=np.array([Z * 0.5, Y * 0.5, X * 0.5 + focal_dist * 0.5]),
            cam_pos=np.array([Z * 0.5, Y * 0.5, X * 0.5]),
            world_up=np.array([0.0, 1.0, 0.0]),
        )
        self.update_image((Z * 0.5, Y * 0.5, X * 0.5), (0.0, 0.0, -1.0))
        logger.info(
            "OrthoPanel3DBillboard READY  volume=(Z=%d,Y=%d,X=%d)  "
            "focal_dist=%.1f  fwd=%.2f  left=%.2f  ang=%.1f",
            Z, Y, X, self.focal_dist, self.forward_frac,
            self.left_frac, self._angular_size_deg,
        )

    # ── pixel injection ──────────────────────────────────────────────────

    def _push_pixels(self, rgb: np.ndarray) -> None:
        """Push ``(H, W, 3)`` uint8 numpy array into the VTK texture pipeline."""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        # Flip vertically: numpy row-0 = top; VTK image y=0 = bottom.
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

    # ── bottom-right label tile ──────────────────────────────────────────

    def _make_br_label_tile(self, speed_text: str) -> np.ndarray:
        """Render a ``(cell, cell, 3)`` tile: 'Map' header + travel speed lines.

        Layout (top → bottom):
          • "Map"   — 2× bigger than the former title font
          • blank line gap
          • "Speed" — 2× bigger than the former speed font
          • speed value (same large font)
        """
        tile = np.zeros((self.cell, self.cell, 3), dtype=np.uint8)
        try:
            from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font  # noqa: PLC0415
            _fg_header = (200, 200, 200)
            _fg_speed  = (160, 220, 255)
            _lm        = max(3, self.cell // 32)
            # 2× the former sizes (were cell/8 and cell/12).
            _fsize_title = max(8, round(self.cell / 4))   # was cell/8
            _fsize_spd   = max(7, round(self.cell / 6))   # was cell/12
            try:
                _font_title = _PIL_Font.load_default(size=_fsize_title)
                _font_spd   = _PIL_Font.load_default(size=_fsize_spd)
            except TypeError:
                _font_title = _font_spd = _PIL_Font.load_default()
            _pil = _PIL_Image.new("RGB", (self.cell, self.cell), (0, 0, 0))
            _d   = _PIL_Draw.Draw(_pil)
            _gap  = max(2, self.cell // 32)
            _bw   = self.cell - 2 * _lm

            def _tw(text, font):
                try:
                    bb = _d.textbbox((0, 0), text, font=font)
                    return bb[2] - bb[0], bb[3] - bb[1]
                except AttributeError:
                    return self.cell - 2 * _lm, _fsize_title

            _w_map, _h_map   = _tw("Map",   _font_title)
            _w_spd, _h_spd   = _tw("Speed", _font_spd)
            _val_text = speed_text if speed_text else "\u2014"
            _w_val, _h_val   = _tw(_val_text, _font_spd)

            # Total height: "Map" + gap + blank-line + "Speed" + gap + value.
            _blank = _h_spd   # blank line = one speed-font height
            _total = _h_map + _gap + _blank + _gap + _h_spd + _gap + _h_val
            _vy    = max(_gap, (self.cell - _total) // 2)

            _d.text((_lm + (_bw - _w_map) // 2, _vy),
                    "Map", font=_font_title, fill=_fg_header)

            _y_spd = _vy + _h_map + _gap + _blank + _gap
            _d.text((_lm + (_bw - _w_spd) // 2, _y_spd),
                    "Speed", font=_font_spd, fill=_fg_speed)

            _y_val = _y_spd + _h_spd + _gap
            _d.text((_lm + (_bw - _w_val) // 2, _y_val),
                    _val_text, font=_font_spd, fill=_fg_speed)

            tile = np.array(_pil, dtype=np.uint8)
        except Exception:  # noqa: BLE001
            pass
        return tile

    def set_speed_text(self, speed_text: str) -> None:
        """Update the bottom-right speed display tile (called before each frame)."""
        if speed_text == self._br_speed_text:
            return
        self._br_speed_text = speed_text
        self._br_label_tile = self._make_br_label_tile(speed_text)


    # ── geometry ─────────────────────────────────────────────────────────

    def _place_plane(
        self,
        panel_center: np.ndarray,
        cam_pos: np.ndarray,
        world_up: np.ndarray,
    ) -> None:
        """Orient and size the ``vtkPlaneSource`` so it faces ``cam_pos``."""
        normal = np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        n_len = float(np.linalg.norm(normal))
        if n_len < 1e-9:
            return
        normal /= n_len

        # Plane-up: project world_up onto the plane (remove normal component).
        up = np.asarray(world_up, dtype=float).copy()
        up -= np.dot(up, normal) * normal
        up_n = float(np.linalg.norm(up))
        if up_n < 1e-6:
            # world_up nearly parallel to normal → use a fallback perpendicular.
            fallback = (
                np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            up = fallback - np.dot(fallback, normal) * normal
            up_n = float(np.linalg.norm(up))
            if up_n < 1e-9:
                return
        up /= up_n

        # panel_right = cross(up, normal):
        #   right-hand rule gives the viewer's right when looking at the
        #   panel face-on (normal pointing toward the camera).
        panel_right = np.cross(up, normal)
        pr_n = float(np.linalg.norm(panel_right))
        if pr_n < 1e-9:
            return
        panel_right /= pr_n

        # Physical half-size derived from the desired angular size:
        #   half = distance × tan(half_angle)
        #   → panel subtends `angular_size_deg` degrees in width and height.
        d_to_panel = float(np.linalg.norm(
            np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        ))
        half = d_to_panel * self._panel_tan

        # vtkPlaneSource corners:
        #   origin = bottom-left, point1 = bottom-right (u axis),
        #   point2 = top-left   (v axis), as seen from the camera.
        c = np.asarray(panel_center, dtype=float)
        origin = c - panel_right * half - up * half
        point1 = c + panel_right * half - up * half
        point2 = c - panel_right * half + up * half
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
        """Reposition and reorient the billboard for the current frame.

        Parameters
        ----------
        cam_pos:
            Camera position in VTK world coordinates.
        travel_dir:
            Pre-smoothed unit travel-direction vector (from
            ``_compute_travel_dirs``).  Must not be collinear with
            ``world_up`` — the caller guarantees no purely-vertical movement.
        world_up:
            Constant world-up unit vector (camera ``view_up`` of first
            keyframe).  Never derived per-frame.
        """
        travel = np.asarray(travel_dir, dtype=float)
        t_n = float(np.linalg.norm(travel))
        if t_n < 1e-9:
            return  # stationary: caller froze travel_dir, skip geometry update
        travel /= t_n

        up = np.asarray(world_up, dtype=float)
        up_n = float(np.linalg.norm(up))
        up = up / up_n if up_n > 1e-9 else np.array([0.0, 1.0, 0.0])

        # "Left" when facing travel_dir standing upright:
        #   right = cross(travel, up)  →  left = −right
        right = np.cross(travel, up)
        r_n = float(np.linalg.norm(right))
        if r_n < 1e-6:
            # travel nearly parallel to world_up (vertical) — keep previous pose.
            return
        right /= r_n

        D = self.focal_dist
        if self._meters_per_voxel > 0.0:
            fwd_vox  = self._forward_metres / self._meters_per_voxel
            left_vox = self._left_metres    / self._meters_per_voxel
            vert_vox = self._vert_metres    / self._meters_per_voxel
        else:
            fwd_vox  = D * self.forward_frac
            left_vox = D * self.left_frac
            vert_vox = D * self.vert_frac
        panel_center = (
            np.asarray(cam_pos, dtype=float)
            + travel   * fwd_vox
            + (-right) * left_vox
            + up       * vert_vox
        )
        self._place_plane(panel_center, cam_pos, up)

    # ── image content ────────────────────────────────────────────────────
    # (Identical logic to OrthoPanelOverlay — duplicated to keep each
    # class fully self-contained without a shared base class.)

    def _draw_cell_border(self, tile: np.ndarray) -> None:
        pass  # borders removed

    def _draw_dot(self, tile: np.ndarray, cx: float, cy: float) -> None:
        H, W = tile.shape[:2]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r = self.dot_radius
        y0, y1 = max(0, cy_i - r), min(H, cy_i + r + 1)
        x0, x1 = max(0, cx_i - r), min(W, cx_i + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx_i) ** 2 + (yy - cy_i) ** 2 <= r * r
        for ch in range(3):
            tile[y0:y1, x0:x1, ch] = np.where(
                mask, self.dot_color[ch], tile[y0:y1, x0:x1, ch]
            )

    def _draw_cone(
        self, tile: np.ndarray, apex: Tuple[float, float],
        direction: Tuple[float, float],
    ) -> None:
        d = np.asarray(direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return
        d /= n
        perp = np.array([-d[1], d[0]], dtype=np.float32)
        a = np.asarray(apex, dtype=np.float32)
        b = a + self.cone_length * d + self.cone_half_width * perp
        c = a + self.cone_length * d - self.cone_half_width * perp
        pts = np.stack([a, b, c])
        H, W = tile.shape[:2]
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        x1 = min(W, int(np.ceil(pts[:, 0].max())) + 1)
        y1 = min(H, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-6:
            return
        px = xx - pts[0, 0]
        py = yy - pts[0, 1]
        u = (px * v1[1] - v1[0] * py) / den
        v = (v0[0] * py - px * v0[1]) / den
        mask = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if not mask.any():
            return
        alpha = self.cone_alpha
        pink = np.array(self.cone_color, dtype=np.float32)
        sub = tile[y0:y1, x0:x1].astype(np.float32)
        sub[mask] = sub[mask] * (1.0 - alpha) + pink * alpha
        tile[y0:y1, x0:x1] = sub.astype(np.uint8)

    def _fit_aspect(
        self, gray: np.ndarray, dot_rc: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = gray.shape
        S = self.cell
        if w == 0 or h == 0:
            return np.zeros((S, S, 3), dtype=np.uint8), (S / 2.0, S / 2.0)
        scale = min(S / w, S / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = self._nn_resize(gray, nw, nh)
        tile = np.zeros((S, S, 3), dtype=np.uint8)
        ox = (S - nw) // 2
        oy = (S - nh) // 2
        tile[oy:oy + nh, ox:ox + nw, 0] = resized
        tile[oy:oy + nh, ox:ox + nw, 1] = resized
        tile[oy:oy + nh, ox:ox + nw, 2] = resized
        dot_row, dot_col = dot_rc
        return tile, (ox + dot_col * scale, oy + dot_row * scale)

    def _square_crop(
        self, gray: np.ndarray, dot_rc: Tuple[int, int],
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        h, w = gray.shape
        S = self.cell
        if w == 0 or h == 0:
            return np.zeros((S, S, 3), dtype=np.uint8), (S / 2.0, S / 2.0)
        win = int(min(h, w))
        dr, dc = dot_rc
        r0 = int(np.clip(dr - win // 2, 0, max(0, h - win)))
        c0 = int(np.clip(dc - win // 2, 0, max(0, w - win)))
        crop = np.zeros((win, win), dtype=gray.dtype)
        r1 = min(h, r0 + win)
        c1 = min(w, c0 + win)
        crop[: r1 - r0, : c1 - c0] = gray[r0:r1, c0:c1]
        dot_in_crop = (dr - r0, dc - c0)
        scale = S / win
        resized = self._nn_resize(crop, S, S)
        tile = np.zeros((S, S, 3), dtype=np.uint8)
        tile[..., 0] = resized
        tile[..., 1] = resized
        tile[..., 2] = resized
        return tile, (dot_in_crop[1] * scale, dot_in_crop[0] * scale)

    def compose(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float],
    ) -> np.ndarray:
        Z, Y, X = self.volume.shape
        z_idx = int(np.clip(round(cam_world[0]), 0, Z - 1))
        y_idx = int(np.clip(round(cam_world[1]), 0, Y - 1))
        x_idx = int(np.clip(round(cam_world[2]), 0, X - 1))
        d = np.asarray(cam_dir, dtype=np.float32)
        if np.linalg.norm(d) < 1e-6:
            d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        d_z, d_y, d_x = float(d[0]), float(d[1]), float(d[2])
        canvas = self._canvas
        canvas.fill(0)
        S = self.cell

        xy = self.volume[z_idx, :, :]
        tile, (dx, dy) = self._fit_aspect(xy, (y_idx, x_idx))
        _premultiply_tile(tile)
        scale_xy = min(S / xy.shape[1], S / xy.shape[0])
        self._draw_cone(tile, (dx, dy), (d_x * scale_xy, d_y * scale_xy))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[0:S, 0:S] = tile

        yz = self.volume[:, :, x_idx]
        yz_rot = np.ascontiguousarray(np.rot90(yz, k=-1))
        dot_rot = (y_idx, (Z - 1) - z_idx)
        tile, (dx, dy) = self._square_crop(yz_rot, dot_rot)
        _premultiply_tile(tile)
        scale_yz = S / float(min(yz_rot.shape))
        self._draw_cone(tile, (dx, dy), (-d_z * scale_yz, d_y * scale_yz))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[0:S, S:2 * S] = tile

        xz = self.volume[:, y_idx, :]
        tile, (dx, dy) = self._square_crop(xz, (z_idx, x_idx))
        _premultiply_tile(tile)
        scale_xz = S / float(min(xz.shape))
        self._draw_cone(tile, (dx, dy), (d_x * scale_xz, d_z * scale_xz))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[S:2 * S, 0:S] = tile

        canvas[S:2 * S, S:2 * S] = self._br_label_tile
        return canvas

    def update_image(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        """Recompose the panel content for the given camera position/direction."""
        self._push_pixels(self.compose(cam_world, cam_dir))

    def update(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float],
        travel_dir: np.ndarray,
        world_up: np.ndarray,
    ) -> None:
        """Combined per-frame update: reposition the plane and refresh the image."""
        self.update_pose(cam_world, travel_dir, world_up)
        self.update_image(cam_world, cam_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Info panel — two-line (title + mode subtitle) overlay
# ──────────────────────────────────────────────────────────────────────────────

_INFO_W: int = 392   # canvas pixel width  (560 × 0.7)
_INFO_H: int = 54    # canvas pixel height (90 × 0.6)


def _render_info_tile(
    title: str,
    subtitle: str,
    speed_text: str = "",
    w: int = _INFO_W,
    h: int = _INFO_H,
    subtitle_font_scale: float = 1.0,
) -> np.ndarray:
    """PIL-render a ``(h, w, 3)`` uint8 tile: title + speed + subtitle, all centred."""
    tile = np.full((h, w, 3), 10, dtype=np.uint8)   # near-black background
    try:
        from PIL import Image as _PI, ImageDraw as _PD, ImageFont as _PF
        img = _PI.fromarray(tile, "RGB")
        d   = _PD.Draw(img)
        # Scale fonts proportionally to tile height (reference: h=54 → 9/11 px).
        _fscale = max(0.5, h / 54)
        try:
            font_small = _PF.load_default(size=max(8, round(11 * _fscale)))
            font_sub   = _PF.load_default(size=max(9, round(14 * _fscale * subtitle_font_scale)))
        except TypeError:       # Pillow < 9.2: load_default has no 'size' arg
            font_small = font_sub = _PF.load_default()
        line_gap   = max(2, round(4 * _fscale))   # px between the two rows
        inline_gap = max(4, round(6 * _fscale))   # px between title and speed on row 1
        try:
            # ── measure row 1 (title + optional speed, same line) ────────
            bb_t  = d.textbbox((0, 0), title,    font=font_small)
            bb_s  = d.textbbox((0, 0), subtitle, font=font_sub)
            t_w, t_h = bb_t[2] - bb_t[0], bb_t[3] - bb_t[1]
            s_w, s_h = bb_s[2] - bb_s[0], bb_s[3] - bb_s[1]
            if speed_text:
                bb_sp  = d.textbbox((0, 0), speed_text, font=font_small)
                sp_w   = bb_sp[2] - bb_sp[0]
                row1_w = t_w + inline_gap + sp_w
            else:
                sp_w   = 0
                row1_w = t_w
            row1_h = t_h   # title and speed share the same font → same height
            total_h = row1_h + line_gap + s_h
            y = max(2, (h - total_h) // 2)
            # ── draw row 1 ───────────────────────────────────────────────
            x = max(4, (w - row1_w) // 2)
            d.text((x, y), title, fill=(190, 190, 190), font=font_small)
            if speed_text:
                d.text((x + t_w + inline_gap, y), speed_text,
                       fill=(180, 220, 160), font=font_small)
            y += row1_h + line_gap
            # ── draw row 2 (subtitle) ────────────────────────────────────
            d.text((max(4, (w - s_w) // 2), y), subtitle,
                   fill=(255, 255, 255), font=font_sub)
        except AttributeError:  # old Pillow: no textbbox
            y = 4
            spd_suffix = ("  " + speed_text) if speed_text else ""
            d.text((4, y), title + spd_suffix, fill=(190, 190, 190), font=font_small)
            y += 15
            d.text((4, y), subtitle, fill=(255, 255, 255), font=font_sub)
        tile = np.array(img, dtype=np.uint8)
    except Exception:  # noqa: BLE001  (PIL not available)
        pass
    return tile


class InfoPanelOverlay:
    """Flat-mode 2-D HUD: title + mode-subtitle with a thin white border.

    Rendered on VTK layer 1 (same as :class:`OrthoPanelOverlay`), centred at
    the top of the window.  Call :meth:`update` whenever the subtitle changes;
    the PIL image is re-rendered only when the content has actually changed.

    Parameters
    ----------
    plotter:
        A ``vedo.Plotter`` (or any object whose ``.window`` attribute is a
        ``vtkRenderWindow``).
    title:
        First line — fixed specimen/session label.
    initial_subtitle:
        Second line — mode description shown at start.
    opacity:
        Global opacity of the actor (0–1).  Does not pre-multiply pixels,
        so the plain composite is: ``rendered_rgb × opacity`` over the scene.
    """

    def __init__(
        self,
        plotter,
        title: str,
        initial_subtitle: str = "",
        *,
        opacity: float = 0.72,
        width_frac: float = 0.38,
        height_frac: float = 0.064,
        subtitle_font_scale: float = 1.0,
    ) -> None:
        self._title       = title
        self._subtitle    = initial_subtitle
        self._speed_text  = ""
        self._width_frac  = width_frac
        self._height_frac = height_frac
        self._subtitle_font_scale = subtitle_font_scale
        self._cached_bytes: bytes = b""

        # ── VTK pipeline ──────────────────────────────────────────────────
        renwin = plotter.window
        win_w, win_h = renwin.GetSize()
        if win_w < 1:
            win_w = 1920
        if win_h < 1:
            win_h = 1080
        self.panel_w = max(100, int(win_w * width_frac))
        self.panel_h = max(20,  int(win_h * height_frac))
        canvas = _render_info_tile(title, initial_subtitle,
                                   subtitle_font_scale=subtitle_font_scale,
                                   w=self.panel_w, h=self.panel_h)
        n_layers = renwin.GetNumberOfLayers()
        renwin.SetNumberOfLayers(max(n_layers, 2))

        self.ren = vtk.vtkRenderer()
        self.ren.SetViewport(0.0, 0.0, 1.0, 1.0)
        self.ren.SetLayer(1)
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.InteractiveOff()
        self.ren.SetUseDepthPeeling(False)
        try:
            self.ren.EraseOff()
        except Exception:  # noqa: BLE001
            pass
        renwin.AddRenderer(self.ren)

        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._push_pixels(canvas)

        self._mapper = vtk.vtkImageMapper()
        self._mapper.SetInputConnection(self._importer.GetOutputPort())
        self._mapper.SetColorWindow(255.0)
        self._mapper.SetColorLevel(127.5)

        self.image_actor = vtk.vtkActor2D()
        self.image_actor.SetMapper(self._mapper)
        self.image_actor.GetProperty().SetOpacity(opacity)
        pos_coord = self.image_actor.GetPositionCoordinate()
        pos_coord.SetCoordinateSystemToNormalizedDisplay()
        self.ren.AddActor2D(self.image_actor)

        self._reposition(renwin)
        try:
            renwin.AddObserver("ModifiedEvent", lambda *_a: self._reposition(renwin))
        except Exception:  # noqa: BLE001
            pass

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

    def _reposition(self, renwin) -> None:
        win_w, win_h = renwin.GetSize()
        if win_w < 1 or win_h < 1:
            return
        pw = max(100, int(win_w * self._width_frac))
        ph = max(20,  int(win_h * self._height_frac))
        if pw != self.panel_w or ph != self.panel_h:
            self.panel_w = pw
            self.panel_h = ph
            self._importer.SetWholeExtent(0, pw - 1, 0, ph - 1, 0, 0)
            self._importer.SetDataExtent(0, pw - 1, 0, ph - 1, 0, 0)
            self._push_pixels(_render_info_tile(
                self._title, self._subtitle, self._speed_text,
                subtitle_font_scale=self._subtitle_font_scale,
                w=pw, h=ph))
        # Bottom-left of the actor in normalised display coords — actor is
        # rendered upward from this point (VTK convention).
        nx = 0.5 - self.panel_w / (2.0 * win_w)
        ny = 1.0 - self.panel_h / float(win_h) - 0.008
        self.image_actor.GetPositionCoordinate().SetValue(nx, ny)

    def update(self, subtitle: str, speed_text: str = "") -> None:
        """Re-render the panel only when content has changed."""
        if subtitle == self._subtitle and speed_text == self._speed_text:
            return
        self._subtitle   = subtitle
        self._speed_text = speed_text
        self._push_pixels(_render_info_tile(self._title, subtitle, speed_text,
                                            subtitle_font_scale=self._subtitle_font_scale,
                                            w=self.panel_w, h=self.panel_h))

    def set_visible(self, visible: bool) -> None:
        """Show or hide the info-panel overlay."""
        self.image_actor.SetVisibility(1 if visible else 0)


class InfoBillboard3D:
    """VR-mode 3-D billboard: title + mode-subtitle, above the travel direction.

    A rectangular ``vtkPlaneSource`` textured with a PIL-rendered tile is
    placed at::

        cam_pos
          + travel_dir × focal_dist × forward_frac
          + world_up   × focal_dist × vert_frac
          − right_dir  × focal_dist × left_frac

    The plane faces the camera (same *orient-and-size* logic as
    :class:`OrthoPanel3DBillboard`) and subtends ``angular_width_deg``
    horizontally.  Height is ``angular_width_deg / aspect`` degrees.

    The pixel tile is **pre-multiplied** for dark-background transparency
    (identical to the ortho billboard), but the 2-px border is restored to
    pure white (255) after pre-multiply so that the frame remains fully
    opaque in VR.

    Parameters
    ----------
    angular_width_deg:
        Horizontal FOV subtended by the billboard (degrees).  Default 20°.
    aspect:
        Width-to-height ratio of the billboard plane.  Default 4.0 → height
        is ``angular_width_deg / 4`` degrees.
    vert_frac:
        Vertical offset as a fraction of :attr:`focal_dist`.  Positive values
        place the panel above the travel direction.  Default 0.15.
    """

    def __init__(
        self,
        plotter,
        title: str,
        initial_subtitle: str = "",
        *,
        focal_dist: float = 100.0,
        forward_frac: float = 0.50,
        left_frac: float = 0.0,
        vert_frac: float = 0.15,
        angular_width_deg: float = 14.3,
        aspect: float = 4.0,
        tile_scale: float = 1.0,
        meters_per_voxel: float = 0.0,
        forward_metres: float = 2.0,
        left_metres: float = 0.0,
        vert_metres: float = 0.375,
        hide_when_reversed: bool = False,
        subtitle_font_scale: float = 1.0,
    ) -> None:
        self._title       = title
        self._subtitle    = initial_subtitle
        self._speed_text  = ""
        self.focal_dist   = float(focal_dist)
        self.forward_frac = float(forward_frac)
        self.left_frac    = float(left_frac)
        self.vert_frac    = float(vert_frac)
        self._meters_per_voxel = float(meters_per_voxel)
        self._forward_metres   = float(forward_metres)
        self._left_metres      = float(left_metres)
        self._vert_metres      = float(vert_metres)
        self._hide_when_reversed: bool = hide_when_reversed
        self._initial_travel: "np.ndarray | None" = None
        self._subtitle_font_scale: float = float(subtitle_font_scale)
        self._tan_x = math.tan(math.radians(float(angular_width_deg) / 2.0))
        self._tan_y = self._tan_x / float(aspect)
        self.panel_w = max(1, round(_INFO_W * tile_scale))
        self.panel_h = max(1, round(_INFO_H * tile_scale))

        canvas = _render_info_tile(title, initial_subtitle,
                                   subtitle_font_scale=self._subtitle_font_scale,
                                   w=self.panel_w, h=self.panel_h)
        _premultiply_tile(canvas)
        self._cached_bytes: bytes = b""

        # ── VTK image pipeline ────────────────────────────────────────────
        self._importer = vtk.vtkImageImport()
        self._importer.SetDataScalarTypeToUnsignedChar()
        self._importer.SetNumberOfScalarComponents(3)
        self._importer.SetWholeExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataExtent(0, self.panel_w - 1, 0, self.panel_h - 1, 0, 0)
        self._importer.SetDataSpacing(1.0, 1.0, 1.0)
        self._importer.SetDataOrigin(0.0, 0.0, 0.0)
        self._push_pixels(canvas)

        self._texture = vtk.vtkTexture()
        self._texture.SetInputConnection(self._importer.GetOutputPort())
        self._texture.InterpolateOn()
        self._texture.RepeatOff()
        self._texture.EdgeClampOn()

        self._plane = vtk.vtkPlaneSource()
        self._plane.SetResolution(1, 1)
        self._plane.SetOrigin(0.0, 0.0, 0.0)
        self._plane.SetPoint1(1.0, 0.0, 0.0)
        self._plane.SetPoint2(0.0, 1.0, 0.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._plane.GetOutputPort())

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(mapper)
        self._actor.SetTexture(self._texture)
        prop = self._actor.GetProperty()
        prop.SetOpacity(1.0)    # must be 1.0 — vtkOpaquePass constraint in VR
        try:
            prop.LightingOff()
        except Exception:  # noqa: BLE001
            pass
        self._actor.GetProperty().BackfaceCullingOff()

        # Add to the layer-0 renderer (panoramic pass operates on layer 0).
        target_ren = None
        for renderer in plotter.renderers:
            if renderer.GetLayer() == 0:
                target_ren = renderer
                break
        if target_ren is None:
            target_ren = plotter.renderers[0]
        target_ren.AddActor(self._actor)
        # Remove any mapper-level clipping planes so the billboard is
        # never clipped by them in VR.
        self._actor.GetMapper().RemoveAllClippingPlanes()

        logger.debug(
            "InfoBillboard3D attached  (ang_w=%.1f°, aspect=%.1f, vert_frac=%.2f, "
            "hide_when_reversed=%s)",
            math.degrees(2.0 * math.atan(self._tan_x)),
            self._tan_x / self._tan_y,
            self.vert_frac,
            hide_when_reversed,
        )

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
        """Orient and size the vtkPlaneSource as a wide rectangle facing *cam_pos*."""
        normal = np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        n_len  = float(np.linalg.norm(normal))
        if n_len < 1e-9:
            return
        normal /= n_len

        up = np.asarray(world_up, dtype=float).copy()
        up -= np.dot(up, normal) * normal
        up_n = float(np.linalg.norm(up))
        if up_n < 1e-6:
            fallback = (
                np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            up   = fallback - np.dot(fallback, normal) * normal
            up_n = float(np.linalg.norm(up))
            if up_n < 1e-9:
                return
        up /= up_n

        panel_right = np.cross(up, normal)
        pr_n = float(np.linalg.norm(panel_right))
        if pr_n < 1e-9:
            return
        panel_right /= pr_n

        d      = float(np.linalg.norm(
            np.asarray(cam_pos, dtype=float) - np.asarray(panel_center, dtype=float)
        ))
        half_x = d * self._tan_x    # horizontal half-size
        half_y = d * self._tan_y    # vertical half-size (< half_x, aspect > 1)

        c      = np.asarray(panel_center, dtype=float)
        origin = c - panel_right * half_x - up * half_y
        point1 = c + panel_right * half_x - up * half_y
        point2 = c - panel_right * half_x + up * half_y
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
        """Reposition the billboard for the current camera / travel direction."""
        travel = np.asarray(travel_dir, dtype=float)
        t_n    = float(np.linalg.norm(travel))
        if t_n < 1e-9:
            return
        travel /= t_n

        up   = np.asarray(world_up, dtype=float)
        up_n = float(np.linalg.norm(up))
        up   = up / up_n if up_n > 1e-9 else np.array([0.0, 1.0, 0.0])

        right = np.cross(travel, up)
        r_n   = float(np.linalg.norm(right))
        if r_n < 1e-6:
            return
        right /= r_n

        if self._meters_per_voxel > 0.0:
            fwd_vox  = self._forward_metres / self._meters_per_voxel
            left_vox = self._left_metres    / self._meters_per_voxel
            vert_vox = self._vert_metres    / self._meters_per_voxel
        else:
            D = self.focal_dist
            fwd_vox  = D * self.forward_frac
            left_vox = D * self.left_frac
            vert_vox = D * self.vert_frac

        # ── Visibility: hide when travel direction has reversed ───────────
        # On the first call we record the initial travel direction as reference.
        # On subsequent calls, if hide_when_reversed is set and the dot product
        # with the initial direction is negative (angle > 90°), the billboard
        # is hidden to avoid pseudoscopic stereo during U-turns.
        if self._hide_when_reversed:
            if self._initial_travel is None:
                self._initial_travel = travel.copy()
                self._actor.VisibilityOn()
            else:
                dot = float(np.dot(travel, self._initial_travel))
                if dot < 0.0:
                    self._actor.VisibilityOff()
                    return
                else:
                    self._actor.VisibilityOn()

        panel_center = (
            np.asarray(cam_pos, dtype=float)
            + travel   * fwd_vox
            + (-right) * left_vox
            + up       * vert_vox
        )
        self._place_plane(panel_center, cam_pos, up)

    def set_visible(self, visible: bool) -> None:
        """Show or hide the info billboard."""
        if visible:
            self._actor.VisibilityOn()
        else:
            self._actor.VisibilityOff()

    def update_image(self, subtitle: str, speed_text: str = "") -> None:
        """Re-render the PIL tile only when content has changed."""
        if subtitle == self._subtitle and speed_text == self._speed_text:
            return
        self._subtitle   = subtitle
        self._speed_text = speed_text
        canvas = _render_info_tile(self._title, subtitle, speed_text,
                                   subtitle_font_scale=self._subtitle_font_scale,
                                   w=self.panel_w, h=self.panel_h)
        _premultiply_tile(canvas)
        self._push_pixels(canvas)

    def update(
        self,
        cam_pos: np.ndarray,
        travel_dir: np.ndarray,
        world_up: np.ndarray,
        subtitle: str,
        speed_text: str = "",
    ) -> None:
        """Combined per-frame update: reposition + refresh image if needed."""
        self.update_pose(cam_pos, travel_dir, world_up)
        self.update_image(subtitle, speed_text)
