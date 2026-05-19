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

import numpy as np
import tifffile
import vtk

logger = logging.getLogger(__name__)


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
    ) -> None:
        if raw_volume.ndim != 3:
            raise ValueError("raw_volume must be a 3-D array (Z, Y, X).")
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
            renwin.AddObserver("ModifiedEvent", lambda *a, **k: self._reposition())
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
        """Anchor the panel's UPPER-LEFT corner at ``(x0, y1)`` of the
        requested viewport rectangle.

        ``vtkActor2D`` anchors its lower-left corner at the position we
        set, so we have to convert.  Panel pixel size stays fixed; only
        the placement tracks the window size.
        """
        try:
            w, h = self.plotter.window.GetSize()
        except Exception:  # noqa: BLE001
            return
        x0, _y0, _x1, y1 = self._target_viewport
        panel_h_norm = self.panel_h / max(1, h)
        nx = float(x0)
        ny = float(y1) - panel_h_norm
        pos_coord = self.image_actor.GetPositionCoordinate()
        pos_coord.SetValue(nx, ny)
        logger.debug(
            "OrthoPanelOverlay reposition  win=%dx%d  anchor=(%.3f, %.3f)",
            w, h, nx, ny,
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
        col = np.array(self.cell_border_color, dtype=np.uint8)
        tile[0, :, :] = col
        tile[-1, :, :] = col
        tile[:, 0, :] = col
        tile[:, -1, :] = col

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
        scale_xz = S / float(min(xz.shape))
        # Screen x = x_world (+d_x), screen y = z_world (+d_z).
        self._draw_cone(tile, (dx, dy), (d_x * scale_xz, d_z * scale_xz))
        self._draw_dot(tile, dx, dy)
        self._draw_cell_border(tile)
        canvas[S:2 * S, 0:S] = tile

        # Bottom-right cell stays black on purpose.
        return canvas

    def update(
        self,
        cam_world: Sequence[float],
        cam_dir: Sequence[float] = (0.0, 0.0, -1.0),
    ) -> None:
        """Recompose the panel for the given camera position and direction."""
        rgb = self.compose(cam_world, cam_dir)
        self._push_pixels(rgb)


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
