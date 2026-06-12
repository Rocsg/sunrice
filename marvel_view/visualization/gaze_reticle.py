"""Gaze reticle: hollow white circle (+ direction triangle in VR) in 3-D space
or as a fixed 2-D screen overlay for the interactive viewer.

:class:`GazeReticle`
    3-D actors placed in front of the camera (flat movie + VR modes).
    The circle is placed along the supplied *forward* direction, so in VR
    mode passing ``vr_travel_dirs[i]`` (travel direction) instead of the
    focal direction correctly centres the reticle on the cable path.

:class:`GazeReticle2D`
    Fixed 2-D screen overlay (``vtkActor2D``) at the window centre.
    For the interactive viewer where the camera always looks at the focal
    point so the screen centre is always the gaze centre.  Not affected
    by camera position or projection.

Triangle (VR mode only, :class:`GazeReticle`):
  A filled equilateral triangle on the rim of the circle that appears
  when the travel direction is changing, pointing outward in the direction
  of rotation.  Size scales from ½× to 2× the circle radius.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import vtk

__all__ = ["GazeReticle", "GazeReticle2D"]

# ── tunables ──────────────────────────────────────────────────────────────────
# Angular speed threshold (rad/frame) below which the triangle is hidden.
_TRI_SHOW_THRESHOLD = 4e-4
# Angular speed (rad/frame) that maps to the maximum triangle size (2× circle r).
_TRI_FAST_SPEED = 0.012


class GazeReticle:
    """Hollow white circle (and optional direction triangle) at gaze center.

    Parameters
    ----------
    renderer:
        The ``vtkRenderer`` that will own the actors.
    circle_deg:
        Apparent diameter of the circle in degrees (default 4°).
    line_width:
        Width of the circle outline in screen pixels (default 2.0).
    vr_mode:
        If ``True``, enable the direction-triangle indicator.
    n_circle_pts:
        Number of line segments used to approximate the circle.
    """

    def __init__(
        self,
        renderer,
        *,
        circle_deg: float = 4.0,
        line_width: float = 2.0,
        vr_mode: bool = False,
        n_circle_pts: int = 64,
        tri_gap_factor: float = 1.7,
    ) -> None:
        self._renderer = renderer
        self._circle_deg = float(circle_deg)
        self._line_width = float(line_width)
        self._vr_mode = bool(vr_mode)
        self._n_circle_pts = int(n_circle_pts)
        self._tri_gap_factor = float(tri_gap_factor)

        # ── Circle (line-loop polydata) ────────────────────────────────────
        self._circle_pts_vtk = vtk.vtkPoints()
        self._circle_pts_vtk.SetNumberOfPoints(n_circle_pts)
        self._circle_cells = vtk.vtkCellArray()
        for j in range(n_circle_pts):
            self._circle_cells.InsertNextCell(2)
            self._circle_cells.InsertCellPoint(j)
            self._circle_cells.InsertCellPoint((j + 1) % n_circle_pts)
        self._circle_pd = vtk.vtkPolyData()
        self._circle_pd.SetPoints(self._circle_pts_vtk)
        self._circle_pd.SetLines(self._circle_cells)

        circle_mapper = vtk.vtkPolyDataMapper()
        circle_mapper.SetInputData(self._circle_pd)
        circle_mapper.ScalarVisibilityOff()

        self._circle_actor = vtk.vtkActor()
        self._circle_actor.SetMapper(circle_mapper)
        p = self._circle_actor.GetProperty()
        p.SetColor(1.0, 1.0, 1.0)
        p.SetLineWidth(float(line_width))
        p.SetRepresentationToWireframe()
        try:
            p.LightingOff()
        except AttributeError:
            pass
        try:
            p.SetRenderLinesAsTubes(False)
        except AttributeError:
            pass
        renderer.AddActor(self._circle_actor)

        # ── Triangle (filled polydata, VR only) ───────────────────────────
        self._tri_pts_vtk = vtk.vtkPoints()
        self._tri_pts_vtk.SetNumberOfPoints(3)
        tri_cells = vtk.vtkCellArray()
        tri_cells.InsertNextCell(3)
        tri_cells.InsertCellPoint(0)
        tri_cells.InsertCellPoint(1)
        tri_cells.InsertCellPoint(2)
        self._tri_pd = vtk.vtkPolyData()
        self._tri_pd.SetPoints(self._tri_pts_vtk)
        self._tri_pd.SetPolys(tri_cells)

        tri_mapper = vtk.vtkPolyDataMapper()
        tri_mapper.SetInputData(self._tri_pd)
        tri_mapper.ScalarVisibilityOff()

        self._tri_actor = vtk.vtkActor()
        self._tri_actor.SetMapper(tri_mapper)
        tp = self._tri_actor.GetProperty()
        tp.SetColor(1.0, 1.0, 1.0)
        tp.SetOpacity(1.0)
        try:
            tp.LightingOff()
        except AttributeError:
            pass
        self._tri_actor.SetVisibility(0)
        renderer.AddActor(self._tri_actor)

    # ── public API ─────────────────────────────────────────────────────────

    def update(
        self,
        cam_pos: np.ndarray,
        cam_dir: np.ndarray,
        cam_right: np.ndarray,
        cam_up: np.ndarray,
        focal_dist: float,
        *,
        ang_vel_2d: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Rebuild all actors for the current camera frame.

        Parameters
        ----------
        cam_pos, cam_dir, cam_right, cam_up:
            Camera position and orthonormal axes (unit vectors).
        focal_dist:
            Approximate camera→focal-point distance (used to choose the
            placement distance so the reticle is never too close to clip).
        ang_vel_2d:
            ``(dright, dup)`` angular velocity in radians/frame projected
            onto the camera's right/up plane.  If ``None`` or below the
            threshold, the triangle is hidden.  Only meaningful in VR mode.
        """
        cam_pos   = np.asarray(cam_pos,   dtype=float)
        cam_dir   = np.asarray(cam_dir,   dtype=float)
        cam_right = np.asarray(cam_right, dtype=float)
        cam_up    = np.asarray(cam_up,    dtype=float)

        # Placement distance: 5 % of focal distance, minimum 3 units.
        d = max(3.0, float(focal_dist) * 0.05)
        center = cam_pos + cam_dir * d
        radius = d * math.tan(math.radians(self._circle_deg / 2.0))

        self._rebuild_circle(center, cam_right, cam_up, radius)

        if self._vr_mode and ang_vel_2d is not None:
            vr, vu = float(ang_vel_2d[0]), float(ang_vel_2d[1])
            speed = math.sqrt(vr * vr + vu * vu)
            if speed >= _TRI_SHOW_THRESHOLD:
                theta = math.atan2(vu, vr)
                t = min(1.0, (speed - _TRI_SHOW_THRESHOLD) / (_TRI_FAST_SPEED - _TRI_SHOW_THRESHOLD))
                t = max(0.0, t)
                # Size: linearly from 0.5× to 2.0× circle radius.
                tri_size = radius * (0.5 + 1.5 * t)
                self._rebuild_triangle(center, cam_right, cam_up, radius, theta, tri_size)
                self._tri_actor.SetVisibility(1)
            else:
                self._tri_actor.SetVisibility(0)
        else:
            self._tri_actor.SetVisibility(0)

    def set_visible(self, visible: bool) -> None:
        """Show or hide both actors."""
        self._circle_actor.SetVisibility(1 if visible else 0)
        if not visible:
            self._tri_actor.SetVisibility(0)

    def remove(self) -> None:
        """Remove actors from the renderer."""
        self._renderer.RemoveActor(self._circle_actor)
        self._renderer.RemoveActor(self._tri_actor)

    # ── geometry builders ──────────────────────────────────────────────────

    def _rebuild_circle(
        self,
        center: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        radius: float,
    ) -> None:
        n = self._n_circle_pts
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        pts = center + radius * (
            np.outer(cos_a, right) + np.outer(sin_a, up)
        )  # shape (n, 3)
        for j in range(n):
            self._circle_pts_vtk.SetPoint(j, float(pts[j, 0]), float(pts[j, 1]), float(pts[j, 2]))
        self._circle_pts_vtk.Modified()
        self._circle_pd.Modified()

    def _rebuild_triangle(
        self,
        center: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        circle_radius: float,
        theta: float,
        tri_size: float,
    ) -> None:
        """Place an equilateral triangle outside the circle at angle *theta*,
        pointing outward (away from center).

        The centroid is placed at ``circle_radius * tri_gap_factor`` from
        *center*, so there is a visible gap between the circle and the triangle.
        *tri_size* is the circumradius (vertex distance from centroid).
        """
        outward = math.cos(theta) * right + math.sin(theta) * up   # unit
        perp    = -math.sin(theta) * right + math.cos(theta) * up  # unit, tangent

        # Centroid sits outside the circle, separated by a gap.
        centroid = center + circle_radius * self._tri_gap_factor * outward

        # For an equilateral triangle with inscribed-circle radius r:
        #   side length  = r * 2 * sqrt(3)
        #   circumradius = 2 * r
        # We use tri_size as the circumradius (vertex distance from centroid).
        R_circ = tri_size  # circumradius
        # Apex points outward; two base vertices are behind and ±60° from axis.
        apex = centroid + outward * R_circ
        bl   = centroid - outward * (R_circ / 2.0) + perp * (R_circ * math.sqrt(3.0) / 2.0)
        br   = centroid - outward * (R_circ / 2.0) - perp * (R_circ * math.sqrt(3.0) / 2.0)

        self._tri_pts_vtk.SetPoint(0, float(apex[0]), float(apex[1]), float(apex[2]))
        self._tri_pts_vtk.SetPoint(1, float(bl[0]),   float(bl[1]),   float(bl[2]))
        self._tri_pts_vtk.SetPoint(2, float(br[0]),   float(br[1]),   float(br[2]))
        self._tri_pts_vtk.Modified()
        self._tri_pd.Modified()


# ── convenience helper ────────────────────────────────────────────────────────


def camera_axes(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(pos, forward, right, up, dist)`` from a VTK camera."""
    pos = np.asarray(cam.GetPosition(),    dtype=float)
    foc = np.asarray(cam.GetFocalPoint(),  dtype=float)
    vu  = np.asarray(cam.GetViewUp(),      dtype=float)

    fwd = foc - pos
    dist = float(np.linalg.norm(fwd))
    if dist < 1e-12:
        fwd  = np.array([0.0, 0.0, -1.0])
        dist = 1.0
    else:
        fwd = fwd / dist

    right = np.cross(fwd, vu)
    rn = float(np.linalg.norm(right))
    if rn < 1e-12:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn

    up = np.cross(right, fwd)
    un = float(np.linalg.norm(up))
    up = up / un if un > 1e-12 else vu / max(float(np.linalg.norm(vu)), 1e-12)

    return pos, fwd, right, up, dist


def angular_velocity_2d(
    prev_dir: np.ndarray,
    curr_dir: np.ndarray,
    curr_right: np.ndarray,
    curr_up: np.ndarray,
) -> tuple[float, float]:
    """Project the angular change from *prev_dir* to *curr_dir* onto the
    camera's (right, up) screen plane.

    Returns ``(dright, dup)`` in radians (one step = one frame).
    """
    prev_dir = np.asarray(prev_dir, dtype=float)
    curr_dir = np.asarray(curr_dir, dtype=float)

    # Component of prev_dir perpendicular to curr_dir (small-angle delta).
    delta = prev_dir - np.dot(prev_dir, curr_dir) * curr_dir
    dright = float(np.dot(delta, curr_right))
    dup    = float(np.dot(delta, curr_up))
    # Note: sign is "where we came from", so negate to get "where we're going".
    return (-dright, -dup)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D screen-space reticle (interactive viewer)
# ─────────────────────────────────────────────────────────────────────────────


class GazeReticle2D:
    """Fixed hollow circle at the window centre, drawn as a ``vtkActor2D``.

    Unlike :class:`GazeReticle`, this class does not depend on camera position
    or projection — the circle is painted in display (pixel) coordinates,
    always at the exact screen centre.  It is correct for the interactive
    viewer where the camera always looks at the focal point.

    Parameters
    ----------
    renderer:
        The ``vtkRenderer`` that owns the actor.
    radius_px:
        Circle radius in screen pixels.
    line_width:
        Line width in screen pixels.
    n_pts:
        Number of line segments in the circle outline.
    """

    def __init__(
        self,
        renderer,
        *,
        radius_px: float = 18.0,
        line_width: float = 2.0,
        n_pts: int = 64,
    ) -> None:
        self._renderer   = renderer
        self._radius_px  = float(radius_px)
        self._line_width = float(line_width)
        self._n_pts      = int(n_pts)
        self._last_size  = (-1, -1)

        # Allocate VTK objects once; points are updated in _rebuild().
        self._pts_vtk = vtk.vtkPoints()
        self._pts_vtk.SetNumberOfPoints(n_pts)

        cells = vtk.vtkCellArray()
        for j in range(n_pts):
            cells.InsertNextCell(2)
            cells.InsertCellPoint(j)
            cells.InsertCellPoint((j + 1) % n_pts)

        self._pd = vtk.vtkPolyData()
        self._pd.SetPoints(self._pts_vtk)
        self._pd.SetLines(cells)

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(self._pd)

        self._actor = vtk.vtkActor2D()
        self._actor.SetMapper(mapper)
        prop = self._actor.GetProperty()
        prop.SetColor(1.0, 1.0, 1.0)
        prop.SetLineWidth(float(line_width))
        renderer.AddActor2D(self._actor)

        # Build with a sensible fallback size.
        self._rebuild(800, 600)

    # ── public API ─────────────────────────────────────────────────────────

    def update_window_size(self, win_w: int, win_h: int) -> None:
        """Recentre the circle if the window was resized."""
        if (win_w, win_h) != self._last_size:
            self._rebuild(win_w, win_h)

    def set_visible(self, visible: bool) -> None:
        self._actor.SetVisibility(1 if visible else 0)

    def remove(self) -> None:
        self._renderer.RemoveActor2D(self._actor)

    # ── geometry ───────────────────────────────────────────────────────────

    def _rebuild(self, win_w: int, win_h: int) -> None:
        cx = win_w / 2.0
        cy = win_h / 2.0
        r  = self._radius_px
        n  = self._n_pts
        for j in range(n):
            a = 2.0 * math.pi * j / n
            self._pts_vtk.SetPoint(j, cx + r * math.cos(a), cy + r * math.sin(a), 0.0)
        self._pts_vtk.Modified()
        self._pd.Modified()
        self._last_size = (win_w, win_h)
