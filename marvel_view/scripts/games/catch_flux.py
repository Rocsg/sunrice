"""Catch the Flux — lame-column-collecting mini-game.

The water-lames animation cycles through N animation steps; at each step a
subset of the M distinct «flux columns» (connected lame surfaces labelled by
``column_id`` in the packed PolyData) is rendered.

Game rules
----------
  • Any column visible in the *current* step can be caught by flying into its
    bounding volume (outside→inside entry event).
  • On catch: +1 point; the column disappears from **all** steps (its cells
    are stripped out of every step_pd via numpy rebuild — same technique as
    the pipeline's ``_build_lames_step_polydatas``).
  • Goal: collect all M columns.  Time limit: 2 minutes.

Leaderboard key:  points_desc  (more catches = better)

Usage (wired up from _attach_controls via 'G' key)
---------------------------------------------------
    from marvel_view.scripts.games.catch_flux import CatchFluxGame
    game = CatchFluxGame(plt, lames_state, _lames_start_timer, _lames_stop_timer)
    game.start()
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import vedo as _vedo
except ImportError:
    _vedo = None  # type: ignore[assignment]


class CatchFluxGame:
    """Lame-column-collecting mini-game.

    Parameters
    ----------
    plt:
        Running ``vedo.Plotter`` instance.
    lames_state:
        The ``lames_state`` dict from ``_attach_controls``; provides
        ``step``, ``n_steps``, ``step_pds`` and ``actor``.
    packed_pd:
        Raw packed ``vtkPolyData`` from the pipeline loader — must carry
        ``column_id`` and ``step_id`` cell-data arrays.
    start_lames / stop_lames:
        Callables that start / stop the lames animation.
    data_id:
        Short dataset identifier for the leaderboard entry.
    """

    DURATION_S        = 120.0
    TICK_MS           = 50
    CATCH_MARGIN_FRAC = 0.02

    def __init__(
        self,
        plt,
        lames_state: dict,
        packed_pd,
        start_lames: Callable,
        stop_lames: Callable,
        data_id: str = "unknown",
    ) -> None:
        self._plt        = plt
        self._ls         = lames_state
        self._packed_pd  = packed_pd
        self._start_lames = start_lames
        self._stop_lames  = stop_lames
        self._data_id    = data_id

        self._running      = False
        self._score        = 0
        self._elapsed      = 0.0
        self._catch_margin = 5.0

        # Per-column data (built at start)
        self._n_columns:   int              = 0
        self._n_steps:     int              = 0
        self._col_bounds:  dict             = {}   # col_id → (xmin,xmax,ymin,ymax,zmin,zmax)
        self._col_in_step: dict             = {}   # col_id → set of step indices where it appears
        # Per-step: col_arr slice (col_id per cell in that step)
        self._step_col:    list             = []   # list[np.ndarray]
        self._caught:      set[int]         = set()
        self._dirty_steps: set[int]         = set()   # steps needing polydata rebuild
        self._original_pds: list           = []    # backup for restore

        # Entry detection (outside→inside per column)
        self._inside_cols: set[int] = set()   # columns currently "inside"
        self._last_seen_step: int   = -1

        self._timer_id:          Optional[int] = None
        self._obs_tag:           Optional[int] = None
        self._was_lames_visible: bool          = False

        self._hud_timer = None
        self._hud_score = None
        self._hud_flash = None

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise HUD, force lames ON, start the VTK repeating timer."""
        if self._running or _vedo is None:
            return
        if not self._ls.get("step_pds") or self._ls.get("n_steps", 0) <= 0:
            logger.warning("CatchFluxGame: no lames data — cannot start.")
            return
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            logger.warning("CatchFluxGame: no interactor — cannot start.")
            return

        self._score   = 0
        self._elapsed = 0.0
        self._caught  = set()
        self._n_steps = int(self._ls["n_steps"])
        self._inside_cols     = set()
        self._last_seen_step  = -1
        self._catch_margin    = self._compute_catch_margin()

        # Save originals for restore
        step_pds = self._ls["step_pds"]
        self._original_pds = list(step_pds)

        # Pre-compute per-column bounds + step membership
        try:
            self._build_column_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CatchFluxGame: column pre-processing failed: %s", exc)
            self._n_columns = 0

        if self._n_columns <= 0:
            logger.warning("CatchFluxGame: no columns found — cannot start.")
            return

        # Force lames animation ON
        self._was_lames_visible = bool(self._ls.get("visible", False))
        if not self._was_lames_visible:
            try:
                self._start_lames()
                self._ls["visible"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("CatchFluxGame: could not start lames: %s", exc)

        self._hud_timer = _vedo.Text2D(
            "⏱  2:00",
            pos="top-left", c="white", bg="black", alpha=0.75, s=1.1,
        )
        self._hud_score = _vedo.Text2D(
            f"Flux  0 / {self._n_columns}",
            pos="top-right", c="cyan", bg="black", alpha=0.75, s=1.1,
        )
        self._plt.add(self._hud_timer)
        self._plt.add(self._hud_score)

        self._running = True
        self._obs_tag  = iren.AddObserver("TimerEvent", self._on_timer, 2.0)
        self._timer_id = iren.CreateRepeatingTimer(self.TICK_MS)
        self._plt.render()

        logger.info(
            "CatchFluxGame started: n_columns=%d  n_steps=%d  "
            "catch_margin=%.1f  duration=%ds",
            self._n_columns, self._n_steps,
            self._catch_margin, int(self.DURATION_S),
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_catch_margin(self) -> float:
        """Return CATCH_MARGIN_FRAC × scene diagonal (min 2.0)."""
        try:
            b = self._plt.renderer.GetVisibleActorsBounds()
            diag = math.sqrt(
                (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
            )
            m = max(2.0, diag * self.CATCH_MARGIN_FRAC)
            logger.info("CatchFluxGame: scene diag=%.0f  catch_margin=%.1f", diag, m)
            return m
        except Exception:  # noqa: BLE001
            return 8.0

    def _build_column_data(self) -> None:
        """Pre-compute per-column bounding boxes and step-membership from packed_pd.

        Vectorised: uses argsort + ``np.minimum/maximum.reduceat`` instead of
        a per-column Python loop, reducing complexity from O(n_cols × n_tris)
        to O(n_tris log n_tris).
        """
        import time
        from vtkmodules.util.numpy_support import vtk_to_numpy

        t0 = time.perf_counter()
        packed = self._packed_pd

        col_arr  = vtk_to_numpy(
            packed.GetCellData().GetArray("column_id")
        ).astype(np.int32)
        step_arr = vtk_to_numpy(
            packed.GetCellData().GetArray("step_id")
        ).astype(np.int32)

        n_cols  = int(col_arr.max()) + 1
        n_steps = self._n_steps
        n_tris  = len(col_arr)
        logger.info(
            "CatchFluxGame._build_column_data: n_cols=%d  n_tris=%d  "
            "n_steps=%d  t=%.2fs",
            n_cols, n_tris, n_steps, time.perf_counter() - t0,
        )

        # ── Step slice boundaries ────────────────────────────────────────────
        lo_b = np.searchsorted(step_arr, np.arange(n_steps),       side="left")
        hi_b = np.searchsorted(step_arr, np.arange(1, n_steps + 1), side="left")
        step_col = [col_arr[lo_b[s]:hi_b[s]] for s in range(n_steps)]
        logger.info("CatchFluxGame: step slices built  t=%.2fs",
                    time.perf_counter() - t0)

        # ── Per-column: set of steps where it is present ─────────────────────
        col_in_step: dict[int, set] = {c: set() for c in range(n_cols)}
        for s in range(n_steps):
            for c in np.unique(step_col[s]):
                col_in_step[int(c)].add(s)
        logger.info("CatchFluxGame: col_in_step built  t=%.2fs",
                    time.perf_counter() - t0)

        # ── Per-column bounding boxes (vectorised) ───────────────────────────
        # Legacy packed format [3, v0, v1, v2] per triangle, or modern API.
        ca = packed.GetPolys()
        try:
            # VTK9+ modern API: connectivity is just [v0, v1, v2, …]
            conn = vtk_to_numpy(ca.GetConnectivityArray()).reshape(-1, 3)
        except Exception:  # noqa: BLE001
            # Legacy fallback: [3, v0, v1, v2, …]
            conn = vtk_to_numpy(ca.GetData()).reshape(-1, 4)[:, 1:]

        pts_np = vtk_to_numpy(packed.GetPoints().GetData())
        logger.info(
            "CatchFluxGame: geometry loaded  n_pts=%d  t=%.2fs",
            len(pts_np), time.perf_counter() - t0,
        )

        # Sort triangles by column_id → contiguous ranges per column.
        sort_idx  = np.argsort(col_arr, kind="stable")
        sc        = col_arr[sort_idx]
        counts    = np.bincount(col_arr, minlength=n_cols)
        non_empty = np.where(counts > 0)[0]
        ne_starts = np.searchsorted(sc, non_empty, side="left").astype(np.intp)

        col_mins = np.full((n_cols, 3),  np.inf, dtype=np.float32)
        col_maxs = np.full((n_cols, 3), -np.inf, dtype=np.float32)

        # Compute sorted connectivity once, then iterate over 3 vertices × 3 axes.
        conn_sorted = conn[sort_idx]        # (n_tris, 3)  — one allocation
        del sort_idx                        # free ~300 MB

        for vi in range(3):
            sv = conn_sorted[:, vi]         # vertex indices for corner vi, sorted by col
            for ax in range(3):
                coord = np.asarray(pts_np[sv, ax], dtype=np.float32)
                r_min = np.minimum.reduceat(coord, ne_starts)
                r_max = np.maximum.reduceat(coord, ne_starts)
                # Use direct assignment — fancy indexing returns a copy,
                # so out=col_mins[fancy] would write to a discarded copy.
                col_mins[non_empty, ax] = np.minimum(
                    col_mins[non_empty, ax], r_min)
                col_maxs[non_empty, ax] = np.maximum(
                    col_maxs[non_empty, ax], r_max)

        del conn_sorted                     # free geometry temp arrays
        logger.info("CatchFluxGame: bounds computed  t=%.2fs",
                    time.perf_counter() - t0)

        col_bounds: dict[int, Optional[tuple]] = {}
        for c in range(n_cols):
            if counts[c] == 0:
                col_bounds[c] = None
            else:
                col_bounds[c] = (
                    float(col_mins[c, 0]), float(col_maxs[c, 0]),
                    float(col_mins[c, 1]), float(col_maxs[c, 1]),
                    float(col_mins[c, 2]), float(col_maxs[c, 2]),
                )

        # Store everything
        self._n_columns   = n_cols
        self._col_arr     = col_arr
        self._step_arr    = step_arr
        self._lo_b        = lo_b
        self._hi_b        = hi_b
        self._step_col    = step_col
        self._col_in_step = col_in_step
        self._col_bounds  = col_bounds
        logger.info(
            "CatchFluxGame: %d columns indexed across %d steps  total=%.2fs",
            n_cols, n_steps, time.perf_counter() - t0,
        )

    def _vanish_column(self, col_id: int) -> None:
        """Queue all of col_id's steps for lazy rebuild; process current step now."""
        dirty = self._col_in_step.get(col_id, set())
        self._dirty_steps.update(dirty)
        # Immediately vanish the currently displayed step (no visible lag)
        current_step = int(self._ls.get("step", 0))
        if current_step in dirty:
            self._vanish_step(current_step)
            self._dirty_steps.discard(current_step)

    def _vanish_step(self, s: int) -> None:
        """Rebuild step s's polydata removing ALL currently caught columns."""
        import vtk
        from vtk.util import numpy_support as nps

        try:
            pds     = self._ls["step_pds"]
            a, b    = int(self._lo_b[s]), int(self._hi_b[s])
            if a >= b:
                return
            step_cols = self._col_arr[a:b]
            caught_arr = np.array(list(self._caught), dtype=np.int32)
            keep = np.where(~np.isin(step_cols, caught_arr))[0]
            if len(keep) == len(step_cols):
                return  # nothing to remove
            pts_vtk = self._packed_pd.GetPoints()
            rgb_np  = nps.vtk_to_numpy(
                self._packed_pd.GetCellData().GetArray("rgb")
            )
            if len(keep) == 0:
                import vtk as _vtk
                pds[s] = _vtk.vtkPolyData()
                if int(self._ls.get("step", -1)) == s:
                    actor = self._ls.get("actor")
                    if actor is not None:
                        actor.GetMapper().SetInputData(pds[s])
                return
            global_keep = a + keep
            conn_np = nps.vtk_to_numpy(
                self._packed_pd.GetPolys().GetData()
            ).reshape(-1, 4)
            sub_conn = np.ascontiguousarray(conn_np[global_keep].ravel(), dtype=np.int64)
            sub_rgb  = np.ascontiguousarray(rgb_np[global_keep])
            cells = vtk.vtkCellArray()
            cells.SetCells(len(keep), nps.numpy_to_vtkIdTypeArray(sub_conn, deep=True))
            new_pd = vtk.vtkPolyData()
            new_pd.SetPoints(pts_vtk)
            new_pd.SetPolys(cells)
            a_rgb = nps.numpy_to_vtk(sub_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            a_rgb.SetName("rgb")
            a_rgb.SetNumberOfComponents(3)
            new_pd.GetCellData().AddArray(a_rgb)
            pds[s] = new_pd
            if int(self._ls.get("step", -1)) == s:
                actor = self._ls.get("actor")
                if actor is not None:
                    actor.GetMapper().SetInputData(new_pd)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CatchFluxGame._vanish_step(%d): %s", s, exc)

    def _process_dirty(self, n: int = 30) -> None:
        """Process up to n queued dirty steps (lazy vanish background work)."""
        for _ in range(n):
            if not self._dirty_steps:
                return
            self._vanish_step(self._dirty_steps.pop())

    # ── VTK timer callback ────────────────────────────────────────────────────

    def _on_timer(self, obj, _event) -> None:
        if not self._running:
            return
        try:
            if obj.GetTimerId() != self._timer_id:
                return
        except Exception:  # noqa: BLE001
            pass

        dt = self.TICK_MS / 1000.0
        self._elapsed += dt
        remaining = max(0.0, self.DURATION_S - self._elapsed)

        # Countdown HUD
        mins = int(remaining) // 60
        secs = int(remaining) % 60
        self._hud_timer.text(f"⏱  {mins}:{secs:02d}")

        current_step = int(self._ls.get("step", 0))

        # When step changes: update inside-tracking
        if current_step != self._last_seen_step:
            if 0 <= current_step < len(self._step_col):
                visible_now = set(int(c) for c in self._step_col[current_step])
            else:
                visible_now = set()

            self._inside_cols &= visible_now
            self._last_seen_step = current_step

        # Entry-event check per visible uncaught column
        step_caught = False
        if 0 <= current_step < len(self._step_col):
            visible_cols = (
                set(int(c) for c in self._step_col[current_step]) - self._caught
            )
            if visible_cols:
                try:
                    cx, cy, cz = self._plt.camera.GetPosition()
                    mg = self._catch_margin
                    for col_id in visible_cols:
                        bnd = self._col_bounds.get(col_id)
                        if bnd is None:
                            continue
                        inside = (
                            bnd[0] - mg <= cx <= bnd[1] + mg
                            and bnd[2] - mg <= cy <= bnd[3] + mg
                            and bnd[4] - mg <= cz <= bnd[5] + mg
                        )
                        if inside and col_id not in self._inside_cols:
                            # Entry event → catch
                            self._score += 1
                            self._caught.add(col_id)
                            self._inside_cols.discard(col_id)
                            self._vanish_column(col_id)
                            step_caught = True
                        elif inside:
                            self._inside_cols.add(col_id)
                        else:
                            self._inside_cols.discard(col_id)
                except Exception:  # noqa: BLE001
                    pass

        # Background lazy vanish: process a few dirty steps per tick
        self._process_dirty(30)

        if step_caught:
            left = self._n_columns - len(self._caught)
            suffix = f"  ({left} left)" if left > 0 else "  ALL!"
            self._hud_score.text(f"Flux  {self._score} / {self._n_columns}{suffix}")
            self._show_flash()

        self._plt.render()

        if len(self._caught) >= self._n_columns:
            self._end(victory=True)
            return
        if remaining <= 0.0:
            self._end(victory=False)

    def _show_flash(self) -> None:
        """Display a brief '+1' indicator that auto-removes after 400ms."""
        if _vedo is None:
            return
        try:
            if self._hud_flash is not None:
                try:
                    self._plt.remove(self._hud_flash)
                except Exception:  # noqa: BLE001
                    pass
            self._hud_flash = _vedo.Text2D(
                "+1",
                pos="top-right", c="lime", bg=None, alpha=0.9, s=1.4,
            )
            self._plt.add(self._hud_flash)
            iren = getattr(self._plt, "interactor", None)
            if iren is not None:
                import time as _time
                flash_deadline = _time.monotonic() + 0.4
                _remove_flash_tag = [None]

                def _remove_flash(obj, _ev):
                    if _time.monotonic() < flash_deadline:
                        return
                    try:
                        self._plt.remove(self._hud_flash)
                        self._hud_flash = None
                    except Exception:  # noqa: BLE001
                        pass
                    if _remove_flash_tag[0] is not None:
                        try:
                            iren.RemoveObserver(_remove_flash_tag[0])
                        except Exception:  # noqa: BLE001
                            pass
                        _remove_flash_tag[0] = None

                _remove_flash_tag[0] = iren.AddObserver("TimerEvent", _remove_flash)
        except Exception:  # noqa: BLE001
            pass

    # ── end of game ───────────────────────────────────────────────────────────

    def _end(self, *, victory: bool = False) -> None:
        if not self._running:
            return
        self._running = False

        iren = getattr(self._plt, "interactor", None)
        if iren is not None:
            try:
                iren.DestroyTimer(self._timer_id)
            except Exception:  # noqa: BLE001
                pass
            if self._obs_tag is not None:
                try:
                    iren.RemoveObserver(self._obs_tag)
                except Exception:  # noqa: BLE001
                    pass
        self._timer_id = None
        self._obs_tag  = None

        # Restore original PolyData so uncaught columns reappear
        self._restore_pds()

        # Restore lames animation state
        if not self._was_lames_visible:
            try:
                self._stop_lames()
                self._ls["visible"] = False
            except Exception as exc:  # noqa: BLE001
                logger.debug("CatchFluxGame: could not stop lames: %s", exc)

        self._hud_timer.text("⏱  DONE" if victory else "⏱  0:00")
        self._hud_score.text(f"Flux  {self._score} / {self._n_columns}")

        result_msg = (
            "Bravo ! Tous les flux attrapés !"
            if victory
            else f"Temps écoulé — {self._score} / {self._n_columns} flux attrapés"
        )
        result_hud = _vedo.Text2D(
            result_msg,
            pos="bottom-center", c="cyan", bg="black", alpha=0.8, s=1.05,
        )
        self._plt.add(result_hud)
        self._plt.render()

        logger.info(
            "CatchFluxGame ended: score=%d/%d  victory=%s  elapsed=%.1fs",
            self._score, self._n_columns, victory, self._elapsed,
        )

        try:
            from marvel_view.leaderboard import ScoreEntry, fetch_scores, submit_score
            from marvel_view.leaderboard.ui import LeaderboardPanel, NameEntry

            def on_name(name: str) -> None:
                entry = ScoreEntry(
                    game_id="catch_flux",
                    player_name=name,
                    score=float(self._score),
                    data_id=self._data_id,
                    metadata={
                        "n_columns": self._n_columns,
                        "n_steps":   self._n_steps,
                        "victory":   victory,
                        "elapsed_s": round(self._elapsed, 1),
                    },
                )
                submit_score(entry)
                try:
                    self._plt.remove(result_hud)
                except Exception:  # noqa: BLE001
                    pass
                scores = fetch_scores("catch_flux")
                LeaderboardPanel(self._plt, "catch_flux", scores).show()
                self._cleanup()

            NameEntry(
                self._plt,
                prompt="Ton prénom pour le leaderboard ?",
                on_done=on_name,
            ).attach()

        except Exception as exc:  # noqa: BLE001
            logger.warning("CatchFluxGame: leaderboard failed: %s", exc)
            self._cleanup()

    def _restore_pds(self) -> None:
        """Restore original PolyData references so uncaught columns reappear."""
        try:
            pds = self._ls.get("step_pds")
            if pds and self._original_pds:
                for i, orig in enumerate(self._original_pds):
                    if i < len(pds):
                        pds[i] = orig
                step  = int(self._ls.get("step", 0))
                actor = self._ls.get("actor")
                if actor is not None and 0 <= step < len(pds) and pds[step] is not None:
                    actor.GetMapper().SetInputData(pds[step])
        except Exception as exc:  # noqa: BLE001
            logger.debug("CatchFluxGame._restore_pds: %s", exc)

    def _cleanup(self) -> None:
        for hud in (self._hud_timer, self._hud_score, self._hud_flash):
            try:
                if hud is not None:
                    self._plt.remove(hud)
            except Exception:  # noqa: BLE001
                pass
        self._hud_timer = None
        self._hud_score = None
        self._hud_flash = None
        try:
            self._plt.render()
        except Exception:  # noqa: BLE001
            pass
