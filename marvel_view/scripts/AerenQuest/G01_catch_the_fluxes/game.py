"""G01 – Catch the Fluxes game logic.

Refactored from ``marvel_view.scripts.games.catch_flux`` to integrate with
the AerenQuest launcher:

  • HUD timer and score actors are **created by the launcher** and passed in,
    so they can be styled as large game-mode elements.
  • The game calls ``on_end(score, victory)`` when it ends instead of
    directly showing NameEntry / LeaderboardPanel (the launcher handles that).

Public API
----------
    game = CatchFluxGame(
        plt, lames_state, packed_pd,
        start_lames, stop_lames,
        hud_timer=<vedo.Text2D>,
        hud_score=<vedo.Text2D>,
        on_end=callable(score, victory),
        data_id="dataset_name",
    )
    game.start()

Game rules
----------
  • Fly into each visible flux column's bounding box → +1 point.
  • Column disappears from all steps.
  • Goal: catch all M columns.  Time limit: 2 minutes.
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
    """Lame-column-catching game for AerenQuest G01.

    Parameters
    ----------
    plt:
        Running ``vedo.Plotter``.
    lames_state:
        The ``lames_state`` dict from ``_attach_controls``.
    packed_pd:
        Raw packed vtkPolyData with ``column_id`` and ``step_id`` cell arrays.
    start_lames / stop_lames:
        Start / stop the lames animation.
    hud_timer:
        External large Text2D actor for the timer (updated each tick).
        If ``None`` a small built-in actor is created.
    hud_score:
        External large Text2D actor for the score.
    on_end:
        Callback ``on_end(score: int, victory: bool)`` called when done.
    data_id:
        Dataset name for the leaderboard entry.
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
        hud_timer=None,
        hud_score=None,
        on_end: Optional[Callable] = None,
        data_id: str = "unknown",
    ) -> None:
        self._plt         = plt
        self._ls          = lames_state
        self._packed_pd   = packed_pd
        self._start_lames = start_lames
        self._stop_lames  = stop_lames
        self._hud_timer   = hud_timer      # Text2D or None (ext. managed)
        self._hud_score   = hud_score      # Text2D or None (ext. managed)
        self._on_end      = on_end
        self._data_id     = data_id

        self._owns_hud    = False          # True if we created the HUD actors
        self._running     = False
        self._score       = 0
        self._elapsed     = 0.0
        self._catch_margin = 5.0

        self._n_columns:    int  = 0
        self._n_steps:      int  = 0
        self._col_bounds:   dict = {}
        self._step_col:     list = []
        self._caught:       set  = set()
        self._rebuilt_steps: set = set()   # steps whose polydata is clean for current _caught
        self._original_pds: list = []

        self._inside_cols:    set = set()
        self._last_seen_step: int = -1

        self._timer_id:  Optional[int] = None
        self._obs_tag:   Optional[int] = None
        self._was_lames_visible: bool  = False

        self._hud_flash = None
        self._internal_hud_timer = None  # only set when _owns_hud is True
        self._internal_hud_score = None

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise game state, HUD, lames animation, and the VTK timer."""
        if self._running or _vedo is None:
            return
        if not self._ls.get("step_pds") or self._ls.get("n_steps", 0) <= 0:
            logger.warning("CatchFluxGame: no lames data — cannot start.")
            return
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            logger.warning("CatchFluxGame: no interactor — cannot start.")
            return

        self._score    = 0
        self._elapsed  = 0.0
        self._caught   = set()
        self._n_steps  = int(self._ls["n_steps"])
        self._inside_cols    = set()
        self._last_seen_step = -1
        self._rebuilt_steps  = set()
        self._catch_margin   = self._compute_catch_margin()
        self._original_pds   = list(self._ls["step_pds"])

        if self._n_columns <= 0:
            print("[G01] start(): precompute not done yet — running _build_column_data now (may block!)")
            try:
                self._build_column_data()
            except Exception as exc:  # noqa: BLE001
                logger.warning("CatchFluxGame: column pre-processing failed: %s", exc)
                self._n_columns = 0
        else:
            print(f"[G01] start(): precomputed data ready  n_columns={self._n_columns}")

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

        # If no external HUD provided, create small built-in ones
        if self._hud_timer is None:
            self._internal_hud_timer = _vedo.Text2D(
                "⏱  2:00",
                pos=(0.03, 0.95), s=1.1, c="white", bg="black", alpha=0.75,
            )
            self._plt.add(self._internal_hud_timer)
            self._hud_timer = self._internal_hud_timer
            self._owns_hud  = True

        if self._hud_score is None:
            self._internal_hud_score = _vedo.Text2D(
                f"Flux  0 / {self._n_columns}",
                pos=(0.97, 0.95), s=1.1, c="cyan", bg="black", alpha=0.75,
                justify="top-right",
            )
            self._plt.add(self._internal_hud_score)
            self._hud_score = self._internal_hud_score
            self._owns_hud  = True

        # Update score HUD with actual column count now that we know it
        try:
            self._hud_score.text(f"Flux  0 / {self._n_columns}")
        except Exception:
            pass

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

    def stop(self) -> None:
        """Forcibly stop the game (called by launcher on restart/quit)."""
        if not self._running:
            return
        self._end(victory=False, _skip_callback=True)

    def precompute(self) -> None:
        """Pre-build column index and bounding boxes.

        Call this during the countdown (before ``start()``) so the heavy numpy
        work happens while the player is reading "3 / 2 / 1", not at game start.
        ``start()`` will skip ``_build_column_data`` if this was called first.
        """
        if self._n_columns > 0:
            print("[G01] precompute() skipped — already done")
            return
        import time as _time
        print("[G01] precompute() START")
        t0 = _time.perf_counter()
        try:
            self._build_column_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CatchFluxGame.precompute failed: %s", exc)
            print(f"[G01] precompute() FAILED: {exc}")
        print(f"[G01] precompute() DONE  elapsed={_time.perf_counter()-t0:.3f}s  n_columns={self._n_columns}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_catch_margin(self) -> float:
        try:
            b = self._plt.renderer.GetVisibleActorsBounds()
            diag = math.sqrt(
                (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
            )
            return max(2.0, diag * self.CATCH_MARGIN_FRAC)
        except Exception:  # noqa: BLE001
            return 8.0

    def _build_column_data(self) -> None:
        """Pre-compute per-column bounding boxes using the flux mid-step (fast)."""
        import time
        from vtkmodules.util.numpy_support import vtk_to_numpy as _v2n

        t0 = time.perf_counter()
        pd = self._packed_pd
        print(f"[G01] _build_column_data: reading cell arrays  t={time.perf_counter()-t0:.3f}s")
        col_arr  = _v2n(pd.GetCellData().GetArray("column_id")).astype(np.int32)
        step_arr = _v2n(pd.GetCellData().GetArray("step_id")).astype(np.int32)
        n_cells  = len(col_arr)
        n_cols   = int(col_arr.max()) + 1 if n_cells > 0 else 0
        n_steps  = int(step_arr.max()) + 1 if n_cells > 0 else 0
        pts_vtk  = pd.GetPoints()
        print(f"[G01] _build_column_data: n_cells={n_cells}  n_cols={n_cols}  n_steps={n_steps}  t={time.perf_counter()-t0:.3f}s")

        # Per-step slice boundaries (cells pre-sorted by step_id)
        print(f"[G01] _build_column_data: argsort by step  t={time.perf_counter()-t0:.3f}s")
        order      = np.argsort(step_arr, kind="stable")
        col_arr_s  = col_arr[order]
        step_arr_s = step_arr[order]
        boundaries = np.searchsorted(step_arr_s, np.arange(n_steps + 1))
        lo_b = boundaries[:-1]
        hi_b = boundaries[1:]

        # Per-step column list — simple slice references, no Python set insertions
        print(f"[G01] _build_column_data: building step_col list  t={time.perf_counter()-t0:.3f}s")
        step_col: list = [col_arr_s[int(lo_b[s]):int(hi_b[s])] for s in range(n_steps)]

        self._lo_b     = lo_b
        self._hi_b     = hi_b
        self._col_arr  = col_arr
        self._step_arr = step_arr

        # Point coordinates
        print(f"[G01] _build_column_data: reading points  t={time.perf_counter()-t0:.3f}s")
        try:
            pts_np = _v2n(pts_vtk.GetData()).reshape(-1, 3)
        except Exception:
            pts_np = np.zeros((pts_vtk.GetNumberOfPoints(), 3), dtype=np.float32)
            for i in range(pts_vtk.GetNumberOfPoints()):
                pts_np[i] = pts_vtk.GetPoint(i)
        print(f"[G01] _build_column_data: pts_np shape={pts_np.shape}  t={time.perf_counter()-t0:.3f}s")

        # Connectivity (triangles → 3 vertex indices per cell)
        print(f"[G01] _build_column_data: reading connectivity  t={time.perf_counter()-t0:.3f}s")
        conn_flat = _v2n(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
        print(f"[G01] _build_column_data: conn_flat shape={conn_flat.shape}  t={time.perf_counter()-t0:.3f}s")

        # For each column: find the mid-step of its lifecycle (flux fully grown).
        print(f"[G01] _build_column_data: computing mid-steps  t={time.perf_counter()-t0:.3f}s")
        col_min_step = np.full(n_cols, n_steps, dtype=np.int32)
        col_max_step = np.zeros(n_cols, dtype=np.int32)
        np.minimum.at(col_min_step, col_arr, step_arr)
        np.maximum.at(col_max_step, col_arr, step_arr)
        col_mid_step = ((col_min_step.astype(np.int64) + col_max_step.astype(np.int64)) // 2
                        ).astype(np.int32)

        # Select only cells at each column's mid-step
        print(f"[G01] _build_column_data: selecting mid-step cells  t={time.perf_counter()-t0:.3f}s")
        is_mid         = (step_arr == col_mid_step[col_arr])
        mid_indices    = np.where(is_mid)[0]
        mid_col        = col_arr[mid_indices]
        sort_mid       = np.argsort(mid_col, kind="stable")
        mid_idx_sorted = mid_indices[sort_mid]
        mid_col_sorted = mid_col[sort_mid]
        bounds_mid     = np.searchsorted(mid_col_sorted, np.arange(n_cols + 1))
        print(f"[G01] _build_column_data: n_mid_cells={len(mid_indices)}  t={time.perf_counter()-t0:.3f}s")

        # Gather vertex positions for the mid cells (shape: n_mid_cells*3 × 3)
        print(f"[G01] _build_column_data: gathering mid vertex positions  t={time.perf_counter()-t0:.3f}s")
        mid_pts = pts_np[conn_flat[mid_idx_sorted].ravel()].reshape(-1, 3)

        print(f"[G01] _build_column_data: computing per-column bboxes  t={time.perf_counter()-t0:.3f}s")
        col_bounds: dict = {}
        for c in range(n_cols):
            a, b_c = int(bounds_mid[c]), int(bounds_mid[c + 1])
            if a >= b_c:
                col_bounds[c] = None
                continue
            pts_c = mid_pts[a * 3: b_c * 3]
            col_bounds[c] = (
                float(pts_c[:, 0].min()), float(pts_c[:, 0].max()),
                float(pts_c[:, 1].min()), float(pts_c[:, 1].max()),
                float(pts_c[:, 2].min()), float(pts_c[:, 2].max()),
            )

        self._n_columns  = n_cols
        self._col_bounds = col_bounds
        self._step_col   = step_col
        elapsed = time.perf_counter() - t0
        print(f"[G01] _build_column_data: DONE  n_cols={n_cols}  n_steps={n_steps}  elapsed={elapsed:.3f}s")
        logger.info(
            "CatchFluxGame: %d columns indexed across %d steps  total=%.2fs",
            n_cols, n_steps, elapsed,
        )

    # ── VTK timer callback ────────────────────────────────────────────────────

    def _vanish_step(self, s: int) -> None:
        import time
        import vtk
        from vtk.util import numpy_support as nps
        t0 = time.perf_counter()
        print(f"[G01] _vanish_step({s}) START  caught={sorted(self._caught)}")
        try:
            pds   = self._ls["step_pds"]
            a, b  = int(self._lo_b[s]), int(self._hi_b[s])
            if a >= b:
                print(f"[G01] _vanish_step({s}) empty slice, skip  t={time.perf_counter()-t0:.4f}s")
                return
            step_cols  = self._col_arr[a:b]
            caught_arr = np.array(list(self._caught), dtype=np.int32)
            print(f"[G01] _vanish_step({s}): slice [{a}:{b}] n_cells={len(step_cols)}  t={time.perf_counter()-t0:.4f}s")
            keep = np.where(~np.isin(step_cols, caught_arr))[0]
            if len(keep) == len(step_cols):
                print(f"[G01] _vanish_step({s}): nothing to remove  t={time.perf_counter()-t0:.4f}s")
                return
            print(f"[G01] _vanish_step({s}): keeping {len(keep)}/{len(step_cols)} cells  t={time.perf_counter()-t0:.4f}s")
            pts_vtk = self._packed_pd.GetPoints()
            rgb_np  = nps.vtk_to_numpy(
                self._packed_pd.GetCellData().GetArray("rgb")
            )
            print(f"[G01] _vanish_step({s}): got rgb array len={len(rgb_np)}  t={time.perf_counter()-t0:.4f}s")
            if len(keep) == 0:
                pds[s] = vtk.vtkPolyData()
                if int(self._ls.get("step", -1)) == s:
                    actor = self._ls.get("actor")
                    if actor is not None:
                        actor.GetMapper().SetInputData(pds[s])
                print(f"[G01] _vanish_step({s}): cleared (all removed)  t={time.perf_counter()-t0:.4f}s")
                return
            global_keep = a + keep
            print(f"[G01] _vanish_step({s}): rebuilding polydata  t={time.perf_counter()-t0:.4f}s")
            conn_np = nps.vtk_to_numpy(
                self._packed_pd.GetPolys().GetData()
            ).reshape(-1, 4)
            sub_conn = np.ascontiguousarray(conn_np[global_keep].ravel(), dtype=np.int64)
            sub_rgb  = np.ascontiguousarray(rgb_np[global_keep])
            cells = vtk.vtkCellArray()
            cells.SetCells(len(keep),
                           nps.numpy_to_vtkIdTypeArray(sub_conn, deep=True))
            new_pd = vtk.vtkPolyData()
            new_pd.SetPoints(pts_vtk)
            new_pd.SetPolys(cells)
            a_rgb = nps.numpy_to_vtk(sub_rgb, deep=True,
                                     array_type=vtk.VTK_UNSIGNED_CHAR)
            a_rgb.SetName("rgb")
            a_rgb.SetNumberOfComponents(3)
            new_pd.GetCellData().AddArray(a_rgb)
            pds[s] = new_pd
            print(f"[G01] _vanish_step({s}): new polydata built  t={time.perf_counter()-t0:.4f}s")
            if int(self._ls.get("step", -1)) == s:
                actor = self._ls.get("actor")
                if actor is not None:
                    actor.GetMapper().SetInputData(new_pd)
                    print(f"[G01] _vanish_step({s}): mapper updated  t={time.perf_counter()-t0:.4f}s")
        except Exception as exc:  # noqa: BLE001
            logger.debug("CatchFluxGame._vanish_step(%d): %s", s, exc)
            print(f"[G01] _vanish_step({s}) EXCEPTION: {exc}  t={time.perf_counter()-t0:.4f}s")
        print(f"[G01] _vanish_step({s}) END  t={time.perf_counter()-t0:.4f}s")

    def _process_dirty(self, n: int = 30) -> None:
        # Kept for backward compatibility — no longer used; vanishing is now lazy.
        pass

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

        mins = int(remaining) // 60
        secs = int(remaining) % 60
        try:
            self._hud_timer.text(f"⏱  {mins}:{secs:02d}")
        except Exception:
            pass

        current_step = int(self._ls.get("step", 0))

        if current_step != self._last_seen_step:
            if 0 <= current_step < len(self._step_col):
                visible_now = set(int(c) for c in self._step_col[current_step])
                # Lazy vanish: on step change, remove caught columns from this
                # step's polydata only if needed (one rebuild per step visit).
                if self._caught and current_step not in self._rebuilt_steps:
                    if visible_now & self._caught:
                        import time as _t; _tlv = _t.perf_counter()
                        print(f"[G01] lazy vanish step={current_step}  caught_in_step={visible_now & self._caught}")
                        self._vanish_step(current_step)
                        self._rebuilt_steps.add(current_step)
                        print(f"[G01] lazy vanish done  t={_t.perf_counter()-_tlv:.4f}s")
            else:
                visible_now = set()
            self._inside_cols &= visible_now
            self._last_seen_step = current_step

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
                            import time as _t
                            _tc = _t.perf_counter()
                            print(f"[G01] CATCH col_id={col_id}  step={current_step}  score={self._score+1}")
                            self._score += 1
                            self._caught.add(col_id)
                            self._inside_cols.discard(col_id)
                            # Rebuild current step immediately; invalidate all
                            # cached rebuilds so future step visits are clean.
                            print(f"[G01] CATCH calling _vanish_step({current_step})  t={_t.perf_counter()-_tc:.4f}s")
                            self._vanish_step(current_step)
                            print(f"[G01] CATCH _vanish_step done  t={_t.perf_counter()-_tc:.4f}s")
                            self._rebuilt_steps.clear()
                            self._rebuilt_steps.add(current_step)
                            step_caught = True
                            print(f"[G01] CATCH complete  total_catch_time={_t.perf_counter()-_tc:.4f}s")
                        elif inside:
                            self._inside_cols.add(col_id)
                        else:
                            self._inside_cols.discard(col_id)
                except Exception:  # noqa: BLE001
                    pass

        if step_caught:
            left   = self._n_columns - len(self._caught)
            suffix = f"  ({left} left)" if left > 0 else "  ALL!"
            try:
                self._hud_score.text(
                    f"Flux  {self._score} / {self._n_columns}{suffix}"
                )
            except Exception:
                pass
            self._show_flash()

        try:
            self._plt.render()
        except Exception:
            pass

        if len(self._caught) >= self._n_columns:
            self._end(victory=True)
            return
        if remaining <= 0.0:
            self._end(victory=False)

    def _show_flash(self) -> None:
        if _vedo is None:
            return
        try:
            if self._hud_flash is not None:
                try:
                    self._plt.remove(self._hud_flash)
                except Exception:
                    pass
            self._hud_flash = _vedo.Text2D(
                "+1 !", pos=(0.85, 0.87), c="lime", bg=None, alpha=0.9, s=2.0,
            )
            self._plt.add(self._hud_flash)
            iren = getattr(self._plt, "interactor", None)
            if iren is not None:
                import time as _time
                flash_deadline = _time.monotonic() + 0.4
                _tag = [None]

                def _remove(_obj, _ev):
                    if _time.monotonic() < flash_deadline:
                        return
                    try:
                        self._plt.remove(self._hud_flash)
                        self._hud_flash = None
                    except Exception:
                        pass
                    if _tag[0] is not None:
                        try:
                            iren.RemoveObserver(_tag[0])
                        except Exception:
                            pass
                        _tag[0] = None

                _tag[0] = iren.AddObserver("TimerEvent", _remove)
        except Exception:
            pass

    # ── end of game ───────────────────────────────────────────────────────────

    def _end(self, *, victory: bool = False, _skip_callback: bool = False) -> None:
        if not self._running:
            return
        self._running = False

        iren = getattr(self._plt, "interactor", None)
        if iren is not None:
            try:
                iren.DestroyTimer(self._timer_id)
            except Exception:
                pass
            if self._obs_tag is not None:
                try:
                    iren.RemoveObserver(self._obs_tag)
                except Exception:
                    pass
        self._timer_id = None
        self._obs_tag  = None

        self._restore_pds()

        if not self._was_lames_visible:
            try:
                self._stop_lames()
                self._ls["visible"] = False
            except Exception as exc:
                logger.debug("CatchFluxGame: could not stop lames: %s", exc)

        # Clean up any owned HUD actors
        if self._owns_hud:
            for act in (self._internal_hud_timer, self._internal_hud_score):
                try:
                    if act is not None:
                        self._plt.remove(act)
                except Exception:
                    pass
            self._internal_hud_timer = None
            self._internal_hud_score = None

        if self._hud_flash is not None:
            try:
                self._plt.remove(self._hud_flash)
            except Exception:
                pass
            self._hud_flash = None

        logger.info(
            "CatchFluxGame ended: score=%d/%d  victory=%s  elapsed=%.1fs",
            self._score, self._n_columns, victory, self._elapsed,
        )

        if not _skip_callback and self._on_end is not None:
            try:
                self._on_end(self._score, victory)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CatchFluxGame on_end callback failed: %s", exc)

    def _restore_pds(self) -> None:
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
