"""Catch the Pillars — 2-minute pillar-collecting mini-game.

Each connected component of the pillars mesh becomes an independent
catch-target.  Fly within the catch radius to collect it; the pillar
disappears.  Timer runs 2 minutes; score = number of pillars caught.
Leaderboard: points_desc (more = better).

Usage (wired up automatically in _attach_controls via 'G' key)
--------------------------------------------------------------
    from marvel_view.scripts.games.catch_pillars import CatchPillarsGame
    game = CatchPillarsGame(plt, pillars_mesh)
    game.start()
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import vedo as _vedo
except ImportError:
    _vedo = None  # type: ignore[assignment]


class CatchPillarsGame:
    """2-minute game: collect pillar actors by flying close to them.

    Parameters
    ----------
    plt:
        Running ``vedo.Plotter`` instance.
    pillars_mesh:
        The merged pillars vedo Mesh passed from ``_attach_controls``.
        Hidden during play and replaced by individual per-component actors.
    data_id:
        Short name of the dataset (used for the leaderboard entry).
    """

    DURATION_S         = 120.0  # two minutes
    TICK_MS            = 100    # game-loop cadence (10 fps — plenty for proximity)
    CATCH_MARGIN_FRAC  = 0.015  # margin added around each pillar bounding box
    MIN_POINTS         = 30     # discard tiny mesh fragments

    def __init__(
        self, plt, pillars_mesh, data_id: str = "unknown"
    ) -> None:
        self._plt          = plt
        self._pillars_mesh = pillars_mesh
        self._data_id      = data_id

        self._running       = False
        self._actors:  list = []    # individual vedo Mesh catch-targets
        self._bounds:  list = []    # [xmin,xmax,ymin,ymax,zmin,zmax] per actor
        self._caught:  list = []    # bool per actor
        self._score         = 0
        self._elapsed       = 0.0
        self._catch_radius  = 20.0

        self._timer_id:     Optional[int] = None
        self._obs_tag:      Optional[int] = None
        self._original_vis: int           = 1

        self._hud_timer = None
        self._hud_score = None
        self._result_hud = None

    # ── public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Split pillars, build HUD, start the VTK repeating timer."""
        if self._running or _vedo is None:
            return
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            logger.warning("CatchPillarsGame: no interactor — cannot start.")
            return

        # Split the merged pillars mesh into individual CC actors
        self._actors, self._bounds = self._split_pillars()
        if not self._actors:
            logger.warning("CatchPillarsGame: no usable pillar components found.")
            return

        self._caught  = [False] * len(self._actors)
        self._score   = 0
        self._elapsed = 0.0

        # Hide the original merged mesh to avoid visual overlap
        try:
            self._original_vis = self._pillars_mesh.GetVisibility()
            self._pillars_mesh.SetVisibility(0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not hide original pillars mesh: %s", exc)
            self._original_vis = 1

        self._catch_margin = self._compute_catch_margin()

        # HUD actors
        n = len(self._actors)
        self._hud_timer = _vedo.Text2D(
            "⏱  2:00",
            pos="top-left", c="white", bg="black", alpha=0.75, s=1.1,
        )
        self._hud_score = _vedo.Text2D(
            f"Score  0 / {n}",
            pos="top-right", c="yellow", bg="black", alpha=0.75, s=1.1,
        )
        self._plt.add(self._hud_timer)
        self._plt.add(self._hud_score)

        self._running = True

        # Register a dedicated VTK timer (separate from the master scheduler)
        self._obs_tag  = iren.AddObserver("TimerEvent", self._on_timer, 2.0)
        self._timer_id = iren.CreateRepeatingTimer(self.TICK_MS)

        self._plt.render()
        logger.info(
            "CatchPillarsGame started: %d pillars, catch_margin=%.1f, duration=%ds",
            n, self._catch_margin, int(self.DURATION_S),
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _split_pillars(self) -> tuple[list, list]:
        """Split pillars_mesh into per-CC actors.  Returns (actors, bounds)."""
        try:
            parts = self._pillars_mesh.split()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pillars_mesh.split() failed (%s) — treating as single target", exc
            )
            parts = [self._pillars_mesh]

        actors: list = []
        bounds: list = []
        color = (0.20, 0.95, 0.75)

        for m in parts:
            try:
                n_pts = m.npoints
            except Exception:  # noqa: BLE001
                n_pts = 0
            if n_pts < self.MIN_POINTS:
                continue
            try:
                m.c(color).alpha(0.18).lighting("off")
                m.name = "catch_target"
                self._plt.add(m)
                actors.append(m)
                bounds.append(list(m.bounds()))  # [xmin,xmax,ymin,ymax,zmin,zmax]
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not add pillar component: %s", exc)

        logger.info(
            "CatchPillarsGame: %d / %d components usable (>= %d pts)",
            len(actors), len(parts), self.MIN_POINTS,
        )
        return actors, bounds

    def _compute_catch_margin(self) -> float:
        """Return CATCH_MARGIN_FRAC × scene diagonal (min 1.0).

        This margin is added on all six faces of each pillar bounding box,
        so the player just needs to fly *through* the pillar rather than
        aim at its centroid.
        """
        try:
            b = self._plt.renderer.GetVisibleActorsBounds()
            diag = math.sqrt(
                (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
            )
            m = max(1.0, diag * self.CATCH_MARGIN_FRAC)
            logger.info(
                "CatchPillarsGame: scene diag=%.0f  catch_margin=%.1f", diag, m
            )
            return m
        except Exception:  # noqa: BLE001
            return 5.0

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

        # Update countdown HUD
        mins = int(remaining) // 60
        secs = int(remaining) % 60
        self._hud_timer.text(f"⏱  {mins}:{secs:02d}")

        # Proximity check — camera position vs each uncaught pillar centroid
        try:
            cam_pos = self._plt.camera.GetPosition()  # (x, y, z) tuple
        except Exception:  # noqa: BLE001
            cam_pos = None

        changed = False
        if cam_pos is not None:
            mg = self._catch_margin
            for i, (b, caught) in enumerate(zip(self._bounds, self._caught)):
                if caught:
                    continue
                # Catch condition: camera inside expanded bounding box.
                # Works naturally for elongated vertical pillars — the player
                # just needs to fly through the pillar, not aim at its centroid.
                if (
                    b[0] - mg <= cam_pos[0] <= b[1] + mg
                    and b[2] - mg <= cam_pos[1] <= b[3] + mg
                    and b[4] - mg <= cam_pos[2] <= b[5] + mg
                ):
                    self._caught[i] = True
                    self._score += 1
                    try:
                        self._actors[i].SetVisibility(0)
                    except Exception:  # noqa: BLE001
                        pass
                    changed = True

        if changed:
            self._hud_score.text(f"Score  {self._score} / {len(self._actors)}")

        self._plt.render()

        # End: time up OR all pillars caught
        if remaining <= 0.0 or self._score == len(self._actors):
            self._end()

    # ── end of game ───────────────────────────────────────────────────────────

    def _end(self) -> None:
        if not self._running:
            return
        self._running = False

        # Stop the game timer
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

        # Freeze HUD at final values
        remaining = max(0.0, self.DURATION_S - self._elapsed)
        self._hud_timer.text(
            f"⏱  {int(remaining)//60}:{int(remaining)%60:02d}"
        )
        self._hud_score.text(f"Score  {self._score} / {len(self._actors)}")

        all_caught = self._score == len(self._actors)
        msg = (
            "Tous les pillars attrapés !"
            if all_caught
            else f"Temps écoulé !  Score final : {self._score} / {len(self._actors)}"
        )
        self._result_hud = _vedo.Text2D(
            msg, pos="bottom-center", c="yellow", bg="black", alpha=0.8, s=1.05,
        )
        self._plt.add(self._result_hud)
        self._plt.render()

        logger.info(
            "CatchPillarsGame ended: score=%d/%d  elapsed=%.1fs  all=%s",
            self._score, len(self._actors), self._elapsed, all_caught,
        )

        # Name entry → leaderboard submission
        try:
            from marvel_view.leaderboard import ScoreEntry, fetch_scores, submit_score
            from marvel_view.leaderboard.ui import LeaderboardPanel, NameEntry

            def on_name(name: str) -> None:
                entry = ScoreEntry(
                    game_id="catch_pillars",
                    player_name=name,
                    score=float(self._score),
                    data_id=self._data_id,
                    metadata={
                        "total_pillars": len(self._actors),
                        "elapsed_s":     round(self._elapsed, 1),
                    },
                )
                submitted = submit_score(entry)
                logger.info(
                    "CatchPillarsGame: leaderboard submit=%s  player=%r  score=%d",
                    submitted, name, self._score,
                )
                try:
                    self._plt.remove(self._result_hud)
                except Exception:  # noqa: BLE001
                    pass
                scores = fetch_scores("catch_pillars")
                LeaderboardPanel(self._plt, "catch_pillars", scores).show()
                self._cleanup()

            NameEntry(
                self._plt,
                prompt="Ton prénom pour le leaderboard ?",
                on_done=on_name,
            ).attach()

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CatchPillarsGame: leaderboard integration failed: %s", exc
            )
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove all game actors and restore original pillars visibility."""
        for a in self._actors:
            try:
                self._plt.remove(a)
            except Exception:  # noqa: BLE001
                pass
        self._actors  = []
        self._bounds  = []
        self._caught  = []

        for hud in (self._hud_timer, self._hud_score):
            try:
                if hud is not None:
                    self._plt.remove(hud)
            except Exception:  # noqa: BLE001
                pass
        self._hud_timer = None
        self._hud_score = None

        try:
            self._pillars_mesh.SetVisibility(self._original_vis)
        except Exception:  # noqa: BLE001
            pass

        try:
            self._plt.render()
        except Exception:  # noqa: BLE001
            pass
