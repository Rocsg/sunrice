"""AerenQuest launcher — selector UI and game-mode orchestration.

Flow
----
1.  User clicks the "AerenQuest" button (added by app.py).
2.  3D navigation is frozen; a dark overlay + icon selector appears.
3.  ← / → navigate between games; the selected game is highlighted with a
    white rectangle border.  Enter launches it; Escape closes the selector.
4.  Camera is placed at XY-centre / Z-extremity of the mesh.
5.  Countdown  3 → 2 → 1  (large centred text).
6.  In-game HUD appears: large title at top, big timer left, big score right,
    plus the keyboard-controls image overlay.  All other UI is hidden.
7.  When the game ends ``on_end(score, victory)`` fires:
      a. Name-entry widget.
      b. Full black leaderboard screen (top-10 deduplicated by lower-cased name).
      c. "Restart (return)" or "Quit this game (escape)" at bottom.
8.  Enter → restart from step 4; Escape → restore scene and exit game mode.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import vedo as _vedo
except ImportError:
    _vedo = None  # type: ignore[assignment]


class AerenQuestLauncher:
    """Manages the AerenQuest game-selector overlay and game lifecycle.

    Parameters
    ----------
    plt:
        Running ``vedo.Plotter``.
    mesh:
        The main scene mesh (used to compute game-start camera position).
    game_context:
        Dict assembled by ``_attach_controls`` in ``app.py``:
          ``repo_root``            Path to the sunrice repo root.
          ``lames_state``          The lames_state dict.
          ``lames_start_timer``    Callable – start lames animation.
          ``lames_stop_timer``     Callable – stop lames animation.
          ``lames_pd``             Packed vtkPolyData for G01.
          ``data_id``              Dataset short name for leaderboard.
          ``controls_img_actors``  dict mode_idx → vtkActor2D (controls image).
          ``nav_mode``             list[int] current navigation sub-mode.
          ``style_state``          dict with "idx" key.
          ``apply_style``          Callable(idx) – restore VTK interaction style.
    """

    # Icon patch size in normalised viewport coordinates.
    # Designed for a 1600 × 900 window: 256 / 1600 = 0.160, 192 / 900 ≈ 0.213.
    ICON_NRW = 0.160
    ICON_NRH = 0.213
    ICON_GAP = 0.060    # horizontal gap between patches
    ICON_Y   = 0.42     # bottom edge of the icon row (normalised)

    def __init__(self, plt, mesh, game_context: dict) -> None:
        self._plt  = plt
        self._mesh = mesh
        self._ctx  = game_context

        self._selector_open = False
        self._selected_idx  = 0

        # Selector actors (cleaned up when the selector closes)
        self._selector_actors: list = []
        self._border_actor = None
        self._selector_obs_tag: Optional[int] = None

        # Snapshot of all 2D actors hidden at game-mode entry
        self._saved_visible_2d: list = []

        # In-game HUD actors (title, timer, score)
        self._game_hud_actors: list = []

        # End-screen actors
        self._end_screen_actors: list = []
        self._end_screen_obs_tag: Optional[int] = None

        # Active game reference and info
        self._game_ref: list = [None]
        self._current_game_info: Optional[dict] = None

    # ─────────────────────────────────────────────── public entry point ───────

    def open(self) -> None:
        """Show the game selector; freeze 3D navigation."""
        if self._selector_open:
            return
        self._selector_open = True
        self._selected_idx  = 0
        self._freeze_navigation()
        # Hide all existing 2D actors so only the selector overlay is visible.
        self._snapshot_hide_2d()
        # Also hide the layer-1 overlays (map, info panel) — not in main renderer.
        self._set_overlays_visible(False)
        self._build_selector_ui()
        self._register_selector_keys()
        try:
            self._plt.render()
        except Exception:
            pass

    # ─────────────────────────────────────────────── navigation helpers ───────

    def _freeze_navigation(self) -> None:
        """Switch to a no-op interactor style so mouse/keys don't move camera."""
        try:
            import vtk
            iren = getattr(self._plt, "interactor", None)
            if iren:
                iren.SetInteractorStyle(vtk.vtkInteractorStyleUser())
        except Exception as exc:
            logger.warning("AerenQuest: could not freeze navigation: %s", exc)

    def _restore_navigation(self) -> None:
        """Restore the original VTK interaction style from game_context."""
        try:
            apply_style = self._ctx.get("apply_style")
            style_state = self._ctx.get("style_state", {})
            if apply_style and style_state:
                apply_style(style_state.get("idx", 1))
        except Exception as exc:
            logger.warning("AerenQuest: could not restore navigation: %s", exc)

    # ──────────────────────────────────────────────── selector UI ─────────────

    def _build_selector_ui(self) -> None:
        """Create the full-screen selector overlay (header, icons, border)."""
        if _vedo is None:
            return

        from . import GAMES  # noqa: PLC0415

        # ── Header ───────────────────────────────────────────────────────────
        header = _vedo.Text2D(
            "AerenQuest",
            pos=(0.5, 0.90), s=2.5, c="gold", bg=None, alpha=0.95,
            justify="top-center",
        )
        self._plt.add(header)
        self._selector_actors.append(header)

        # ── Sub-header ───────────────────────────────────────────────────────
        nav_hint = _vedo.Text2D(
            "← →  navigate     Enter  launch     Escape  close",
            pos=(0.5, 0.18), s=0.95, c="white", bg=None, alpha=0.7,
            justify="top-center",
        )
        self._plt.add(nav_hint)
        self._selector_actors.append(nav_hint)

        # ── Icon patches and labels ───────────────────────────────────────────
        repo_root = self._ctx.get("repo_root")
        layout    = self._compute_icon_layout(len(GAMES))

        for i, (game, (x1, y1, x2, y2)) in enumerate(zip(GAMES, layout)):
            self._add_game_patch(game, x1, y1, x2, y2, repo_root)
            # Label below icon
            label = _vedo.Text2D(
                game["label"],
                pos=((x1 + x2) / 2, y1 - 0.015),
                s=0.85, c="white", bg=None, alpha=0.95,
                justify="top-center",
            )
            self._plt.add(label)
            self._selector_actors.append(label)

        # ── Selection border around first game ────────────────────────────────
        if layout:
            x1, y1, x2, y2 = layout[0]
            self._border_actor = self._make_border_actor(
                x1 - 0.006, y1 - 0.006, x2 + 0.006, y2 + 0.006,
            )
            if self._border_actor is not None:
                self._plt.renderer.AddActor2D(self._border_actor)
                self._selector_actors.append(self._border_actor)

    def _add_game_patch(
        self, game: dict,
        x1: float, y1: float, x2: float, y2: float,
        repo_root,
    ) -> None:
        """Load the game icon image or show a placeholder rectangle."""
        if _vedo is None:
            return
        icon_actor = None
        if repo_root:
            icon_path = Path(repo_root) / "images" / game["icon"]
            if icon_path.exists():
                icon_actor = self._load_icon_actor(str(icon_path), x1, y1, x2, y2)

        if icon_actor is not None:
            self._plt.renderer.AddActor2D(icon_actor)
            self._selector_actors.append(icon_actor)
        else:
            # Fallback: grey placeholder label
            ph = _vedo.Text2D(
                f"[ {game['label']} ]",
                pos=((x1 + x2) / 2, (y1 + y2) / 2 + 0.05),
                s=1.0, c="gray", bg="black", alpha=0.7,
                justify="center",
            )
            self._plt.add(ph)
            self._selector_actors.append(ph)

    def _compute_icon_layout(self, n: int) -> list:
        """Return [(x1, y1, x2, y2), …] for each game icon."""
        if n == 0:
            return []
        w   = self.ICON_NRW
        h   = self.ICON_NRH
        gap = self.ICON_GAP
        total_w = n * w + (n - 1) * gap
        x_start = (1.0 - total_w) / 2.0
        y_bot   = self.ICON_Y
        y_top   = y_bot + h
        return [(x_start + i * (w + gap), y_bot,
                 x_start + i * (w + gap) + w, y_top)
                for i in range(n)]

    @staticmethod
    def _load_icon_actor(path: str, x1: float, y1: float, x2: float, y2: float):
        """Return a vtkActor2D displaying the image, or None on failure."""
        try:
            import vtk
            rdr = vtk.vtkImageReader2Factory.CreateImageReader2(path)
            if rdr is None:
                return None
            rdr.SetFileName(path)
            rdr.Update()
            mapper = vtk.vtkImageMapper()
            mapper.SetInputData(rdr.GetOutput())
            mapper.SetColorWindow(255)
            mapper.SetColorLevel(127.5)
            mapper.RenderToRectangleOn()
            act = vtk.vtkActor2D()
            act.SetMapper(mapper)
            act.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            act.GetPositionCoordinate().SetValue(x1, y1)
            act.GetPosition2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            act.GetPosition2Coordinate().SetValue(x2, y2)
            return act
        except Exception as exc:
            logger.warning("AerenQuest: could not load icon %s: %s", path, exc)
            return None

    @staticmethod
    def _make_border_actor(
        x1: float, y1: float, x2: float, y2: float,
        color: tuple = (1.0, 1.0, 1.0),
        lw: float = 3.0,
    ):
        """Draw a white rectangle border using vtkPolyDataMapper2D."""
        try:
            import vtk
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(x1, y1, 0)
            pts.InsertNextPoint(x2, y1, 0)
            pts.InsertNextPoint(x2, y2, 0)
            pts.InsertNextPoint(x1, y2, 0)
            lines = vtk.vtkCellArray()
            for i in range(4):
                ln = vtk.vtkLine()
                ln.GetPointIds().SetId(0, i)
                ln.GetPointIds().SetId(1, (i + 1) % 4)
                lines.InsertNextCell(ln)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetLines(lines)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToNormalizedDisplay()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(pd)
            mapper.SetTransformCoordinate(coord)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(lw)
            return actor
        except Exception as exc:
            logger.warning("AerenQuest: could not create border: %s", exc)
            return None

    def _move_border(self, idx: int) -> None:
        """Reposition the selection border to game *idx*."""
        from . import GAMES  # noqa: PLC0415
        layout = self._compute_icon_layout(len(GAMES))
        if idx >= len(layout) or self._border_actor is None:
            return
        x1, y1, x2, y2 = layout[idx]
        pad = 0.006
        try:
            pd  = self._border_actor.GetMapper().GetInput()
            pts = pd.GetPoints()
            pts.SetPoint(0, x1 - pad, y1 - pad, 0)
            pts.SetPoint(1, x2 + pad, y1 - pad, 0)
            pts.SetPoint(2, x2 + pad, y2 + pad, 0)
            pts.SetPoint(3, x1 - pad, y2 + pad, 0)
            pts.Modified()
            pd.Modified()
        except Exception as exc:
            logger.debug("AerenQuest: _move_border failed: %s", exc)

    def _register_selector_keys(self) -> None:
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            return
        self._selector_obs_tag = iren.AddObserver(
            "KeyPressEvent", self._on_selector_key, 100.0,
        )

    def _deregister_selector_keys(self) -> None:
        iren = getattr(self._plt, "interactor", None)
        if iren is not None and self._selector_obs_tag is not None:
            try:
                iren.RemoveObserver(self._selector_obs_tag)
            except Exception:
                pass
        self._selector_obs_tag = None

    def _on_selector_key(self, obj, _event) -> None:
        from . import GAMES  # noqa: PLC0415
        try:
            key = obj.GetKeySym()
        except Exception:
            return

        if key in ("Left", "KP_Left"):
            self._selected_idx = (self._selected_idx - 1) % max(len(GAMES), 1)
            self._move_border(self._selected_idx)
            try:
                self._plt.render()
            except Exception:
                pass
        elif key in ("Right", "KP_Right"):
            self._selected_idx = (self._selected_idx + 1) % max(len(GAMES), 1)
            self._move_border(self._selected_idx)
            try:
                self._plt.render()
            except Exception:
                pass
        elif key in ("Return", "KP_Enter"):
            self._launch_selected()
        elif key == "Escape":
            self._close_selector()

        # Consume the key
        try:
            cmd = obj.GetCommand(self._selector_obs_tag)
            if cmd is not None:
                cmd.AbortFlagOn()
        except Exception:
            pass
        try:
            obj.SetKeySym("")
        except Exception:
            pass

    def _clear_selector_ui(self) -> None:
        ren = getattr(self._plt, "renderer", None)
        for act in self._selector_actors:
            try:
                # vedo wrapper
                if hasattr(act, "actor") or hasattr(act, "text"):
                    self._plt.remove(act)
                elif ren is not None:
                    ren.RemoveActor2D(act)
            except Exception:
                pass
        self._selector_actors.clear()
        self._border_actor = None

    def _close_selector(self) -> None:
        if not self._selector_open:
            return
        self._selector_open = False
        self._deregister_selector_keys()
        self._clear_selector_ui()
        self._restore_2d()          # restore the 2D actors hidden at open()
        self._set_overlays_visible(True)   # restore map + info panel
        self._restore_navigation()
        try:
            self._plt.render()
        except Exception:
            pass

    # ──────────────────────────────────────────────── game launch ─────────────

    def _launch_selected(self) -> None:
        from . import GAMES  # noqa: PLC0415
        if not GAMES:
            return
        game_info = GAMES[self._selected_idx]
        self._current_game_info = game_info
        self._deregister_selector_keys()
        self._clear_selector_ui()
        self._selector_open = False
        self._enter_game_mode(game_info)

    def _enter_game_mode(self, game_info: dict) -> None:
        """Position camera, start lames, countdown, then launch game."""
        # 2D actors already hidden by open() → just position and launch.
        self._start_lames_if_needed()
        self._position_camera()
        self._run_countdown(lambda: self._actually_launch(game_info))

    def _start_lames_if_needed(self) -> None:
        """Ensure lames animation is running before the game starts."""
        try:
            ls = self._ctx.get("lames_state", {})
            if ls and not ls.get("visible", False):
                start_fn = self._ctx.get("lames_start_timer")
                if start_fn:
                    start_fn()
                    ls["visible"] = True
        except Exception as exc:
            logger.debug("AerenQuest: _start_lames_if_needed: %s", exc)

    # ── UI snapshot / restore ────────────────────────────────────────────────

    def _snapshot_hide_2d(self) -> None:
        """Hide all currently visible 2D actors and save the list."""
        ren = getattr(self._plt, "renderer", None)
        if ren is None:
            return
        col = ren.GetActors2D()
        col.InitTraversal()
        self._saved_visible_2d = []
        while True:
            act = col.GetNextActor2D()
            if act is None:
                break
            if act.GetVisibility():
                self._saved_visible_2d.append(act)
                act.SetVisibility(0)

    def _restore_2d(self) -> None:
        """Restore all 2D actors hidden during game mode."""
        for act in self._saved_visible_2d:
            try:
                act.SetVisibility(1)
            except Exception:
                pass
        self._saved_visible_2d.clear()

    def _show_controls_image(self) -> None:
        """Re-show the keyboard controls image for the current nav mode."""
        ctrl_imgs = self._ctx.get("controls_img_actors", {})
        nav_mode  = self._ctx.get("nav_mode", [0])
        cur_mode  = nav_mode[0] if nav_mode else 0
        for mi, act in ctrl_imgs.items():
            if act is not None:
                try:
                    act.SetVisibility(1 if mi == cur_mode else 0)
                except Exception:
                    pass

    def _set_overlays_visible(self, visible: bool) -> None:
        """Show/hide the layer-1 info panel and ortho map overlays."""
        for key in ("info_panel", "ortho_overlay"):
            obj = self._ctx.get(key)
            if obj is not None:
                try:
                    obj.set_visible(visible)
                except Exception as exc:
                    logger.debug("AerenQuest: set_visible(%s) on %s: %s",
                                 visible, key, exc)

    # ── camera positioning ───────────────────────────────────────────────────

    def _position_camera(self) -> None:
        """Place camera at the centre of the 3D volume, looking along +Z."""
        try:
            bds     = self._mesh.bounds()   # xmin, xmax, ymin, ymax, zmin, zmax
            cx      = (bds[0] + bds[1]) / 2.0
            cy      = (bds[2] + bds[3]) / 2.0
            cz      = (bds[4] + bds[5]) / 2.0
            z_range = max(float(bds[5]) - float(bds[4]), 1.0)
            cam = self._plt.camera
            # Start at volume centre, look slightly forward along +Z
            cam.SetPosition(cx, cy, cz)
            cam.SetFocalPoint(cx, cy, cz + z_range * 0.1)
            cam.SetViewUp(0.0, 1.0, 0.0)
            cam.SetViewAngle(75.0)     # wide FOV for first-person flight
            try:
                self._plt.renderer.ResetCameraClippingRange()
            except Exception:
                pass
        except Exception as exc:
            logger.warning("AerenQuest: camera positioning failed: %s", exc)

    # ── countdown ────────────────────────────────────────────────────────────

    def _run_countdown(self, on_done: Callable) -> None:
        """Show 3 → 2 → 1 in large text, then call *on_done*."""
        if _vedo is None:
            on_done()
            return

        cd_text = _vedo.Text2D(
            "3",
            pos=(0.5, 0.55), s=7.0, c="white", bg=None, alpha=0.95,
            justify="center",
        )
        self._plt.add(cd_text)
        try:
            self._plt.render()
        except Exception:
            pass

        iren       = getattr(self._plt, "interactor", None)
        start_t    = time.monotonic()
        last_n     = [3]
        timer_id   = [None]
        obs_tag    = [None]

        def _tick(obj, _ev):
            try:
                if obj.GetTimerId() != timer_id[0]:
                    return
            except Exception:
                pass
            elapsed = time.monotonic() - start_t
            n       = 3 - int(elapsed)
            if n > 0:
                if n != last_n[0]:
                    last_n[0] = n
                    try:
                        cd_text.text(str(n))
                        self._plt.render()
                    except Exception:
                        pass
            else:
                # Countdown done — tear down
                if iren is not None:
                    try:
                        iren.DestroyTimer(timer_id[0])
                    except Exception:
                        pass
                    if obs_tag[0] is not None:
                        try:
                            iren.RemoveObserver(obs_tag[0])
                        except Exception:
                            pass
                        obs_tag[0] = None
                try:
                    self._plt.remove(cd_text)
                    self._plt.render()
                except Exception:
                    pass
                on_done()

        if iren is not None:
            obs_tag[0]  = iren.AddObserver("TimerEvent", _tick, 5.0)
            timer_id[0] = iren.CreateRepeatingTimer(100)
        else:
            # Fallback: skip countdown
            try:
                self._plt.remove(cd_text)
            except Exception:
                pass
            on_done()

    # ── actual game start ────────────────────────────────────────────────────

    def _actually_launch(self, game_info: dict) -> None:
        """After countdown: create HUD, show controls image, start game."""
        if _vedo is None:
            return

        title = game_info.get("title", "Go !")

        # ── Title bar (top centre, stays all game) ────────────────────────
        title_hud = _vedo.Text2D(
            title,
            pos=(0.5, 0.978), s=1.6, c="cyan", bg="black", alpha=0.82,
            justify="top-center",
        )
        self._plt.add(title_hud)
        self._game_hud_actors.append(title_hud)

        # ── Timer (big, left) ─────────────────────────────────────────────
        hud_timer = _vedo.Text2D(
            "⏱  2:00",
            pos=(0.015, 0.96), s=2.2, c="white", bg="black", alpha=0.82,
            justify="top-left",
        )
        self._plt.add(hud_timer)
        self._game_hud_actors.append(hud_timer)

        # ── Score (big, right) ────────────────────────────────────────────
        hud_score = _vedo.Text2D(
            "Flux  0 / ?",
            pos=(0.985, 0.96), s=2.2, c="cyan", bg="black", alpha=0.82,
            justify="top-right",
        )
        self._plt.add(hud_score)
        self._game_hud_actors.append(hud_score)

        # Restore navigation so the player can fly during the game
        self._restore_navigation()
        # Re-show keyboard controls image
        self._show_controls_image()
        try:
            self._plt.render()
        except Exception:
            pass

        # ── Dynamically import and instantiate the game ───────────────────
        try:
            import importlib
            mod = importlib.import_module(game_info["module"])
            cls = getattr(mod, game_info["cls"])
        except Exception as exc:
            logger.warning("AerenQuest: could not import game %s: %s",
                           game_info["id"], exc)
            self._restore_scene()
            return

        ctx  = self._ctx
        game = cls(
            plt         = self._plt,
            lames_state = ctx.get("lames_state", {}),
            packed_pd   = ctx.get("lames_pd"),
            start_lames = ctx.get("lames_start_timer", lambda: None),
            stop_lames  = ctx.get("lames_stop_timer",  lambda: None),
            hud_timer   = hud_timer,
            hud_score   = hud_score,
            on_end      = self._on_game_end,
            data_id     = ctx.get("data_id", "unknown"),
        )
        self._game_ref[0] = game
        game.start()

    # ── game-end flow ────────────────────────────────────────────────────────

    def _on_game_end(self, score: int, victory: bool) -> None:
        """Called by the game when it finishes (time up or all caught)."""
        # Remove in-game HUD
        self._clear_game_hud()

        # Name entry → submit → end screen
        try:
            from marvel_view.leaderboard.ui import NameEntry  # noqa: PLC0415

            def _on_name(name: str) -> None:
                try:
                    from marvel_view.leaderboard import (  # noqa: PLC0415
                        ScoreEntry, submit_score,
                    )
                    entry = ScoreEntry(
                        game_id     = self._current_game_info["id"],
                        player_name = (name.lower().strip() or "anonyme"),
                        score       = float(score),
                        data_id     = self._ctx.get("data_id", "unknown"),
                        metadata    = {"victory": victory},
                    )
                    submit_score(entry)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("AerenQuest: score submit failed: %s", exc)
                self._show_end_screen(score, victory)

            NameEntry(
                self._plt,
                prompt  = "Ton prénom pour le leaderboard ?",
                on_done = _on_name,
            ).attach()

        except Exception as exc:  # noqa: BLE001
            logger.warning("AerenQuest: NameEntry failed: %s", exc)
            self._show_end_screen(score, victory)

    def _clear_game_hud(self) -> None:
        for act in self._game_hud_actors:
            try:
                self._plt.remove(act)
            except Exception:
                pass
        self._game_hud_actors.clear()

    # ── end screen ───────────────────────────────────────────────────────────

    def _show_end_screen(self, score: int, victory: bool) -> None:
        """Full-screen black leaderboard + Restart / Quit buttons."""
        if _vedo is None:
            self._restore_scene()
            return

        # Fetch and deduplicate scores (best per lowercased name)
        try:
            from marvel_view.leaderboard import fetch_scores  # noqa: PLC0415
            raw = fetch_scores(self._current_game_info["id"])
        except Exception:
            raw = []

        best: dict[str, float] = {}
        for s in raw:
            name = (s.get("player_name") or "anonyme").lower().strip()
            val  = float(s.get("score", 0))
            if name not in best or val > best[name]:
                best[name] = val
        top10 = sorted(best.items(), key=lambda x: -x[1])[:10]

        # ── Black fullscreen background ───────────────────────────────────
        bg = self._make_fullscreen_black_bg()
        if bg is not None:
            self._plt.renderer.AddActor2D(bg)
            self._end_screen_actors.append(bg)

        # ── Leaderboard text ─────────────────────────────────────────────
        game_label = (self._current_game_info or {}).get("title", "Leaderboard")
        lines = [f"  ══  {game_label}  ══\n"]
        if not top10:
            lines.append("   (no scores yet)\n")
        else:
            for i, (name, val) in enumerate(top10, 1):
                medal = ("🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3
                         else f"{i:2d}.")
                lines.append(f"  {medal}  {name:<20s}  {int(val):>5d} pts")
        lines.append(f"\n  Your score this run: {score} pts")

        lb_text = _vedo.Text2D(
            "\n".join(lines),
            pos=(0.5, 0.88), s=1.3, c="white", bg=None, alpha=0.95,
            justify="top-center", font="Calco",
        )
        self._plt.add(lb_text)
        self._end_screen_actors.append(lb_text)

        # ── Bottom buttons ────────────────────────────────────────────────
        btn_restart = _vedo.Text2D(
            "  Restart  (Return)  ",
            pos=(0.35, 0.08), s=1.1, c="black", bg="lime", alpha=0.9,
            justify="bottom-center",
        )
        btn_quit = _vedo.Text2D(
            "  Quit this game  (Escape)  ",
            pos=(0.65, 0.08), s=1.1, c="black", bg="tomato", alpha=0.9,
            justify="bottom-center",
        )
        for btn in (btn_restart, btn_quit):
            self._plt.add(btn)
            self._end_screen_actors.append(btn)

        try:
            self._plt.render()
        except Exception:
            pass

        # ── Keyboard handler for Restart / Quit ───────────────────────────
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            return

        self._end_screen_obs_tag = iren.AddObserver(
            "KeyPressEvent", self._on_end_screen_key, 100.0,
        )

    @staticmethod
    def _make_fullscreen_black_bg():
        """vtkActor2D that fills the entire normalised viewport with black."""
        try:
            import vtk
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(0.0, 0.0, 0)
            pts.InsertNextPoint(1.0, 0.0, 0)
            pts.InsertNextPoint(1.0, 1.0, 0)
            pts.InsertNextPoint(0.0, 1.0, 0)
            quad = vtk.vtkCellArray()
            quad.InsertNextCell(4)
            for i in range(4):
                quad.InsertCellPoint(i)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetPolys(quad)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToNormalizedDisplay()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(pd)
            mapper.SetTransformCoordinate(coord)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0, 0, 0)
            actor.GetProperty().SetOpacity(1.0)
            return actor
        except Exception as exc:
            logger.warning("AerenQuest: could not create black bg: %s", exc)
            return None

    def _on_end_screen_key(self, obj, _event) -> None:
        try:
            key = obj.GetKeySym()
        except Exception:
            return

        handled = False
        if key in ("Return", "KP_Enter"):
            handled = True
            self._restart_game()
        elif key == "Escape":
            handled = True
            self._quit_game()

        if handled:
            try:
                cmd = obj.GetCommand(self._end_screen_obs_tag)
                if cmd is not None:
                    cmd.AbortFlagOn()
            except Exception:
                pass
            try:
                obj.SetKeySym("")
            except Exception:
                pass

    def _clear_end_screen(self) -> None:
        ren = getattr(self._plt, "renderer", None)
        for act in self._end_screen_actors:
            try:
                if hasattr(act, "actor") or hasattr(act, "text"):
                    self._plt.remove(act)
                elif ren is not None:
                    ren.RemoveActor2D(act)
            except Exception:
                pass
        self._end_screen_actors.clear()

        iren = getattr(self._plt, "interactor", None)
        if iren is not None and self._end_screen_obs_tag is not None:
            try:
                iren.RemoveObserver(self._end_screen_obs_tag)
            except Exception:
                pass
        self._end_screen_obs_tag = None

    # ── restart / quit ───────────────────────────────────────────────────────

    def _restart_game(self) -> None:
        """Clear end screen and re-run the same game from scratch."""
        self._clear_end_screen()
        if self._current_game_info is not None:
            self._position_camera()
            self._run_countdown(
                lambda: self._actually_launch(self._current_game_info)
            )

    def _quit_game(self) -> None:
        """Exit game mode: clear everything, restore original scene."""
        self._clear_end_screen()
        self._restore_scene()

    def _restore_scene(self) -> None:
        """Restore all hidden UI and navigation after game mode ends."""
        self._clear_game_hud()
        self._restore_2d()
        self._set_overlays_visible(True)   # restore map + info panel
        self._restore_navigation()
        # Refresh nav panel visibility (controls image) via context callback
        refresh = self._ctx.get("refresh_nav_panel_visibility")
        if refresh is not None:
            try:
                refresh()
            except Exception:
                pass
        try:
            self._plt.render()
        except Exception:
            pass
