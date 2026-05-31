"""G02 – Safari Photo game.

Once selected from the AerenQuest menu, this game restores the normal
scene view (free navigation, all UI visible) and adds two HUD overlays:

  • "Use P to take a photo  ·  M to quit safari"
  • "0 photos taken"  (counter increments after each photo)

**P key** – takes a clean screenshot:
    - All 2D UI widgets (buttons, text, controls image) are temporarily
      hidden so they don't appear in the photo.
    - Colormap bars (``ColormapBar2D`` objects on layer-1 renderers) are
      left visible so scientific context is preserved.
    - The PNG is saved to ``<data_path>/safari/safari_YYYYMMDD_HHMM/``.
    - After the *first* photo the folder is opened in the system file
      manager (``xdg-open`` on Linux); the VTK window is then raised back
      to the foreground so subsequent photos keep filling the folder.

**M key** – quit safari mode:
    - HUD texts are removed.
    - NameEntry prompt is shown.
    - Score (number of photos taken) is submitted to the leaderboard.
    - Full leaderboard end screen is shown.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import vedo as _vedo
except ImportError:
    _vedo = None  # type: ignore[assignment]

# Numpad key → exponent: scale = 2**power  (numlock ON and OFF keysyms)
_NUMPAD_POWER: dict[str, int] = {
    "KP_1": 1, "KP_End":       1,
    "KP_2": 2, "KP_Down":      2,
    "KP_3": 3, "KP_Next":      3,   # Page Down with numlock off
}


class SafariPhotoGame:
    """Freeform Safari Photo game for AerenQuest G02.

    Parameters
    ----------
    plt:
        Running ``vedo.Plotter``.
    on_end:
        Callback ``on_end(score: int, victory: bool)`` fired when the
        player quits with M.  ``score`` = number of photos taken.
    data_id:
        Dataset short name for the leaderboard.
    data_path:
        Base data directory; screenshots go to
        ``<data_path>/safari/safari_YYYYMMDD_HHMM/``.
    cbar_objects:
        List of ``ColormapBar2D`` instances.  Their actors are excluded
        from the "hide everything" pass during screenshots so they remain
        visible in the photos.
    game_context:
        The full game context dict from the launcher (used to hide/restore
        info_panel and ortho_overlay during screenshots).
    **kwargs:
        Absorbs any extra keyword arguments passed by the launcher for
        forward-compatibility.
    """

    def __init__(
        self,
        plt,
        on_end: Optional[Callable] = None,
        data_id: str = "unknown",
        data_path=None,
        cbar_objects=None,
        game_context: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self._plt          = plt
        self._on_end       = on_end
        self._data_id      = data_id
        self._data_path    = Path(data_path) if data_path else Path.home()
        self._cbar_objects = list(cbar_objects or [])
        self._ctx          = game_context or {}

        self._n_photos:      int            = 0
        self._safari_folder: Optional[Path] = None
        self._hud_actors:    list           = []
        self._hint_text                     = None
        self._counter_text                  = None
        self._obs_tag:             Optional[int]  = None
        self._key_release_obs_tag: Optional[int]  = None
        self._held_keys:           set             = set()
        self._folder_opened:       bool            = False
        self._ended:               bool            = False   # guard against double-end

        # Pre-compute the set of colormap bar actor object-ids so we can
        # skip them during the screenshot hide-pass.
        self._cbar_actor_ids: set = {
            id(c.image_actor)
            for c in self._cbar_objects
            if hasattr(c, "image_actor")
        }

    # ── precompute ────────────────────────────────────────────────────────────

    def precompute(self) -> None:
        """Nothing heavy to precompute for safari mode."""

    # ── start ─────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Add HUD overlays and register the P / M key observer."""
        if _vedo is None:
            return

        # Create the timestamped safari folder.
        now = datetime.now()
        folder_name = f"safari_{now.strftime('%Y%m%d_%H%M')}"
        self._safari_folder = self._data_path / "safari" / folder_name
        try:
            self._safari_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Safari folder: %s", self._safari_folder)
        except Exception as exc:
            logger.warning("Safari: could not create folder: %s", exc)
            self._safari_folder = Path.home()

        # ── HUD: instructions ──────────────────────────────────────────────
        self._hint_text = _vedo.Text2D(
            "P  photo  ·  KP1/2/3 = ×2/×4/×8  ·  M  quitter",

            pos=(0.5, 0.977),
            s=0.95, c="yellow", bg="black", alpha=0.82,
            justify="top-center",
        )
        self._plt.add(self._hint_text)
        self._hud_actors.append(self._hint_text)

        # ── HUD: photo counter ─────────────────────────────────────────────
        self._counter_text = _vedo.Text2D(
            "0 photos taken",
            pos=(0.5, 0.935),
            s=1.25, c="white", bg="black", alpha=0.78,
            justify="top-center",
        )
        self._plt.add(self._counter_text)
        self._hud_actors.append(self._counter_text)

        try:
            self._plt.render()
        except Exception:
            pass

        # ── Register key observers (priority 90 = below selector at 100) ──
        iren = getattr(self._plt, "interactor", None)
        if iren is not None:
            self._obs_tag = iren.AddObserver(
                "KeyPressEvent", self._on_key, 90.0,
            )
            self._key_release_obs_tag = iren.AddObserver(
                "KeyReleaseEvent", self._on_key_release, 90.0,
            )

    # ── key handler ───────────────────────────────────────────────────────────

    def _on_key(self, obj, _event) -> None:
        try:
            key = obj.GetKeySym()
        except Exception:
            return
        self._held_keys.add(key)

        if key == "p":
            self._take_screenshot()
            self._consume_key(obj)
        elif key == "m":
            self._quit()
            self._consume_key(obj)

    def _on_key_release(self, obj, _event) -> None:
        try:
            self._held_keys.discard(obj.GetKeySym())
        except Exception:
            pass

    def _screenshot_scale(self) -> int:
        """Return 1, 2, 4 or 8 depending on which numpad key is held with P."""
        for key, power in _NUMPAD_POWER.items():
            if key in self._held_keys:
                return 1 << power   # 2, 4 or 8
        return 1

    def _consume_key(self, obj) -> None:
        try:
            cmd = obj.GetCommand(self._obs_tag)
            if cmd is not None:
                cmd.AbortFlagOn()
        except Exception:
            pass
        try:
            obj.SetKeySym("")
        except Exception:
            pass

    # ── screenshot ────────────────────────────────────────────────────────────

    def _take_screenshot(self) -> None:
        """Capture a clean screenshot (no UI; colorbars + ortho panel kept) to the safari folder."""
        try:
            import vtk
        except ImportError:
            logger.warning("Safari: vtk not available, cannot screenshot")
            return

        ren_win = getattr(self._plt, "window", None)
        if ren_win is None:
            logger.warning("Safari: plt.window not available")
            return

        # ── 0. Calcul du facteur + feedback visuel ────────────────────────
        scale = self._screenshot_scale()
        scale_label = f"×{scale}" if scale > 1 else "native"
        capture_banner = None
        if _vedo is not None:
            try:
                capture_banner = _vedo.Text2D(
                    f"📸  Capture {scale_label} en cours…",
                    pos=(0.5, 0.5), s=1.8, c="white", bg="black", alpha=0.88,
                    justify="center",
                )
                self._plt.add(capture_banner)
                ren_win.Render()   # affiche la bannière avant le freeze
            except Exception:
                capture_banner = None

        # ── 1. Build the set of actor IDs to keep visible ─────────────────
        # Always keep colormap bars (layer-1 renderers).
        # Also keep the ortho panel map (layer-1 renderer) — it gives
        # spatial context in the photo.
        safe_actor_ids = set(self._cbar_actor_ids)
        ortho = self._ctx.get("ortho_overlay")
        if ortho is not None and hasattr(ortho, "image_actor"):
            safe_actor_ids.add(id(ortho.image_actor))

        # ── 2. Hide all visible 2D actors NOT in the safe set ─────────────
        hidden_actors: list = []
        rens = ren_win.GetRenderers()
        rens.InitTraversal()
        while True:
            ren = rens.GetNextItem()
            if ren is None:
                break
            col = ren.GetActors2D()
            col.InitTraversal()
            while True:
                act = col.GetNextActor2D()
                if act is None:
                    break
                if id(act) in safe_actor_ids:
                    continue   # keep colorbars and ortho map
                if act.GetVisibility():
                    hidden_actors.append(act)
                    act.SetVisibility(0)

        # Also explicitly hide the info_panel (subtitle/speed banner).
        info_panel = self._ctx.get("info_panel")
        info_was_visible = False
        if info_panel is not None:
            try:
                info_was_visible = bool(getattr(info_panel, "image_actor",
                                                None) and
                                        info_panel.image_actor.GetVisibility())
                if info_was_visible:
                    info_panel.set_visible(False)
            except Exception:
                pass

        # ── 3. Render the clean scene ──────────────────────────────────────
        ren_win.Render()

        # ── 4. Capture via vtkWindowToImageFilter at ~3 K width ───────────
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(ren_win)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        try:
            w2i.SetScale(scale)
        except (AttributeError, TypeError):
            try:
                w2i.SetMagnification(scale)   # older VTK API
            except AttributeError:
                pass
        w2i.Modified()
        w2i.Update()

        # ── 5. Write PNG ───────────────────────────────────────────────────
        now = datetime.now()
        fname = (
            self._safari_folder
            / f"photo_{self._n_photos + 1:04d}_{now.strftime('%H%M%S')}.png"
        )
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(fname))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        logger.info("Safari: photo saved → %s", fname)

        # ── 6. Restore all hidden 2D actors ────────────────────────────────
        for act in hidden_actors:
            act.SetVisibility(1)

        # Supprime la bannière de feedback (elle a été restaurée par la boucle).
        if capture_banner is not None:
            try:
                self._plt.remove(capture_banner)
            except Exception:
                pass
            capture_banner = None

        if info_was_visible and info_panel is not None:
            try:
                info_panel.set_visible(True)
            except Exception:
                pass

        # ── 7. Increment counter and re-render ─────────────────────────────
        self._n_photos += 1
        self._update_counter()

        # Force ortho panel to reclaim its correct position (ConfigureEvent
        # doesn't fire during screenshots, so we do it explicitly here).
        ortho = self._ctx.get("ortho_overlay")
        if ortho is not None and hasattr(ortho, "_reposition"):
            try:
                ortho._reposition()
            except Exception:
                pass

        try:
            self._plt.render()
        except Exception:
            pass

        # ── 8. Open folder in system file manager on first photo ───────────
        if self._n_photos == 1 and not self._folder_opened:
            self._folder_opened = True
            self._open_folder_async()

    def _update_counter(self) -> None:
        if self._counter_text is None:
            return
        n = self._n_photos
        label = f"{n} photo{'s' if n != 1 else ''} taken"
        try:
            self._counter_text.text(label)
        except Exception:
            pass

    def _open_folder_async(self) -> None:
        """Open the safari folder in the system file manager (non-blocking),
        then raise the VTK window back to the foreground after ~1.5 s."""
        if self._safari_folder is None:
            return
        folder = str(self._safari_folder)
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["explorer", folder])
        except Exception as exc:
            logger.warning("Safari: could not open folder: %s", exc)
            return

        # Schedule a VTK timer to raise the render window after 1.5 s.
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            return
        raise_tag: list = [None]
        timer_id:  list = [None]

        def _raise(obj, _ev):
            try:
                if obj.GetTimerId() != timer_id[0]:
                    return
                iren.DestroyTimer(timer_id[0])
                iren.RemoveObserver(raise_tag[0])
            except Exception:
                pass
            try:
                ren_win = getattr(self._plt, "window", None)
                if ren_win is not None:
                    ren_win.Raise()
            except Exception as exc:
                logger.debug("Safari: Raise() failed: %s", exc)

        raise_tag[0] = iren.AddObserver("TimerEvent", _raise, 1.0)
        timer_id[0]  = iren.CreateOneShotTimer(1500)

    # ── quit / abort ───────────────────────────────────────────────────

    def abort(self) -> None:
        """Abort the game immediately (e.g. user pressed Escape)."""
        self._quit()

    def _quit(self) -> None:
        """Deregister keys, remove HUD texts, call on_end(n_photos, True)."""
        if self._ended:
            return
        self._ended = True
        iren = getattr(self._plt, "interactor", None)
        if iren is not None:
            for tag_attr in ("_obs_tag", "_key_release_obs_tag"):
                tag = getattr(self, tag_attr, None)
                if tag is not None:
                    try:
                        iren.RemoveObserver(tag)
                    except Exception:
                        pass
                    setattr(self, tag_attr, None)
        self._held_keys.clear()

        # Remove HUD text actors.
        for act in self._hud_actors:
            try:
                self._plt.remove(act)
            except Exception:
                pass
        self._hud_actors.clear()
        self._hint_text    = None
        self._counter_text = None

        try:
            self._plt.render()
        except Exception:
            pass

        # Fire the end callback — score = number of photos taken.
        if self._on_end is not None:
            self._on_end(self._n_photos, True)
