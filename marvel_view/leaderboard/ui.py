"""Leaderboard UI helpers — vedo/VTK overlays for name entry and score display.

Usage pattern (inside a game end handler)
------------------------------------------
    from marvel_view.leaderboard.ui import NameEntry, LeaderboardPanel
    from marvel_view.leaderboard import fetch_scores

    def on_game_end(plt, elapsed, game_id, data_id):
        def on_name(name):
            entry = ScoreEntry(game_id, name, elapsed, data_id)
            ok = submit_score(entry)
            scores = fetch_scores(game_id)
            panel = LeaderboardPanel(plt, game_id, scores)
            panel.show()

        entry_widget = NameEntry(plt, prompt="Ton prénom ?", on_done=on_name)
        entry_widget.attach()
"""

from __future__ import annotations

from typing import Callable, Optional

# vedo is imported lazily so this module can be imported without a display.
try:
    import vedo as _vedo
except ImportError:
    _vedo = None  # type: ignore[assignment]

from .client import GAME_REGISTRY


# ── NameEntry ─────────────────────────────────────────────────────────────────

class NameEntry:
    """Capture a player name via keyboard inside the running vedo window.

    Attaches a high-priority KeyPressEvent observer that intercepts all
    keystrokes while active.  Calls *on_done(name)* when Enter is pressed
    (or when the name reaches *max_len* characters).  Detaches itself
    automatically once done.

    Parameters
    ----------
    plt:
        The running ``vedo.Plotter`` instance.
    prompt:
        Text shown above the input field.
    on_done:
        Callback receiving the final name string.
    max_len:
        Maximum number of characters allowed (default 24).
    """

    _ALLOWED = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 -_.'éèêëàâùûüîïôçÉÈÀÂÙÎÔÇ"
    )

    def __init__(
        self,
        plt,
        prompt: str = "Ton prénom ?",
        on_done: Optional[Callable[[str], None]] = None,
        max_len: int = 24,
    ) -> None:
        self._plt = plt
        self._prompt = prompt
        self._on_done = on_done
        self._max_len = max_len
        self._name: list[str] = []
        self._observer_tag: Optional[int] = None
        self._text_actor = None

    def attach(self) -> None:
        """Start intercepting keyboard input."""
        if _vedo is None:
            return
        iren = getattr(self._plt, "interactor", None)
        if iren is None:
            return
        self._text_actor = _vedo.Text2D(
            self._prompt + "\n> _",
            pos="bottom-center",
            c="white",
            bg="black",
            alpha=0.7,
            s=1.0,
        )
        self._plt.add(self._text_actor)
        self._plt.render()
        self._observer_tag = iren.AddObserver(
            "KeyPressEvent", self._on_key, 100.0  # very high priority — eats all keys
        )

    def _on_key(self, obj, _event) -> None:
        key_sym = obj.GetKeySym()
        key_char = obj.GetKeyCode()

        if key_sym in ("Return", "KP_Enter"):
            self._finish()
            return
        if key_sym == "BackSpace":
            if self._name:
                self._name.pop()
        elif key_char and key_char in self._ALLOWED and len(self._name) < self._max_len:
            self._name.append(key_char)

        self._refresh()

        # Abort event propagation so the game controls don't fire
        try:
            cmd = obj.GetCommand(self._observer_tag)
            if cmd is not None:
                cmd.AbortFlagOn()
        except Exception:
            pass

    def _refresh(self) -> None:
        name_str = "".join(self._name) or ""
        if self._text_actor is not None:
            self._text_actor.text(f"{self._prompt}\n> {name_str}_")
            self._plt.render()

    def _finish(self) -> None:
        self._detach()
        name = "".join(self._name).strip() or "anonyme"
        if self._on_done:
            self._on_done(name)

    def _detach(self) -> None:
        iren = getattr(self._plt, "interactor", None)
        if iren is not None and self._observer_tag is not None:
            try:
                iren.RemoveObserver(self._observer_tag)
            except Exception:
                pass
        if self._text_actor is not None:
            try:
                self._plt.remove(self._text_actor)
            except Exception:
                pass
            self._text_actor = None
        self._observer_tag = None


# ── LeaderboardPanel ──────────────────────────────────────────────────────────

class LeaderboardPanel:
    """Display a top-N leaderboard as a vedo Text2D overlay.

    Parameters
    ----------
    plt:
        The running ``vedo.Plotter`` instance.
    game_id:
        Game identifier (used for the title).
    scores:
        List of score dicts as returned by ``fetch_scores()``.
    auto_hide_s:
        Seconds before the panel removes itself.  ``None`` = stay until
        ``hide()`` is called explicitly.
    """

    def __init__(
        self,
        plt,
        game_id: str,
        scores: list[dict],
        auto_hide_s: Optional[float] = 12.0,
    ) -> None:
        self._plt = plt
        self._game_id = game_id
        self._scores = scores
        self._auto_hide_s = auto_hide_s
        self._actor = None
        self._obs_tag: Optional[int] = None
        self._hide_deadline: float = float("inf")

    def show(self) -> None:
        if _vedo is None:
            return
        game_info = GAME_REGISTRY.get(self._game_id, {})
        label = game_info.get("label", self._game_id)
        score_type = game_info.get("score_type", "time_asc")
        unit = "s" if score_type == "time_asc" else "pts"

        lines = [f"── {label} ──"]
        if not self._scores:
            lines.append("(pas encore de scores)")
        else:
            for i, s in enumerate(self._scores, 1):
                name = s.get("player_name", "?")
                val = s.get("score", 0)
                val_str = f"{val:.1f}{unit}"
                lines.append(f"{i:2d}. {name:<18s} {val_str:>8s}")

        self._actor = _vedo.Text2D(
            "\n".join(lines),
            pos="top-right",
            c="yellow",
            bg="black",
            alpha=0.75,
            s=0.85,
            font="Calco",
        )
        self._plt.add(self._actor)
        self._plt.render()

        if self._auto_hide_s is not None:
            iren = getattr(self._plt, "interactor", None)
            if iren is not None:
                import time as _time
                self._hide_deadline = _time.monotonic() + self._auto_hide_s
                self._obs_tag = iren.AddObserver("TimerEvent", self._on_timer)

    def hide(self) -> None:
        iren = getattr(self._plt, "interactor", None)
        if iren is not None and self._obs_tag is not None:
            try:
                iren.RemoveObserver(self._obs_tag)
            except Exception:
                pass
            self._obs_tag = None
        if self._actor is not None:
            try:
                self._plt.remove(self._actor)
                self._plt.render()
            except Exception:
                pass
            self._actor = None

    def _on_timer(self, obj, _event) -> None:
        import time as _time
        if _time.monotonic() >= self._hide_deadline:
            self.hide()
