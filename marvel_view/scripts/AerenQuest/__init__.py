"""AerenQuest — mini-game registry.

Each game entry:
  id          snake-case leaderboard key
  label       short display label shown under the icon: "G01: Catch the fluxes"
  title       long title shown in selector header and during gameplay
  icon        filename in <repo>/images/  (256 × 192 px PNG)
  module      dotted import path of the game module
  cls         class name to instantiate
  score_type  "points_desc" | "time_asc"

Add new games (G02, G03 …) by appending to GAMES.  The rest of the
launcher picks them up automatically.
"""
from __future__ import annotations

GAMES: list[dict] = [
    {
        "id":         "catch_flux",
        "label":      "G01: Catch the fluxes",
        "title":      "Catch the fluxes !",
        "icon":       "G01_icon.png",
        "module":     "marvel_view.scripts.AerenQuest.G01_catch_the_fluxes.game",
        "cls":        "CatchFluxGame",
        "score_type": "points_desc",
    },
    {
        "id":         "safari_photo",
        "label":      "G02: Safari Photo",
        "title":      "Safari Photo",
        "icon":       "G02_icon.png",
        "module":     "marvel_view.scripts.AerenQuest.G02_Safari_photo.game",
        "cls":        "SafariPhotoGame",
        "score_type": "points_desc",
        "score_unit": "photos",
        # Freeform games skip countdown and restore the normal scene view.
        # The launcher calls _freeform_enter() instead of _enter_game_mode().
        "freeform":   True,
    },
]
