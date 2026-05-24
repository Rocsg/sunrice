"""``marvel_view.scripts.water_conductance`` — interactive vedo viewer.

This package was historically a single ~6000-line module.  It has been
split into focused sub-modules:

* :mod:`.constants` — all ``DEFAULT_*`` paths, colours and animation tunings.
* :mod:`.cli` — :func:`parse_args` argument parser.
* :mod:`.pipeline` — pure mesh/arrow/track build & cache loaders.
* :mod:`.styling` — lighting / shading helpers.
* :mod:`.app` — viewer, ``_attach_controls`` interactor closures, ``main``.

The package itself re-exports every public name that was importable from
the old monolithic module so the sibling scripts
(``water_conductance_build_meshes``, ``water_movie``, ``wind_field_build``)
keep working without any change.
"""
from __future__ import annotations

# ── Public entry point ─────────────────────────────────────────────────────
from .app import main

# ── Constants (re-exported wholesale so ``from … import DEFAULT_*`` works) ─
from .constants import *  # noqa: F401,F403

# ── CLI ────────────────────────────────────────────────────────────────────
from .cli import parse_args  # noqa: F401

# ── Pipeline helpers used by sibling scripts ───────────────────────────────
from .pipeline import (  # noqa: F401
    _build_arrow_field,
    _build_crown_dijkstra_tracks,
    _build_density_facet_scalars,
    _build_lames_step_polydatas,
    _build_mesh,
    _build_tracks_polydata,
    _load_lames,
    _load_or_build_arrows,
    _load_or_build_crown_tracks,
    _load_or_build_dual_arrows,
    _load_or_build_membranes,
    _load_or_build_mesh,
    _load_or_build_overlay_mesh,
    _parse_central_axis,
    _write_tracks_arrows_vtp,
    _write_tracks_vtp,
)

# ── Styling helpers ────────────────────────────────────────────────────────
from .styling import (  # noqa: F401
    _hue_shifted_rgb,
    _install_scene_lights,
    _set_lighting,
    _set_shading,
    _style_mesh,
)

__all__ = [
    "main",
    "parse_args",
    # pipeline
    "_build_arrow_field",
    "_build_crown_dijkstra_tracks",
    "_build_density_facet_scalars",
    "_build_lames_step_polydatas",
    "_build_mesh",
    "_build_tracks_polydata",
    "_load_lames",
    "_load_or_build_arrows",
    "_load_or_build_crown_tracks",
    "_load_or_build_dual_arrows",
    "_load_or_build_membranes",
    "_load_or_build_mesh",
    "_load_or_build_overlay_mesh",
    "_parse_central_axis",
    "_write_tracks_arrows_vtp",
    "_write_tracks_vtp",
    # styling
    "_hue_shifted_rgb",
    "_install_scene_lights",
    "_set_lighting",
    "_set_shading",
    "_style_mesh",
]
