#!/usr/bin/env python3
"""
Pre-build the wind harmonic field + O₂ / CH₄ particle trajectory
templates used by the ``marvel-water-conductance`` viewer's wind toggles.

Reads (binary 8-bit TIFF, threshold 127, shape ``(Z, Y, X)``):

    1_Intermediate_computed_images/wind_area.tif    (air mask)
    1_Intermediate_computed_images/wind_source.tif  (source slice @ high axis-0)
    1_Intermediate_computed_images/wind_target.tif  (target slice @ low  axis-0)

Writes (one set per speed level: slow / med / fast):

    2_Vtk_files/wind_field.npz   (harmonic u, ∇u, wall_dist, masks)
    2_Vtk_files/wind_o2_slow.npz (O₂  trajectory templates — slow / 10 substeps)
    2_Vtk_files/wind_o2_med.npz  (O₂  trajectory templates — med  / 20 substeps)
    2_Vtk_files/wind_o2_fast.npz (O₂  trajectory templates — fast / 40 substeps)
    2_Vtk_files/wind_ch4_slow.npz(CH₄ trajectory templates — slow / 10 substeps)
    2_Vtk_files/wind_ch4_med.npz (CH₄ trajectory templates — med  / 20 substeps)
    2_Vtk_files/wind_ch4_fast.npz(CH₄ trajectory templates — fast / 40 substeps)

Usage
-----
::

    marvel-wind-field-build
    # … or, equivalently:
    python -m marvel_view.scripts.wind_field_build

    # Force a full rebuild even if caches are up-to-date:
    marvel-wind-field-build --force

    # Build only specific speed levels:
    marvel-wind-field-build --speeds slow med

    # Tune the trajectory generation:
    marvel-wind-field-build --o2-templates 5000 --ch4-templates 2500 \\
        --fps 25 --seconds 40 --lifespan 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running the file directly without pip install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view.preprocessing.wind_field import (  # noqa: E402
    DEFAULT_CG_MAXITER,
    DEFAULT_CG_TOL,
    DEFAULT_FPS,
    DEFAULT_GRAD_SIGMA,
    DEFAULT_LIFESPAN_S,
    DEFAULT_N_TEMPLATES,
    DEFAULT_SECONDS,
    build_particle_templates,
    build_wind_field_from_tifs,
    load_wind_field,
    save_particle_templates,
    save_wind_field,
)
from marvel_view.scripts.water_conductance import (  # noqa: E402
    DEFAULT_WIND_AREA_PATH,
    DEFAULT_WIND_FIELD_CACHE,
    DEFAULT_WIND_SOURCE_PATH,
    DEFAULT_WIND_TARGET_PATH,
    DEFAULT_WIND_O2_CACHES,
    DEFAULT_WIND_CH4_CACHES,
    WIND_SUBSTEP_VALUES,
    WIND_SPEED_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.wind_field_build")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-build the wind harmonic field and the O₂/CH₄ "
                    "particle trajectory templates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs.
    p.add_argument("--wind-area",   default=str(DEFAULT_WIND_AREA_PATH),
                   help="Binary 8-bit TIFF of the air mask (where wind passes).")
    p.add_argument("--wind-source", default=str(DEFAULT_WIND_SOURCE_PATH),
                   help="Binary 8-bit TIFF of the source slice (u=0, "
                        "axis-0 = high X end of the cylinder).")
    p.add_argument("--wind-target", default=str(DEFAULT_WIND_TARGET_PATH),
                   help="Binary 8-bit TIFF of the target slice (u=1, "
                        "axis-0 = low X end of the cylinder).")
    # Outputs.
    p.add_argument("--field-cache", default=str(DEFAULT_WIND_FIELD_CACHE),
                   help="Path to the harmonic-field .npz cache.")
    # Field params.
    p.add_argument("--grad-sigma", type=float, default=DEFAULT_GRAD_SIGMA,
                   help="Masked-Gaussian sigma applied to u before gradient.")
    p.add_argument("--cg-tol",     type=float, default=DEFAULT_CG_TOL,
                   help="Relative tolerance for the CG Laplace solver.")
    p.add_argument("--cg-maxiter", type=int,   default=DEFAULT_CG_MAXITER,
                   help="Maximum CG iterations.")
    # Templates params.
    p.add_argument("--fps",       type=int,   default=DEFAULT_FPS,
                   help="Frames per second of the precomputed sequence.")
    p.add_argument("--seconds",   type=float, default=DEFAULT_SECONDS,
                   help="Length of the periodic precomputed loop, in seconds.")
    p.add_argument("--lifespan",  type=float, default=DEFAULT_LIFESPAN_S,
                   help="Per-particle lifespan, in seconds.")
    p.add_argument("--o2-templates",  type=int, default=DEFAULT_N_TEMPLATES,
                   help="Number of O₂  trajectory templates to precompute.")
    p.add_argument("--ch4-templates", type=int, default=DEFAULT_N_TEMPLATES // 2,
                   help="Number of CH₄ trajectory templates to precompute.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible templates.")
    # Behaviour.
    p.add_argument("--force",         action="store_true",
                   help="Recompute everything even if caches exist.")
    p.add_argument("--field-only",    action="store_true",
                   help="Build only the harmonic field; skip the particles.")
    p.add_argument("--templates-only", action="store_true",
                   help="Build only the particle templates; reuse the cached field.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    wind_area   = Path(args.wind_area).expanduser().resolve()
    wind_source = Path(args.wind_source).expanduser().resolve()
    wind_target = Path(args.wind_target).expanduser().resolve()
    field_cache = Path(args.field_cache).expanduser().resolve()

    speed_idx_map = {lbl: i for i, lbl in enumerate(WIND_SPEED_LABELS)}

    logger.info("═" * 64)
    logger.info("Marvel Wind-Field Build")
    logger.info("  area    : %s", wind_area)
    logger.info("  source  : %s", wind_source)
    logger.info("  target  : %s", wind_target)
    logger.info("  → field : %s", field_cache)
    for _lbl in WIND_SPEED_LABELS:
        _i = speed_idx_map[_lbl]
        logger.info("  → O₂  %-4s : %s", _lbl, DEFAULT_WIND_O2_CACHES[_i])
        logger.info("  → CH₄ %-4s : %s", _lbl, DEFAULT_WIND_CH4_CACHES[_i])
    logger.info("═" * 64)

    # ── 1. Harmonic field ────────────────────────────────────────────
    need_field = (args.force or not field_cache.exists()) and not args.templates_only
    if need_field:
        for p in (wind_area, wind_source, wind_target):
            if not p.exists():
                logger.error("Required input missing: %s", p)
                return 2
        field = build_wind_field_from_tifs(
            wind_area, wind_source, wind_target,
            grad_sigma=args.grad_sigma,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
        )
        save_wind_field(field, field_cache)
    else:
        if not field_cache.exists():
            logger.error("Field cache not found and --templates-only set: %s",
                         field_cache)
            return 2
        logger.info("Loading cached harmonic field from %s", field_cache)
        field = load_wind_field(field_cache)

    if args.field_only:
        logger.info("Done (field only).")
        return 0

    # ── 2. Particle templates — one set per speed level ─────────────────
    # Each level is built with a different n_substeps (10 / 20 / 40).
    # speed_vox_per_frame is the same for all three so the particle paths
    # cover the same physical distance but with finer curvature resolution
    # at higher levels.  frames_per_tick is always 1 in the viewer.

    _DEFAULT_SPEED = 1.4  # vox/frame — same for all three levels

    for _lbl in WIND_SPEED_LABELS:
        _i        = speed_idx_map[_lbl]
        _nsub     = int(WIND_SUBSTEP_VALUES[_i])
        _o2_cache  = Path(DEFAULT_WIND_O2_CACHES[_i]).expanduser().resolve()
        _ch4_cache = Path(DEFAULT_WIND_CH4_CACHES[_i]).expanduser().resolve()

        logger.info("── Speed level: %s (n_substeps=%d) ──", _lbl, _nsub)

        if args.force or not _o2_cache.exists():
            _tpl_o2 = build_particle_templates(
                field, species="o2",
                n_templates=args.o2_templates,
                fps=args.fps,
                seconds=args.seconds,
                lifespan_s=args.lifespan,
                speed_vox_per_frame=_DEFAULT_SPEED,
                seed=args.seed,
                n_substeps=_nsub,
            )
            save_particle_templates(_tpl_o2, _o2_cache)
        else:
            logger.info("Skipping O₂ %s templates (cache exists: %s).",
                        _lbl, _o2_cache)

        if args.force or not _ch4_cache.exists():
            _tpl_ch4 = build_particle_templates(
                field, species="ch4",
                n_templates=args.ch4_templates,
                fps=args.fps,
                seconds=args.seconds,
                lifespan_s=args.lifespan,
                speed_vox_per_frame=_DEFAULT_SPEED,
                seed=args.seed + 1,
                n_substeps=_nsub,
            )
            save_particle_templates(_tpl_ch4, _ch4_cache)
        else:
            logger.info("Skipping CH₄ %s templates (cache exists: %s).",
                        _lbl, _ch4_cache)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
