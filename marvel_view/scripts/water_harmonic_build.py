#!/usr/bin/env python3
"""
Pre-build the water harmonic potential field, passage-time maps, stream
tracks and dual (water + air) arrow fields used by the
``marvel-water-conductance`` viewer.

Reads (binary 8-bit TIFF, threshold 127, shape ``(Z, Y, X)``):

    1_Intermediate_computed_images/Wat_Norm_Cortex.tif  (water-conducting mask)
    1_Intermediate_computed_images/Source_crown.tif     (source mask, u=0)
    1_Intermediate_computed_images/Target_crown.tif     (target mask, u=1)

Optionally for dual arrows (requires wind field to be pre-built first):

    2_Vtk_files/wind_field.npz   (air harmonic field from marvel-wind-field-build)

Writes:

    2_Vtk_files/water_harmonic.npz     — u, vec, speed, area, source, target,
                                          T_simple[, T_eikonal]
    2_Vtk_files/crown_tracks.vtp       — stream-line tracks (replaces Dijkstra)
    2_Vtk_files/water_dual_arrows.npz  — water arrows for dual overlay
    2_Vtk_files/air_dual_arrows.npz    — air arrows for dual overlay

Usage
-----
::

    marvel-water-harmonic-build
    # … or equivalently:
    python -m marvel_view.scripts.water_harmonic_build

    # Skip the slow Eikonal passage-time computation:
    marvel-water-harmonic-build --skip-eikonal

    # Force full rebuild even if caches exist:
    marvel-water-harmonic-build --force

    # Tune stream tracks:
    marvel-water-harmonic-build --n-seeds 800 --step-size 0.4 --max-steps 8000
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

from marvel_view.preprocessing.water_harmonic import (  # noqa: E402
    DEFAULT_CG_MAXITER,
    DEFAULT_CG_TOL,
    DEFAULT_GRAD_SIGMA,
    DEFAULT_MAX_STEPS,
    DEFAULT_N_SEEDS,
    DEFAULT_SEPARATION_VOX,
    DEFAULT_STEP_VOX,
    build_dual_arrows_filtered,
    build_water_harmonic_field_from_tifs,
    build_water_stream_tracks,
    load_water_harmonic_field,
    save_water_harmonic_field,
)
from marvel_view.scripts.water_conductance import (  # noqa: E402
    DEFAULT_WATER_AREA_PATH,
    DEFAULT_WATER_SOURCE_PATH,
    DEFAULT_WATER_TARGET_PATH,
    DEFAULT_WATER_HARMONIC_CACHE,
    DEFAULT_CROWN_TRACKS_VTP_CACHE,
    DEFAULT_WIND_FIELD_CACHE,
    DEFAULT_WATER_DUAL_ARROWS_CACHE,
    DEFAULT_AIR_DUAL_ARROWS_CACHE,
    DEFAULT_SPACING,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.water_harmonic_build")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-build the water harmonic field, passage times, "
                    "stream tracks and dual arrow caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs.
    p.add_argument("--water-area",   default=str(DEFAULT_WATER_AREA_PATH),
                   help="Binary 8-bit TIFF: water-conducting domain mask.")
    p.add_argument("--water-source", default=str(DEFAULT_WATER_SOURCE_PATH),
                   help="Binary 8-bit TIFF: source crown (u=0).")
    p.add_argument("--water-target", default=str(DEFAULT_WATER_TARGET_PATH),
                   help="Binary 8-bit TIFF: target / stele crown (u=1).")
    p.add_argument("--wind-field",   default=str(DEFAULT_WIND_FIELD_CACHE),
                   help="Path to wind_field.npz for dual-arrow overlay.  "
                        "Omit / missing → dual arrows skipped.")
    # Outputs.
    p.add_argument("--field-cache",  default=str(DEFAULT_WATER_HARMONIC_CACHE),
                   help="Path to water_harmonic.npz output.")
    p.add_argument("--tracks-cache", default=str(DEFAULT_CROWN_TRACKS_VTP_CACHE),
                   help="Path to crown_tracks.vtp output.")
    p.add_argument("--dual-water-cache",
                   default=str(DEFAULT_WATER_DUAL_ARROWS_CACHE),
                   help="Path to water_dual_arrows.npz output.")
    p.add_argument("--dual-air-cache",
                   default=str(DEFAULT_AIR_DUAL_ARROWS_CACHE),
                   help="Path to air_dual_arrows.npz output.")
    # Field params.
    p.add_argument("--grad-sigma", type=float, default=DEFAULT_GRAD_SIGMA,
                   help="Masked-Gaussian sigma applied to u before gradient.")
    p.add_argument("--cg-tol",     type=float, default=DEFAULT_CG_TOL,
                   help="Relative tolerance for the CG Laplace solver.")
    p.add_argument("--cg-maxiter", type=int,   default=DEFAULT_CG_MAXITER,
                   help="Maximum CG / PyAMG iterations.")
    # Stream tracks.
    p.add_argument("--n-seeds",    type=int,   default=DEFAULT_N_SEEDS,
                   help="Number of stream-track seeds (k-medoids).")
    p.add_argument("--step-size",  type=float, default=DEFAULT_STEP_VOX,
                   help="Euler integration step size (voxels).")
    p.add_argument("--max-steps",  type=int,   default=DEFAULT_MAX_STEPS,
                   help="Maximum integration steps per track (safety cap).")
    # Dual arrows.
    p.add_argument("--fine-stride", type=int, default=4,
                   help="Sub-sampling stride for dual-arrow placement.")
    p.add_argument("--separation",  type=int, default=DEFAULT_SEPARATION_VOX,
                   help="Minimum voxel distance from opposite domain (EDT).")
    # Behaviour.
    p.add_argument("--skip-eikonal",    action="store_true",
                   help="Skip Eikonal passage-time computation (fast but "
                        "T_eikonal will be absent from the cache).")
    p.add_argument("--skip-tracks",     action="store_true",
                   help="Skip stream-track computation.")
    p.add_argument("--skip-dual",       action="store_true",
                   help="Skip dual-arrow computation.")
    p.add_argument("--field-only",      action="store_true",
                   help="Build only the harmonic field + passage times.")
    p.add_argument("--tracks-only",     action="store_true",
                   help="Load field from cache and rebuild only crown_tracks.vtp "
                        "(skips Laplace solve, Eikonal and dual arrows).")
    p.add_argument("--dual-only",       action="store_true",
                   help="Load field from cache and rebuild only the dual-arrow "
                        "caches (fast — skips Laplace solve and tracks).")
    p.add_argument("--force", "-f",     action="store_true",
                   help="Recompute everything even if caches exist.")
    p.add_argument("--verbose", "-v",   action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # --tracks-only: load field from cache, rebuild tracks only.
    if getattr(args, "tracks_only", False):
        args.skip_eikonal = True
        args.skip_dual = True

    # --dual-only: convenience shortcut — implies --skip-tracks + force-dual.
    if getattr(args, "dual_only", False):
        args.skip_tracks = True
        args.skip_eikonal = True

    water_area   = Path(args.water_area).expanduser().resolve()
    water_source = Path(args.water_source).expanduser().resolve()
    water_target = Path(args.water_target).expanduser().resolve()
    field_cache  = Path(args.field_cache).expanduser().resolve()
    tracks_cache = Path(args.tracks_cache).expanduser().resolve()
    dual_water   = Path(args.dual_water_cache).expanduser().resolve()
    dual_air     = Path(args.dual_air_cache).expanduser().resolve()
    wind_path    = Path(args.wind_field).expanduser().resolve()

    logger.info("═" * 64)
    logger.info("Marvel Water-Harmonic Build")
    logger.info("  area     : %s", water_area)
    logger.info("  source   : %s", water_source)
    logger.info("  target   : %s", water_target)
    logger.info("  → field  : %s", field_cache)
    logger.info("  → tracks : %s", tracks_cache)
    logger.info("  → dual-W : %s", dual_water)
    logger.info("  → dual-A : %s", dual_air)
    logger.info("  skip_eikonal=%s  skip_tracks=%s  skip_dual=%s",
                args.skip_eikonal, args.skip_tracks, args.skip_dual)
    logger.info("═" * 64)

    # ── 1. Harmonic field + passage times ────────────────────────────
    need_field = args.force or not field_cache.exists()
    if getattr(args, "dual_only", False) or getattr(args, "tracks_only", False):
        need_field = False   # always load from cache in --dual-only / --tracks-only mode
    if need_field:
        for fp in (water_area, water_source, water_target):
            if not fp.exists():
                logger.error("Required input missing: %s", fp)
                return 2
        field = build_water_harmonic_field_from_tifs(
            water_area, water_source, water_target,
            spacing=DEFAULT_SPACING,
            grad_sigma=args.grad_sigma,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
            skip_eikonal=args.skip_eikonal,
        )
        save_water_harmonic_field(field, field_cache)
    else:
        logger.info("Loading cached harmonic field from %s", field_cache)
        field = load_water_harmonic_field(field_cache)

    if args.field_only:
        logger.info("Done (field only).")
        return 0

    # ── 2. Stream tracks (replaces Dijkstra / MCP) ────────────────────
    if not args.skip_tracks:
        need_tracks = (args.force or getattr(args, "tracks_only", False)
                       or not tracks_cache.exists())
        if need_tracks:
            logger.info("Building stream tracks …")
            from marvel_view.scripts.water_conductance.pipeline import (
                _write_tracks_vtp,
            )
            tracks_data = build_water_stream_tracks(
                vec_field=field["vec"],
                u_field=field["u"],
                source_vol=(field["source"].astype(np.uint8) * 255),
                spacing=DEFAULT_SPACING,
                n_seeds=args.n_seeds,
                step_vox=args.step_size,
                max_steps=args.max_steps,
            )
            if tracks_data is not None:
                _write_tracks_vtp(tracks_data, tracks_cache)
            else:
                logger.warning("Stream tracks: no tracks produced — VTP not written.")
        else:
            logger.info("Skipping stream tracks (cache exists: %s).", tracks_cache)
    else:
        logger.info("Skipping stream tracks (--skip-tracks).")

    # ── 3. Dual arrow fields ──────────────────────────────────────────
    if not args.skip_dual:
        need_dual = (args.force or getattr(args, "dual_only", False)
                     or not dual_water.exists() or not dual_air.exists())
        if need_dual:
            if not wind_path.exists():
                logger.warning(
                    "Wind-field cache not found (%s) — dual arrows skipped.  "
                    "Run marvel-wind-field-build first.", wind_path,
                )
            else:
                logger.info("Building dual arrow fields …")
                from marvel_view.preprocessing.wind_field import load_wind_field
                wind = load_wind_field(wind_path)

                w_arrows, a_arrows = build_dual_arrows_filtered(
                    water_vec=field["vec"],
                    water_area=field["area"],
                    air_vec=wind["vec"],
                    air_area=wind["area"],
                    fine_stride=args.fine_stride,
                    separation_vox=args.separation,
                    spacing=DEFAULT_SPACING,
                )

                dual_water.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(dual_water),
                    pts=w_arrows["pts"],
                    dirs=w_arrows["dirs"],
                    scores=w_arrows["scores"],
                    boost=w_arrows["boost"],
                    stride=np.array([w_arrows["stride"]], dtype=np.int32),
                )
                logger.info("Water dual arrows saved → %s  (%d arrows)",
                            dual_water, len(w_arrows["pts"]))

                dual_air.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(dual_air),
                    pts=a_arrows["pts"],
                    dirs=a_arrows["dirs"],
                    scores=a_arrows["scores"],
                    boost=a_arrows["boost"],
                    stride=np.array([a_arrows["stride"]], dtype=np.int32),
                )
                logger.info("Air dual arrows saved → %s  (%d arrows)",
                            dual_air, len(a_arrows["pts"]))
        else:
            logger.info("Skipping dual arrows (caches exist).")
    else:
        logger.info("Skipping dual arrows (--skip-dual).")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
