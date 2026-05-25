#!/usr/bin/env python3
"""
marvel-lame2-normals-build — pre-compute and cache per-step point normals.

Loads the packed ``lame2.vtp`` produced by
``marvel-water-conductance-build-meshes``, splits it into per-animation-step
PolyDatas, runs ``vtkPolyDataNormals`` on each (exactly as the viewer would
do at startup), then saves each result as an individual binary VTP into
``lame2_normals_cache/``.

On the next launch of ``marvel-water-conductance`` the viewer detects the
normals cache and loads the pre-baked VTPs directly, skipping the per-frame
normals filter — this shaves several seconds off the cold-start time for
large datasets.

Usage
-----
::

    # Build with default paths (uses MARVEL_DATA_DIR env var):
    marvel-lame2-normals-build

    # Explicit paths:
    marvel-lame2-normals-build \\
        --lame2-vtp  /data/vtk/lame2.vtp \\
        --lame2-meta /data/vtk/lame2_meta.json \\
        --output-dir /data/vtk/lame2_normals_cache

    # Force rebuild even if cache already exists:
    marvel-lame2-normals-build --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── allow running directly without pip install ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view.scripts.water_conductance.constants import (  # noqa: E402
    DEFAULT_LAME2_VTP_CACHE,
    DEFAULT_LAME2_META_CACHE,
    DEFAULT_LAME2_NORMALS_CACHE_DIR,
)
from marvel_view.scripts.water_conductance.pipeline import (  # noqa: E402
    _load_lames,
    _build_lames_step_polydatas,
    save_lame2_normals_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="marvel-lame2-normals-build",
        description="Pre-compute per-step point normals for the lame2 animation "
                    "and save them to a cache directory.",
    )
    p.add_argument(
        "--lame2-vtp",
        default=str(DEFAULT_LAME2_VTP_CACHE),
        help="Path to the packed lame2.vtp (default: %(default)s)",
    )
    p.add_argument(
        "--lame2-meta",
        default=str(DEFAULT_LAME2_META_CACHE),
        help="Path to the lame2_meta.json sidecar (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAME2_NORMALS_CACHE_DIR),
        help="Directory where per-step VTPs are written (default: %(default)s)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the normals cache already exists.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    vtp_path  = Path(args.lame2_vtp).expanduser().resolve()
    meta_path = Path(args.lame2_meta).expanduser().resolve()
    out_dir   = Path(args.output_dir).expanduser().resolve()

    # ── Early sentinel check (read meta JSON only, no VTP load) ────────────
    import json as _json
    try:
        _pre_meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        _pre_n = int(_pre_meta.get("n_steps", 0))
    except Exception:
        _pre_n = 0
    if _pre_n > 0:
        from marvel_view.scripts.water_conductance.pipeline import _lame2_cache_subdir
        _sentinel = _lame2_cache_subdir(out_dir, _pre_n) / "_normals_meta.json"
        if _sentinel.exists() and not args.force:
            logger.info(
                "Normals cache already exists at %s — nothing to do.  "
                "Pass --force to rebuild.",
                _sentinel.parent,
            )
            return 0

    # ── Load packed lame2 VTP ───────────────────────────────────────────────
    logger.info("Loading lame2 VTP: %s", vtp_path)
    packed_pd, meta = _load_lames(vtp_path, meta_path)
    if packed_pd is None:
        logger.error(
            "Could not load lame2 VTP/meta — make sure "
            "marvel-water-conductance-build-meshes has been run first."
        )
        return 1

    n_steps = int(meta.get("n_steps", 0))
    if n_steps <= 0:
        logger.error("lame2_meta.json reports n_steps=%d — nothing to build.", n_steps)
        return 1

    logger.info("n_steps=%d  — computing point normals per step …", n_steps)

    # ── Compute normals (same filter settings as the viewer uses) ────────────
    step_pds, _actor = _build_lames_step_polydatas(packed_pd, n_steps)

    # ── Save to normals cache directory ─────────────────────────────────────
    save_lame2_normals_cache(step_pds, out_dir)
    logger.info("Done. Cache written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
