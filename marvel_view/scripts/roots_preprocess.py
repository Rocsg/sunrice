#!/usr/bin/env python3
"""
Marvel View – Roots preprocessing pipeline.

Standalone twin of :mod:`marvel_view.scripts.preprocess_all`, dedicated to
the *small_roots-2* dataset.  Differences:

* Reads paths and per-label settings from
  :mod:`marvel_view.roots_config` (does **not** touch the original
  ``marvel_view.config``).
* After loading the segmentation volume, **seals the 6 outer faces** of
  the volume to value ``SEAL_BORDERS_LABEL`` (default ``2`` = background)
  so non-background labels never touch the boundary; this prevents
  open-edge / singular surfaces in the marching-cubes output.
* Writes meshes to a separate output directory
  (default ``./roots_output``) so previous SunRice meshes are untouched.

Usage
-----
::

    python -m marvel_view.scripts.roots_preprocess
    # or, after pip install -e .:
    marvel-roots-preprocess
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import roots_config as rcfg
from marvel_view.preprocessing import load_segmentation
from marvel_view.scripts.preprocess_all import _process_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.roots_preprocess")


def _seal_borders(volume: np.ndarray, fill_value: int) -> None:
    """In-place: set the 6 outer faces of the volume to ``fill_value``.

    Logs the unique label values found on each face *before* and *after*
    sealing so any forgotten face / wrong dtype / view-vs-copy issue is
    immediately visible.
    """
    logger.info(
        "Sealing 6 outer faces of the volume to value %d  (shape=%s, dtype=%s) …",
        fill_value, volume.shape, volume.dtype,
    )
    faces = {
        "Z=0   ":   (slice(0, 1),    slice(None), slice(None)),
        "Z=-1  ":   (slice(-1, None), slice(None), slice(None)),
        "Y=0   ":   (slice(None), slice(0, 1),    slice(None)),
        "Y=-1  ":   (slice(None), slice(-1, None), slice(None)),
        "X=0   ":   (slice(None), slice(None), slice(0, 1)),
        "X=-1  ":   (slice(None), slice(None), slice(-1, None)),
    }
    logger.info("  Unique values per face BEFORE sealing:")
    for tag, sl in faces.items():
        logger.info("    %s  →  %s", tag, np.unique(volume[sl]).tolist())

    fv = np.uint8(fill_value)
    for sl in faces.values():
        volume[sl] = fv

    logger.info("  Unique values per face AFTER sealing:")
    for tag, sl in faces.items():
        logger.info("    %s  →  %s", tag, np.unique(volume[sl]).tolist())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the roots segmentation TIFF into per-label VTK meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", "-i", default=str(rcfg.DEFAULT_SEG_PATH),
                   help="Path to the multi-label segmentation TIFF.")
    p.add_argument("--output-dir", "-o", default=str(rcfg.DEFAULT_OUTPUT_DIR),
                   help="Root directory for VTK output.")
    p.add_argument("--spacing", nargs=3, type=float, metavar=("Z", "Y", "X"),
                   default=list(rcfg.DEFAULT_SPACING),
                   help="Voxel spacing in z y x order.")
    p.add_argument("--smooth-iter", type=int, default=rcfg.SMOOTH_ITERATIONS,
                   help="Laplacian smoothing iterations.")
    p.add_argument("--decimate", type=float, default=rcfg.DECIMATE_FRACTION,
                   dest="decimate_fraction",
                   help="Fallback fraction of faces to keep (only when no face_budget).")
    p.add_argument("--workers", type=int,
                   default=(rcfg.DEFAULT_NUM_WORKERS or 0),
                   help="Parallel workers for all_cc; 0 = cpu_count, 1 = serial.")
    p.add_argument("--step-size", type=int, default=0, metavar="N",
                   help="Marching-cubes stride.  0 = auto from face_budget.")
    p.add_argument("--labels", nargs="+", type=int, default=None,
                   help="Restrict to these label IDs (default: all configured).")
    p.add_argument("--max-components", type=int, default=None, metavar="N",
                   help="For all_cc labels: mesh only the N largest components.")
    p.add_argument("--no-seal-borders", action="store_true",
                   help="Disable the 6-face border-sealing step.")
    p.add_argument("--resume", action="store_true",
                   help="Skip labels / components already on disk.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    image_path = Path(args.image)
    out_dir    = Path(args.output_dir)
    spacing    = tuple(args.spacing)
    label_ids  = args.labels or list(rcfg.LABEL_CONFIG.keys())

    num_workers = int(args.workers) if args.workers and args.workers > 0 \
        else (os.cpu_count() or 1)
    step_size: Optional[int] = int(args.step_size) if args.step_size and args.step_size > 0 else None

    logger.info("═" * 64)
    logger.info("Marvel View – Roots Preprocessing")
    logger.info("  Segmentation : %s", image_path)
    logger.info("  Output dir   : %s", out_dir.resolve())
    logger.info("  Spacing      : %s", spacing)
    logger.info("  Labels       : %s", label_ids)
    logger.info("  Seal borders : %s (label %d)",
                not args.no_seal_borders, rcfg.SEAL_BORDERS_LABEL)
    logger.info("  Workers      : %d", num_workers)
    logger.info("  Step size    : %s", step_size or "auto")
    logger.info("═" * 64)

    volume = load_segmentation(image_path)

    if not args.no_seal_borders:
        _seal_borders(volume, rcfg.SEAL_BORDERS_LABEL)

    t_total = time.perf_counter()
    for lid in label_ids:
        if lid not in rcfg.LABEL_CONFIG:
            logger.warning("Label %d not in roots LABEL_CONFIG — skipping.", lid)
            continue
        cfg = rcfg.LABEL_CONFIG[lid]
        # Per-label face_budget is mandatory in roots_config; resolve it here.
        target_faces: Optional[int] = cfg.get("face_budget")
        _process_label(
            volume=volume,
            label_id=lid,
            cfg=cfg,
            out_dir=out_dir,
            spacing=spacing,
            smooth_iter=args.smooth_iter,
            decimate_fraction=args.decimate_fraction,
            resume=args.resume,
            max_components=args.max_components,
            target_faces=target_faces,
            num_workers=num_workers,
            step_size=step_size,
        )

    logger.info("═" * 64)
    logger.info(
        "All done in %.1f s.  Output: %s",
        time.perf_counter() - t_total, out_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
