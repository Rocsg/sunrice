#!/usr/bin/env python3
"""
Marvel View – Interactive 3-D scene viewer.

Loads all VTK meshes produced by ``preprocess_all.py``, applies the
colours and opacities defined in ``config.py``, and opens an interactive
vedo window.

Usage
-----
Direct::

    python -m marvel_view.scripts.visualize_scene

After ``pip install -e .``::

    marvel-view

Key options
-----------
  -d / --vtk-dir    Directory that contains the per-label VTK sub-dirs
                    (default: ./marvel_output).
  --bg              Background colour name or hex (default: black).
  --labels 1 3 5    Show only these label IDs (default: all).
  -s / --screenshot Save a PNG and exit instead of opening the viewer.
  -v / --verbose    Enable DEBUG logging.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── allow running the file directly without pip install ───────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import config
from marvel_view.visualization import render_from_vtk_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.visualize")


# ──────────────────────────────── CLI ─────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open an interactive 3-D viewer for preprocessed segmentation meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vtk-dir", "-d",
        default=str(config.DEFAULT_OUTPUT_DIR),
        help="Root directory containing per-label VTK sub-directories.",
    )
    p.add_argument(
        "--bg",
        default="black",
        help="Viewer background colour (name or #hex).",
    )
    p.add_argument(
        "--labels",
        nargs="+", type=int, default=None,
        help="Show only these label IDs (default: all configured labels).",
    )
    p.add_argument(
        "--screenshot", "-s",
        default=None,
        metavar="FILE.png",
        help="Save a PNG screenshot to this path and exit (non-interactive).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    label_cfg = config.LABEL_CONFIG
    if args.labels:
        label_cfg = {k: v for k, v in label_cfg.items() if k in args.labels}
        if not label_cfg:
            logger.error("None of the requested labels are in LABEL_CONFIG: %s", args.labels)
            return 1

    interactive = args.screenshot is None

    logger.info("═" * 64)
    logger.info("Marvel View – Visualisation")
    logger.info("  VTK dir  : %s", Path(args.vtk_dir).resolve())
    logger.info("  Labels   : %s", list(label_cfg.keys()))
    logger.info("  Mode     : %s", "interactive" if interactive else f"screenshot → {args.screenshot}")
    logger.info("═" * 64)

    render_from_vtk_dir(
        vtk_dir=args.vtk_dir,
        label_config=label_cfg,
        bg=args.bg,
        interactive=interactive,
        screenshot_path=args.screenshot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
