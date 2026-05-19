#!/usr/bin/env python3
"""
Marvel Aerench – Interactive viewer.

Loads the per-label VTK meshes produced by ``aerench_preprocess`` and
opens the shared :class:`RootsViewer` in **mesh-only** mode (no source
8-bit volume in this pipeline).

Usage
-----
::

    python -m marvel_view.scripts.aerench_visualize
    # or, after pip install -e .:
    marvel-aerench-view
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import aerench_config as acfg
from marvel_view.visualization.roots_viewer import RootsViewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.aerench_visualize")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive viewer for aerench meshes (mesh-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vtk-dir", "-d", default=str(acfg.DEFAULT_OUTPUT_DIR),
                   help="Directory with per-label VTK sub-directories.")
    p.add_argument("--settings", default=str(acfg.DEFAULT_SETTINGS_PATH),
                   help="JSON settings sidecar (auto save / load).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("═" * 64)
    logger.info("Marvel Aerench – Viewer")
    logger.info("  VTK dir  : %s", Path(args.vtk_dir).resolve())
    logger.info("  Settings : %s", Path(args.settings).resolve())
    logger.info("═" * 64)

    # source_path=None disables volume / split modes; the viewer stays in
    # mesh mode (acfg.VIEW_MODE_DEFAULT == "mesh").
    viewer = RootsViewer(
        rcfg=acfg,
        vtk_dir=Path(args.vtk_dir),
        source_path=None,
        settings_path=Path(args.settings),
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
