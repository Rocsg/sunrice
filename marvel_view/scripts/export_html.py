#!/usr/bin/env python3
"""
Marvel View – Export the 3-D scene as a standalone HTML page.

Loads the per-label VTK meshes produced by ``marvel-preprocess``
(directory layout below), applies the colours / opacities defined in
``marvel_view.config.LABEL_CONFIG``, and writes a single self-contained
HTML file that can be opened in any browser to manipulate the model
(rotate / zoom / pan).

Directory layout (from ``preprocess_all.py``)::

    marvel_output/
    ├── label1_background_cavities/<mesh.vtk | component_*.vtk>
    ├── label2_aqueous_tissues/    <mesh.vtk | component_*.vtk>
    ├── label3_iodine/             <mesh.vtk | component_*.vtk>
    ├── label4_scaffold/           <mesh.vtk | component_*.vtk>
    └── label5_membranes/          <mesh.vtk | component_*.vtk>

Usage
-----
::

    python -m marvel_view.scripts.export_html
    # or, after pip install -e .:
    marvel-export-html -o scene.html
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.export_html")


# ──────────────────────────────── helpers ─────────────────────────────────────


def _rgb_to_float(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, float(c) / 255.0)) for c in rgb)  # type: ignore[return-value]


def _label_dir(vtk_dir: Path, label_id: int, name: str) -> Path:
    return vtk_dir / f"label{label_id}_{name}"


# ──────────────────────────────── core ────────────────────────────────────────


def build_plotter(
    vtk_dir: Path,
    label_config: Dict,
    bg: str = "black",
    fallback_iodine_only: bool = False,
):
    """Build a ``pyvista.Plotter`` populated with per-label meshes.

    Parameters
    ----------
    vtk_dir:
        Root directory containing the per-label sub-directories.
    label_config:
        Slice of :data:`marvel_view.config.LABEL_CONFIG` (label ids → cfg).
    bg:
        Background colour name or hex string.
    fallback_iodine_only:
        If ``True``, render only iodine (label 3) fully opaque and the
        outer envelope (label 2 aqueous_tissues) at 0.10 opacity — the
        minimal "as-close-as-possible" fallback requested.
    """
    import pyvista as pv

    if fallback_iodine_only:
        # Hard-coded minimal scene: iodine opaque + envelope at 0.10.
        overrides = {
            3: {"opacity": 1.0},
            2: {"opacity": 0.10},
        }
        label_config = {
            lid: {**label_config[lid], **overrides[lid]}
            for lid in (2, 3) if lid in label_config
        }

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
    plotter.set_background(bg)

    n_added = 0
    for label_id, cfg in label_config.items():
        ldir = _label_dir(vtk_dir, label_id, cfg["name"])
        if not ldir.exists():
            logger.warning("No directory for label %d (%s) — skipping.",
                           label_id, cfg["name"])
            continue
        vtk_files: List[Path] = sorted(ldir.glob("*.vtk"))
        if not vtk_files:
            logger.warning("No VTK files in %s — skipping.", ldir)
            continue

        color = _rgb_to_float(cfg["color"])
        opacity = float(cfg["opacity"])
        ambient = float(cfg.get("ambient", 0.15))
        diffuse = float(cfg.get("diffuse", 0.85))
        specular = float(cfg.get("specular", 0.20))

        for f in vtk_files:
            mesh = pv.read(str(f))
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                ambient=ambient,
                diffuse=diffuse,
                specular=specular,
                specular_power=20.0,
                smooth_shading=True,
                name=f"label{label_id}_{f.stem}",
            )
            n_added += 1
            logger.info("  [label %d] added %s  (%d cells)",
                        label_id, f.name, mesh.n_cells)

    if n_added == 0:
        raise RuntimeError(
            f"No meshes found under {vtk_dir}. "
            "Run `marvel-preprocess` first."
        )

    plotter.reset_camera()
    return plotter


# ──────────────────────────────── CLI ─────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export the marvel-view 3-D scene as a standalone HTML page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vtk-dir", "-d", default=str(config.DEFAULT_OUTPUT_DIR),
                   help="Directory with per-label VTK sub-directories.")
    p.add_argument("--output", "-o", default="marvel_view.html",
                   help="Path to the output HTML file.")
    p.add_argument("--bg", default="black",
                   help="Background colour (name or #hex).")
    p.add_argument("--labels", nargs="+", type=int, default=None,
                   help="Show only these label IDs (default: all configured).")
    p.add_argument("--fallback", action="store_true",
                   help="Minimal scene: iodine opaque + envelope at 0.10 opacity.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    label_cfg = config.LABEL_CONFIG
    if args.labels:
        label_cfg = {k: v for k, v in label_cfg.items() if k in args.labels}
        if not label_cfg:
            logger.error("None of the requested labels are in LABEL_CONFIG: %s",
                         args.labels)
            return 1

    vtk_dir = Path(args.vtk_dir)
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("═" * 64)
    logger.info("Marvel View – HTML Export")
    logger.info("  VTK dir : %s", vtk_dir.resolve())
    logger.info("  Labels  : %s", list(label_cfg.keys()))
    logger.info("  Output  : %s", out_path)
    logger.info("  Mode    : %s", "fallback (iodine + envelope @0.10)"
                if args.fallback else "full scene")
    logger.info("═" * 64)

    plotter = build_plotter(
        vtk_dir=vtk_dir,
        label_config=label_cfg,
        bg=args.bg,
        fallback_iodine_only=args.fallback,
    )

    plotter.export_html(str(out_path))
    plotter.close()
    logger.info("Wrote interactive HTML scene to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
