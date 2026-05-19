#!/usr/bin/env python3
"""
Marvel View – Lightweight HTML export (iodine + surrounding medium only).

Loads only label 3 (iodine) and label 2 (aqueous_tissues, the surrounding
medium / outer envelope) from ``marvel_output/``, decimates each mesh to
~1/5 of its triangles, then writes a single self-contained HTML page you
can open in any browser to rotate / zoom / pan the model.

Verbose progress is printed at every step so you can see exactly where
it's at (loading, decimating, adding to the scene, exporting).

Usage
-----
::

    # with the venv activated:
    python marvel_view/scripts/export_html_light.py
    # or:
    python -m marvel_view.scripts.export_html_light

Options
-------
    -o, --output FILE   Output HTML path (default: ./marvel_view_light.html)
    -d, --vtk-dir DIR   Input mesh directory (default: ./marvel_output)
    -r, --reduction F   Decimation reduction fraction in [0, 1).
                        0.8 = keep 1/5 of triangles (default).
    --bg COLOR          Background colour (default: black).
    --iodine-opacity F  Opacity for the iodine mesh   (default: 1.0).
    --medium-opacity F  Opacity for the medium mesh   (default: 0.10).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import config


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rgb01(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c / 255.0)) for c in rgb)  # type: ignore[return-value]


def _load_label(vtk_dir: Path, label_id: int, name: str):
    """Read every *.vtk in label dir and merge into a single PolyData."""
    import pyvista as pv

    ldir = vtk_dir / f"label{label_id}_{name}"
    if not ldir.exists():
        raise FileNotFoundError(f"Missing label directory: {ldir}")
    files = sorted(ldir.glob("*.vtk"))
    if not files:
        raise FileNotFoundError(f"No .vtk files in {ldir}")

    log(f"  label {label_id} ({name}): {len(files)} VTK file(s) to read")
    parts = []
    total_cells = 0
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        m = pv.read(str(f))
        # Ensure PolyData (surface mesh) – marching-cubes output already is.
        if not isinstance(m, pv.PolyData):
            m = m.extract_surface()
        parts.append(m)
        total_cells += m.n_cells
        if i == 1 or i == len(files) or i % 20 == 0:
            log(f"    read {i}/{len(files)}  cumulative cells={total_cells:,}"
                f"  ({time.perf_counter() - t0:.1f}s)")

    if len(parts) == 1:
        merged = parts[0]
    else:
        log(f"  merging {len(parts)} parts …")
        t1 = time.perf_counter()
        merged = parts[0].merge(parts[1:])
        log(f"  merged  cells={merged.n_cells:,}  pts={merged.n_points:,}"
            f"  ({time.perf_counter() - t1:.1f}s)")
    return merged


def _decimate(mesh, reduction: float):
    import pyvista as pv

    if reduction <= 0:
        return mesh
    log(f"  decimating (reduction={reduction:.2f})  before:"
        f" cells={mesh.n_cells:,} pts={mesh.n_points:,}")
    t0 = time.perf_counter()
    # decimate_pro handles non-watertight surfaces better than decimate.
    out = mesh.decimate_pro(reduction, preserve_topology=False)
    log(f"  decimated  after:  cells={out.n_cells:,} pts={out.n_points:,}"
        f"  ({time.perf_counter() - t0:.1f}s)")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--vtk-dir", default=str(config.DEFAULT_OUTPUT_DIR))
    p.add_argument("-o", "--output", default="marvel_view_light.html")
    p.add_argument("-r", "--reduction", type=float, default=0.8,
                   help="Decimation reduction (0.8 = keep 1/5).")
    p.add_argument("--bg", default="black")
    p.add_argument("--iodine-opacity", type=float, default=1.0)
    p.add_argument("--medium-opacity", type=float, default=0.05)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    log("═" * 60)
    log("Marvel View – Lightweight HTML export")
    log(f"  vtk_dir        : {Path(args.vtk_dir).resolve()}")
    log(f"  output         : {Path(args.output).resolve()}")
    log(f"  reduction      : {args.reduction}")
    log(f"  iodine opacity : {args.iodine_opacity}")
    log(f"  medium opacity : {args.medium_opacity}")
    log("═" * 60)

    log("Importing pyvista …")
    t0 = time.perf_counter()
    import pyvista as pv
    log(f"  pyvista {pv.__version__}  ({time.perf_counter() - t0:.1f}s)")

    vtk_dir = Path(args.vtk_dir)

    # Label 3 = iodine, Label 2 = aqueous_tissues (surrounding medium envelope).
    cfg_iodine = config.LABEL_CONFIG[3]
    cfg_medium = config.LABEL_CONFIG[2]

    log("─" * 60)
    log("Loading IODINE (label 3) …")
    iodine = _load_label(vtk_dir, 3, cfg_iodine["name"])
    iodine = _decimate(iodine, args.reduction)

    log("─" * 60)
    log("Loading SURROUNDING MEDIUM (label 2 aqueous_tissues) …")
    medium = _load_label(vtk_dir, 2, cfg_medium["name"])
    medium = _decimate(medium, args.reduction)

    log("─" * 60)
    log("Building scene …")
    plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
    plotter.set_background(args.bg)

    plotter.add_mesh(
        medium,
        color=_rgb01(cfg_medium["color"]),
        opacity=float(args.medium_opacity),
        smooth_shading=True,
        ambient=0.20, diffuse=0.85, specular=0.10,
        name="medium",
    )
    log(f"  added medium  cells={medium.n_cells:,}")

    plotter.add_mesh(
        iodine,
        color=_rgb01(cfg_iodine["color"]),
        opacity=float(args.iodine_opacity),
        smooth_shading=True,
        ambient=0.13, diffuse=0.78, specular=0.55,
        specular_power=50.0,
        name="iodine",
    )
    log(f"  added iodine  cells={iodine.n_cells:,}")

    plotter.reset_camera()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log("─" * 60)
    log(f"Exporting HTML → {out_path}")
    t1 = time.perf_counter()
    plotter.export_html(str(out_path))
    plotter.close()
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log(f"Done.  {size_mb:.1f} MB  ({time.perf_counter() - t1:.1f}s)")
    log("Open the file in a browser to rotate / zoom / pan the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
