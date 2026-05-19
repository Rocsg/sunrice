#!/usr/bin/env python3
"""
Marvel Aerench – Preprocessing pipeline.

Reads three 32-bit "norm" TIFFs (one per class) from an input directory
plus a shared 8-bit ``Outer.tif`` (exterior mask) and writes per class:

* Gas    (label 1) — connected components of ``Aerench_norm > 0``
                     (after masking out ``Outer == 255``).
* Méats  (label 2) — connected components of ``Meat_norm > 0`` …
* Water  (label 3) — connected components of ``Wat_norm > 0`` …

For every kept component a sub-volume is cropped to the component's
bbox (+ padding), voxels not belonging to the CC are set to a negative
sentinel, and marching cubes is run at iso-level ``0`` directly on the
float values — yielding a smooth sub-voxel-accurate surface without
the usual binary-mask staircase.

Outputs sit under ``aerench_output/labelN_<name>/component_NNNN.vtk``,
with a ``_done`` marker per label and per-component resume support.

Usage
-----
::

    python -m marvel_view.scripts.aerench_preprocess
    # or, after pip install -e .:
    marvel-aerench-preprocess
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── allow running the file directly without pip install ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import aerench_config as acfg
from marvel_view.preprocessing import load_float_volume, load_segmentation
from marvel_view.preprocessing.connected_components import label_components
from marvel_view.preprocessing.meshing import mask_to_mesh
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.aerench_preprocess")

_DONE_MARKER = "_done"

# Negative sentinel used to "blank out" voxels inside a CC's bbox that
# don't belong to the current CC.  Must be < 0 so marching cubes at
# level=0 stays inside the actual CC.
_OUTSIDE_VALUE = -1.0


# ──────────────────────────────── helpers ─────────────────────────────────────


def _gb(arr: np.ndarray) -> float:
    return arr.nbytes / 1024 ** 3


def _padded_bbox(
    sl: Tuple[slice, slice, slice], shape: Tuple[int, int, int], pad: int
) -> Tuple[Tuple[slice, slice, slice], Tuple[int, int, int]]:
    """Return ``(padded_slice_tuple, offset)`` for a single component bbox."""
    z0 = max(0, sl[0].start - pad)
    y0 = max(0, sl[1].start - pad)
    x0 = max(0, sl[2].start - pad)
    z1 = min(shape[0], sl[0].stop + pad)
    y1 = min(shape[1], sl[1].stop + pad)
    x1 = min(shape[2], sl[2].stop + pad)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1)), (z0, y0, x0)


def _allocate_component_budget(
    voxel_counts: List[int], total_budget: Optional[int]
) -> List[Optional[int]]:
    """Distribute a label's face budget across components ∝ voxel_count**(2/3)."""
    if not voxel_counts or not total_budget or total_budget <= 0:
        return [None] * len(voxel_counts)
    weights = np.asarray(voxel_counts, dtype=np.float64) ** (2.0 / 3.0)
    total_w = float(weights.sum())
    if total_w <= 0:
        return [None] * len(voxel_counts)
    shares = weights / total_w
    targets = np.maximum(1, np.round(shares * total_budget)).astype(np.int64)
    return [int(x) for x in targets]


# ──────────────────────────────── worker ──────────────────────────────────────


def _mesh_component_worker(
    sub_volume: np.ndarray,
    offset: Tuple[int, int, int],
    spacing: tuple,
    smooth_iter: int,
    gaussian_sigma: float,
    target_faces: Optional[int],
    decimate_fraction: float,
    out_path: str,
    step_size: Optional[int],
    level: float,
) -> Tuple[str, Optional[int], float]:
    """Mesh one float sub-volume (marching cubes at ``level``)."""
    t = time.perf_counter()
    mesh = mask_to_mesh(
        sub_volume,
        spacing=spacing,
        smooth_iter=smooth_iter,
        decimate_fraction=decimate_fraction,
        level=level,
        gaussian_sigma=gaussian_sigma,
        offset=offset,
        target_faces=target_faces,
        step_size=step_size,
    )
    elapsed = time.perf_counter() - t
    if mesh is None:
        return out_path, None, elapsed
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(p))
    return out_path, int(mesh.ncells), elapsed


# ──────────────────────────────── per-label dispatcher ───────────────────────


def _process_aerench_label(
    label_id: int,
    cfg: dict,
    input_dir: Path,
    outer_mask: Optional[np.ndarray],
    out_dir: Path,
    spacing: tuple,
    smooth_iter: int,
    decimate_fraction: float,
    resume: bool,
    max_components: int | None,
    num_workers: int,
    step_size: Optional[int],
    bbox_pad: int,
    level: float,
) -> None:
    """Process a single label end-to-end."""
    name = cfg["name"]
    input_file = cfg.get("input_file")
    if not input_file:
        logger.warning("Label %d (%s) has no input_file — skipping.", label_id, name)
        return
    input_path = input_dir / input_file

    min_voxels = int(cfg.get("min_voxels", acfg.DEFAULT_MIN_VOXELS))
    target_faces: Optional[int] = cfg.get("face_budget")
    raw_sigma = cfg.get("gaussian_sigma", 0.0)
    if isinstance(raw_sigma, (tuple, list)):
        gaussian_sigma = tuple(float(s) for s in raw_sigma)
        sigma_repr = "(" + ", ".join(f"{s:.2f}" for s in gaussian_sigma) + ")"
    else:
        gaussian_sigma = float(raw_sigma)
        sigma_repr = f"{gaussian_sigma:.2f}"

    logger.info("━" * 64)
    logger.info("Label %d  ·  %s  ·  input=%s  ·  level=%.2f  ·  sigma=%s",
                label_id, name, input_path.name, level, sigma_repr)
    if target_faces:
        logger.info("  Face budget: %d triangles", target_faces)

    label_dir = out_dir / f"label{label_id}_{name}"
    label_dir.mkdir(parents=True, exist_ok=True)
    done_marker = label_dir / _DONE_MARKER

    if resume and done_marker.exists():
        n_vtk = len(list(label_dir.glob("*.vtk")))
        logger.info("  [SKIP] _done marker found — %d VTK file(s) on disk.", n_vtk)
        return

    t_label = time.perf_counter()
    volume = load_float_volume(input_path)

    # Mask out the outside of the sample (Outer.tif == 255) so it never
    # contributes to either the iso-surface or the connected components.
    if outer_mask is not None:
        if outer_mask.shape != volume.shape:
            raise ValueError(
                f"Outer mask shape {outer_mask.shape} does not match "
                f"volume shape {volume.shape} for label {label_id}."
            )
        n_outside = int((outer_mask == 255).sum())
        logger.info(
            "  Masking out %d Outer==255 voxel(s)  (%.2f%% of volume).",
            n_outside, 100.0 * n_outside / volume.size,
        )
        # Force these voxels well below the iso-level so they never count
        # as inside, no matter what value they had in the float TIFF.
        volume[outer_mask == 255] = _OUTSIDE_VALUE

    logger.info("  Building binary mask (volume > %.2f) …", level)
    t = time.perf_counter()
    binary = (volume > level).view(np.uint8) if (volume > level).dtype == np.bool_ \
        else (volume > level).astype(np.uint8)
    n_pos = int(np.count_nonzero(binary))
    logger.info(
        "  └─ %d positive voxels  (%.1f s, mask %.1f GiB)",
        n_pos, time.perf_counter() - t, _gb(binary),
    )
    if n_pos == 0:
        logger.warning("  Label %d has no positive voxels — skipping.", label_id)
        done_marker.touch()
        return

    labeled, n_cc = label_components(binary)
    del binary

    logger.info("  Sizing components (bincount) …")
    sizes = np.bincount(labeled.ravel())
    bboxes = ndimage.find_objects(labeled, max_label=n_cc)

    candidates: List[Tuple[int, int]] = [
        (int(sizes[cid]), cid)
        for cid in range(1, n_cc + 1)
        if int(sizes[cid]) >= min_voxels and bboxes[cid - 1] is not None
    ]
    candidates.sort(reverse=True)
    logger.info(
        "  %d / %d components pass min_voxels=%d filter.",
        len(candidates), n_cc, min_voxels,
    )

    if not candidates:
        logger.warning("  No components left after filtering — skipping.")
        done_marker.touch()
        return

    if max_components is not None and len(candidates) > max_components:
        logger.info("  Limiting to %d largest components.", max_components)
        candidates = candidates[:max_components]

    voxel_counts = [vx for vx, _ in candidates]
    per_component_budget = _allocate_component_budget(voxel_counts, target_faces)
    if target_faces:
        logger.info(
            "  Budget allocation  →  min/median/max per CC: %d / %d / %d",
            int(min(per_component_budget)),
            int(np.median(per_component_budget)),
            int(max(per_component_budget)),
        )

    total = len(candidates)
    width = len(str(total))
    logger.info("  Meshing %d component(s) with %d worker(s) …", total, num_workers)

    jobs: List[tuple] = []
    n_skip = 0
    for i, (_vx, cid) in enumerate(candidates):
        out_path = label_dir / f"component_{i:04d}.vtk"
        if resume and out_path.exists():
            n_skip += 1
            continue
        padded_sl, offset = _padded_bbox(
            bboxes[cid - 1], labeled.shape, pad=bbox_pad,
        )
        # Crop the float volume to the padded bbox.  Keep one voxel of
        # *negative* boundary around the CC (via a 1-voxel dilation of
        # the CC mask) so marching cubes can interpolate the iso-surface
        # smoothly against those negative values.  Everything outside
        # that dilated mask (other CCs, far-away negative regions, and
        # Outer==255 voxels) is overwritten with a negative sentinel so
        # it can never contribute to this component's surface.
        sub_volume = volume[padded_sl].copy()
        cc_mask = (labeled[padded_sl] == cid)
        keep_mask = ndimage.binary_dilation(cc_mask, iterations=1)
        sub_volume[~keep_mask] = _OUTSIDE_VALUE
        jobs.append((i, sub_volume, offset, str(out_path), per_component_budget[i]))

    if n_skip:
        logger.info("  Skipping %d component(s) already on disk (resume).", n_skip)

    del labeled  # free the int32 label array before spawning workers
    del volume   # free the float32 source volume

    n_ok = n_skip
    if num_workers <= 1 or len(jobs) <= 1:
        for i, sub_volume, offset, out_path, budget in jobs:
            tag = f"    [{i + 1:{width}}/{total}]"
            path, n_faces, elapsed = _mesh_component_worker(
                sub_volume=sub_volume,
                offset=offset,
                spacing=spacing,
                smooth_iter=smooth_iter,
                gaussian_sigma=gaussian_sigma,
                target_faces=budget,
                decimate_fraction=decimate_fraction,
                out_path=out_path,
                step_size=step_size,
                level=level,
            )
            if n_faces is None:
                logger.warning("%s meshing failed — skipping.", tag)
                continue
            logger.info("%s %s  (%d faces, %.1f s)",
                        tag, Path(path).name, n_faces, elapsed)
            n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _mesh_component_worker,
                    sub_volume, offset, spacing, smooth_iter,
                    gaussian_sigma, budget, decimate_fraction, out_path,
                    step_size, level,
                ): i
                for (i, sub_volume, offset, out_path, budget) in jobs
            }
            for fut in as_completed(futures):
                i = futures[fut]
                tag = f"    [{i + 1:{width}}/{total}]"
                try:
                    path, n_faces, elapsed = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s worker crashed: %s", tag, exc)
                    continue
                if n_faces is None:
                    logger.warning("%s meshing failed — skipping.", tag)
                    continue
                logger.info("%s %s  (%d faces, %.1f s)",
                            tag, Path(path).name, n_faces, elapsed)
                n_ok += 1

    elapsed = time.perf_counter() - t_label
    logger.info("  Meshed %d / %d components in %.1f s.", n_ok, total, elapsed)
    if n_ok > 0:
        done_marker.touch()
        logger.info("  Label %d done.  (_done marker written)", label_id)


# ──────────────────────────────── CLI ─────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aerench preprocessing: per-class 32-bit norm TIFFs → VTK meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", "-i", default=str(acfg.DEFAULT_INPUT_DIR),
                   help="Directory containing the per-class *_norm.tif files.")
    p.add_argument("--outer", default=str(acfg.DEFAULT_OUTER_PATH),
                   help="8-bit Outer.tif mask (255 = outside the sample, excluded).")
    p.add_argument("--output-dir", "-o", default=str(acfg.DEFAULT_OUTPUT_DIR),
                   help="Root directory for VTK output.")
    p.add_argument("--spacing", nargs=3, type=float, metavar=("Z", "Y", "X"),
                   default=list(acfg.DEFAULT_SPACING),
                   help="Voxel spacing in z y x order.")
    p.add_argument("--bbox-pad", type=int, default=acfg.DEFAULT_BBOX_PAD,
                   help="Voxels of padding added around each component's bbox.")
    p.add_argument("--level", type=float, default=acfg.DEFAULT_LEVEL,
                   help="Marching-cubes iso-level on the float volume.")
    p.add_argument("--smooth-iter", type=int, default=acfg.SMOOTH_ITERATIONS,
                   help="Laplacian smoothing iterations.")
    p.add_argument("--decimate", type=float, default=acfg.DECIMATE_FRACTION,
                   dest="decimate_fraction",
                   help="Fallback fraction of faces to keep (only when no face_budget).")
    p.add_argument("--workers", type=int,
                   default=(acfg.DEFAULT_NUM_WORKERS or 0),
                   help="Parallel workers; 0 = cpu_count, 1 = serial.")
    p.add_argument("--step-size", type=int, default=0, metavar="N",
                   help="Marching-cubes stride.  0 = auto from face_budget.")
    p.add_argument("--labels", nargs="+", type=int, default=None,
                   help="Restrict to these label IDs (default: all configured).")
    p.add_argument("--max-components", type=int, default=None, metavar="N",
                   help="Mesh only the N largest components per label.")
    p.add_argument("--resume", action="store_true",
                   help="Skip labels / components already on disk.")
    p.add_argument("--no-outer", action="store_true",
                   help="Skip loading Outer.tif (process the full float volume).")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir  = Path(args.input_dir)
    outer_path = Path(args.outer)
    out_dir    = Path(args.output_dir)
    spacing    = tuple(args.spacing)
    label_ids  = args.labels or list(acfg.LABEL_CONFIG.keys())

    num_workers = int(args.workers) if args.workers and args.workers > 0 \
        else (os.cpu_count() or 1)
    step_size: Optional[int] = (
        int(args.step_size) if args.step_size and args.step_size > 0 else None
    )

    logger.info("═" * 64)
    logger.info("Marvel Aerench – Preprocessing")
    logger.info("  Input dir    : %s", input_dir.resolve())
    logger.info("  Outer mask   : %s", outer_path if not args.no_outer else "(disabled)")
    logger.info("  Output dir   : %s", out_dir.resolve())
    logger.info("  Spacing      : %s", spacing)
    logger.info("  BBox padding : %d", args.bbox_pad)
    logger.info("  MC level     : %.3f", args.level)
    logger.info("  Labels       : %s", label_ids)
    logger.info("  Workers      : %d", num_workers)
    logger.info("  Step size    : %s", step_size or "auto")
    logger.info("═" * 64)

    outer_mask: Optional[np.ndarray] = None
    if not args.no_outer:
        if not outer_path.exists():
            logger.warning(
                "Outer mask %s not found — running without exterior masking.",
                outer_path,
            )
        else:
            outer_mask = load_segmentation(outer_path)
            logger.info(
                "Outer mask loaded — %d voxels at 255 (%.2f%% of volume).",
                int((outer_mask == 255).sum()),
                100.0 * (outer_mask == 255).sum() / outer_mask.size,
            )

    t_total = time.perf_counter()
    for lid in label_ids:
        if lid not in acfg.LABEL_CONFIG:
            logger.warning("Label %d not in aerench LABEL_CONFIG — skipping.", lid)
            continue
        _process_aerench_label(
            label_id=lid,
            cfg=acfg.LABEL_CONFIG[lid],
            input_dir=input_dir,
            outer_mask=outer_mask,
            out_dir=out_dir,
            spacing=spacing,
            smooth_iter=args.smooth_iter,
            decimate_fraction=args.decimate_fraction,
            resume=args.resume,
            max_components=args.max_components,
            num_workers=num_workers,
            step_size=step_size,
            bbox_pad=int(args.bbox_pad),
            level=float(args.level),
        )

    logger.info("═" * 64)
    logger.info(
        "All done in %.1f s.  Output: %s",
        time.perf_counter() - t_total, out_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
