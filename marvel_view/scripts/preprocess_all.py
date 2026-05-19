#!/usr/bin/env python3
"""
Marvel View – Preprocessing pipeline.

Reads a segmentation TIFF, extracts surfaces for every configured label,
and saves them as VTK files on disk for later interactive rendering.

Pipeline steps per label
------------------------
  full_mask   : [1] marching cubes
  interior_cc : [1] CC analysis  →  [2] filter (interior only)  →  [3] mesh
  all_cc      : [1] CC analysis  →  [2] split components (bbox-cropped)
                                 →  [3] parallel mesh each component

Face-budget allocation
----------------------
A global face budget (``--face-budget``, default: ``DEFAULT_FACE_BUDGET``
from the config) caps the total number of triangles across *all* labels.
Each label receives a share proportional to its voxel count; for
``all_cc`` labels the share is further distributed across components
proportional to ``voxel_count ** (2/3)`` (≈ surface area).  Individual
workers compute an effective decimation ratio once the post-marching-cubes
face count is known.  Per-label ``face_budget`` in ``LABEL_CONFIG`` or the
legacy ``decimate_fraction`` are respected when no budget is active.

Checkpointing and resume
------------------------
* After finishing a label entirely, a marker ``_done`` is written to the
  label's output directory.  Subsequent runs with ``--resume`` skip labels
  that carry this marker.
* For the ``interior_cc`` strategy (typically the expensive background
  label), the filtered binary mask is saved as ``_ckpt_mask.npy`` after
  step 2.  On re-run, steps 1-2 are skipped and the cached mask is loaded
  directly.
* For ``all_cc``, individual component VTK files that already exist are
  skipped when ``--resume`` is active.

Usage
-----
Direct::

    python -m marvel_view.scripts.preprocess_all

After ``pip install -e .``::

    marvel-preprocess

Key options
-----------
  -i / --image          Path to the .tiff segmentation file.
  -o / --output-dir     Directory for VTK output (created if absent).
  --spacing Z Y X       Physical voxel size (default: 1 1 1).
  --smooth-iter N       Laplacian smoothing iterations (default: 80).
  --decimate F          Fallback face-keep fraction 0-1 when no face budget.
  --face-budget N       Total triangle budget across all labels.
  --workers N           Parallel workers for all_cc meshing.
  --labels 1 3 5        Process only these label IDs (default: all).
  --max-components N    Cap on components to mesh for all_cc labels.
  --resume              Skip labels / components already done on disk.
  -v / --verbose        Enable DEBUG logging.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── allow running the file directly without pip install ───────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marvel_view import config
from marvel_view.preprocessing import (
    load_segmentation,
    label_components,
    interior_mask,
    split_components,
    mask_to_mesh,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marvel_view.preprocess")

# ──────────────────────────────── constants ───────────────────────────────────

_DONE_MARKER = "_done"           # written when a label is fully processed
_CKPT_MASK   = "_ckpt_mask.npy"  # cached filtered mask for interior_cc labels


# ──────────────────────────────── helpers ─────────────────────────────────────


def _gb(arr: np.ndarray) -> float:
    """Array RAM in GiB."""
    return arr.nbytes / 1024 ** 3


def _shape_gb(shape: tuple, dtype=np.int32) -> float:
    """Estimated RAM in GiB for an array of given shape and dtype."""
    return int(np.prod(shape)) * np.dtype(dtype).itemsize / 1024 ** 3


def _save_mesh(mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(path))
    logger.info("    ✓ Saved  %s  (%d faces)", path.name, mesh.ncells)


def _tick(label: str, t0: float) -> None:
    logger.info("  └─ %s  (%.1f s)", label, time.perf_counter() - t0)


def _allocate_component_budget(
    voxel_counts: List[int], total_budget: int
) -> List[int]:
    """Distribute ``total_budget`` across components ∝ voxel_count**(2/3)."""
    if not voxel_counts or total_budget <= 0:
        return [0] * len(voxel_counts)
    weights = np.asarray(voxel_counts, dtype=np.float64) ** (2.0 / 3.0)
    total_w = float(weights.sum())
    if total_w <= 0:
        return [0] * len(voxel_counts)
    shares = weights / total_w
    targets = np.maximum(1, np.round(shares * total_budget)).astype(np.int64)
    return [int(x) for x in targets]


def _allocate_label_budgets(
    volume: np.ndarray,
    label_ids: List[int],
    total_budget: Optional[int],
) -> Dict[int, Optional[int]]:
    """Resolve a per-label face budget.

    Priority per label:
      1. ``face_budget`` override in ``LABEL_CONFIG[label_id]``
      2. share of ``total_budget`` proportional to the label's voxel count
      3. ``None`` — fall back to ``decimate_fraction``
    """
    budgets: Dict[int, Optional[int]] = {}

    pooled_ids: List[int] = []
    pooled_voxels: List[int] = []

    for lid in label_ids:
        cfg = config.LABEL_CONFIG.get(lid)
        if cfg is None:
            budgets[lid] = None
            continue
        override = cfg.get("face_budget")
        if override is not None:
            budgets[lid] = int(override)
            continue
        if total_budget is None:
            budgets[lid] = None
            continue
        vox = int(np.count_nonzero(volume == lid))
        if vox == 0:
            budgets[lid] = None
            continue
        pooled_ids.append(lid)
        pooled_voxels.append(vox)

    if pooled_ids and total_budget is not None:
        total_v = sum(pooled_voxels)
        for lid, v in zip(pooled_ids, pooled_voxels):
            budgets[lid] = int(round(total_budget * v / total_v))

    return budgets


# ──────────────────────────────── parallel worker ────────────────────────────


def _mesh_component_worker(
    sub_mask: np.ndarray,
    offset: Tuple[int, int, int],
    spacing: tuple,
    smooth_iter: int,
    gaussian_sigma: float,
    target_faces: Optional[int],
    decimate_fraction: float,
    out_path: str,
    step_size: Optional[int],
) -> Tuple[str, Optional[int], float]:
    """Mesh one component and write it to disk.

    Returns ``(out_path, n_faces_or_None, elapsed_seconds)``.
    Runs in a subprocess; imports vedo/skimage lazily via ``mask_to_mesh``.
    """
    t = time.perf_counter()
    mesh = mask_to_mesh(
        sub_mask,
        spacing=spacing,
        smooth_iter=smooth_iter,
        decimate_fraction=decimate_fraction,
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


# ──────────────────────────────── per-strategy sub-pipelines ─────────────────


def _process_full_mask(
    mask: np.ndarray,
    label_dir: Path,
    spacing: tuple,
    smooth_iter: int,
    decimate_fraction: float,
    gaussian_sigma: float,
    target_faces: Optional[int],
    step_size: Optional[int],
) -> bool:
    logger.info(
        "  [Step 1/1] Marching cubes + smoothing  (target_faces=%s, step=%s) …",
        target_faces, step_size,
    )
    t = time.perf_counter()
    mesh = mask_to_mesh(
        mask,
        spacing=spacing,
        smooth_iter=smooth_iter,
        decimate_fraction=decimate_fraction,
        gaussian_sigma=gaussian_sigma,
        target_faces=target_faces,
        step_size=step_size,
    )
    if mesh is None:
        logger.warning("  Meshing failed (empty surface).")
        return False
    _tick(f"mesh ready  ({mesh.ncells} faces)", t)
    _save_mesh(mesh, label_dir / "mesh.vtk")
    return True


def _process_interior_cc(
    mask: np.ndarray,
    label_dir: Path,
    spacing: tuple,
    smooth_iter: int,
    decimate_fraction: float,
    gaussian_sigma: float,
    min_voxels: int,
    resume: bool,
    target_faces: Optional[int],
    step_size: Optional[int],
) -> bool:
    ckpt_path = label_dir / _CKPT_MASK

    if ckpt_path.exists():
        logger.info("  [Step 1+2] CHECKPOINT — loading %s …", ckpt_path.name)
        t = time.perf_counter()
        int_mask = np.load(ckpt_path)
        n_kept = int(np.count_nonzero(int_mask))
        _tick(f"mask loaded  ({n_kept:,} voxels,  {_gb(int_mask):.1f} GiB)", t)
    else:
        logger.info(
            "  [Step 1/3] CC analysis  (~%.1f GiB for labeled array) …",
            _shape_gb(mask.shape, np.int32),
        )
        t = time.perf_counter()
        labeled, n = label_components(mask)
        del mask
        _tick(f"{n:,} components found", t)

        logger.info(
            "  [Step 2/3] Filtering interior components  (min_voxels=%d) …",
            min_voxels,
        )
        t = time.perf_counter()
        int_mask = interior_mask(labeled, n, min_voxels=min_voxels)
        del labeled
        _tick("filter done", t)

        if int(np.count_nonzero(int_mask)) == 0:
            logger.warning("  No interior components survived — skipping label.")
            return False

        logger.info(
            "  [CHECKPOINT] Saving %s  (%.1f GiB) …",
            ckpt_path.name, _gb(int_mask),
        )
        t = time.perf_counter()
        np.save(ckpt_path, int_mask)
        _tick("checkpoint saved", t)

    logger.info(
        "  [Step 3/3] Marching cubes + smoothing  (target_faces=%s, step=%s) …",
        target_faces, step_size,
    )
    t = time.perf_counter()
    mesh = mask_to_mesh(
        int_mask,
        spacing=spacing,
        smooth_iter=smooth_iter,
        decimate_fraction=decimate_fraction,
        gaussian_sigma=gaussian_sigma,
        target_faces=target_faces,
        step_size=step_size,
    )
    if mesh is None:
        logger.warning("  Meshing failed (empty surface).")
        return False
    _tick(f"mesh ready  ({mesh.ncells} faces)", t)
    _save_mesh(mesh, label_dir / "mesh.vtk")
    return True


def _process_all_cc(
    mask: np.ndarray,
    label_dir: Path,
    spacing: tuple,
    smooth_iter: int,
    decimate_fraction: float,
    gaussian_sigma: float,
    min_voxels: int,
    max_components: int | None,
    resume: bool,
    target_faces: Optional[int],
    num_workers: int,
    step_size: Optional[int],
) -> bool:
    logger.info(
        "  [Step 1/3] CC analysis  (~%.1f GiB for labeled array) …",
        _shape_gb(mask.shape, np.int32),
    )
    t = time.perf_counter()
    labeled, n = label_components(mask)
    del mask
    _tick(f"{n:,} components found", t)

    logger.info(
        "  [Step 2/3] Splitting components (bbox crop)  (min_voxels=%d, max=%s) …",
        min_voxels,
        str(max_components) if max_components else "all",
    )
    t = time.perf_counter()
    components = split_components(
        labeled, n, min_voxels=min_voxels, max_components=max_components
    )
    del labeled
    _tick(f"{len(components)} components to mesh", t)

    if not components:
        logger.warning("  No valid components — skipping label.")
        return False

    # Allocate per-component face budget when one is active.
    if target_faces is not None and target_faces > 0:
        voxel_counts = [vx for _, _, vx in components]
        per_component_budget: List[Optional[int]] = [
            int(x) for x in _allocate_component_budget(voxel_counts, target_faces)
        ]
        logger.info(
            "  Budget: %d faces total  →  min/median/max per component: %d / %d / %d",
            target_faces,
            int(min(per_component_budget)),
            int(np.median(per_component_budget)),
            int(max(per_component_budget)),
        )
    else:
        per_component_budget = [None] * len(components)

    total = len(components)
    width = len(str(total))
    logger.info(
        "  [Step 3/3] Meshing %d component(s) with %d worker(s) …",
        total, num_workers,
    )

    # Build job list, skipping already-done files in resume mode.
    jobs: List[tuple] = []
    n_skip = 0
    for i, (sub_mask, offset, _vx) in enumerate(components):
        out_path = label_dir / f"component_{i:04d}.vtk"
        if resume and out_path.exists():
            n_skip += 1
            continue
        jobs.append((i, sub_mask, offset, str(out_path), per_component_budget[i]))

    if n_skip:
        logger.info("  Skipping %d component(s) already on disk (resume).", n_skip)

    n_ok = n_skip
    del components  # free references before spawning workers

    if num_workers <= 1 or len(jobs) <= 1:
        for i, sub_mask, offset, out_path, budget in jobs:
            tag = f"    [{i + 1:{width}}/{total}]"
            path, n_faces, elapsed = _mesh_component_worker(
                sub_mask=sub_mask,
                offset=offset,
                spacing=spacing,
                smooth_iter=smooth_iter,
                gaussian_sigma=gaussian_sigma,
                target_faces=budget,
                decimate_fraction=decimate_fraction,
                out_path=out_path,
                step_size=step_size,
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
                    sub_mask, offset, spacing, smooth_iter,
                    gaussian_sigma, budget, decimate_fraction, out_path,
                    step_size,
                ): i
                for (i, sub_mask, offset, out_path, budget) in jobs
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

    logger.info("  Meshed %d / %d components.", n_ok, total)
    return n_ok > 0


# ──────────────────────────────── top-level dispatcher ───────────────────────


def _process_label(
    volume: np.ndarray,
    label_id: int,
    cfg: dict,
    out_dir: Path,
    spacing: tuple,
    smooth_iter: int,
    decimate_fraction: float,
    resume: bool,
    max_components: int | None,
    target_faces: Optional[int],
    num_workers: int,
    step_size: Optional[int],
) -> None:
    label_name     = cfg["name"]
    processing     = cfg["processing"]
    min_voxels     = cfg.get("min_voxels", config.DEFAULT_MIN_VOXELS)
    label_smooth   = cfg.get("smooth_iter", smooth_iter)
    label_decimate = cfg.get("decimate_fraction", decimate_fraction)
    label_gaussian = cfg.get("gaussian_sigma", 0.0)
    label_step     = cfg.get("step_size", step_size)

    logger.info("━" * 64)
    logger.info("Label %d  ·  %s  ·  strategy: %s", label_id, label_name, processing)
    if target_faces is not None:
        logger.info("  Face budget: %d triangles", target_faces)

    label_dir = out_dir / f"label{label_id}_{label_name}"
    label_dir.mkdir(parents=True, exist_ok=True)
    done_marker = label_dir / _DONE_MARKER

    if resume and done_marker.exists():
        vtk_files = sorted(label_dir.glob("*.vtk"))
        logger.info(
            "  [SKIP] _done marker found — %d VTK file(s) on disk.  "
            "Delete %s to reprocess.",
            len(vtk_files), done_marker,
        )
        return

    logger.info("  Extracting label %d mask …", label_id)
    t = time.perf_counter()
    # Bool and uint8 share the same itemsize; skip a redundant copy when safe.
    eq = np.equal(volume, label_id)
    mask = eq.view(np.uint8) if eq.dtype == np.bool_ else eq.astype(np.uint8)
    n_total = int(np.count_nonzero(mask))
    logger.info(
        "  Voxels: %d  (mask %.1f GiB)  [%.1f s]",
        n_total, _gb(mask), time.perf_counter() - t,
    )

    if n_total == 0:
        logger.warning("  Label %d not present in volume — skipping.", label_id)
        done_marker.touch()
        return

    t_label = time.perf_counter()
    ok = False

    if processing == "full_mask":
        ok = _process_full_mask(
            mask, label_dir, spacing, label_smooth, label_decimate,
            label_gaussian, target_faces, label_step,
        )
    elif processing == "interior_cc":
        ok = _process_interior_cc(
            mask, label_dir, spacing, label_smooth, label_decimate,
            label_gaussian, min_voxels, resume, target_faces, label_step,
        )
    elif processing == "all_cc":
        ok = _process_all_cc(
            mask, label_dir, spacing, label_smooth, label_decimate,
            label_gaussian, min_voxels, max_components, resume,
            target_faces, num_workers, label_step,
        )
    else:
        raise ValueError(f"Unknown processing strategy: {processing!r}")

    elapsed = time.perf_counter() - t_label
    if ok:
        done_marker.touch()
        logger.info("  Label %d done in %.1f s.  (_done marker written)", label_id, elapsed)
    else:
        logger.warning(
            "  Label %d finished with errors after %.1f s — no _done marker written.",
            label_id, elapsed,
        )


# ──────────────────────────────── CLI ─────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess a segmentation TIFF and save surface meshes as VTK files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", "-i",
        default=str(config.DEFAULT_IMAGE_PATH),
        help="Path to the multipage TIFF segmentation file.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=str(config.DEFAULT_OUTPUT_DIR),
        help="Root directory for VTK output.",
    )
    p.add_argument(
        "--spacing",
        nargs=3, type=float, metavar=("Z", "Y", "X"),
        default=list(config.DEFAULT_SPACING),
        help="Voxel spacing in z y x order.",
    )
    p.add_argument(
        "--smooth-iter",
        type=int, default=config.SMOOTH_ITERATIONS,
        help="Laplacian smoothing iterations (per-label config may override).",
    )
    p.add_argument(
        "--decimate",
        type=float, default=config.DECIMATE_FRACTION,
        dest="decimate_fraction",
        help="Fallback fraction of faces to keep when no face budget is active.",
    )
    p.add_argument(
        "--face-budget",
        type=int, default=(config.DEFAULT_FACE_BUDGET or 0),
        dest="face_budget",
        help=(
            "Total triangle budget across all labels.  Distributed per label "
            "by voxel count, then per component by voxel_count**(2/3).  "
            "Set to 0 (or negative) to disable and use decimate_fraction."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=(config.DEFAULT_NUM_WORKERS or 0),
        help=(
            "Parallel worker processes for all_cc meshing.  "
            "0 = use os.cpu_count(); 1 = serial."
        ),
    )
    p.add_argument(
        "--step-size",
        type=int, default=0, metavar="N",
        help=(
            "Marching-cubes stride.  0 (default) → auto-pick from face budget "
            "so MC emits ~3× the target; 1 = full resolution; >1 = coarser/faster."
        ),
    )
    p.add_argument(
        "--labels",
        nargs="+", type=int, default=None,
        help="Process only these label IDs (default: all labels in config).",
    )
    p.add_argument(
        "--max-components",
        type=int, default=None, metavar="N",
        help="For all_cc labels: mesh only the N largest components.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip labels with a _done marker.  "
            "For all_cc, also skip individual VTK files that already exist.  "
            "For interior_cc, re-use _ckpt_mask.npy if present."
        ),
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

    image_path = Path(args.image)
    out_dir    = Path(args.output_dir)
    spacing    = tuple(args.spacing)
    label_ids  = args.labels or list(config.LABEL_CONFIG.keys())

    face_budget: Optional[int] = (
        args.face_budget if args.face_budget and args.face_budget > 0 else None
    )
    num_workers = int(args.workers) if args.workers and args.workers > 0 \
        else (os.cpu_count() or 1)
    step_size: Optional[int] = int(args.step_size) if args.step_size and args.step_size > 0 else None

    logger.info("═" * 64)
    logger.info("Marvel View – Preprocessing")
    logger.info("  Image          : %s", image_path)
    logger.info("  Output dir     : %s", out_dir.resolve())
    logger.info("  Spacing (z,y,x): %s", spacing)
    logger.info("  Labels         : %s", label_ids)
    logger.info("  Resume         : %s", args.resume)
    logger.info("  Max components : %s", args.max_components or "all")
    logger.info("  Face budget    : %s", face_budget or "off (decimate_fraction)")
    logger.info("  Workers        : %d", num_workers)
    logger.info("  Step size      : %s", step_size or "auto")
    logger.info("═" * 64)

    volume = load_segmentation(image_path)

    # Resolve a face budget per label once we know the volume.
    label_budgets = _allocate_label_budgets(volume, label_ids, face_budget)
    if face_budget:
        logger.info("Per-label face budget allocation:")
        for lid in label_ids:
            logger.info(
                "  label %d (%s): %s",
                lid,
                config.LABEL_CONFIG.get(lid, {}).get("name", "?"),
                label_budgets.get(lid),
            )

    t_total = time.perf_counter()
    for lid in label_ids:
        if lid not in config.LABEL_CONFIG:
            logger.warning("Label %d not in LABEL_CONFIG — skipping.", lid)
            continue
        _process_label(
            volume=volume,
            label_id=lid,
            cfg=config.LABEL_CONFIG[lid],
            out_dir=out_dir,
            spacing=spacing,
            smooth_iter=args.smooth_iter,
            decimate_fraction=args.decimate_fraction,
            resume=args.resume,
            max_components=args.max_components,
            target_faces=label_budgets.get(lid),
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
