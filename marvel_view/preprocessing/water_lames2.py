"""
Water-descent "lame2" (V3) precomputation.

Parallel to ``water_lames.py`` (V2) — produces a separate set of cache
files (``lame2.vtp`` / ``lame2_meta.json`` / ``lame2_iso_cache/`` / …)
so the V2 outputs are preserved.

Per-column progression (new schedule)
-------------------------------------

For every water column we extract a sequence of *thick* iso-shells
parameterised by ``(d0, d1)`` with ``d0 > 0.1``, ``d0`` strictly
increasing and ``d1 > d0 + 0.1``.  Unlike V2, the schedule no longer
follows a central iso-distance; instead it has three phases:

* Phase A — *growth of d1*.  ``d0 = 0.1`` is held constant while
  ``d1`` grows from ``0.2`` to ``d_max_col - 0.1`` in increments of
  ``step_growth`` (default 2.0 distance units).  The shell starts as a
  thin envelope near the column wall and inflates inward.

* Phase B — *hold*.  ``d0`` and ``d1`` remain at their phase-A end
  values for a number of frames equal to ``hold_ratio * (N_A + N_C)``
  (default 0.064 → ≈ 6 % of the total when A and C share ≈ 47 % each).
  No marching cubes is performed; the last phase-A mesh is reused.

* Phase C — *growth of d0*.  ``d1 = d_max_col - 0.1`` is held while
  ``d0`` grows from ``0.1 + step_growth`` to ``d1 - 0.1`` in
  increments of ``step_growth``.  The shell shrinks inward, eating
  itself.

Other differences vs V2
-----------------------

* Total animation length is roughly doubled by using a larger
  ``phase_factor`` default.

* Glow envelope is short and only fires at the very start of phase A
  and the very end of phase C — the middle is glow-free.

* The viewer side picks up the new files via ``--lame2-vtp-cache``
  defaults, and bumps the actor specular property slightly.

All heavy imports (skimage, scipy, vtk) are deferred to ``build()`` so
this module stays cheap to import.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

# Reuse helpers from the V2 module to avoid duplication.
from . import water_lames as _v2

logger = logging.getLogger(__name__)


# ── defaults ─────────────────────────────────────────────────────────────

DEFAULT_N_SEEDS:               int   = 300
# Doubled w.r.t. V2 to accommodate the longer per-column lifetime.
DEFAULT_PHASE_FACTOR:          int   = 10
DEFAULT_STEP_GROWTH:           float = 2.0
DEFAULT_HOLD_RATIO:            float = 0.064      # B ≈ 6 % of total (≈ 47/6/47 split)
DEFAULT_TARGET_TRIS_PER_SHELL: int   = 1500
DEFAULT_SMOOTH_ITER:           int   = _v2.DEFAULT_SMOOTH_ITER
# Short glow envelope: 3 frames at start of A and 3 frames at end of C.
DEFAULT_FADE_FRAMES:           int   = 3
DEFAULT_KMEANS_ITERS:          int   = _v2.DEFAULT_KMEANS_ITERS
DEFAULT_SEED:                  int   = _v2.DEFAULT_SEED

_BASE_RGB     = np.array((140, 195, 255), dtype=np.float32)
_GLOW_PEAK_T  = 0.45


# ── Per-column schedule ──────────────────────────────────────────────────


def _build_schedule(
    d_max_col: float,
    step_growth: float,
    hold_ratio: float,
) -> tuple[list[tuple[float, float]], int, int, int]:
    """Return ``(frames, n_A, n_B, n_C)``.

    ``frames`` is the ordered list of ``(d0, d1)`` for phase A + phase C
    only (phase B reuses the last A frame).  Returns an empty list if
    the column cannot host any valid shell.
    """
    d_max_eff = float(d_max_col) - 0.1
    if d_max_eff < 0.3:
        return [], 0, 0, 0

    step = float(step_growth)
    if step <= 0.0:
        step = 0.5

    # Phase A: d0 = 0.1, d1 = 0.2 → d_max_eff
    d1_vals = list(np.arange(0.2, d_max_eff + 1e-6, step, dtype=np.float64))
    # Always end exactly at d_max_eff.
    if not d1_vals or abs(d1_vals[-1] - d_max_eff) > 1e-6:
        if d1_vals and d_max_eff - d1_vals[-1] < step * 0.25:
            d1_vals[-1] = d_max_eff
        else:
            d1_vals.append(d_max_eff)
    phase_a = [(0.1, float(d1)) for d1 in d1_vals]

    # Phase C: d1 = d_max_eff fixed, d0 = (0.1 + step) → d_max_eff - 0.1.
    d0_end = d_max_eff - 0.1
    if d0_end <= 0.1 + 1e-6:
        phase_c = []
    else:
        d0_vals = list(np.arange(0.1 + step, d0_end + 1e-6, step,
                                  dtype=np.float64))
        if not d0_vals or abs(d0_vals[-1] - d0_end) > 1e-6:
            if d0_vals and d0_end - d0_vals[-1] < step * 0.25:
                d0_vals[-1] = d0_end
            else:
                d0_vals.append(d0_end)
        phase_c = [(float(d0), d_max_eff) for d0 in d0_vals]

    n_A = len(phase_a)
    n_C = len(phase_c)
    n_B = int(round((n_A + n_C) * float(hold_ratio)))
    n_B = max(0, n_B)

    return phase_a + phase_c, n_A, n_B, n_C


# ── Glow envelope (V3) ───────────────────────────────────────────────────


def _glow_v3(
    f_local: int,
    n_A: int,
    n_B: int,
    n_C: int,
    fade: int,
) -> float:
    """Short glow at start of A and end of C only; middle is dark."""
    fade = max(1, int(fade))
    if f_local < n_A:
        # Distance from start of A.
        return float(np.clip(1.0 - f_local / fade, 0.0, 1.0))
    elif f_local < n_A + n_B:
        return 0.0
    else:
        # Distance from end of C.
        c_local = f_local - (n_A + n_B)         # 0..n_C-1
        return float(np.clip(1.0 - (n_C - 1 - c_local) / fade, 0.0, 1.0))


# ── ISO cache layout — separate dir, identical NPZ format ───────────────


def _shell_cache_path(cache_dir: Path, label: int, fi: int) -> Path:
    return cache_dir / f"L{int(label):04d}" / f"F{int(fi):04d}.npz"


def _shell_cache_save(cache_dir: Path, label: int, fi: int,
                      verts: np.ndarray, faces: np.ndarray) -> None:
    p = _shell_cache_path(cache_dir, label, fi)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p,
                        verts=verts.astype(np.float32),
                        faces=faces.astype(np.int64))


def _shell_cache_load(cache_dir: Path, label: int, fi: int):
    p = _shell_cache_path(cache_dir, label, fi)
    if not p.exists():
        return None, None
    with np.load(p) as z:
        return z["verts"].astype(np.float32), z["faces"].astype(np.int64)


def _cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "lame2_cache_meta.json"


def _read_cache_meta(cache_dir: Path) -> dict:
    p = _cache_meta_path(cache_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_cache_meta(cache_dir: Path, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_meta_path(cache_dir).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _cache_is_valid(cache_dir: Path, *, n_columns: int,
                    step_growth: float, hold_ratio: float,
                    target_tris: int) -> bool:
    m = _read_cache_meta(cache_dir)
    if not m:
        return False
    return (
        m.get("n_columns", -1) == int(n_columns)
        and abs(m.get("step_growth", -1.0) - float(step_growth)) < 1e-6
        and abs(m.get("hold_ratio", -1.0) - float(hold_ratio))   < 1e-6
        and m.get("target_tris_per_shell", -1) == int(target_tris)
    )


# ── Per-label worker ─────────────────────────────────────────────────────


def _process_label(
    *,
    label:        int,
    bbox,
    geoddist:     np.ndarray,
    label_vol:    np.ndarray,
    step_growth:  float,
    hold_ratio:   float,
    iso_cache_dir: Path,
    target_tris:  int,
    smooth_iter:  int,
):
    """Process one column: marching cubes per phase-A/C frame, save NPZ.

    Returns ``(label, n_A, n_B, n_C, n_geo, d_max_col)``.
    """
    pad = 1
    z0 = max(0, bbox[0].start - pad); z1 = min(geoddist.shape[0], bbox[0].stop + pad)
    y0 = max(0, bbox[1].start - pad); y1 = min(geoddist.shape[1], bbox[1].stop + pad)
    x0 = max(0, bbox[2].start - pad); x1 = min(geoddist.shape[2], bbox[2].stop + pad)
    offset = np.array((z0, y0, x0), dtype=np.float32)

    D_sub = geoddist [z0:z1, y0:y1, x0:x1]
    L_sub = label_vol[z0:z1, y0:y1, x0:x1]

    d_col_vals = D_sub[L_sub == label]
    d_col_vals = d_col_vals[(d_col_vals > 0) & np.isfinite(d_col_vals)]
    if d_col_vals.size == 0:
        return int(label), 0, 0, 0, 0, 0.0

    d_max_col = float(d_col_vals.max())
    frames, n_A, n_B, n_C = _build_schedule(d_max_col, step_growth, hold_ratio)
    if not frames:
        return int(label), 0, 0, 0, 0, d_max_col

    n_geo = 0
    for fi, (lo, hi) in enumerate(frames):
        F = (D_sub - lo) * (hi - D_sub)
        verts, faces = _v2._extract_shell_mesh(
            F, L_sub, label, target_tris, smooth_iter,
        )
        if verts is None or faces is None or faces.shape[0] == 0:
            continue
        verts = verts + offset
        _shell_cache_save(iso_cache_dir, label, fi, verts, faces)
        n_geo += 1

    return int(label), int(n_A), int(n_B), int(n_C), int(n_geo), d_max_col


# ── Main entry point ─────────────────────────────────────────────────────


def build_water_lame2(
    *,
    geoddist_path:   Path,
    bg_dist_path:    Path,
    object_path:     Path,
    crown_path:      Path,
    output_vtp:      Path,
    output_labels:   Path | None = None,
    output_meta:     Path | None = None,
    iso_cache_dir:   Path | None = None,
    rebuild_iso_cache: bool = False,
    iso_only:        bool  = False,
    n_seeds:         int   = DEFAULT_N_SEEDS,
    phase_factor:    int   = DEFAULT_PHASE_FACTOR,
    fps:             int   = 25,
    step_growth:     float = DEFAULT_STEP_GROWTH,
    hold_ratio:      float = DEFAULT_HOLD_RATIO,
    target_tris_per_shell: int = DEFAULT_TARGET_TRIS_PER_SHELL,
    smooth_iter:     int   = DEFAULT_SMOOTH_ITER,
    fade_frames:     int   = DEFAULT_FADE_FRAMES,
    kmeans_iters:    int   = DEFAULT_KMEANS_ITERS,
    seed:            int   = DEFAULT_SEED,
    n_workers:       int | None = None,
) -> dict:
    """Build the lame2 (V3) cache.  See module docstring."""
    import tifffile
    import vtk
    from vtk.util import numpy_support as nps  # type: ignore
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        import ctypes
        _libc = ctypes.CDLL("libc.so.6")
        def _malloc_trim() -> None: _libc.malloc_trim(0)
    except OSError:
        def _malloc_trim() -> None: return

    geoddist_path = Path(geoddist_path)
    bg_dist_path  = Path(bg_dist_path)
    object_path   = Path(object_path)
    crown_path    = Path(crown_path)
    output_vtp    = Path(output_vtp)
    if output_meta is None:
        output_meta = output_vtp.with_suffix("").with_name(
            output_vtp.stem + "_meta.json")
    if iso_cache_dir is None:
        iso_cache_dir = output_vtp.parent / "lame2_iso_cache"
    iso_cache_dir = Path(iso_cache_dir)

    logger.info("─" * 60)
    logger.info("Water lame2 (V3) build")
    logger.info("  geoddist     : %s", geoddist_path)
    logger.info("  bg_dist      : %s", bg_dist_path)
    logger.info("  object       : %s", object_path)
    logger.info("  crown        : %s", crown_path)
    logger.info("  output (.vtp): %s", output_vtp)
    logger.info("  n_seeds=%d  phase_factor=%d  step_growth=%.2f  hold_ratio=%.2f",
                n_seeds, phase_factor, step_growth, hold_ratio)
    logger.info("  target_tris_per_shell=%d  smooth_iter=%d  fade=%d  seed=%d",
                target_tris_per_shell, smooth_iter, fade_frames, seed)

    rng = np.random.default_rng(int(seed))

    # ── 1. Load + clean inputs ──────────────────────────────────────
    t0_all = time.perf_counter()
    geoddist    = _v2._clean_distance(_v2._load_float32(geoddist_path),
                                       name="geoddist")
    bg_dist     = _v2._clean_distance(_v2._load_float32(bg_dist_path),
                                       name="bg_dist")
    object_mask = _v2._load_uint8(object_path)
    crown_mask  = _v2._load_uint8(crown_path)
    if not (geoddist.shape == bg_dist.shape == object_mask.shape == crown_mask.shape):
        raise ValueError(
            f"Input volumes have mismatching shapes: "
            f"geoddist={geoddist.shape}, bg_dist={bg_dist.shape}, "
            f"object={object_mask.shape}, crown={crown_mask.shape}"
        )

    # ── 2. K-means seeds + watershed ────────────────────────────────
    seeds = _v2._kmeans_medoid_crown(crown_mask, n_seeds,
                                      n_iter=kmeans_iters, rng=rng)
    n_columns = int(seeds.shape[0])
    label_vol = _v2._geodesic_watershed(object_mask, geoddist, seeds)
    if output_labels is not None:
        Path(output_labels).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(output_labels), label_vol.astype(np.int32),
                         compression="zlib")
        logger.info("Labels TIFF written: %s", output_labels)

    # ── 3. Bounding boxes ───────────────────────────────────────────
    from scipy.ndimage import find_objects
    bboxes = find_objects(label_vol)
    active_labels = [(k + 1, b) for k, b in enumerate(bboxes) if b is not None]
    logger.info("Active labels with non-empty bbox: %d / %d",
                len(active_labels), n_columns)

    # ── 4. ISO cache validity check ─────────────────────────────────
    schedule_per_col: dict[int, tuple[int, int, int, float]] = {}
    # (n_A, n_B, n_C, d_max_col) per label.

    _iso_valid = (
        not rebuild_iso_cache
        and iso_cache_dir.exists()
        and _cache_is_valid(
            iso_cache_dir,
            n_columns=n_columns, step_growth=step_growth,
            hold_ratio=hold_ratio,
            target_tris=target_tris_per_shell,
        )
    )
    if _iso_valid:
        logger.info("Lame2 ISO cache valid (%s); skipping per-label MC.",
                    iso_cache_dir)
        # Recompute schedules from d_max_col so packing knows phase splits.
        for label, bbox in active_labels:
            pad = 1
            z0 = max(0, bbox[0].start - pad); z1 = min(geoddist.shape[0], bbox[0].stop + pad)
            y0 = max(0, bbox[1].start - pad); y1 = min(geoddist.shape[1], bbox[1].stop + pad)
            x0 = max(0, bbox[2].start - pad); x1 = min(geoddist.shape[2], bbox[2].stop + pad)
            d_col_vals = geoddist[z0:z1, y0:y1, x0:x1][
                label_vol[z0:z1, y0:y1, x0:x1] == label
            ]
            d_col_vals = d_col_vals[(d_col_vals > 0) & np.isfinite(d_col_vals)]
            if d_col_vals.size == 0:
                continue
            d_max_col = float(d_col_vals.max())
            _, n_A, n_B, n_C = _build_schedule(d_max_col, step_growth, hold_ratio)
            if n_A + n_C == 0:
                continue
            schedule_per_col[int(label)] = (n_A, n_B, n_C, d_max_col)
    else:
        if rebuild_iso_cache and iso_cache_dir.exists():
            import shutil
            shutil.rmtree(iso_cache_dir)
            logger.info("Lame2 ISO cache cleared (rebuild requested).")
        iso_cache_dir.mkdir(parents=True, exist_ok=True)

        n_workers_actual = _v2._resolve_n_workers(n_workers)
        logger.info("Per-label shell extraction on %d thread(s) …",
                    n_workers_actual)

        # Biggest bbox first (high-water-mark).
        def _bbox_vol(b):
            return ((b[0].stop - b[0].start)
                    * (b[1].stop - b[1].start)
                    * (b[2].stop - b[2].start))
        active_labels.sort(key=lambda lb: _bbox_vol(lb[1]), reverse=True)

        t0_phase = time.perf_counter()
        n_done = 0
        GC_EVERY = n_workers_actual
        with ThreadPoolExecutor(max_workers=n_workers_actual) as ex:
            futures = {
                ex.submit(
                    _process_label,
                    label=lb, bbox=bb,
                    geoddist=geoddist, label_vol=label_vol,
                    step_growth=step_growth, hold_ratio=hold_ratio,
                    iso_cache_dir=iso_cache_dir,
                    target_tris=target_tris_per_shell,
                    smooth_iter=smooth_iter,
                ): lb
                for lb, bb in active_labels
            }
            for fut in as_completed(futures):
                lb, n_A, n_B, n_C, n_geo, d_max_col = fut.result()
                n_done += 1
                if n_geo > 0:
                    schedule_per_col[int(lb)] = (n_A, n_B, n_C, d_max_col)
                logger.info(
                    "  [%4d/%d] L=%4d  d_max=%.2f  A=%3d  B=%3d  C=%3d  shells=%3d",
                    n_done, len(active_labels), lb,
                    d_max_col, n_A, n_B, n_C, n_geo,
                )
                if n_done % GC_EVERY == 0:
                    gc.collect()
                    _malloc_trim()
        logger.info("Per-label phase done in %.1f s",
                    time.perf_counter() - t0_phase)

        _write_cache_meta(iso_cache_dir, {
            "n_columns":             int(n_columns),
            "step_growth":           float(step_growth),
            "hold_ratio":            float(hold_ratio),
            "target_tris_per_shell": int(target_tris_per_shell),
            "smooth_iter":           int(smooth_iter),
            "seed":                  int(seed),
        })
        logger.info("Lame2 ISO cache written: %s", iso_cache_dir)

    if iso_only:
        logger.info("iso-only mode: stopping after Phase 1a.")
        return {}

    # ── 5. Compute per-column lifetimes + global Nstep ──────────────
    if not schedule_per_col:
        raise RuntimeError("No surviving shells; check inputs / thresholds.")
    lifetimes = {lb: (s[0] + s[1] + s[2])
                  for lb, s in schedule_per_col.items()}
    max_lifetime = max(lifetimes.values())
    Nstep = int(phase_factor) * int(max_lifetime)
    logger.info("Max per-column lifetime=%d  Nstep=%d  active_columns=%d",
                max_lifetime, Nstep, len(schedule_per_col))

    # ── 6. Per-column phase offset (uniform in [0, Nstep)) ──────────
    tstart = np.zeros(n_columns + 1, dtype=np.int32)
    active_cols = sorted(schedule_per_col.keys())
    tstart[active_cols] = rng.integers(
        0, Nstep, size=len(active_cols)
    ).astype(np.int32)

    # ── 7. Packing pass: walk every (label, animation-step) ─────────
    # First count totals — phase-B reuses last A mesh so its cells are
    # duplicated across n_B step-buckets.
    shell_index: list[tuple[int, int, int, int]] = []
    # (label, fi_cached, step_id, n_tris)
    total_tris  = 0
    total_verts = 0
    for label in active_cols:
        n_A, n_B, n_C, _ = schedule_per_col[label]
        lifetime = n_A + n_B + n_C
        ts = int(tstart[label])
        # Phase A
        for f in range(n_A):
            v, fa = _shell_cache_load(iso_cache_dir, int(label), f)
            if v is None or fa is None or fa.shape[0] == 0:
                continue
            step = (ts + f) % Nstep
            shell_index.append((int(label), f, int(step), int(fa.shape[0])))
            total_tris  += int(fa.shape[0])
            total_verts += int(v.shape[0])
        # Phase B — reuse the last A mesh, duplicated per step
        if n_B > 0 and n_A > 0:
            # Find the last A frame with a cached mesh.
            last_a = -1
            for f in range(n_A - 1, -1, -1):
                p = _shell_cache_path(iso_cache_dir, int(label), f)
                if p.exists():
                    last_a = f
                    break
            if last_a >= 0:
                v, fa = _shell_cache_load(iso_cache_dir, int(label), last_a)
                if v is not None and fa is not None and fa.shape[0] > 0:
                    for b in range(n_B):
                        step = (ts + n_A + b) % Nstep
                        shell_index.append(
                            (int(label), last_a, int(step), int(fa.shape[0])))
                        total_tris  += int(fa.shape[0])
                        total_verts += int(v.shape[0])
        # Phase C
        for c in range(n_C):
            f = n_A + c
            v, fa = _shell_cache_load(iso_cache_dir, int(label), f)
            if v is None or fa is None or fa.shape[0] == 0:
                continue
            step = (ts + n_A + n_B + c) % Nstep
            shell_index.append((int(label), f, int(step), int(fa.shape[0])))
            total_tris  += int(fa.shape[0])
            total_verts += int(v.shape[0])

    if total_tris == 0:
        raise RuntimeError("No triangles in any shell cache.")
    logger.info("Packing: total verts=%d  tris=%d  shells=%d",
                total_verts, total_tris, len(shell_index))

    merged_verts = np.empty((total_verts, 3), dtype=np.float32)
    merged_faces = np.empty((total_tris,  3), dtype=np.int64)
    merged_time  = np.empty(total_tris, dtype=np.int32)
    merged_col   = np.empty(total_tris, dtype=np.int32)
    merged_step  = np.empty(total_tris, dtype=np.int32)
    merged_rgb   = np.empty((total_tris, 3), dtype=np.uint8)

    voff = 0
    foff = 0
    # Cumulative frame offset per (label, animation-step) so we can
    # determine f_local (position inside the column's lifetime) for
    # glow lookups.  Walk again in the same order as above.
    for label in active_cols:
        n_A, n_B, n_C, _ = schedule_per_col[label]
        ts = int(tstart[label])

        def _emit(fi_cached: int, step: int, f_local: int):
            nonlocal voff, foff
            v, fa = _shell_cache_load(iso_cache_dir, int(label), fi_cached)
            if v is None or fa is None or fa.shape[0] == 0:
                return
            n_v = int(v.shape[0]); n_f = int(fa.shape[0])
            merged_verts[voff:voff + n_v] = v
            merged_faces[foff:foff + n_f] = fa + voff
            merged_time [foff:foff + n_f] = f_local
            merged_col  [foff:foff + n_f] = label
            merged_step [foff:foff + n_f] = step

            glow = _glow_v3(f_local, n_A, n_B, n_C, fade_frames)
            rgb = _BASE_RGB * (1.0 - _GLOW_PEAK_T * glow) \
                  + np.array((255.0, 255.0, 255.0), dtype=np.float32) \
                    * (_GLOW_PEAK_T * glow)
            merged_rgb[foff:foff + n_f] = np.clip(rgb, 0, 255).astype(np.uint8)

            voff += n_v
            foff += n_f

        # Phase A
        for f in range(n_A):
            _emit(f, (ts + f) % Nstep, f)
        # Phase B (reuse last A)
        if n_B > 0 and n_A > 0:
            last_a = -1
            for f in range(n_A - 1, -1, -1):
                if _shell_cache_path(iso_cache_dir, int(label), f).exists():
                    last_a = f
                    break
            if last_a >= 0:
                for b in range(n_B):
                    _emit(last_a, (ts + n_A + b) % Nstep, n_A + b)
        # Phase C
        for c in range(n_C):
            _emit(n_A + c, (ts + n_A + n_B + c) % Nstep, n_A + n_B + c)

    # Trim possibly-unused tail (some shells failed to load second time).
    merged_verts = merged_verts[:voff]
    merged_faces = merged_faces[:foff]
    merged_time  = merged_time [:foff]
    merged_col   = merged_col  [:foff]
    merged_step  = merged_step [:foff]
    merged_rgb   = merged_rgb  [:foff]

    gc.collect()
    _malloc_trim()

    # ── 8. Sort by step_id ──────────────────────────────────────────
    order = np.argsort(merged_step, kind="stable")
    merged_faces = merged_faces[order]
    merged_time  = merged_time [order]
    merged_col   = merged_col  [order]
    merged_step  = merged_step [order]
    merged_rgb   = merged_rgb  [order]
    del order

    # ── 9. Build packed PolyData ────────────────────────────────────
    pd = _v2._polydata_from_vf(merged_verts, merged_faces)
    cd = pd.GetCellData()
    for name, arr, vtype in (
        ("step_id",   merged_step, vtk.VTK_INT),
        ("column_id", merged_col,  vtk.VTK_INT),
        ("time_id",   merged_time, vtk.VTK_INT),
    ):
        a = nps.numpy_to_vtk(np.ascontiguousarray(arr), deep=True,
                             array_type=vtype)
        a.SetName(name)
        cd.AddArray(a)
    a_rgb = nps.numpy_to_vtk(np.ascontiguousarray(merged_rgb),
                             deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    a_rgb.SetName("rgb")
    a_rgb.SetNumberOfComponents(3)
    cd.AddArray(a_rgb)
    del merged_verts, merged_faces, merged_time, merged_col, merged_step, merged_rgb

    # ── 10. Write VTP + meta ────────────────────────────────────────
    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(output_vtp))
    w.SetInputData(pd)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToZLib()
    w.Write()
    logger.info("Lame2 .vtp written to %s (%.1f MB)",
                output_vtp, output_vtp.stat().st_size / 1e6)

    meta = {
        "version":              3,
        "fps":                  int(max(1, fps)),
        "n_steps":              int(Nstep),
        "max_lifetime":         int(max_lifetime),
        "n_columns":            int(n_columns),
        "n_active_columns":     int(len(active_cols)),
        "n_triangles_total":    int(total_tris),
        "phase_factor":         int(phase_factor),
        "step_growth":          float(step_growth),
        "hold_ratio":           float(hold_ratio),
        "target_tris_per_shell": int(target_tris_per_shell),
        "smooth_iter":          int(smooth_iter),
        "fade_frames":          int(fade_frames),
        "seed":                 int(seed),
        "tstart":               tstart.tolist(),
        "schedule":             {str(lb): list(s)
                                  for lb, s in schedule_per_col.items()},
    }
    Path(output_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(output_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Lame2 meta written to %s", output_meta)
    logger.info("Total build time: %.1f s", time.perf_counter() - t0_all)
    return meta
