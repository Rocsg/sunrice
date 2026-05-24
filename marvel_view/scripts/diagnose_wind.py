#!/usr/bin/env python3
"""
Diagnostic script for wind particle trajectory caches.

Loads all wind_o2 / wind_ch4 npz files (slow / med / fast) and prints
statistics about how much particles actually move across frames.

Run::

    python marvel_view/scripts/diagnose_wind.py
    # or:
    python marvel_view/scripts/diagnose_wind.py --all     # include per-frame table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from marvel_view.scripts.water_conductance.constants import (
    DEFAULT_WIND_O2_CACHES,
    DEFAULT_WIND_CH4_CACHES,
    WIND_SPEED_LABELS,
    WIND_SPEED_LEVELS,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 30, vmax: float = 1.0) -> str:
    filled = int(round(width * min(v / max(vmax, 1e-9), 1.0)))
    return "█" * filled + "░" * (width - filled)


def _analyse(path: Path, verbose: bool = False) -> None:
    if not path.exists():
        print(f"  ⚠  not found: {path}")
        return

    z = np.load(str(path), allow_pickle=True)
    import json as _json
    positions = z["positions"].astype(np.float32)    # (T, F, 3)
    alive     = z["alive"].astype(bool)              # (T, F)
    try:
        fps = int(_json.loads(str(z["meta"][0])).get("fps", 25))
    except Exception:
        fps = 25

    T, F, _ = positions.shape

    # Per-step displacement for every template.
    # diff[t, f] = distance moved from frame f to f+1 for template t
    diff      = np.diff(positions, axis=1)                   # (T, F-1, 3)
    step_dist = np.linalg.norm(diff, axis=2).astype(np.float32)  # (T, F-1)

    # Only count steps where the particle was alive at both endpoints.
    alive_both = alive[:, :-1] & alive[:, 1:]                # (T, F-1)
    step_dist_alive = np.where(alive_both, step_dist, np.nan)

    # Per-frame mean speed (vox/frame), averaging over alive templates.
    mean_speed_per_frame = np.nanmean(step_dist_alive, axis=0)  # (F-1,)

    # Overall stats.
    all_steps = step_dist_alive[alive_both]
    if len(all_steps) == 0:
        print("  ⚠  no alive steps found — dead templates?")
        return

    n_alive_steps = int(alive_both.sum())
    total_steps   = alive_both.size
    alive_frac    = 100.0 * n_alive_steps / max(total_steps, 1)

    spd_mean  = float(np.nanmean(step_dist_alive))
    spd_p5    = float(np.nanpercentile(step_dist_alive, 5))
    spd_p50   = float(np.nanpercentile(step_dist_alive, 50))
    spd_p95   = float(np.nanpercentile(step_dist_alive, 95))
    spd_max   = float(np.nanmax(step_dist_alive))

    # Total path length per template (alive steps only).
    total_path = np.nansum(step_dist_alive, axis=1)           # (T,)
    path_mean  = float(total_path.mean())
    path_max   = float(total_path.max())

    # Check frozen frames: fraction of alive steps with displacement < 0.01 vox.
    frozen_frac = 100.0 * float((step_dist_alive[alive_both] < 0.01).mean())

    print(f"  templates  : {T}  ×  {F} frames  (fps={fps})")
    print(f"  alive steps: {n_alive_steps:,}/{total_steps:,}  ({alive_frac:.1f}%)")
    print(f"  speed (vox/frame):  mean={spd_mean:.3f}  p5={spd_p5:.3f}"
          f"  median={spd_p50:.3f}  p95={spd_p95:.3f}  max={spd_max:.3f}")
    print(f"  total path / template:  mean={path_mean:.1f} vox  max={path_max:.1f} vox")
    print(f"  FROZEN steps (< 0.01 vox): {frozen_frac:.1f}%"
          + ("  ← PROBLEM!" if frozen_frac > 50 else "  ✓"))

    if verbose:
        # Print per-frame mean speed (binned to 20 rows for readability).
        n_rows = 20
        chunk  = max(1, (F - 1) // n_rows)
        vmax   = float(np.nanmax(mean_speed_per_frame))
        print(f"\n  Per-frame mean speed (binned × {chunk} frames each, max={vmax:.3f} vox/frm):")
        for row in range(n_rows):
            lo = row * chunk
            hi = min(lo + chunk, F - 1)
            v  = float(np.nanmean(mean_speed_per_frame[lo:hi]))
            print(f"    f{lo:4d}-{hi:<4d}  {v:6.3f}  {_bar(v, vmax=vmax)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose wind trajectory caches.")
    p.add_argument("--all", action="store_true",
                   help="Print per-frame speed table.")
    p.add_argument("--species", choices=["o2", "ch4", "both"], default="both")
    args = p.parse_args(argv)

    species_map = {
        "o2":  ("O₂",  DEFAULT_WIND_O2_CACHES),
        "ch4": ("CH₄", DEFAULT_WIND_CH4_CACHES),
    }
    targets = (["o2", "ch4"] if args.species == "both"
               else [args.species])

    for sp in targets:
        label, caches = species_map[sp]
        for i, (lbl, speed) in enumerate(zip(WIND_SPEED_LABELS, WIND_SPEED_LEVELS)):
            path = Path(caches[i])
            print(f"\n{'═'*60}")
            print(f"  {label}  [{lbl}]  speed={speed} vox/frm   {path.name}")
            print(f"{'─'*60}")
            _analyse(path, verbose=args.all)

    print(f"\n{'═'*60}")


if __name__ == "__main__":
    main()
