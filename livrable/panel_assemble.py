#!/usr/bin/env python3
"""
panel_assemble.py

Assembles 4 TIFF panel images into a single 360° composite by blending
in azimuth around the image centre.

Azimuth convention: 0° = top (North), clockwise, range [0°, 360°)

Panel layout  (soft boundaries are 10° past each cardinal point):
  Panel4:   0° (hard) → 100° (soft)   North hard-start, East+10°S soft-end
  Panel3: 100° (soft) → 190° (soft)   South+10°W soft-end
  Panel2: 190° (soft) → 280° (soft)   West+10°N soft-end
  Panel1: 280° (soft) → 360°/0° (hard)  sharp cut into Panel4

Transitions:
  - Variable-width cross-fade at 100°, 190°, 280° boundaries
  - Width: 15° at periphery (r=900), 40° at centre (r=0), linear in r
  - Hard cuts at 0° (Panel4 start = Panel1 end)

Dependencies: numpy, Pillow  (no VTK required)
"""

import numpy as np
from pathlib import Path
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = Path("/home/rfernandez/Data/Arize/Livrable_Arize")
OUTPUT_PATH = DATA_DIR / "Panel_assembled.tif"

IMG_SIZE  = 1800
CENTER    = IMG_SIZE // 2   # 900

T_PERIPH  = 15.0   # transition width (°) at periphery (r = CENTER)
T_CENTER  = 40.0   # transition width (°) at centre    (r = 0)

# Soft-boundary centres  (10° past each cardinal point, clockwise)
B_4_3 = 100.0    # Panel4 → Panel3  (East + 10° toward South)
B_3_2 = 190.0    # Panel3 → Panel2  (South + 10° toward West)
B_2_1 = 280.0    # Panel2 → Panel1  (West  + 10° toward North)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_panel(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    assert arr.shape == (IMG_SIZE, IMG_SIZE, 3), (
        f"Expected {IMG_SIZE}×{IMG_SIZE} RGB, got {arr.shape} for {path.name}"
    )
    return arr


def azimuth_map() -> np.ndarray:
    """Clockwise azimuth from North (top), range [0, 360°)."""
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dx = (xs - CENTER).astype(np.float32)
    dy = -(ys - CENTER).astype(np.float32)   # flip Y so up = positive
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def radius_map() -> np.ndarray:
    """Distance from centre, in pixels."""
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dx = (xs - CENTER).astype(np.float32)
    dy = (ys - CENTER).astype(np.float32)
    return np.sqrt(dx ** 2 + dy ** 2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading panels…")
    panels = {i: load_panel(DATA_DIR / f"Panel{i}.tif") for i in range(1, 5)}

    print("Building maps…")
    angles = azimuth_map()
    r      = radius_map()

    # Per-pixel transition width: 40° at centre, 15° at periphery
    T      = T_CENTER - (T_CENTER - T_PERIPH) * np.clip(r / CENTER, 0.0, 1.0)
    half_T = T / 2.0

    def alpha(B: float) -> np.ndarray:
        """Blend alpha at boundary B: 0 on incoming side, 1 on outgoing side."""
        return np.clip((angles - B + half_T) / T, 0.0, 1.0)

    a_4_3 = alpha(B_4_3)   # 0 = Panel4 side,  1 = Panel3 side
    a_3_2 = alpha(B_3_2)   # 0 = Panel3 side,  1 = Panel2 side
    a_2_1 = alpha(B_2_1)   # 0 = Panel2 side,  1 = Panel1 side

    # ── Weight maps ───────────────────────────────────────────────────────────
    # Product formula: each panel fades in through its left boundary and
    # fades out through its right boundary.  Weights sum to 1 in all
    # transition zones; the 350°–360° black gap has total weight 0.

    # Panel4: hard start at 0°, soft end at B_4_3
    w4 = np.clip(1.0 - a_4_3, 0.0, 1.0)

    # Panel3: soft start at B_4_3, soft end at B_3_2
    w3 = np.clip(a_4_3, 0.0, 1.0) * np.clip(1.0 - a_3_2, 0.0, 1.0)

    # Panel2: soft start at B_3_2, soft end at B_2_1
    w2 = np.clip(a_3_2, 0.0, 1.0) * np.clip(1.0 - a_2_1, 0.0, 1.0)

    # Panel1: soft start at B_2_1, hard cut at 360°/0° (sharp join with Panel4)
    w1 = np.clip(a_2_1, 0.0, 1.0)

    # ── Composite ─────────────────────────────────────────────────────────────
    print("Compositing…")
    result = (
          w4[:, :, None] * panels[4]
        + w3[:, :, None] * panels[3]
        + w2[:, :, None] * panels[2]
        + w1[:, :, None] * panels[1]
    )
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)

    print(f"Saving → {OUTPUT_PATH}")
    Image.fromarray(result).save(str(OUTPUT_PATH))
    print("Done.")


if __name__ == "__main__":
    main()
