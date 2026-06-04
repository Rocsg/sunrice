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
  Panel1: 280° (soft) → 0°/360° (soft)  back to Panel4

Transitions:
  - Variable-width cross-fade at ALL 4 boundaries: 0°, 100°, 190°, 280°
  - Width: 15° at periphery (r=900), 40° at centre (r=0), linear in r

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

    # Wrap-aware alpha at the 0°/360° boundary (Panel1 → Panel4).
    # Map angles in (180°, 360°) to negative values so the boundary at 0°
    # is treated symmetrically: e.g. 355° → -5°, 5° → +5°.
    d_1_4 = np.where(angles > 180.0, angles - 360.0, angles)
    a_1_4 = np.clip((d_1_4 + half_T) / T, 0.0, 1.0)  # 0 = Panel1 side, 1 = Panel4 side

    # In the wrap-transition zone around 0°/360°, a_2_1 is erroneously 0
    # (angle 0° is far from boundary 280°), but Panel1 is fully faded in there.
    # Force it to 1 so the product formula gives the correct fade-out via a_1_4.
    # In the wrap-transition zone (|d_1_4| < half_T):
    #   - Panel1 is fully faded-in from its left boundary → force a_2_1 = 1
    #   - Panel4 hasn't yet passed its right boundary (B_4_3) → force a_4_3 = 0
    in_wrap = np.abs(d_1_4) < half_T
    a_2_1 = np.where(in_wrap, 1.0, a_2_1)
    a_4_3 = np.where(in_wrap, 0.0, a_4_3)

    # ── Weight maps ───────────────────────────────────────────────────────────
    # Product formula: each panel fades in at its left boundary and out at its
    # right boundary.  Weights sum to 1 everywhere.

    # Panel4: soft start at 0° (from Panel1), soft end at B_4_3
    w4 = np.clip(a_1_4, 0.0, 1.0) * np.clip(1.0 - a_4_3, 0.0, 1.0)

    # Panel3: soft start at B_4_3, soft end at B_3_2
    w3 = np.clip(a_4_3, 0.0, 1.0) * np.clip(1.0 - a_3_2, 0.0, 1.0)

    # Panel2: soft start at B_3_2, soft end at B_2_1
    w2 = np.clip(a_3_2, 0.0, 1.0) * np.clip(1.0 - a_2_1, 0.0, 1.0)

    # Panel1: soft start at B_2_1, soft end at 0°/360°
    w1 = np.clip(a_2_1, 0.0, 1.0) * np.clip(1.0 - a_1_4, 0.0, 1.0)

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
