#!/usr/bin/env python3
"""
Improve_roi.py

Reads an ImageJ polygon ROI, inserts Catmull-Rom intermediate points on
gently-curved segments, and preserves sharp corners (turn angle > --threshold,
default 40°).

The improved ROI is written next to the original as <stem>_improved.roi.

Usage:
    python livrable/Improve_roi.py [input.roi] [--threshold 40] [--interp 4]
"""

import argparse
import struct
from pathlib import Path

import numpy as np

# ── ImageJ ROI binary layout ──────────────────────────────────────────────────
MAGIC       = b'Iout'
HEADER_SIZE = 64          # BASE_HEADER_SIZE in ImageJ source

OFF_VERSION  = 4          # >H  unsigned short
OFF_TYPE     = 6          # B   unsigned byte
OFF_TOP      = 8          # >h  signed short
OFF_LEFT     = 10         # >h  signed short
OFF_BOTTOM   = 12         # >h  signed short
OFF_RIGHT    = 14         # >h  signed short
OFF_N_COORDS = 16         # >H  unsigned short
OFF_OPTIONS  = 50         # >h  signed short  (bit 128 = SUB_PIXEL_RESOLUTION)
OFF_HEADER2  = 60         # >i  signed int    (0 = absent)

SUB_PIXEL_FLAG = 128      # options bit: coordinates stored as float32

ROI_TYPE_NAMES = {
    0: 'Polygon', 1: 'Rect', 2: 'Oval', 3: 'Line',
    4: 'FreeLine', 5: 'PolyLine', 6: 'NoRoi',
    7: 'Freehand', 8: 'Traced', 9: 'Angle', 10: 'Point',
}


# ── ROI I/O ───────────────────────────────────────────────────────────────────

def read_roi(path: Path):
    """Return (meta dict, Nx2 float array of absolute coords, raw bytes)."""
    raw = path.read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"Not an ImageJ ROI file (bad magic): {path}")

    version   = struct.unpack_from('>H', raw, OFF_VERSION)[0]
    roi_type  = raw[OFF_TYPE]
    top       = struct.unpack_from('>h', raw, OFF_TOP)[0]
    left      = struct.unpack_from('>h', raw, OFF_LEFT)[0]
    bottom    = struct.unpack_from('>h', raw, OFF_BOTTOM)[0]
    right     = struct.unpack_from('>h', raw, OFF_RIGHT)[0]
    n         = struct.unpack_from('>H', raw, OFF_N_COORDS)[0]
    options   = struct.unpack_from('>h', raw, OFF_OPTIONS)[0]
    h2_offset = struct.unpack_from('>i', raw, OFF_HEADER2)[0]

    if n == 0:
        raise ValueError("ROI contains no coordinate points.")

    subpixel = bool(options & SUB_PIXEL_FLAG)

    if subpixel:
        # float32 layout at HEADER_SIZE: x[0..n-1] then y[0..n-1]
        x = np.array(struct.unpack_from(f'>{n}f', raw, HEADER_SIZE), dtype=float)
        y = np.array(struct.unpack_from(f'>{n}f', raw, HEADER_SIZE + 4 * n), dtype=float)
    else:
        # int16 layout at HEADER_SIZE: x[0..n-1] then y[0..n-1]
        x = np.array(struct.unpack_from(f'>{n}h', raw, HEADER_SIZE), dtype=float)
        y = np.array(struct.unpack_from(f'>{n}h', raw, HEADER_SIZE + 2 * n), dtype=float)

    # Stored as offsets from (left, top) → convert to absolute pixel coords
    points = np.column_stack([x + left, y + top])

    meta = dict(
        version=version, roi_type=roi_type,
        top=top, left=left, bottom=bottom, right=right,
        n=n, options=options, subpixel=subpixel, h2_offset=h2_offset,
    )
    return meta, points, raw


def write_roi(path: Path, meta: dict, points: np.ndarray, raw: bytes) -> None:
    """Write an ImageJ ROI file with updated polygon coordinates."""
    n        = len(points)
    subpixel = meta['subpixel']

    # New bounding box
    new_left   = int(np.floor(points[:, 0].min()))
    new_top    = int(np.floor(points[:, 1].min()))
    new_right  = int(np.ceil( points[:, 0].max()))
    new_bottom = int(np.ceil( points[:, 1].max()))

    # Relative coords
    xr = points[:, 0] - new_left
    yr = points[:, 1] - new_top

    # Build coordinate section
    if subpixel:
        coord_sec  = struct.pack(f'>{n}f', *xr.astype(np.float32).tolist())
        coord_sec += struct.pack(f'>{n}f', *yr.astype(np.float32).tolist())
        coord_size = 8 * n
    else:
        coord_sec  = struct.pack(f'>{n}h', *np.round(xr).astype(np.int16).tolist())
        coord_sec += struct.pack(f'>{n}h', *np.round(yr).astype(np.int16).tolist())
        coord_size = 4 * n

    # Patch fixed header (copy original, update fields that changed)
    hdr = bytearray(raw[:HEADER_SIZE])
    struct.pack_into('>H', hdr, OFF_N_COORDS, n)
    struct.pack_into('>h', hdr, OFF_TOP,      new_top)
    struct.pack_into('>h', hdr, OFF_LEFT,     new_left)
    struct.pack_into('>h', hdr, OFF_BOTTOM,   new_bottom)
    struct.pack_into('>h', hdr, OFF_RIGHT,    new_right)

    # header2 block: update its offset, copy its content verbatim
    old_h2 = meta['h2_offset']
    if old_h2 > 0:
        new_h2 = HEADER_SIZE + coord_size
        struct.pack_into('>i', hdr, OFF_HEADER2, new_h2)
        tail = raw[old_h2:]   # header2 metadata (channel/z/t position, name…)
    else:
        tail = b''

    path.write_bytes(bytes(hdr) + coord_sec + tail)
    print(f"  Saved {n} points → {path}")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def turn_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Deflection angle (°) at vertex b.
    0° = straight line; 90° = right-angle turn; 180° = U-turn.
    """
    v1 = b - a;  n1 = np.linalg.norm(v1)
    v2 = c - b;  n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))))


def catmull_rom_pts(p0: np.ndarray, p1: np.ndarray,
                    p2: np.ndarray, p3: np.ndarray,
                    n_pts: int) -> list:
    """
    Return n_pts points strictly between p1 and p2 using uniform
    Catmull-Rom interpolation.  The spline passes exactly through p1 and p2.
    """
    out = []
    for k in range(1, n_pts + 1):
        t  = k / (n_pts + 1)
        t2 = t * t
        t3 = t2 * t
        out.append(0.5 * (
              2.0 * p1
            + (-p0 + p2) * t
            + ( 2*p0 - 5*p1 + 4*p2 - p3) * t2
            + (-p0 + 3*p1 - 3*p2 + p3) * t3
        ))
    return out


# ── Core smoothing ────────────────────────────────────────────────────────────

def smooth_polygon(points: np.ndarray,
                   threshold: float = 40.0,
                   n_interp: int = 4) -> np.ndarray:
    """
    Insert Catmull-Rom intermediate points on every segment whose two
    endpoints are both non-corner vertices.

    A vertex is a *corner* when its turn angle exceeds `threshold` degrees.
    Segments adjacent to a corner are kept as straight lines, so the sharp
    angle is fully preserved.
    """
    n = len(points)

    # Classify every vertex
    is_corner = np.array([
        turn_angle_deg(points[(i - 1) % n], points[i], points[(i + 1) % n]) > threshold
        for i in range(n)
    ])

    n_corners = int(is_corner.sum())
    print(f"  {n_corners} corner(s) with turn > {threshold}°  ({n - n_corners} smooth vertices)")

    out = []
    for i in range(n):
        out.append(points[i])
        j = (i + 1) % n
        # Only interpolate on smooth-to-smooth segments
        if not is_corner[i] and not is_corner[j]:
            p0 = points[(i - 1) % n]
            p2 = points[j]
            p3 = points[(j + 1) % n]
            out.extend(catmull_rom_pts(p0, points[i], p2, p3, n_interp))

    return np.asarray(out)


# ── Entry point ───────────────────────────────────────────────────────────────

DEFAULT_INPUT = '/home/rfernandez/Data/Arize/Livrable_Arize/arrow.roi'


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('input', nargs='?', default=DEFAULT_INPUT,
                    help='Input .roi file (default: %(default)s)')
    ap.add_argument('--threshold', type=float, default=40.0,
                    help='Turn angle (°) above which a vertex is a corner (default: 40)')
    ap.add_argument('--interp', type=int, default=4,
                    help='Intermediate points per smooth segment (default: 4)')
    args = ap.parse_args()

    inp = Path(args.input)
    out = inp.with_name(inp.stem + '_improved.roi')

    print(f"Reading  : {inp}")
    meta, pts, raw = read_roi(inp)
    print(f"  Type     : {ROI_TYPE_NAMES.get(meta['roi_type'], meta['roi_type'])}")
    print(f"  Points   : {meta['n']}")
    print(f"  SubPixel : {meta['subpixel']}")
    print(f"  Bounds   : top={meta['top']}  left={meta['left']}  "
          f"bottom={meta['bottom']}  right={meta['right']}")

    new_pts = smooth_polygon(pts, threshold=args.threshold, n_interp=args.interp)
    print(f"  → {len(new_pts)} points after smoothing")

    write_roi(out, meta, new_pts, raw)
    print(f"Output   : {out}")


if __name__ == '__main__':
    main()
