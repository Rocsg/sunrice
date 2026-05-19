"""
Connected-component analysis helpers for binary segmentation masks.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

# A component is represented as (sub_mask, offset_zyx, voxel_count) where
# sub_mask is the cropped uint8 binary mask restricted to the component's
# bounding box and offset_zyx is the (z, y, x) origin of that box in the
# full-resolution volume.
Component = Tuple[np.ndarray, Tuple[int, int, int], int]

logger = logging.getLogger(__name__)

# Full 3-D 26-connectivity structuring element (captures diagonal neighbours).
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)


def label_components(
    mask: np.ndarray, connectivity: int = 26
) -> tuple[np.ndarray, int]:
    """Find connected components in a binary mask.

    Parameters
    ----------
    mask:
        Binary (0 / 1) 3-D array.
    connectivity:
        ``26`` for full 3-D connectivity (default) or ``6`` for
        face-adjacent-only connectivity.

    Returns
    -------
    labeled : np.ndarray  (int32)
        Each voxel holds the integer ID of its component; 0 = background.
    n_components : int
        Total number of components found.
    """
    struct = _STRUCT_26 if connectivity == 26 else None
    logger.info(
        "  label_components  shape=%s  connectivity=%d …", mask.shape, connectivity
    )
    t = time.perf_counter()
    labeled, n = ndimage.label(mask, structure=struct)
    logger.info("  └─ found %d components  (%.1f s)", n, time.perf_counter() - t)
    return labeled.astype(np.int32), int(n)


def _border_touching_ids(labeled: np.ndarray) -> set[int]:
    """Return IDs of components that touch at least one face of the volume."""
    ids: set[int] = set()
    for face in (
        labeled[0, :, :],
        labeled[-1, :, :],
        labeled[:, 0, :],
        labeled[:, -1, :],
        labeled[:, :, 0],
        labeled[:, :, -1],
    ):
        ids.update(np.unique(face).tolist())
    ids.discard(0)  # 0 is the unlabelled background
    return ids


def interior_mask(
    labeled: np.ndarray,
    n_components: int,
    min_voxels: int = 0,
) -> np.ndarray:
    """Build a binary mask of components that do **not** touch the image border.

    Uses ``np.bincount`` + ``np.isin`` for O(n_voxels) complexity instead of
    the naive O(n_components * n_voxels) per-component loop.

    Parameters
    ----------
    labeled:
        Output of :func:`label_components`.
    n_components:
        Number of components (second output of :func:`label_components`).
    min_voxels:
        Components with fewer voxels are also discarded.

    Returns
    -------
    np.ndarray (uint8, values 0/1)
    """
    logger.info("  Computing component sizes (bincount, single pass) …")
    # One O(n_voxels) pass — vastly faster than iterating per component.
    sizes = np.bincount(labeled.ravel(), minlength=n_components + 1)
    # sizes[0] = unlabelled background; sizes[i] = voxel count of component i.

    border_ids = _border_touching_ids(labeled)
    logger.info(
        "  %d / %d components touch the image border.", len(border_ids), n_components
    )

    kept: list[int] = []
    n_border = 0
    n_small = 0
    for cid in range(1, n_components + 1):
        if cid in border_ids:
            n_border += 1
        elif int(sizes[cid]) < min_voxels:
            n_small += 1
        else:
            kept.append(cid)

    logger.info(
        "  Kept: %d  |  border-touching: %d  |  too small (< %d vx): %d",
        len(kept), n_border, min_voxels, n_small,
    )

    if not kept:
        logger.warning("  No interior components survived the filters.")
        return np.zeros(labeled.shape, dtype=np.uint8)

    total_kept_vx = int(sum(sizes[cid] for cid in kept))
    logger.info("  Total kept voxels: %d", total_kept_vx)

    # LUT-based selection: single O(n_voxels) gather, much faster than
    # np.isin (which sorts internally) on very large labeled arrays.
    logger.info("  Building output mask (LUT gather) …")
    lut = np.zeros(n_components + 1, dtype=np.uint8)
    lut[np.asarray(kept, dtype=np.int64)] = 1
    return lut[labeled]


def split_components(
    labeled: np.ndarray,
    n_components: int,
    min_voxels: int = 0,
    max_components: Optional[int] = None,
) -> List[Component]:
    """Return per-component cropped binary masks with their bbox offset.

    Uses ``ndimage.find_objects`` to obtain each component's bounding box in
    a single pass, then extracts only the sub-volume — orders of magnitude
    faster than ``(labeled == cid)`` on the full array when there are many
    small components.  Results sorted by descending size.

    Parameters
    ----------
    labeled:
        Output of :func:`label_components`.
    n_components:
        Number of components.
    min_voxels:
        Skip components with fewer voxels than this threshold.
    max_components:
        If given, keep only the *N* largest components.

    Returns
    -------
    List[Component]
        Each element is ``(sub_mask, (z0, y0, x0), voxel_count)``.
        ``sub_mask`` is a cropped ``uint8`` array with values 0/1,
        ``(z0, y0, x0)`` is the offset of its bbox in the full volume,
        and ``voxel_count`` is the component size.  Sorted by descending
        size (largest first).
    """
    # Single O(n_voxels) pass to size all components.
    sizes = np.bincount(labeled.ravel(), minlength=n_components + 1)

    # Single O(n_voxels) pass to locate each component's bbox.
    bboxes = ndimage.find_objects(labeled, max_label=n_components)

    keep: list[tuple[int, int]] = sorted(
        [
            (int(sizes[cid]), cid)
            for cid in range(1, n_components + 1)
            if int(sizes[cid]) >= min_voxels and bboxes[cid - 1] is not None
        ],
        reverse=True,
    )

    if max_components is not None and len(keep) > max_components:
        logger.info(
            "  Limiting to %d largest components (out of %d eligible).",
            max_components, len(keep),
        )
        keep = keep[:max_components]

    logger.info(
        "  split_components: %d / %d kept  "
        "(largest: %d vx | smallest kept: %d vx)",
        len(keep), n_components,
        keep[0][0] if keep else 0,
        keep[-1][0] if keep else 0,
    )

    components: List[Component] = []
    for vx, cid in keep:
        sl = bboxes[cid - 1]
        offset = (sl[0].start, sl[1].start, sl[2].start)
        sub = (labeled[sl] == cid).astype(np.uint8)
        components.append((sub, offset, vx))
    return components


# 2-D 8-connectivity structuring element (for per-slice operations).
_STRUCT_8_2D = np.ones((3, 3), dtype=np.uint8)


def outside_mask_2d(
    binary: np.ndarray,
    erode_iter: int = 2,
    dilate_iter: int = 2,
) -> np.ndarray:
    """Detect the "outside" (background air) region of a gas mask, slice-by-slice.

    Strategy (per Z-slice independently):

    1. Morphological opening: erode by ``erode_iter`` pixels, then dilate by
       ``dilate_iter`` pixels.  This severs thin connections such as
       segmentation leaks between aerenchyma cavities and the air around
       the sample at the tube borders.
    2. Largest 2-D connected component on the opened slice — this is the
       air enveloping the sample on that slice.

    Stacking the per-slice 2-D masks yields a 3-D mask of voxels to exclude
    from the gas channel (image background plus its narrow leaks into the
    aerenchymes through the top/bottom holes and tube borders).

    Parameters
    ----------
    binary:
        3-D uint8 mask (values 0/1) of the gas label.
    erode_iter, dilate_iter:
        Number of binary-erosion / dilation iterations applied per slice
        (8-connectivity, ``3 × 3`` square structuring element).

    Returns
    -------
    np.ndarray (uint8, 0/1)
        Mask, same shape as ``binary``, marking the outside region.
    """
    if binary.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {binary.shape}.")

    logger.info(
        "  outside_mask_2d  shape=%s  erode=%d  dilate=%d …",
        binary.shape, erode_iter, dilate_iter,
    )
    t = time.perf_counter()

    out = np.zeros_like(binary, dtype=np.uint8)
    n_z = binary.shape[0]
    total_vx = 0

    for z in range(n_z):
        sl = binary[z].astype(bool, copy=False)
        if not sl.any():
            continue

        if erode_iter > 0:
            opened = ndimage.binary_erosion(
                sl, structure=_STRUCT_8_2D, iterations=erode_iter
            )
        else:
            opened = sl
        if dilate_iter > 0:
            opened = ndimage.binary_dilation(
                opened, structure=_STRUCT_8_2D, iterations=dilate_iter
            )
        if not opened.any():
            continue

        lbl, n = ndimage.label(opened, structure=_STRUCT_8_2D)
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0  # ignore background
        biggest = int(sizes.argmax())
        if sizes[biggest] == 0:
            continue

        # Restrict the biggest opened-CC back to the *original* gas mask
        # (the dilation can spill outside, we only want voxels that were
        # actually labelled "gas").
        slice_out = ((lbl == biggest) & sl).astype(np.uint8)
        out[z] = slice_out
        total_vx += int(slice_out.sum())

    logger.info(
        "  └─ outside mask: %d voxels  (%.1f s)",
        total_vx, time.perf_counter() - t,
    )
    return out
