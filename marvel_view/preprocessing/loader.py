"""Load multipage TIFF segmentation files as 3-D numpy arrays."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def load_segmentation(path: str | Path) -> np.ndarray:
    """Load a multipage TIFF segmentation file.

    Parameters
    ----------
    path:
        Absolute path to the ``.tiff`` / ``.tif`` file.

    Returns
    -------
    np.ndarray
        3-D array shaped ``(Z, Y, X)`` with dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the array is not 3-D after loading (wrong file or wrong axes order).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {path}")

    logger.info("Loading segmentation from %s …", path)
    t = time.perf_counter()
    volume: np.ndarray = tifffile.imread(str(path))
    logger.info("  └─ file read in %.1f s  (raw shape: %s)", time.perf_counter() - t, volume.shape)

    # Some TIFF writers add trailing singleton dims (e.g. CZYX with C=1).
    volume = volume.squeeze()

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D volume after loading, got shape {volume.shape}. "
            "Check that the file is a valid multipage TIFF (Z-stack)."
        )

    volume = volume.astype(np.uint8)
    unique_labels = np.unique(volume)
    logger.info(
        "Volume loaded  –  shape: %s  dtype: %s  labels: %s",
        volume.shape,
        volume.dtype,
        unique_labels.tolist(),
    )
    return volume


def load_float_volume(path: str | Path) -> np.ndarray:
    """Load a multipage TIFF as a 3-D ``float32`` array.

    Intended for the 32-bit "norm" outputs of an upstream pipeline where
    positive values mean *inside* the class and negative values mean
    *outside*.  Marching cubes can then be run at ``level=0`` on the raw
    array to obtain a smooth sub-voxel-accurate iso-surface.

    Parameters
    ----------
    path:
        Absolute path to the ``.tif`` / ``.tiff`` file.

    Returns
    -------
    np.ndarray
        3-D ``float32`` array of shape ``(Z, Y, X)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Float volume not found: {path}")

    logger.info("Loading float volume from %s …", path)
    t = time.perf_counter()
    volume: np.ndarray = tifffile.imread(str(path))
    logger.info(
        "  └─ file read in %.1f s  (raw shape: %s, dtype: %s)",
        time.perf_counter() - t, volume.shape, volume.dtype,
    )

    volume = volume.squeeze()
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D volume after loading, got shape {volume.shape}."
        )

    volume = volume.astype(np.float32, copy=False)
    logger.info(
        "Volume loaded  –  shape: %s  dtype: %s  min: %.3f  max: %.3f  positive: %d",
        volume.shape, volume.dtype,
        float(volume.min()), float(volume.max()),
        int(np.count_nonzero(volume > 0)),
    )
    return volume
