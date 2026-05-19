"""
marvel_view – 3-D visualisation of volumetric segmentations.

Quick-start
-----------
Preprocess (generates VTK meshes on disk)::

    python -m marvel_view.scripts.preprocess_all

Visualise::

    python -m marvel_view.scripts.visualize_scene
"""

__version__ = "0.1.0"

from .preprocessing import (
    load_segmentation,
    label_components,
    interior_mask,
    split_components,
    mask_to_mesh,
    masks_to_meshes,
)
from .visualization import style_mesh, style_meshes, render, render_from_vtk_dir

__all__ = [
    "__version__",
    # preprocessing
    "load_segmentation",
    "label_components",
    "interior_mask",
    "split_components",
    "mask_to_mesh",
    "masks_to_meshes",
    # visualization
    "style_mesh",
    "style_meshes",
    "render",
    "render_from_vtk_dir",
]
