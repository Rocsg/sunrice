"""
Scene assembly and rendering with vedo.

Two entry points are provided:

* :func:`render`              – render a list of already-styled vedo meshes.
* :func:`render_from_vtk_dir` – load VTK files written by the preprocessing
  script, apply colours / opacities from the config, and render.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .actors import style_mesh, _normalise_color

logger = logging.getLogger(__name__)


def render(
    actors: List["vedo.Mesh"],
    bg: str = "black",
    title: str = "Marvel View – 3-D Segmentation",
    axes: int = 1,
    interactive: bool = True,
    screenshot_path: Optional[Union[str, Path]] = None,
) -> None:
    """Render a list of vedo meshes in a 3-D viewer.

    Parameters
    ----------
    actors:
        Already-styled vedo meshes to display.
    bg:
        Background colour (name or hex string).
    title:
        Window title.
    axes:
        vedo axes style: 0 = none, 1 = box, 4 = ruler, 8 = compass …
    interactive:
        If ``True`` the window stays open for mouse navigation.
        Set to ``False`` for off-screen / batch rendering.
    screenshot_path:
        If given, write a PNG to this path after showing.
    """
    import vedo

    if not actors:
        logger.warning("render() received an empty actor list – nothing to show.")
        return

    plt = vedo.Plotter(bg=bg, title=title, axes=axes)
    plt.show(actors, interactive=interactive, resetcam=True)
    if screenshot_path:
        plt.screenshot(str(screenshot_path))
        logger.info("Screenshot saved to %s", screenshot_path)
    if not interactive:
        plt.close()


def render_from_vtk_dir(
    vtk_dir: Union[str, Path],
    label_config: Dict,
    bg: str = "black",
    title: str = "Marvel View – 3-D Segmentation",
    axes: int = 1,
    interactive: bool = True,
    screenshot_path: Optional[Union[str, Path]] = None,
) -> None:
    """Load VTK meshes produced by the preprocessing script and render them.

    Expected directory layout (produced by
    :mod:`marvel_view.scripts.preprocess_all`)::

        vtk_dir/
        ├── label1_background_cavities/
        │   └── mesh.vtk
        ├── label2_aqueous_tissues/
        │   └── mesh.vtk
        ├── label3_iodine/
        │   ├── component_0000.vtk
        │   └── component_0001.vtk
        ├── label4_scaffold/
        │   └── mesh.vtk
        └── label5_membranes/
            ├── component_0000.vtk
            └── …

    Parameters
    ----------
    vtk_dir:
        Root directory of the per-label sub-directories.
    label_config:
        Dict matching :data:`marvel_view.config.LABEL_CONFIG`.
    bg:
        Background colour.
    title:
        Window title.
    axes:
        vedo axes style.
    interactive:
        Keep the window open for mouse navigation.
    screenshot_path:
        Optional path to save a PNG screenshot.
    """
    import vedo

    vtk_dir = Path(vtk_dir)
    actors: List[vedo.Mesh] = []
    actors_by_label: Dict[int, List[vedo.Mesh]] = {}
    rendered_cfg: Dict[int, dict] = {}

    for label_id, cfg in label_config.items():
        label_dir = vtk_dir / f"label{label_id}_{cfg['name']}"
        if not label_dir.exists():
            logger.warning(
                "No directory for label %d (%s), skipping.", label_id, cfg["name"]
            )
            continue

        vtk_files = sorted(label_dir.glob("*.vtk"))
        if not vtk_files:
            logger.warning("No VTK files in %s – skipping.", label_dir)
            continue

        color = cfg["color"]
        opacity = cfg["opacity"]

        group: List[vedo.Mesh] = []
        for vtk_file in vtk_files:
            m = vedo.Mesh(str(vtk_file))
            style_mesh(
                m,
                color=color,
                opacity=opacity,
                name=f"{cfg['name']}_{vtk_file.stem}",
            )
            actors.append(m)
            group.append(m)
            logger.info(
                "  [label %d] loaded %s  (%d faces)", label_id, vtk_file.name, m.ncells
            )
        if group:
            actors_by_label[label_id] = group
            rendered_cfg[label_id] = cfg

    if not actors:
        raise RuntimeError(
            f"No meshes found under {vtk_dir}.\n"
            "Run `marvel-preprocess` (or `preprocess_all.py`) first."
        )

    logger.info(
        "Rendering %d mesh(es) across %d label(s) …",
        len(actors), len(actors_by_label),
    )

    # When interactive, build the plotter ourselves so we can attach the
    # control panel (visibility toggles + per-label lighting sliders)
    # before show() enters the event loop.  Screenshot / headless paths
    # keep the simple render() helper.
    if interactive:
        from .panel import attach_panel
        plt = vedo.Plotter(bg=bg, title=title, axes=axes)
        plt.add(*actors)
        attach_panel(plt, actors_by_label, rendered_cfg)
        plt.show(interactive=True, resetcam=True)
        if screenshot_path:
            plt.screenshot(str(screenshot_path))
            logger.info("Screenshot saved to %s", screenshot_path)
        plt.close()
    else:
        render(
            actors,
            bg=bg,
            title=title,
            axes=axes,
            interactive=False,
            screenshot_path=screenshot_path,
        )
