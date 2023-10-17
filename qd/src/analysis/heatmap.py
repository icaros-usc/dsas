"""Visualizes heatmaps from an experiment.

Usage:
    python -m src.analysis.heatmap -l LOGDIR
"""
import os
import shutil

import fire
import gin
import matplotlib.pyplot as plt
import numpy as np
import ribs.visualize
from loguru import logger

from src.analysis.utils import load_archive_gen, load_experiment, load_metrics
from src.mpl_styles.utils import mpl_style_file


def plot_generation_on_axis(ax, mode, logdir, gen, plot_kwargs):
    # pylint: disable = unused-argument
    archive = load_archive_gen(logdir, gen)
    ribs.visualize.grid_archive_heatmap(archive, ax, **plot_kwargs)


def plot_generation(mode, logdir, gen, plot_kwargs, filenames, mario, maze,
                    maze_walls_explore, maze_rep_explore):
    with mpl_style_file("heatmap.mplstyle") as f:
        with plt.style.context(f):
            if mario:
                figsize = (6, 3.3)
            elif maze:
                figsize = (6, 3.3)
            elif maze_walls_explore:
                figsize = (3, 3.3)
            elif maze_rep_explore:
                figsize = (3, 3.3)
            else:
                figsize = (6, 5.5)

            # Figure should be created inside of the style context so that all
            # settings are handled properly, e.g. setting fonts for axis labels.
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            plot_generation_on_axis(ax, mode, logdir, gen, plot_kwargs)

            if mario:
                ax.set_ylabel("Number of jumps")
                ax.set_xlabel("Sky tiles")
            elif maze:
                ax.set_ylabel("Mean agent path length")
                ax.set_xlabel("Number of wall cells")
            elif maze_walls_explore:
                ax.set_ylabel("Number of wall cells")
                ax.set_xlabel("Maze exploration")
            elif maze_rep_explore:
                ax.set_ylabel("Repeated visits")
                ax.set_xlabel("Maze exploration")

            if maze:
                ax.set_aspect(0.25)
            elif maze_walls_explore:
                ax.set_aspect(0.01)
            elif maze_rep_explore:
                ax.set_aspect(0.01 / 4)

            fig.tight_layout(pad=2)

            for filename in filenames:
                # Videos should use lower DPI because they have a lot of images.
                fig.savefig(logdir.file(filename),
                            dpi="figure" if mode == "video" else 300)
                logger.info(f"Plotted {filename}")

    plt.close(fig)


def heatmap(logdir: str,
            mode: str = "single",
            skip_plot: bool = False,
            freq: int = 100,
            framerate: int = 6,
            gen: int = None,
            mario: bool = False,
            maze: bool = False,
            maze_walls_explore: bool = False,
            maze_rep_explore: bool = False):
    """Plots the heatmaps for archives in a logdir.

    Args:
        logdir: Path to experiment logging directory.
        mode:
          - "single": plot the archive and save to logdir /
            `heatmap_archive_{gen}.{pdf,png,svg}`
          - "video": plot every `freq` generations and save to the directory
            logdir / `heatmap_archive`; logdir / `heatmap_archive.mp4` is also
            created from these images with ffmpeg.
        skip_plot: Skip plotting the heatmaps and just make the video. Only
            applies to "video" mode.
        freq: Frequency (in terms of generations) to plot heatmaps for video.
            Only applies to "video" mode.
        framerate: Framerate for the video. Only applies to "video" mode.
        gen: Generation to plot -- only applies to "single" mode.
            None indicates the final gen.
        mario: Pass this to activate several special settings for Mario.
        maze: Pass this to activate several special settings for Maze.
        maze_walls_explore: Pass this to activate several special settings for
            Maze with walls and exploration measures.
        maze_rep_explore: Pass this to activate several special settings for
            Maze with repeated visits and exploration measures.
    """
    logdir = load_experiment(logdir)

    if len(gin.query_parameter("GridArchive.dims")) != 2:
        logger.error("Heatmaps not supported for non-2D archives")
        return

    # TODO: Settings are currently for the old shared autonomy scenario (BC1 &
    # BC2 in Matt's paper).
    # Plotting arguments for grid_archive_heatmap.
    plot_kwargs = {
        "aspect": 0.11 / 0.32,
        "cmap": "coolwarm",
        "pcm_kwargs": {
            # Looks much better in PDF viewers because the heatmap is not drawn
            # as individual rectangles. See here:
            # https://stackoverflow.com/questions/27092991/white-lines-in-matplotlibs-pcolor
            "rasterized": True,
        },
        "vmin": 0.0,
        "vmax": 10.0,
        # Make the archive show up vertically.
        "transpose_measures": True,
    }

    total_gens = load_metrics(logdir).total_itrs
    gen = total_gens if gen is None else gen

    if mode == "single":
        plot_generation(
            mode,
            logdir,
            gen,
            plot_kwargs,
            [
                f"heatmap_archive_{gen}.pdf",
                f"heatmap_archive_{gen}.png",
                f"heatmap_archive_{gen}.svg",
            ],
            mario,
            maze,
            maze_walls_explore,
            maze_rep_explore,
        )
    elif mode == "video":  # pylint: disable = too-many-nested-blocks
        if not skip_plot:
            # Remove existing heatmaps.
            shutil.rmtree(logdir.pdir("heatmap_archive/"), ignore_errors=True)

            digits = int(np.ceil(np.log10(total_gens + 1)))
            for g in range(total_gens + 1):  # 0...total_gens
                try:
                    if g % freq == 0 or g == total_gens:
                        plot_generation(
                            mode,
                            logdir,
                            g,
                            plot_kwargs,
                            [f"heatmap_archive/{g:0{digits}}.png"],
                            mario,
                            maze,
                            maze_walls_explore,
                            maze_rep_explore,
                        )
                except ValueError as e:
                    logger.error(
                        "ValueError caught. Have you tried setting the max "
                        "objective in objectives/__init__.py ?")
                    raise e

        # The extra options make the video web-compatible - see
        # https://gist.github.com/Vestride/278e13915894821e1d6f
        os.system(f"""\
ffmpeg -an -r {framerate} -i "{logdir.file('heatmap_archive/%*.png')}" \
    -vcodec libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3 \
    "{logdir.file('heatmap_archive.mp4')}" \
    -y \
""")
    else:
        raise ValueError(f"Unknown mode '{mode}'")


if __name__ == "__main__":
    fire.Fire(heatmap)
