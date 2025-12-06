from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_boxplot(
    figsize: tuple,
    title: str,
    ylabel: str,
    labels: list[str],
    x: ArrayLike | Sequence[ArrayLike],
    colors: list[str] | None = None,
    legend: bool = False,
    legend_labels: list[str] | None = None,
    legend_title: str | None = None,
    legend_bbox_to_anchor: tuple | None = None,
    legend_loc: str = "best",
    legend_fontsize: int | str = "medium",
    legend_ncol: int = 1,
    xlabel: str | None = None,
    xticklabel_rotation: float = 45,
    xticklabel_ha: str = "right",
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 300,
    show: bool = False,
    save: bool = True,
    save_path: str = "plot_boxplot.png",
    return_fig_ax: bool = False,
    # ============================================
    notch: bool | None = None,
    sym: str | None = None,
    vert: bool | None = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    whis: float | tuple[float, float] | None = None,
    positions: ArrayLike | None = None,
    widths: float | ArrayLike | None = None,
    patch_artist: bool | None = None,
    bootstrap: int | None = None,
    usermedians: ArrayLike | None = None,
    conf_intervals: ArrayLike | None = None,
    meanline: bool | None = None,
    showmeans: bool | None = None,
    showcaps: bool | None = None,
    showbox: bool | None = None,
    showfliers: bool | None = None,
    boxprops: dict[str, Any] | None = None,
    tick_labels: Sequence[str] | None = None,
    flierprops: dict[str, Any] | None = None,
    medianprops: dict[str, Any] | None = None,
    meanprops: dict[str, Any] | None = None,
    capprops: dict[str, Any] | None = None,
    whiskerprops: dict[str, Any] | None = None,
    manage_ticks: bool = True,
    autorange: bool = False,
    zorder: float | None = None,
    capwidths: float | ArrayLike | None = None,
    label: Sequence[str] | None = None,
    *,
    data=None,
) -> tuple[plt.Figure, plt.Axes] | None:
    fig, ax = plt.subplots(figsize=figsize)

    # Set default patch_artist to True if colors are provided
    if colors and patch_artist is None:
        patch_artist = True

    box = ax.boxplot(
        x=x,
        notch=notch,
        sym=sym,
        vert=vert,
        orientation=orientation,
        whis=whis,
        positions=positions,
        widths=widths,
        patch_artist=patch_artist,
        bootstrap=bootstrap,
        usermedians=usermedians,
        conf_intervals=conf_intervals,
        meanline=meanline,
        showmeans=showmeans,
        showcaps=showcaps,
        showbox=showbox,
        showfliers=showfliers,
        boxprops=boxprops,
        tick_labels=tick_labels,
        flierprops=flierprops,
        medianprops=medianprops,
        meanprops=meanprops,
        capprops=capprops,
        whiskerprops=whiskerprops,
        manage_ticks=manage_ticks,
        autorange=autorange,
        zorder=zorder,
        capwidths=capwidths,
        label=label,
        data=None,
    )

    # Set tick labels
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(labels, rotation=xticklabel_rotation, ha=xticklabel_ha)

    # Set labels and title
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Add grid if requested
    if grid:
        ax.grid(True, alpha=grid_alpha, axis="y")

    # Handle colors and legend
    legend_handles = None
    if colors and patch_artist:
        # Apply colors to boxes
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        # Create legend if requested
        if legend:
            from matplotlib.patches import Patch

            # Use legend_labels if provided, otherwise use labels
            legend_text = legend_labels if legend_labels is not None else labels
            legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, legend_text)]

            # Set up legend parameters
            legend_kwargs = {
                "handles": legend_handles,
                "title": legend_title,
                "fontsize": legend_fontsize,
                "ncol": legend_ncol,
                "loc": legend_loc,
            }

            if legend_bbox_to_anchor:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

            legend_obj = ax.legend(**legend_kwargs)

    # Save the plot
    if save:
        save_kwargs = {"bbox_inches": "tight", "dpi": dpi}
        if legend and legend_handles and legend_bbox_to_anchor:
            # Include legend in the saved area
            legend_obj = ax.get_legend()
            if legend_obj:
                save_kwargs["bbox_extra_artists"] = [legend_obj]
        plt.savefig(save_path, **save_kwargs)

    # Show the plot
    if show:
        plt.show()

    # Return figure and axes if requested
    if return_fig_ax:
        return fig, ax

    # Close the figure if not returning it
    if not return_fig_ax:
        plt.close(fig)


if __name__ == "__main__":
    """
    In all the box plots, the minimum is the smallest value within the data set, marked at the end of the lower whisker. The first quartile (Q1), or 25th percentile, forms the lower edge of the box. The median (50th percentile) is represented by a line inside the box, indicating the midpoint of the data. The third quartile (Q3), or 75th percentile, forms the upper edge of the box. The maximum is the largest value within the data set, marked at the end of the upper whisker. The whiskers extend to the smallest and largest values within 1.5 times the interquartile range (IQR).
    """

    model1 = np.random.randn(10) + 1.0
    model2 = np.random.randn(10) + 2.0
    model3 = np.random.randn(10) + 0.5

    data = [model1, model2, model3]
    labels = ["Model 1", "Model 2", "Model 3"]

    plot_boxplot(
        figsize=(8, 6),
        x=data,
        labels=labels,
        title="Box-plot Comparison",
        ylabel="Performance Metric",
        colors=["lightblue", "lightgreen", "lightcoral"],
        legend=True,
        legend_title="Models",
        legend_bbox_to_anchor=(1.05, 1),
        legend_loc="upper left",
        whis=1.5,
        show=False,
        save=True,
        save_path="plot_boxplot.png",
        medianprops={"color": "black", "linewidth": 1, "linestyle": "-"},
    )
