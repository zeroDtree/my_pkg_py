import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Sequence, Literal
from numpy.typing import ArrayLike


def plot_boxplot(
    figsize: tuple,
    title: str,
    ylabel: str,
    labels: list[str],
    x: ArrayLike | Sequence[ArrayLike],
    colors: list[str] | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    show: bool = False,
    save: bool = True,
    save_path: str = "plot_boxplot.png",
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
) -> None:
    plt.figure(figsize=figsize)

    box = plt.boxplot(
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

    plt.xticks(list(range(1, len(labels) + 1)), labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    if colors:
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        if legend:
            from matplotlib.patches import Patch

            legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
            plt.legend(handles=legend_handles, title=legend_title)
    if save:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()


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
        title="box-plot Comparison",
        ylabel="Performance Metric",
        patch_artist=True,
        colors=["lightblue", "lightgreen", "lightcoral"],
        legend=True,
        legend_title="Models",
        whis=1.5,
    )
