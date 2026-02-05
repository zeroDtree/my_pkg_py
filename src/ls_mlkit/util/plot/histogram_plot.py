from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histogram_and_kde(
    data=None,
    title: str = "Histogram and KDE Plot",
    xlabel: str = "",
    ylabel: str = "",
    save_path: str = "histogram_plot.png",
    show: bool = False,
    save: bool = True,
    figsize: tuple = (10, 6),
    dpi: int = 300,
    # Grid parameters
    grid: bool = True,
    grid_alpha: float = 0.3,
    # Legend parameters
    legend: bool = True,
    legend_title: str | None = None,
    legend_bbox_to_anchor: tuple | None = None,
    legend_loc: str = "best",
    legend_fontsize: int | str = "medium",
    legend_ncol: int = 1,
    # Style parameters
    style: str = "whitegrid",
    context: str = "notebook",
    font_scale: float = 1.0,
    return_fig_ax: bool = False,
    *,
    # Vector variables
    x=None,
    y=None,
    hue=None,
    weights=None,
    # Histogram computation parameters
    stat: str = "count",
    bins: str | int | Sequence = "auto",
    binwidth: float | None = None,
    binrange: tuple | None = None,
    discrete: bool | None = None,
    cumulative: bool = False,
    common_bins: bool = True,
    common_norm: bool = True,
    # Histogram appearance parameters
    multiple: str = "layer",
    element: str = "bars",
    fill: bool = True,
    shrink: float = 1,
    alpha: float | None = None,
    # Histogram smoothing with a kernel density estimate
    kde: bool = False,
    kde_kws: dict[str, Any] | None = None,
    line_kws: dict[str, Any] | None = None,
    # Bivariate histogram parameters
    thresh: float = 0,
    pthresh: float | None = None,
    pmax: float | None = None,
    cbar: bool = False,
    cbar_ax=None,
    cbar_kws: dict[str, Any] | None = None,
    # Hue mapping parameters
    palette=None,
    hue_order: Sequence | None = None,
    hue_norm: tuple | None = None,
    color: str | None = None,
    # Axes information
    log_scale: bool | dict | None = None,
    ax=None,
    # Other appearance keywords
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    # Set seaborn style and context
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Create histogram plot
    # Note: seaborn's histplot legend parameter controls automatic legend creation
    sns.histplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        weights=weights,
        # Histogram computation parameters
        stat=stat,
        bins=bins,
        binwidth=binwidth,
        binrange=binrange,
        discrete=discrete,
        cumulative=cumulative,
        common_bins=common_bins,
        common_norm=common_norm,
        # Histogram appearance parameters
        multiple=multiple,
        element=element,
        fill=fill,
        shrink=shrink,
        alpha=alpha,
        # Histogram smoothing with a kernel density estimate
        kde=kde,
        kde_kws=kde_kws,
        line_kws=line_kws,
        # Bivariate histogram parameters
        thresh=thresh,
        pthresh=pthresh,
        pmax=pmax,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        # Hue mapping parameters
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        color=color,
        # Axes information
        log_scale=log_scale,
        legend=True,  # Let seaborn create the legend, we'll customize it later
        ax=ax,
        # Other appearance keywords
        **kwargs,
    )

    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add grid if requested
    if grid:
        ax.grid(True, alpha=grid_alpha, axis="y")

    # Handle legend customization
    if legend and hue is not None:
        # Get the existing legend created by seaborn
        existing_legend = ax.get_legend()
        if existing_legend:
            # Get handles and labels from the existing legend
            handles = existing_legend.legend_handles  # Use correct attribute name
            labels = [t.get_text() for t in existing_legend.get_texts()]

            # Set up legend parameters
            legend_kwargs = {
                "title": legend_title if legend_title is not None else hue,
                "fontsize": legend_fontsize,
                "ncol": legend_ncol,
                "loc": legend_loc,
            }

            if legend_bbox_to_anchor:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor

            # Remove the old legend and create a new one with custom parameters
            existing_legend.remove()
            ax.legend(handles=handles, labels=labels, **legend_kwargs)
    elif not legend and hue is not None:
        # Remove legend if legend=False but hue is provided
        existing_legend = ax.get_legend()
        if existing_legend:
            existing_legend.remove()

    # Save the plot
    if save:
        save_kwargs = {"bbox_inches": "tight", "dpi": dpi}
        if legend and legend_bbox_to_anchor:
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
    import numpy as np
    import pandas as pd

    # 假设有三组数据
    data1 = np.random.randn(1000)
    data2 = np.random.randn(1000) + 2
    data3 = np.random.randn(1000) - 2

    data = [data1, data2, data3]

    group_names = ["A", "B", "C"]
    group_colors = ["skyblue", "orange", "red"]

    group_flag = []
    for idx, group in enumerate(group_names):
        group_flag.extend([group] * len(data[idx]))
    custom_palette = {}
    for idx, group in enumerate(group_names):
        custom_palette[group] = group_colors[idx]

    df = pd.DataFrame({"Value": np.concatenate([data1, data2, data3]), "Group": group_flag})

    plot_histogram_and_kde(
        data=df,
        x="Value",
        hue="Group",
        kde=True,
        bins=30,
        title="Histogram with Multiple Groups",
        xlabel="Value",
        ylabel="Density",
        palette=custom_palette,
        show=True,
    )
