import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram_and_kde(
    data=None,
    title="Histogram and KDE Plot",
    xlabel="",
    ylable="",
    save_path="",
    show=False,
    save=True,
    *,
    # Vector variables
    x=None,
    y=None,
    hue=None,
    weights=None,
    # Histogram computation parameters
    stat="count",
    bins="auto",
    binwidth=None,
    binrange=None,
    discrete=None,
    cumulative=False,
    common_bins=True,
    common_norm=True,
    # Histogram appearance parameters
    multiple="layer",
    element="bars",
    fill=True,
    shrink=1,
    # Histogram smoothing with a kernel density estimate
    kde=False,
    kde_kws=None,
    line_kws=None,
    # Bivariate histogram parameters
    thresh=0,
    pthresh=None,
    pmax=None,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    # Hue mapping parameters
    palette=None,
    hue_order=None,
    hue_norm=None,
    color=None,
    # Axes information
    log_scale=None,
    legend=True,
    ax=None,
    # Other appearance keywords
    **kwargs,
):
    sns.histplot(
        data,  # Vector variables
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
        legend=legend,
        ax=ax,
        # Other appearance keywords
        **kwargs,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    if save:
        plt.savefig(save_path)
    if show:
        plt.show()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

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
        ylable="Density",
        palette=custom_palette,
        show=True,
    )
