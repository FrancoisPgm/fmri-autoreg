import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import seaborn as sns
from ipywidgets import interact_manual, IntSlider
from neuromaps.transforms import mni152_to_fslr
from neuromaps.datasets import fetch_fslr
from surfplot import Plot


def plot_from_paths(paths, figsize=(15, 8)):
    if isinstance(paths, str):
        paths = [paths]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).rename(columns={"t+": "lag"})
    make_interact_plot(df, figsize)


def remove_nan(x):
    filtered_x = []
    for v in x:
        try:
            is_nan = np.isnan(v)
        except TypeError:
            is_nan = False
        if not is_nan:
            filtered_x.append(v)
    return filtered_x


def make_interact_plot(df, figsize=(15, 8), all_columns=True, ylim=(-0.1, 0.6)):
    cols = list(df.columns)
    kwargs = {}
    for c in cols:
        if c != "checkpoints" and not "file" in c and not "r2" in c and not "loss" in c:
            if all_columns or len(remove_nan(df[c].unique())) > 1:
                kwargs[c] = ["all"] + list(df[c].unique())
    score_columns = [column for column in df.columns if "r2" in column or "loss" in column]
    absciss_columns = [
        column for column in df.columns if not "r2" in column and not "loss" in column
    ]
    if not all_columns:
        absciss_columns = [col for col in absciss_columns if len(remove_nan(df[col].unique())) > 1]

    def plot_fn(
        score,
        absciss,
        abs_log,
        plot_type,
        max_compare,
        tick_rotation,
        legend_y,
        legend_x,
        show_legend,
        save,
        **args,
    ):
        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)
        plt.xlabel(absciss)
        plt.ylabel(score)
        filtered_df = df
        param_str = ""
        for key in args:
            if absciss != key and args[key] != "all":
                if isinstance(args[key], str):
                    filtered_df = filtered_df.query(f"{key}=='{args[key]}'")
                else:
                    filtered_df = filtered_df.query(f"{key}=={args[key]}")
                param_str += f"_{key}={args[key]}"
        param_str = "_with" + param_str if param_str else param_str
        filtered_df = filtered_df.sort_values(by=[absciss])
        x_values = filtered_df[absciss].values
        if abs_log:
            plt.xscale("log")
        plt.xticks(rotation=tick_rotation)
        if "r2" in score:
            plt.ylim(*ylim)

        if plot_type == "max_only":
            if max_compare is not None:
                for val in remove_nan(filtered_df[max_compare].unique()):
                    if isinstance(val, str):
                        query_string = f"{max_compare}=='{val}'"
                    else:
                        query_string = f"{max_compare}=={val}"
                    max_filtered_df = filtered_df.query(query_string)
                    x_values = max_filtered_df[absciss].unique()
                    x_values = remove_nan(x_values)
                    y_values = max_filtered_df.groupby([absciss])[score].max().values
                    plt.scatter(x_values, y_values, label=query_string)
                if show_legend:
                    plt.legend(loc=f"{legend_y} {legend_x}")
                if save:
                    plt.savefig(f"max_{score}_{absciss}_per_{max_compare}{param_str}.png")
            else:
                x_values = filtered_df[absciss].unique()
                x_values = remove_nan(x_values)
                y_values = filtered_df.groupby([absciss])[score].max().values
                plt.scatter(x_values, y_values)
                if save:
                    plt.savefig(f"max_{score}_{absciss}{param_str}.png")
            return None

        elif plot_type == "swarm":
            sns.swarmplot(x=absciss, y=score, data=filtered_df)
        elif plot_type == "violin":
            sns.violinplot(x=absciss, y=score, data=filtered_df)
        elif plot_type == "barplot":
            x_values = filtered_df[absciss].unique()
            x_values = remove_nan(x_values)
            sns.barplot(x=x_values, y=score, data=filtered_df)
        else:
            x_values = filtered_df[absciss].unique()
            x_values = remove_nan(x_values)
            plt.plot(x_values, filtered_df[score].values, "-o")
        plt.show()
        plt.savefig(f"{score}_{absciss}{param_str}.png")
        return None

    interact_manual(
        plot_fn,
        score=score_columns,
        absciss=absciss_columns,
        abs_log=False,
        plot_type=["max_only", "violin", "swarm", "plot", "barplot"],
        max_compare=[None] + absciss_columns,
        tick_rotation=IntSlider(min=0, max=90, step=5, value=20),
        legend_y=["lower", "upper"],
        legend_x=["right", "left"],
        show_legend=True,
        save=False,
        **kwargs,
    )


def make_surf_fig(
    img,
    colorbar=False,
    cmap="cold_hot",
    color_range=(-0.8, 0.8),
    size=(800, 600),
    n_ticks=3,
    fontsize=10,
):
    gii_lh, gii_rh = mni152_to_fslr(img, method="nearest")

    data_lh = gii_lh.agg_data()
    data_rh = gii_rh.agg_data()

    surfaces = fetch_fslr()
    lh, rh = surfaces["inflated"]
    sulc_lh, sulc_rh = surfaces["sulc"]

    p = Plot(lh, rh, size=size)
    p.add_layer({"left": data_lh, "right": data_rh}, cmap=cmap, color_range=color_range)

    kws = dict(
        location="right",
        draw_border=True,
        aspect=10,
        shrink=0.8,
        pad=0.08,
        n_ticks=n_ticks,
        fontsize=fontsize,
    )
    fig = p.build(colorbar=colorbar, cbar_kws=kws)

    return fig
