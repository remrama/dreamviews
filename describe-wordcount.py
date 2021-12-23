"""
plot word counts
lemmas or tokens
"""
import os
import numpy as np
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


### handle i/o and load in data
export_fname = os.path.join(c.DATA_DIR, "results", "describe-wordcount.png")

df, _ = c.load_dreamviews_data()


# token counts column already exists, but need to add lemma one
df["lemmacount"] = df["post_lemmas"].str.split().str.len()

# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().rename("n_posts").reset_index()

# # # this is a rather useful table
# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().reset_index(
# #     ).pivot(index="user_id", columns="lucidity", values=0
# #     ).fillna(0).astype(int)


LUCIDITY_ORDER = ["unspecified", "ambiguous", "nonlucid", "lucid"]

COUNT_ORDER = ["word", "lemma"]

LINE_ARGS = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
    "zorder"    : 0,
}

legend_handles = [ plt.matplotlib.patches.Patch(edgecolor="none",
        facecolor=c.COLORS[cond], label=cond)
    for cond in LUCIDITY_ORDER ]

# bins/ticks
# generate bins
N_BINS = 20
bin_min = 0
bin_max = c.MAX_WORDCOUNT
bins = np.linspace(0, bin_max, N_BINS+1)
minor_tick_loc = np.diff(bins).mean()


fig, axes = plt.subplots(nrows=2, figsize=(4, 4),
    sharex=False, sharey=False,
    constrained_layout=True)

for ax, countvar in zip(axes, COUNT_ORDER):

    sea.histplot(data=df,
        x=f"{countvar}count", hue="lucidity",
        multiple="stack", palette=c.COLORS,
        hue_order=LUCIDITY_ORDER,
        stat="count", element="bars",
        bins=bins, discrete=False,
        linewidth=.5, edgecolor="black",
        legend=False,
        ax=ax)

    ax.set_xlabel(f"# {countvar}s", fontsize=10)

    ax.set_ylabel("# posts", fontsize=10)
    ymax = 10000
    ax.set_ybound(upper=ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, c.MAX_WORDCOUNT)
    ax.tick_params(axis="both", which="both", labelsize=10)
    # ax.tick_params(axis="y", which="both", direction="in")
    ax.xaxis.set(major_locator=plt.MultipleLocator(200),
                 minor_locator=plt.MultipleLocator(minor_tick_loc))
    ax.yaxis.set(major_locator=plt.MultipleLocator(5000),
                 minor_locator=plt.MultipleLocator(1000))

    legend = ax.legend(handles=legend_handles,
        loc="upper left", bbox_to_anchor=(.55, .9),
        borderaxespad=0, frameon=False,
        labelspacing=.2, handletextpad=.2)


    # txt flags
    if countvar == "word":
        ax.axvline(c.MIN_WORDCOUNT, **LINE_ARGS)
        # ax.axvline(c.MAX_WORDCOUNT, **LINE_ARGS)
        ax.text(c.MIN_WORDCOUNT+10, 1, "min word cutoff",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10)
        # ax.text(c.MAX_WORDCOUNT-10, 1, "max word cutoff",
        #     transform=ax.get_xaxis_transform(),
        #     ha="right", va="top", fontsize=10)


plt.savefig(export_fname)
c.save_hires_figs(export_fname, [".svg", ".pdf"])
plt.close()
