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
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


### handle i/o and load in data
import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "describe-wordcount.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")


# token counts column already exists, but need to add lemma one
df["n_lemmas"] = df["post_lemmas"].str.split().str.len()

# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().rename("n_posts").reset_index()

# # # this is a rather useful table
# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().reset_index(
# #     ).pivot(index="user_id", columns="lucidity", values=0
# #     ).fillna(0).astype(int)


LUCIDITY_ORDER = ["unspecified", "ambiguous", "non-lucid", "lucid"]

COUNT_ORDER = ["tokens", "lemmas"]

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
bin_max = c.MAX_TOKEN_COUNT
bins = np.linspace(0, bin_max, N_BINS+1)
minor_tick_loc = np.diff(bins).mean()
# I'm gonna do something stupid here. With a min lemma count
# of 3, bins are gross and weird. To make it viewable, I'll
# cut the first bin to be 3-5 while the rest are in gaps of 5.
# This has to be mentioned in fig caption if it stays.
# it doesn't matter if applied to tokens too bc that bin will be empty
bins[0] = c.MIN_LEMMA_COUNT


fig, axes = plt.subplots(nrows=2, figsize=(4, 4),
    sharex=False, sharey=False,
    constrained_layout=True)

for ax, countvar in zip(axes, COUNT_ORDER):

    sea.histplot(data=df,
        x=f"n_{countvar}", hue="lucidity",
        multiple="stack", palette=c.COLORS,
        hue_order=LUCIDITY_ORDER,
        stat="count", element="bars",
        bins=bins, discrete=False,
        linewidth=.5, edgecolor="black",
        legend=False,
        ax=ax)

    ax.set_xlabel(f"# {countvar}", fontsize=10)

    ax.set_ylabel("# posts", fontsize=10)
    ymax = 10000 if countvar=="tokens" else 20000
    ax.set_ybound(upper=ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, c.MAX_TOKEN_COUNT+minor_tick_loc)
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
    if countvar == "lemmas":
        ax.axvline(c.MIN_LEMMA_COUNT, **LINE_ARGS)
        ax.text(c.MIN_LEMMA_COUNT+10, 1, "min lemma cutoff",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10)
    else:
        ax.axvline(c.MIN_TOKEN_COUNT, **LINE_ARGS)
        ax.axvline(c.MAX_TOKEN_COUNT, **LINE_ARGS)
        ax.text(c.MIN_TOKEN_COUNT+10, 1, "min word cutoff",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10)
        ax.text(c.MAX_TOKEN_COUNT-10, 1, "max word cutoff",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=10)


plt.savefig(export_fname)
plt.close()
