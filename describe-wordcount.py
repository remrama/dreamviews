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
export_fname_plot = os.path.join(c.DATA_DIR, "results", "describe-wordcount.png")
export_fname_table1 = os.path.join(c.DATA_DIR, "results", "describe-wordcount.tsv")
export_fname_table2 = os.path.join(c.DATA_DIR, "results", "describe-wordcount.tex")

df, _ = c.load_dreamviews_data()


# token counts column already exists, but need to add lemma one
df["lemmacount"] = df["post_lemmas"].str.split().str.len()


# a table counting how many words per dream type
df[["wordcount","lemmacount"]].describe().round(1)
lucidity_wc = df.groupby("lucidity"
    )[["wordcount","lemmacount"]
    ].agg(["mean", "std", "median", "min", "max"]
    ).rename_axis(["tokentype", "stat"], axis=1)

lucidity_wc = lucidity_wc.T.pivot_table(columns="tokentype", index="stat"
    )[["nonlucid", "lucid"]
    ].swaplevel(axis=1).sort_index(axis=1)

lucidity_wc.to_csv(export_fname_table1, sep="\t", encoding="utf-8",
    index=True, float_format="%.1f")

lucidity_wc.columns.names = [None, None]
lucidity_wc.index.name = None

lucidity_wc.to_latex(buf=export_fname_table2, index=True, encoding="utf-8",
    float_format="%.0f")

# counts = df.groupby(["user_id","lucidity"]
#     ).size().rename("n_posts").reset_index()
# # this is a rather useful table
# counts = df.groupby(["user_id","lucidity"]
#     ).size().reset_index(
#     ).pivot(index="user_id", columns="lucidity", values=0
#     ).fillna(0).astype(int)


# LUCIDITY_ORDER = ["unspecified", "ambiguous", "nonlucid", "lucid"]
# COUNT_ORDER = ["word", "lemma"]

LINE_ARGS = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
    "zorder"    : 0,
}

# legend_handles = [ plt.matplotlib.patches.Patch(edgecolor="none",
#         facecolor=c.COLORS[cond], label=cond)
#     for cond in LUCIDITY_ORDER ]

# bins/ticks
# generate bins
N_BINS = 20
bin_min = 0
bin_max = c.MAX_WORDCOUNT
bins = np.linspace(0, bin_max, N_BINS+1)
minor_tick_loc = np.diff(bins).mean()


fig, axes = plt.subplots(nrows=3, figsize=(3, 3),
    sharex=True, sharey=False,
    constrained_layout=True)


BAR_ARGS = {
    "alpha" : 1,
    "color" : "gainsboro",
    "clip_on" : False,
    "bins" : bins,
}

for i, ax in enumerate(axes):

    if i == 2:
        ymax = .003
        for lucidity, series in df.groupby(["user_id", "lucidity"]
                                 ).wordcount.mean(
                                 ).groupby("lucidity"):
            distvals = series.values
            if "lucid" in lucidity:
                linecolor = c.COLORS[lucidity]
                linewidth = 1
                zorder = 2 if lucidity == "lucid" else 1
                ax.hist(distvals, density=True, histtype="step", zorder=zorder,
                    edgecolor=linecolor, linewidth=linewidth, **BAR_ARGS)
    else:
        linecolor = "black"
        linewidth = .5
        alpha = 1
        if i == 0:
            ser = df.wordcount
            ymax = 10000
        else:
            ser = df.groupby("user_id").wordcount.mean()
            ymax = 1000
        distvals = ser.values
        ax.hist(distvals, edgecolor=linecolor, linewidth=linewidth, **BAR_ARGS)

    ax.set_ylabel("# posts", fontsize=10)
    ax.set_ybound(upper=ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, c.MAX_WORDCOUNT)
    ax.tick_params(axis="both", which="both", labelsize=10)
    ax.tick_params(axis="y", which="both", direction="in")
    ax.xaxis.set(major_locator=plt.MultipleLocator(200),
                 minor_locator=plt.MultipleLocator(minor_tick_loc))
    if i == 2:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax),
                     major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    else:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax/2),
                     minor_locator=plt.MultipleLocator(ymax/10))

    # grab some values for text
    n, mean, sd, median = ser.describe().loc[["count", "mean", "std", "50%"]]
    txt_list = [
        fr"$n={n:.0f}$",
        fr"$\bar{{x}}={mean:.1f}$",
        fr"$\sigma_{{\bar{{x}}}}={sd:.1f}$",
        fr"$\tilde{{x}}={median:.0f}$",
    ]
    # txt = "\n".join(txt_list)
    # ax.text(1, 1, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10, linespacing=1)
    if i != 2:
        for j, txt in enumerate(txt_list):
            ytop = 1-.2*j
            ax.text(1, ytop, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10)


    # txt flags
    if i == 0:
        ax.axvline(c.MIN_WORDCOUNT, **LINE_ARGS)
        ax.text(c.MIN_WORDCOUNT+10, 1, "min word cutoff",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10)
        # ax.axvline(c.MAX_WORDCOUNT, **LINE_ARGS)
        # ax.text(c.MAX_WORDCOUNT-10, 1, "max word cutoff",
        #     transform=ax.get_xaxis_transform(),
        #     ha="right", va="top", fontsize=10)


ax.set_xlabel("# words", fontsize=10)

plt.savefig(export_fname_plot)
c.save_hires_figs(export_fname_plot)
plt.close()
