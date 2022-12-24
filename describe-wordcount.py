"""Count the word (and lemma) frequencies across all posts
and broken up by lucidity.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - table of all word and lemma counts,               results/describe-wordcount.tsv
    - visualization of total word counts per post,      results/describe-wordcount_perpost.png
    - visualization of total word counts per user,      results/describe-wordcount_peruser.png
    - visualization of lucid and non-lucid word counts, results/describe-wordcount_lucidity.png
"""
import numpy as np
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


# I/O
export_path_table = c.DATA_DIR / "results" / "describe-wordcount.tsv"
df = c.load_dreamviews_posts()

# token counts column already exists, but need to add lemma one
df["lemmacount"] = df["post_lemmas"].str.split().str.len()


################################ generate tables for export (not used for plotting)

# first make life easier by melting the word and lemma counts
token_melt = df.melt(value_vars=["wordcount", "lemmacount"],
    var_name="token_type", value_name="n",
    id_vars=["post_id", "user_id", "lucidity"])
token_melt["token_type"] = token_melt["token_type"].str.rstrip("count")

# get a table of descriptives across the whole corpus
# that averages within users to account for that bias
total_descr = token_melt.groupby(["user_id", "token_type"]).mean(
    ).groupby("token_type").describe(
    ).droplevel(level=0, axis=1).rename_axis("metric", axis=1)
# # without averaging across users
# total_descr = token_melt.groupby("token_type").n.describe()

# same thing but get for lucid and non-lucid labeled posts
lucid_descr = token_melt.groupby(["user_id", "lucidity", "token_type"]).mean(
    ).groupby(["lucidity", "token_type"]).describe(
    ).droplevel(level=0, axis=1).rename_axis("metric", axis=1)

## merge them

# modify index on total_descr by giving it a lucidity value to merge
total_descr = pd.concat({"combined": total_descr}, names=["lucidity"])
# total_descr = pd.concat([total_descr], keys=["total"], names=["lucidity"])

descriptives = pd.concat([total_descr, lucid_descr], axis=0)

# export
descriptives.to_csv(export_path_table, float_format="%.1f", index=True, sep="\t", encoding="utf-8")

# # a table counting how many words per dream type
# lucidity_wc = df.groupby("lucidity"
#     )[["wordcount","lemmacount"]
#     ].agg(["mean", "std", "median", "min", "max"]
#     ).rename_axis(["tokentype", "stat"], axis=1)

# lucidity_wc = lucidity_wc.T.pivot_table(columns="tokentype", index="stat"
#     )[["nonlucid", "lucid"]
#     ].swaplevel(axis=1).sort_index(axis=1)

# lucidity_wc.to_csv(export_fname_table, float_format="%.1f", index=True, sep="\t", encoding="utf-8")

# counts = df.groupby(["user_id","lucidity"]
#     ).size().rename("n_posts").reset_index()
# # this is a rather useful table
# counts = df.groupby(["user_id","lucidity"]
#     ).size().reset_index(
#     ).pivot(index="user_id", columns="lucidity", values=0
#     ).fillna(0).astype(int)

################################ visualizeeeeee

# LUCIDITY_ORDER = ["unspecified", "ambiguous", "nonlucid", "lucid"]
# COUNT_ORDER = ["word", "lemma"]
# legend_handles = [ plt.matplotlib.patches.Patch(edgecolor="none",
#         facecolor=c.COLORS[cond], label=cond)
#     for cond in LUCIDITY_ORDER ]

# define some plotting variables
N_BINS = 20
bin_min = 0
bin_max = c.MAX_WORDCOUNT
bins = np.linspace(0, bin_max, N_BINS+1)
minor_tick_loc = np.diff(bins).mean()
MAJOR_XTICK_LOC = 200
LINE_ARGS = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
    "zorder"    : 0,
}
BAR_ARGS = {
    "alpha" : 1,
    "color" : "gainsboro",
    "clip_on" : False,
    "bins" : bins,
}

# crappy quick fix
export_paths = [
    c.DATA_DIR / "results" / "describe-wordcount_perpost.png",
    c.DATA_DIR / "results" / "describe-wordcount_peruser.png",
    c.DATA_DIR / "results" / "describe-wordcount_lucidity.png",
]

# open figure
# fig, axes = plt.subplots(nrows=3, figsize=(2.5, 3),
#     sharex=True, sharey=False,
#     constrained_layout=True)

# Loop over axes and word vs lemma counts.
for i, path in enumerate(export_paths):

    fig, ax = plt.subplots(figsize=(2, 1), constrained_layout=True)

    if i == 2:
        ymax = .003
        ylabel = "density"
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
        # legend
        handles = [ plt.matplotlib.patches.Patch(edgecolor="none",
                facecolor=c.COLORS[x], label=x.replace("nl", "n-l"))
            for x in ["nonlucid", "lucid"] ]
        legend = ax.legend(handles=handles,
            bbox_to_anchor=(1.05, .85), loc="upper right",
            frameon=False, borderaxespad=0,
            labelspacing=.1, handletextpad=.2)

    else:
        ylabel = r"$n$ posts"
        linecolor = "black"
        linewidth = .5
        # alpha = 1
        if i == 0:
            ser = df.wordcount
            ymax = 10000
        else:
            ser = df.groupby("user_id").wordcount.mean()
            ymax = 1000
        distvals = ser.values
        ax.hist(distvals, edgecolor=linecolor, linewidth=linewidth, **BAR_ARGS)

    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_ybound(upper=ymax)
    ax.set_xlabel(r"$n$ words", labelpad=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, c.MAX_WORDCOUNT)
    ax.tick_params(axis="y", which="both", direction="in")
    ax.xaxis.set(major_locator=plt.MultipleLocator(MAJOR_XTICK_LOC),
                 minor_locator=plt.MultipleLocator(minor_tick_loc))
    if i == 2:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax),
                     major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    else:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax/2),
                     minor_locator=plt.MultipleLocator(ymax/10))

    # Grab some values for text.
    n, mean, sd, median = ser.describe().loc[["count", "mean", "std", "50%"]]
    txt_list = [
        fr"$n={n:.0f}$",
        fr"$\bar{{x}}={mean:.0f}$",
        fr"$\sigma_{{\bar{{x}}}}={sd:.1f}$",
        fr"$\tilde{{x}}={median:.0f}$",
    ]
    # txt = "\n".join(txt_list)
    # ax.text(1, 1, txt, transform=ax.transAxes, ha="right", va="top", fontsize=10, linespacing=1)
    if i != 2:
        for j, txt in enumerate(txt_list):
            ytop = 1-.2*j
            ax.text(1, ytop, txt, transform=ax.transAxes, ha="right", va="top")

    # txt flags
    # if i == 0:
    ax.axvline(c.MIN_WORDCOUNT, **LINE_ARGS)
    ax.text(c.MIN_WORDCOUNT+10, 1, "min word cutoff",
        transform=ax.get_xaxis_transform(),
        ha="left", va="top")
    # ax.axvline(c.MAX_WORDCOUNT, **LINE_ARGS)
    # ax.text(c.MAX_WORDCOUNT-10, 1, "max cutoff",
    #     transform=ax.get_xaxis_transform(),
    #     ha="right", va="top")

    # Export.
    plt.savefig(path)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
