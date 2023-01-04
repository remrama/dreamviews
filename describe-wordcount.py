"""Count the word (and lemma) frequencies across all posts and broken up by lucidity.

IMPORTS
=======
    - posts, dreamviews-posts.tsv
EXPORTS
=======
    - table of all word and lemma counts,               describe-wordcount.tsv
    - visualization of total word counts per post,      describe-wordcount_perpost.png
    - visualization of total word counts per user,      describe-wordcount_peruser.png
    - visualization of lucid and non-lucid word counts, describe-wordcount_lucidity.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea


import config as c


################################################################################
# SETUP
################################################################################

# Load custom plotting aesthetics.
c.load_matplotlib_settings()

# Pick export locations.
export_path_table = c.DATA_DIR / "derivatives" / "describe-wordcount.tsv"
export_path_plots = [
    c.DATA_DIR / "derivatives" / "describe-wordcount_perpost.png",
    c.DATA_DIR / "derivatives" / "describe-wordcount_peruser.png",
    c.DATA_DIR / "derivatives" / "describe-wordcount_lucidity.png",
]

# Load data.
df = c.load_dreamviews_posts()


################################################################################
# GET FREQUENCIES
################################################################################

# Add column with lemma count (token count column already exists).
df["lemmacount"] = df["post_lemmas"].str.split().str.len()

# Go from wide-to-long format.
token_melt = df.melt(
    value_vars=["wordcount", "lemmacount"],
    var_name="token_type",
    value_name="n",
    id_vars=["post_id", "user_id", "lucidity"],
)
token_melt["token_type"] = token_melt["token_type"].str.rstrip("count")

# Get table of descriptives across the whole corpus that averages within users to account for bias.
total_descr = (token_melt
    .groupby(["user_id", "token_type"]).mean()
    .groupby("token_type").describe()
    .droplevel(level=0, axis=1).rename_axis("metric", axis=1)
)

# Same thing but for lucid and non-lucid labeled posts.
lucid_descr = (token_melt
    .groupby(["user_id", "lucidity", "token_type"]).mean()
    .groupby(["lucidity", "token_type"]).describe()
    .droplevel(level=0, axis=1).rename_axis("metric", axis=1)
)

# Merge them.
total_descr = pd.concat({"combined": total_descr}, names=["lucidity"])
descriptives = pd.concat([total_descr, lucid_descr], axis=0)

# Export.
descriptives.to_csv(export_path_table, float_format="%.1f", index=True, sep="\t", encoding="utf-8")


################################################################################
# PLOTTING
################################################################################

# Define some plotting variables.
n_bins = 20
bin_min = 0
bin_max = c.MAX_WORDCOUNT
bins = np.linspace(0, bin_max, n_bins + 1)
minor_xtick_loc = np.diff(bins).mean()
major_xtick_loc = 200
line_kwargs = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
    "zorder"    : 0,
}
bar_kwargs = {
    "alpha" : 1,
    "color" : "gainsboro",
    "clip_on" : False,
    "bins" : bins,
}

# Loop over axes and word vs lemma counts to draw and export individual plots.
for i, path in enumerate(export_path_plots):

    # Open figure.
    fig, ax = plt.subplots(figsize=(2, 1), constrained_layout=True)

    if i == 2:
        ymax = 0.003
        ylabel = "density"
        for lucidity, series in df.groupby(["user_id", "lucidity"]
                                 ).wordcount.mean(
                                 ).groupby("lucidity"):
            distvals = series.values
            if "lucid" in lucidity:
                linecolor = c.COLORS[lucidity]
                linewidth = 1
                zorder = 2 if lucidity == "lucid" else 1
                ax.hist(
                    distvals,
                    density=True,
                    histtype="step",
                    zorder=zorder,
                    edgecolor=linecolor,
                    linewidth=linewidth,
                    **bar_kwargs,
                )

        # Add legend.
        handles = [
            plt.matplotlib.patches.Patch(
                edgecolor="none", facecolor=c.COLORS[x], label=x.replace("nl", "n-l")
            )
            for x in ["nonlucid", "lucid"]
        ]
        legend = ax.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 0.85),
            loc="upper right",
            frameon=False,
            borderaxespad=0,
            labelspacing=0.1,
            handletextpad=0.2,
        )

    else:
        ylabel = r"$n$ posts"
        linecolor = "black"
        linewidth = 0.5
        if i == 0:
            ser = df.wordcount
            ymax = 10000
        else:
            ser = df.groupby("user_id").wordcount.mean()
            ymax = 1000
        distvals = ser.values
        ax.hist(distvals, edgecolor=linecolor, linewidth=linewidth, **bar_kwargs)

    # Adjust aesthetics.
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_ybound(upper=ymax)
    ax.set_xlabel(r"$n$ words", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, c.MAX_WORDCOUNT)
    ax.tick_params(axis="y", which="both", direction="in")
    ax.xaxis.set(major_locator=plt.MultipleLocator(major_xtick_loc),
                 minor_locator=plt.MultipleLocator(minor_xtick_loc))
    if i == 2:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax),
                     major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    else:
        ax.yaxis.set(major_locator=plt.MultipleLocator(ymax / 2),
                     minor_locator=plt.MultipleLocator(ymax / 10))

    # Grab some values for text.
    n, mean, sd, median = ser.describe().loc[["count", "mean", "std", "50%"]]

    # Draw text.
    text_list = [
        fr"$n={n:.0f}$",
        fr"$\bar{{x}}={mean:.0f}$",
        fr"$\sigma_{{\bar{{x}}}}={sd:.1f}$",
        fr"$\tilde{{x}}={median:.0f}$",
    ]
    if i != 2:
        for j, txt in enumerate(text_list):
            ytop = 1 - 0.2 * j
            ax.text(1, ytop, txt, transform=ax.transAxes, ha="right", va="top")

    # Draw a line and other text.
    ax.axvline(c.MIN_WORDCOUNT, **line_kwargs)
    ax.text(
        c.MIN_WORDCOUNT + 10,
        1,
        "min word cutoff",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
    )

    # Export.
    plt.savefig(path)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
