"""Analyze traditional LIWC scores between lucid and non-lucid dreams.

IMPORTS
=======
    - original post info,         dreamviews-posts.tsv
    - traditionally LIWCed posts, validate-liwc_scores.tsv
EXPORTS
=======
    - descriptives table, validate-liwc_scores-descr.tsv
    - statistics table,   validate-liwc_scores-stats.tsv
    - visualization,      validate-liwc_scores-plot.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from tqdm import tqdm

import config as c

c.load_matplotlib_settings()

LIWC_CATS = ["insight", "agency"]
LUCID_ORDER = ["nonlucid", "lucid"]

EXPORT_STEM = "validate-liwc_scores"
import_path_liwc = c.tables_dir / "validate-liwc_scores.tsv"
export_stem_descr = f"{EXPORT_STEM}-descr"
export_stem_stats = f"{EXPORT_STEM}-stats"
export_stem_plot = f"{EXPORT_STEM}-plot"

########################## I/O

# merge the clean data file and all its attributes with the liwc results
df = c.load_dreamviews_posts()
df_attr = df.set_index("post_id")
df_liwc = pd.read_csv(import_path_liwc, index_col="post_id", sep="\t", encoding="utf-8")
df = df_attr.join(df_liwc, how="inner")
assert len(df) == len(df_attr) == len(df_liwc), "Should all be same length after joining"

# Average the LD and NLD scores for each user
# Users without both dream types will be removed
avgs = (
    df.groupby(["user_id", "lucidity"])[LIWC_CATS]
    .mean()
    .drop(["ambiguous", "unspecified"], level="lucidity")
    .rename_axis(columns="category")
    .pivot_table(index="user_id", columns="lucidity")
    .dropna()
    .multiply(100)
)  # convert to percentages
# avgs.index.get_level_values("user_id").duplicated(keep=False)

########################## get descriptives

descriptives = avgs.agg(["mean", "std", "sem", "min", "max"]).T

# export
c.export_table(descriptives, export_stem_descr, float_format="%.3f")

########################## run statistics
#### Repeated-measures test
#### (subjects were already averaged and such, in the I/O section)

# loop over each LIWC category, running test and getting effect size at each
COMPUTE_BOOTCI_KWARGS = dict(
    paired=True,
    func="cohen",
    method="cper",
    confidence=0.95,
    n_boot=2000,
    decimals=5,
    seed=5,
)

wilcoxon_results = []
for cat in tqdm(LIWC_CATS, desc="LIWC stats"):
    ld, nld = avgs[cat][["lucid", "nonlucid"]].T.values
    stats_ = pg.wilcoxon(ld, nld, alternative="two-sided")
    stats_.index = [cat]
    stats_["cohen-d"] = pg.compute_effsize(ld, nld, paired=True, eftype="cohen")
    stats_["cohen-d_lo"], stats_["cohen-d_hi"] = pg.compute_bootci(ld, nld, **COMPUTE_BOOTCI_KWARGS)
    stats_["n"] = len(ld)  # should be the same every time
    wilcoxon_results.append(stats_)

# combine into one dataframe
stats = pd.concat(wilcoxon_results).rename_axis("category").drop(columns="alternative")

# export
c.export_table(stats, export_stem_stats)

########################## plot visualization
#### Barplot,
#### one bar for Agency the other for Insight
#### It's a little overly complicated to get
#### the hatches in the y-axis (that's why there are two different axes)

# define some values for aesthetics
BAR_ARGS = dict(width=1, linewidth=1, edgecolor="k", alpha=1)
ERROR_ARGS = dict(ecolor="black", elinewidth=1, capsize=1, capthick=1)
SIG_LINEWIDTH = 1
YMIN = 1.9
YMAX = 3.4
FIGSIZE = (1.7, 2)
GRIDSPEC_ARGS = dict(
    height_ratios=[15, 1], hspace=0.04, top=0.99, right=0.96, bottom=0.1, left=0.16
)

# generate data for plotting
row_indx = pd.MultiIndex.from_product([LIWC_CATS, LUCID_ORDER])
yvals = descriptives.loc[row_indx, "mean"].values
yerrs = descriptives.loc[row_indx, "sem"].values
xvals = [x + 0.5 if x > 1 else x for x in range(yvals.size)]
cvals = [c.COLORS[luc] for cat, luc in row_indx]
xticks = [sum(xvals[:2]) / 2, sum(xvals[2:]) / 2]

# open the figure
fig, axes = plt.subplots(
    nrows=2, figsize=FIGSIZE, sharex=True, sharey=False, gridspec_kw=GRIDSPEC_ARGS
)

# plot data on both axes
for ax in axes:
    ax.bar(xvals, yvals, yerr=yerrs, color=cvals, error_kw=ERROR_ARGS, **BAR_ARGS)

# play with axis stuff - labels, ticks, and limits
ax1, ax2 = axes  # axis 2 is for the low/blanked axes, to make hatches
ax2.set_xticks(xticks)
ax2.set_xticklabels(LIWC_CATS)
ax2.set_xlim(min(xvals) - 1, max(xvals) + 1)
ax1.set_ylabel(r"total word category %")
ax1.set_ylim(YMIN, YMAX)
ax2.set_ylim(0, 0.01)  # arbitrarily low, just wanna avoid any ticks
ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(bottom=False)

# draw the slanted hatch lines
D = 0.5  # proportion vertical to horizontal extent of slanted line (0=flat, increasing rotates ccw)
kwargs = dict(
    marker=[(-1, -D), (1, D)],
    markersize=8,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# more tick stuff
for ax in axes:
    ax.yaxis.set(
        major_locator=plt.MultipleLocator(1),
        minor_locator=plt.MultipleLocator(0.2),
        major_formatter=plt.FuncFormatter(c.no_leading_zeros),
    )
    ax.tick_params(which="both", axis="y", direction="in")
    ax.yaxis.grid(True, which="major", linewidth=1, clip_on=False, color="gainsboro")
    ax.yaxis.grid(True, which="minor", linewidth=0.3, clip_on=False, color="gainsboro")
    ax.set_axisbelow(True)

# draw significance markers
for cat, xloc in zip(LIWC_CATS, xticks, strict=True):
    pval = stats.loc[cat, "p_val"]
    sigchars = "*" * sum([pval < cutoff for cutoff in (0.05, 0.01, 0.001)])
    yloc = descriptives.loc[cat, ["mean", "sem"]].sum(axis=1).max()
    yloc += 0.1
    ax1.text(xloc, yloc + 0.01, sigchars, fontsize=10, weight="bold", ha="center", va="center")
    ax1.hlines(
        y=yloc,
        xmin=xloc - BAR_ARGS["width"] / 2,
        xmax=xloc + BAR_ARGS["width"] / 2,
        lw=SIG_LINEWIDTH,
        color="k",
        capstyle="round",
    )

# legend
legend_handles = [
    plt.matplotlib.patches.Patch(facecolor=c.COLORS[label], edgecolor="none", label=label)
    for label in LUCID_ORDER
]
legend = ax1.legend(
    handles=legend_handles,
    frameon=False,
    borderaxespad=0,
    loc="upper left",
    bbox_to_anchor=(0.05, 0.98),
    labelspacing=0.1,
    handletextpad=0.2,
)

# Export
c.export_fig(fig, export_stem_plot)
