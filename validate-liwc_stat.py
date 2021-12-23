"""stats and bargraph for total
insight and agency LIWC LD vs NLD
"""
import os
import tqdm
import pandas as pd
import pingouin as pg
import config as c

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


LIWC_CATS = ["insight", "agency"]
LUCID_ORDER = ["non-lucid", "lucid"]


import_fname_liwc = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc.tsv")
import_fname_attr = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname_table = os.path.join(c.DATA_DIR, "derivatives", "validate-liwc.tsv")
export_fname_stats = os.path.join(c.DATA_DIR, "results", "validate-liwc.tsv")
export_fname_plot  = os.path.join(c.DATA_DIR, "results", "validate-liwc.png")


# merge the clean data file and all its attributes with the liwc results
df_liwc = pd.read_csv(import_fname_liwc, sep="\t", encoding="utf-8", index_col="post_id")
df_attr = pd.read_csv(import_fname_attr, sep="\t", encoding="utf-8", index_col="post_id")
df = df_attr.join(df_liwc, how="inner")
assert len(df) == len(df_attr) == len(df_liwc)


# Average the LD and NLD scores for each user.
# Some users might not have both dream types
# and they'll be removed.
avgs = df.groupby(["user_id", "lucidity"]
    )[LIWC_CATS].mean(
    ).drop(["ambiguous", "unspecified"], level="lucidity"
    ).rename_axis(columns="category"
    ).pivot_table(index="user_id", columns="lucidity"
    ).dropna(
    ).multiply(100) # convert to percentages
# avgs.index.get_level_values("user_id").duplicated(keep=False)


############### descriptives

# init the results with descriptives, then add stats
descriptives = avgs.agg(["mean", "std", "sem", "min", "max"]).T

# export
descriptives.to_csv(export_fname_table, sep="\t", encoding="utf-8",
    na_rep="NA", index=True, float_format="%.3f")


############### stats

wilcoxon_results = []
for cat in tqdm.tqdm(LIWC_CATS, desc="wilcoxon tests"):
    ld, nld = avgs[cat][["lucid", "non-lucid"]].T.values
    stats_ = pg.wilcoxon(ld, nld, alternative="two-sided")
    stats_.index = [cat]
    stats_["cohen-d"] = pg.compute_effsize(ld, nld, paired=True, eftype="cohen")
    stats_["cohen-d_lo"], stats_["cohen-d_hi"] = pg.compute_bootci(ld, nld,
        paired=True, func="cohen", method="cper",
        confidence=.95, n_boot=2000, decimals=4)
    stats_["n"] = len(ld) # should be the same every time
    wilcoxon_results.append(stats_)
stats = pd.concat(wilcoxon_results
    ).rename_axis("category"
    ).drop(columns="alternative")

# export
stats.to_csv(export_fname_stats, sep="\t", encoding="utf-8",
    na_rep="NA", index=True, float_format="%.5f")



########### plot


BAR_ARGS = dict(width=1, linewidth=1, edgecolor="k", alpha=1)
ERROR_ARGS = dict(ecolor="black", elinewidth=1, capsize=1, capthick=1)

row_indx = pd.MultiIndex.from_product([LIWC_CATS, LUCID_ORDER])
yvals = descriptives.loc[row_indx, "mean"].values
yerrs = descriptives.loc[row_indx, "sem"].values
xvals = [ x+.5 if x>1 else x for x in range(yvals.size) ]
cvals = [ c.COLORS[luc] for cat, luc in row_indx ]

xticks = [ sum(xvals[:2])/2, sum(xvals[2:])/2 ]

YMIN = 1.9
YMAX = 3.4
HEIGHT_RATIOS = [15, 1]

fig, axes = plt.subplots(nrows=2, figsize=(3, 4),
    sharex=True, sharey=False,
    gridspec_kw=dict(hspace=.04, height_ratios=HEIGHT_RATIOS,
        top=.98, right=.95, bottom=.1, left=.15))

# plot data on both axes
for ax in axes:
    ax.bar(xvals, yvals, yerr=yerrs, color=cvals, error_kw=ERROR_ARGS, **BAR_ARGS)


ax1, ax2 = axes

ax2.set_xticks(xticks)
ax2.set_xticklabels(LIWC_CATS, fontsize=10)
ax2.set_xlim(min(xvals)-1, max(xvals)+1)

ax1.set_ylabel(r"% of total word count", fontsize=10)
ax1.set_ylim(YMIN, YMAX)
ax2.set_ylim(0, .01) # arbitrarily low, just avoid any ticks

ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.tick_params(bottom=False)

D = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -D), (1, D)], markersize=12,
              linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

for ax in axes:
    ax.yaxis.set(major_locator=plt.MultipleLocator(1),
        minor_locator=plt.MultipleLocator(.2),
        major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    ax.tick_params(which="both", axis="y", direction="in")
    ax.yaxis.grid(True, which="major", linewidth=1, clip_on=False, color="gainsboro")
    ax.yaxis.grid(True, which="minor", linewidth=.3, clip_on=False, color="gainsboro")
    ax.set_axisbelow(True)

# asterisks
for cat, xloc in zip(LIWC_CATS, xticks):
    pval = stats.loc[cat, "p-val"]
    sigchars = "*" * sum([ pval<cutoff for cutoff in (.05, .01, .001) ])
    yloc = descriptives.loc[cat, ["mean", "sem"]].sort_values("mean").sum(axis=1)[-1]
    yloc += .1
    ax1.text(xloc, yloc, sigchars, fontsize=14,
        weight="bold", ha="center", va="center")
    ax1.hlines(y=yloc, xmin=xloc-BAR_ARGS["width"]/2,
        xmax=xloc+BAR_ARGS["width"]/2, lw=2, color="k", capstyle="round")

# legend
legend_handles = [ plt.matplotlib.patches.Patch(
        facecolor=c.COLORS[l], edgecolor="none", label=l)
    for l in LUCID_ORDER ]
legend = ax1.legend(handles=legend_handles,
    frameon=False, borderaxespad=0,
    fontsize=10,
    loc="upper left", bbox_to_anchor=(.05, .98),
    labelspacing=.2, handletextpad=.2)



# export
plt.savefig(export_fname_plot)
plt.close()    
