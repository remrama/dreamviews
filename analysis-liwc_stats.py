"""
draw a single plot of all the LIWC effects.
"""
import os
import numpy as np
import pandas as pd
import pingouin as pg
import config as c

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

import_fname_liwc = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc.tsv")
import_fname_attr = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname_stat = os.path.join(c.DATA_DIR, "results", "analysis-liwc.tsv")
export_fname_plot = os.path.join(c.DATA_DIR, "results", "analysis-liwc.png")

# merge the clean data file and all its attributes with the liwc results
df_liwc = pd.read_csv(import_fname_liwc, sep="\t", encoding="utf-8", index_col="post_id")
df_attr = pd.read_csv(import_fname_attr, sep="\t", encoding="utf-8", index_col="post_id")
df = df_attr.join(df_liwc, how="inner")
assert len(df) == len(df_attr) == len(df_liwc)



##################### statistics

liwc_cats = df_liwc.columns.tolist()

# Average the LD and NLD scores for each user.
# Some users might not have both dream types
# and they'll be removed.
avgs = df.groupby(["user_id", "lucidity"]
    )[liwc_cats].mean(
    ).drop(["ambiguous", "unspecified"], level="lucidity"
    ).rename_axis(columns="category"
    ).pivot_table(index="user_id", columns="lucidity"
    ).dropna()
# avgs.index.get_level_values("user_id").duplicated(keep=False)

# init the results with descriptives, then add stats
descrpt = avgs.agg(["mean", "std", "sem", "min", "max"]).T

wilcoxon_results = []
for cat in liwc_cats:
    ld, nld = avgs[cat][["lucid", "non-lucid"]].T.values
    stats = pg.wilcoxon(ld, nld, alternative="two-sided")
    stats.index = [cat]
    stats["cohen-d"] = pg.compute_effsize(ld, nld, paired=True, eftype="cohen")
    stats["cohen-d_lo"], stats["cohen-d_hi"] = pg.compute_bootci(ld, nld,
        paired=True, func="cohen", method="cper",
        confidence=.95, n_boot=2000, decimals=4)
    stats["n"] = len(ld) # should be the same every time
    wilcoxon_results.append(stats)
wilc = pd.concat(wilcoxon_results
    ).rename_axis("category"
    ).drop(columns="alternative")

# merge descriptives and statistices
results = descrpt.join(wilc, how="inner")

# correct for multiple comparisons
_, fdr = pg.multicomp(results["p-val"].values, method="fdr_bh")
insert_indx = 1 + results.columns.tolist().index("p-val")
results.insert(insert_indx, "p-val_fdr", fdr)

# sort values by the effect size
results = results.sort_values("cohen-d", ascending=False)

results.to_csv(export_fname_stat, sep="\t", encoding="utf-8",
    index=True)



##################### draw

# this assumes no 2 test have the exact same effect size, fair
plot_df = results.drop_duplicates(subset=["cohen-d"]
    ).droplevel("lucidity")

BAR_ARGS = dict(height=1, edgecolor="k", lw=.5, alpha=1)
ERROR_ARGS = dict(ecolor="gainsboro", elinewidth=.5, capsize=0)
GRID_COLOR = "gainsboro"

fig, ax = plt.subplots(figsize=(4.5, 6.5), constrained_layout=True)

yvals = plot_df["cohen-d"].values
y_cis = plot_df[["cohen-d_lo", "cohen-d_hi"]].values
yerrs = np.abs(yvals-y_cis.T)
xvals = np.arange(len(yvals))
pvals = plot_df["p-val"].values
pvals_fdr = plot_df["p-val_fdr"].values
xlabels = plot_df.index.to_numpy()

pass_mcc_mask = pvals_fdr < .01
ld_effects_mask  = pass_mcc_mask & (yvals>0)
nld_effects_mask = pass_mcc_mask & (yvals<0)

NEUTRAL_COLOR = "gainsboro"

for mask, color in zip([ ~pass_mcc_mask, ld_effects_mask, nld_effects_mask],
                       [ NEUTRAL_COLOR, c.COLORS["lucid"], c.COLORS["non-lucid"]]):
    ax.barh(xvals[mask], yvals[mask], xerr=yerrs[:,mask],
        color=color, error_kw=ERROR_ARGS, **BAR_ARGS)


ax.axvline(0, ls="-", color="k", lw=1, zorder=5)
ax.xaxis.grid(True, which="major", linewidth=1, clip_on=False, color=GRID_COLOR)
ax.xaxis.grid(True, which="minor", linewidth=.3, clip_on=False, color=GRID_COLOR)
ax.set_axisbelow(True)

# xmax = max(map(abs, ax.get_xlim()))
xmax = 1.5
xtickmajor = 1
xtickminor = .2
ax.set_xlim(-xmax, xmax)
ax.xaxis.set(major_locator=plt.MultipleLocator(xtickmajor),
    minor_locator=plt.MultipleLocator(xtickminor),
    major_formatter=plt.FuncFormatter(c.no_leading_zeros))

ax.set_xlabel("Effect size (Cohen's $\it{d}$) " +
    "\nnon-lucid$\leftarrow$   $\\rightarrow$lucid         ")
ax.set_ylim(-1, len(xvals))
ax.set_yticks(xvals)
ax.set_yticklabels(xlabels, fontsize=7)
ax.set_ylabel("LIWC category", labelpad=0)
for text in ax.get_yticklabels():
    if text.get_text() in xlabels[pass_mcc_mask]:
        text.set_weight("bold")

# significance markers
for x, p in enumerate(pvals):
    # d = yvals[x]
    # peak = hi if d > 0 else lo
    ci = y_cis[x]
    peak = max(ci, key=abs)
    pchars = "*" * sum([ p < cutoff for cutoff in [.05, .01, .001] ])
    ha = "left" if peak > 0 else "right"
    # offset wrt tip/peak of the bar (values in <textcoords>)
    x_offset = 1 if peak > 0 else -1
    y_offset = -3 # constant, just to get the asterisk centered
    ax.annotate(pchars, xy=(peak, x), xytext=(x_offset, y_offset),
        clip_on=True, textcoords="offset points", ha=ha, va="center")

ax.invert_yaxis()


# export
plt.savefig(export_fname_plot)
plt.savefig(export_fname_plot.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()
