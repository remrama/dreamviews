"""plot liwc results based on stats output"""

import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


import_fname = os.path.join(c.DATA_DIR, "results", "analysis-liwc.tsv")
export_fname_plot = os.path.join(c.DATA_DIR, "results", "analysis-liwc.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8", index_col="post_id")


##################### draw

# this assumes no 2 test have the exact same effect size, fair
df = results.drop_duplicates(subset=["cohen-d"]
    ).droplevel("lucidity")

BAR_ARGS = dict(height=1, edgecolor="k", lw=.5, alpha=1)
ERROR_ARGS = dict(ecolor="gainsboro", elinewidth=.5, capsize=0)
GRID_COLOR = "gainsboro"

fig, ax = plt.subplots(figsize=(5, 7.5), constrained_layout=True)

yvals = df["cohen-d"].values
y_cis = df[["cohen-d_lo", "cohen-d_hi"]].values
yerrs = np.abs(yvals-y_cis.T)
xvals = np.arange(len(yvals))
pvals = df["p-val"].values
pvals_fdr = df["p-val_fdr"].values
xlabels = df.index.to_numpy()

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
plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()
