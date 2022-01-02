"""
"""
import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
c.load_matplotlib_settings()

N_WORDS = 30 # number of top ranking words to plot

METRIC = "jsd"
import_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_scores.tsv")

METRIC = "fear"
import_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_scores_nightmares.tsv")

export_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_scores.png")

df = pd.read_csv(import_fname, header=[0,1], index_col=0, sep="\t", encoding="utf-8")


shift_col = f"{METRIC}-type2shift_score"

# sort for main contributors of interest
df = df.sort_values((shift_col, "mean"), key=abs, ascending=False)

################ plotting

# make a trimmed copy for plotting
plot_df = df[:N_WORDS].copy()

labels = plot_df.index.tolist()
means  = plot_df[(shift_col, "mean")].values
ci     = plot_df[[(shift_col, "ci_lo"), (shift_col, "ci_hi")]].values.T
errors = abs(means-ci)
locs   = np.arange(means.size) + 1

colors = [ c.COLORS["lucid"] if x>0 else c.COLORS["nonlucid"] for x in means]


ERROR_ARGS = {
    "capsize"    : 0,
    "elinewidth" : .5,
}

BAR_ARGS = {
    "linewidth" : .5,
    "height"    : .8,
}

fig, ax = plt.subplots(figsize=(2.5, N_WORDS/6), constrained_layout=True)
ax.invert_yaxis()
ax.barh(locs, means, xerr=errors, color=colors,
    error_kw=ERROR_ARGS, **BAR_ARGS)

TXT_BUFF = .005
for i, txt in enumerate(labels):
    yloc = i + 1
    if means[i] > 0:
        # to the right, align left of txt against high CI
        ha = "left"
        xloc = ci[1, i] + TXT_BUFF
    else:
        # to the left, align right of txt against low CI
        ha = "right"
        xloc = ci[0, i] - TXT_BUFF
    ax.text(xloc, yloc, txt, ha=ha, va="center", fontsize=8)

ax.axvline(0, linewidth=1, color="black")

xlabel = f"{METRIC} shift contribution score\n" + r"non-lucid$\leftarrow$$\rightarrow$lucid       "
ax.set_xlabel(xlabel)
ax.set_ylabel("lemma contribution rank")
XLIM = .04
ax.set_xlim(-XLIM, XLIM)
ax.set_ylim(N_WORDS+1, 0)

yticklocs = np.linspace(0, N_WORDS, int(N_WORDS/10+1))
yticklocs[0] = 1
ax.yaxis.set(major_locator=plt.FixedLocator(yticklocs))
ax.xaxis.set(major_locator=plt.MultipleLocator(XLIM),
             minor_locator=plt.MultipleLocator(.01),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))



########## export

# plt.savefig(export_fname)
# c.save_hires_figs(export_fname)
# plt.close()
