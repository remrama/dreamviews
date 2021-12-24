"""
plot the wordshift graph, from analysis-wordshift output
which is run separately because it takes a long time.

Do some permutation stats too, though
the actual permutations were run in the analysis script.
"""
import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
c.load_matplotlib_settings()


N_WORDS = 30 # number of top ranking words to plot


import_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-wordshift.tsv")
export_fname_table = os.path.join(c.DATA_DIR, "results", "validate-wordshift.tsv")
export_fname_plot  = os.path.join(c.DATA_DIR, "results", "validate-wordshift.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")


####### new dataframe with stats for each token
####### across all sampling iterations

def ci_lo(x):
    return np.quantile(x, .025)
def ci_hi(x):
    return np.quantile(x, .975)

def pval(x):
    fracs = [ np.mean(x<0), np.mean(x>0) ]
    lowest = np.min(fracs)
    return lowest / 2

# add sign to the shift score
df["type2shift_score"] *= df["type2s_ref_diff"].transform(np.sign)

stats_df = df.groupby("token")["type2shift_score"].agg(
    ["mean", ci_lo, ci_hi, pval])

# sort by absolute mean
stats_df = stats_df.sort_values("mean", key=abs, ascending=False)



################ plotting

# make a trimmed copy for plotting
plot_df = stats_df[:N_WORDS].copy()

labels = plot_df.index.tolist()
means  = plot_df["mean"].values
ci     = plot_df[["ci_lo", "ci_hi"]].values.T
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

xlabel = "JSD shift contribution score\n" + r"non-lucid$\leftarrow$$\rightarrow$lucid       "
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

stats_df.to_csv(export_fname_table, index=True, sep="\t", encoding="utf-8")

plt.savefig(export_fname_plot)
c.save_hires_figs(export_fname_plot)
plt.close()
