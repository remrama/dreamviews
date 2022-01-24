"""Visualize word-level LIWC scores between lucid and non-lucid dreams.

IMPORTS
=======
    - effect sizes (d) for top words from each category, results/validate-liwc_wordscores-stats.tsv
    - stats output for traditional LIWC,                 results/validate-liwc_scores-stats.tsv
EXPORTS
=======
    - visualization of a category, results/validate-liwc_wordscores_<category>-plot.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
c.load_matplotlib_settings()


# handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--category", choices=["insight", "agency"], type=str, required=True)
args = parser.parse_args()

CATEGORY = args.category

TOP_N = 10 # might be more stored, only plot this many


######################## I/O

import_fname = os.path.join(c.DATA_DIR, "results", "validate-liwc_wordscores-stats.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", f"validate-liwc_wordscores_{CATEGORY}-plot.png")
import_fname_full_liwc = os.path.join(c.DATA_DIR, "results", "validate-liwc_scores-stats.tsv")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# load full category liwc results (to draw on top row)
liwccats = pd.read_csv(import_fname_full_liwc, sep="\t", encoding="utf-8",
    index_col="category", usecols=["category", "cohen-d", "cohen-d_lo", "cohen-d_hi"])


######################## plot

# # identify categories to plot
# category_columns = [ c for c in df if "rank" in c ]
# n_cats = len(category_columns)

# define some variables for aesthetics
BAR_ARGS = dict(height=.8, edgecolor="k", lw=.5, alpha=1)
ERROR_ARGS = dict(ecolor="k", elinewidth=.5, capsize=0)
GRID_COLOR = "gainsboro"
YTICK_GAP = 10
yticklocs = np.linspace(0, TOP_N, int(TOP_N/YTICK_GAP+1))
yticklocs[0] = 1
XLIM = 1.4 # xaxis is centered at 0, this will be balanced on either side
TXT_BUFF = .05 # space between tip of bar and text
FIG_SIZE = (1.7, 2.5) #(3.2, 2.5)
GRIDSPEC_ARGS = {
    "height_ratios" : [2, TOP_N],
    "hspace"        : 0,
    # "wspace"        : .05,
    "top"           : .99,
    "right"         : .99,
    "bottom"        : .18,
    "left"          : .15,
}

# open figure
# fig, axes = plt.subplots(nrows=2, ncols=n_cats,
#     figsize=FIG_SIZE, sharex=True, gridspec_kw=GRIDSPEC_ARGS)
fig, (topax, ax) = plt.subplots(nrows=2, figsize=FIG_SIZE,
    sharex=True, gridspec_kw=GRIDSPEC_ARGS)


# # loop over categories and draw
# for i, col in enumerate(category_columns):

# # grab axes, top is for single total effect, bottom for words
# topax, ax = axes[:, i]

## draw total LIWC effect on the top axis
# category = col.split("_")[0]
dval = liwccats.loc[CATEGORY, "cohen-d"]
d_ci = liwccats.loc[CATEGORY, ["cohen-d_lo", "cohen-d_hi"]].values
derr = np.abs(dval-d_ci).reshape(2,1)
color = c.COLORS["lucid"] if dval>0 else c.COLORS["nonlucid"]
topax.barh(0, dval, xerr=derr,
    color=color, error_kw=ERROR_ARGS, **BAR_ARGS)

## draw word-level LIWC effects on bottom

# grab top N tokens for THIS category and sort it
column_name = CATEGORY + "_rank"
subdf = df.loc[df[column_name].notna()
    ].sort_values(column_name, ascending=True
    )[:TOP_N]

# generate relevant plot info
dvals = subdf["cohen-d"].values
d_cis = subdf[["cohen-d_lo", "cohen-d_hi"]].values
derrs = np.abs(dvals-d_cis.T)
labelvals = np.arange(len(dvals)) + 1
labels = subdf["token"].tolist()
colors = [ c.COLORS["lucid"] if d>0 else c.COLORS["nonlucid"] for d in dvals ]
# draw word-level effects
ax.barh(labelvals, dvals, xerr=derrs, color=colors, error_kw=ERROR_ARGS, **BAR_ARGS)

# draw the text of each word next to its appopriate bar
for i, txt in enumerate(labels):
    yloc = i + 1
    if dvals[i] > 0: # to the right, align left of txt against high CI
        xloc = d_cis[i,1] + TXT_BUFF
        ha_align = "left"
    else: # to the left, align right of txt against low CI
        xloc = d_cis[i,0] - TXT_BUFF
        ha_align = "right"
    # replace asterisks with better (unicode) asterisks
    txt = txt.replace("*", "\u204E")
    ax.text(xloc, yloc, txt,
        fontstyle="italic", ha=ha_align, va="center")

##### a e s t h e t i c s
for ax_ in [ax, topax]:
    ax_.axvline(0, color="black", linewidth=1, zorder=5)
    ax_.set_axisbelow(True)
# ax.invert_yaxis()
ax.set_ylim(labelvals.max()+1, labelvals.min()-1)
ax.xaxis.grid(True, which="major", linewidth=1, clip_on=False, color=GRID_COLOR)
ax.xaxis.grid(True, which="minor", linewidth=.3, clip_on=False, color=GRID_COLOR)
# xmax = max(map(abs, ax.get_xlim()))
ax.set_xlim(-XLIM, XLIM)
ax.set_xlabel("Cohen's $\it{d}$ effect size"
    + "\nnon-lucid$\leftarrow$   $\\rightarrow$lucid        ", labelpad=2)
ax.xaxis.set(major_locator=plt.MultipleLocator(.5),
             minor_locator=plt.MultipleLocator(.1),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))
ax.set_ylabel("word rank", labelpad=-6)
ax.yaxis.set(major_locator=plt.FixedLocator(yticklocs))
topax.set_ylim(-1, 1)
topax.yaxis.set(major_locator=plt.NullLocator())
topax.tick_params(which="both", left=False, bottom=False)
topax.spines["bottom"].set_visible(False)
for side in ["left", "top", "right"]:
    topax.spines[side].set_visible(False)
xloc_txt = .02 if dval > 0 else .98
ha_align = "left" if dval > 0 else "right"
topax.text(xloc_txt, .95,
    f"{CATEGORY} total",
    transform=topax.transAxes,
    fontweight="bold",
    ha=ha_align, va="top")
ax.text(xloc_txt, .99,
    f"Individual\nword effects",
    transform=ax.transAxes,
    ha=ha_align, va="top")


# export
plt.savefig(export_fname)
c.save_hires_figs(export_fname)
plt.close()