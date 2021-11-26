"""plot main liwc effects and their individual word contributions
"""
import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


N_WORDS = 10 # might be more stored, only plot this many

import_fname = os.path.join(c.DATA_DIR, "results", "analysis-liwc_tokens.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "analysis-liwc_tokens.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# also add in full category liwc results (for top row)
import_fname_fullliwc = os.path.join(c.DATA_DIR, "results", "analysis-liwc.tsv")
liwccats = pd.read_csv(import_fname_fullliwc, sep="\t", encoding="utf-8",
    index_col="category",
    usecols=["category", "cohen-d", "cohen-d_lo", "cohen-d_hi"])
liwccats = liwccats.drop_duplicates()



category_columns = [ c for c in df if "rank" in c ]

n_cats = len(category_columns)

BAR_ARGS = dict(height=.8, edgecolor="k", lw=.5, alpha=1)
ERROR_ARGS = dict(ecolor="k", elinewidth=.5, capsize=0)
GRID_COLOR = "gainsboro"

GRIDSPEC_KWS = {
    "height_ratios" : [2, N_WORDS],
    "hspace"        : 0,
    "wspace"        : .4,
    "top"           : .98,
    "right"         : .98,
    "bottom"        : .15,
    "left"          : .08,
}

YTICK_GAP = 10
yticklocs = np.linspace(0, N_WORDS, int(N_WORDS/YTICK_GAP+1))
yticklocs[0] = 1


fig, axes = plt.subplots(nrows=2, ncols=n_cats,
    figsize=(7, 3), sharex=True,
    gridspec_kw=GRIDSPEC_KWS)

for i, col in enumerate(category_columns):

    # grab axes
    topax, ax = axes[:, i]

    # top ax is from the main liwc results and just one bar
    category = col.split("_")[0]
    dval = liwccats.loc[category, "cohen-d"]
    d_ci = liwccats.loc[category, ["cohen-d_lo", "cohen-d_hi"]].values
    derr = np.abs(dval-d_ci).reshape(2,1)
    color = c.COLORS["lucid"] if dval>0 else c.COLORS["non-lucid"]
    topax.barh(0, dval, xerr=derr,
        color=color, error_kw=ERROR_ARGS, **BAR_ARGS)

    # grab top N tokens for THIS category and sort it
    subdf = df.loc[ df[col].notna()
        ].sort_values(col, ascending=True
        )[:N_WORDS]

    dvals = subdf["cohen-d"].values
    d_cis = subdf[["cohen-d_lo", "cohen-d_hi"]].values
    derrs = np.abs(dvals-d_cis.T)
    labelvals = np.arange(len(dvals)) + 1
    labels = subdf["token"].tolist()
    colors = [ c.COLORS["lucid"] if d>0 else c.COLORS["non-lucid"] for d in dvals ]

    ax.barh(labelvals, dvals, xerr=derrs,
        color=colors, error_kw=ERROR_ARGS, **BAR_ARGS)

    TXT_BUFF = .05
    ha_align = "left" if dval > 0 else "right"
    for i, txt in enumerate(labels):
        yloc = i + 1
        if dval > 0: # to the right, align left of txt against high CI
            xloc = d_cis[i,1] + TXT_BUFF
        else: # to the left, align right of txt against low CI
            xloc = d_cis[i,0] - TXT_BUFF
        ax.text(xloc, yloc, txt.replace("*","_"),
            fontstyle="italic", fontsize=8,
            ha=ha_align, va="center")

    ##### a e s t h e t i c s
    for ax_ in [ax, topax]:
        ax_.axvline(0, color="black", linewidth=1, zorder=5)
        ax_.set_axisbelow(True)
    # ax.invert_yaxis()
    ax.set_ylim(labelvals.max()+1, labelvals.min()-1)
    ax.xaxis.grid(True, which="major", linewidth=1, clip_on=False, color=GRID_COLOR)
    ax.xaxis.grid(True, which="minor", linewidth=.3, clip_on=False, color=GRID_COLOR)
    # xmax = max(map(abs, ax.get_xlim()))
    ax.set_xlim(-1.2, 1.2)
    ax.set_xlabel("Effect size (Cohen's $\it{d}$) " +
        "\nnon-lucid$\leftarrow$   $\\rightarrow$lucid         ")
    ax.yaxis.set(major_locator=plt.FixedLocator(yticklocs))
    ax.xaxis.set(major_locator=plt.MultipleLocator(.5),
                 minor_locator=plt.MultipleLocator(.1),
                 major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    # if ax.get_subplotspec().is_first_col():
    ax.set_ylabel("word contribution rank", labelpad=-5)

    topax.set_ylim(-1, 1)
    topax.yaxis.set(major_locator=plt.NullLocator())
    topax.tick_params(which="both", left=False, bottom=False)
    topax.spines["bottom"].set_visible(False)
    for side in ["left", "top", "right"]:
        topax.spines[side].set_visible(False)
    xloc_txt = .02 if dval > 0 else .98
    topax.text(xloc_txt, .95,
        f"Total {category}\neffect",
        transform=topax.transAxes,
        fontsize=8, fontweight="bold",
        ha=ha_align, va="top")
    ax.text(xloc_txt, .99,
        f"Individual word\ncontributions",
        transform=ax.transAxes,
        fontsize=8,
        ha=ha_align, va="top")


# export
plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()