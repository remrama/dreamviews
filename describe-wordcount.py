"""
plot word counts
"""
import os
import numpy as np
import pandas as pd

import seaborn as sea
import matplotlib.pyplot as plt

import config as c

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


### handle i/o and load in data
import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "describe-wordcount.png")
df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# counts = df["user_id"].value_counts(
#     ).rename_axis("user_id").rename("n_posts"
#     ).to_frame()
# # # use this to color by lucidity
# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().rename("n_posts").reset_index()

# # # this is a rather useful table
# # counts = df.groupby(["user_id","lucidity"]
# #     ).size().reset_index(
# #     ).pivot(index="user_id", columns="lucidity", values=0
# #     ).fillna(0).astype(int)

# generate bins
N_BINS = 38
bins = np.linspace(c.MIN_TOKEN_COUNT, c.MAX_TOKEN_COUNT, N_BINS+1)

fig, ax = plt.subplots(figsize=(4, 2.5),
    constrained_layout=True)

LUCIDITY_ORDER = ["unspecified", "ambiguous", "non-lucid", "lucid"]

sea.histplot(data=df,
    x="n_tokens", hue="lucidity",
    multiple="stack", palette=c.COLORS,
    hue_order=LUCIDITY_ORDER,
    stat="count", element="bars",
    bins=bins, discrete=False,
    linewidth=.5, edgecolor="black",
    legend=False,
    ax=ax)

ax.set_xlabel("# words/tokens", fontsize=10)
ax.set_ylabel("# posts", fontsize=10)
ax.set_ybound(upper=4000)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

LINE_ARGS = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
}
ax.axvline(c.MIN_TOKEN_COUNT, **LINE_ARGS)
ax.axvline(c.MAX_TOKEN_COUNT, **LINE_ARGS)
ax.text(c.MIN_TOKEN_COUNT+10, 1, "min word cutoff",
    transform=ax.get_xaxis_transform(),
    ha="left", va="top", fontsize=10)
ax.text(c.MAX_TOKEN_COUNT-10, 1, "max word cutoff",
    transform=ax.get_xaxis_transform(),
    ha="right", va="top", fontsize=10)

minor_tick_loc = np.diff(bins).mean()
ax.set_xlim(0, c.MAX_TOKEN_COUNT+minor_tick_loc)
ax.xaxis.set(major_locator=plt.MultipleLocator(200),
             minor_locator=plt.MultipleLocator(minor_tick_loc))
ax.yaxis.set(major_locator=plt.MultipleLocator(1000),
             minor_locator=plt.MultipleLocator(200))
ax.tick_params(axis="both", which="both", labelsize=10)
ax.tick_params(axis="y", which="both", direction="in")
handles = [ plt.matplotlib.patches.Patch(edgecolor="none",
        facecolor=c.COLORS[cond], label=cond)
    for cond in LUCIDITY_ORDER ]
legend = ax.legend(handles=handles,
    loc="upper left", bbox_to_anchor=(.55, .9),
    borderaxespad=0, frameon=False,
    labelspacing=.2, handletextpad=.2)


plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()
