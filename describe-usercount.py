"""
# of dream reports per user
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
export_fname = os.path.join(c.DATA_DIR, "results", "describe-usercount.png")
df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")


counts = df["user_id"].value_counts(
    ).rename_axis("user_id").rename("n_posts"
    ).to_frame()

# # use this to color by lucidity
# counts = df.groupby(["user_id","lucidity"]
#     ).size().rename("n_posts").reset_index()

# # this is a rather useful table
# counts = df.groupby(["user_id","lucidity"]
#     ).size().reset_index(
#     ).pivot(index="user_id", columns="lucidity", values=0
#     ).fillna(0).astype(int)

# generate bins
N_BINS = 50
bins = np.linspace(0, c.MAX_POST_COUNT, N_BINS+1)

fig, ax = plt.subplots(figsize=(4, 2.5),
    constrained_layout=True)

sea.histplot(data=counts, x="n_posts",
    stat="count", element="bars",
    # hue="lucidity", multiple="dodge", palette=c.COLORS,
    discrete=False,
    bins=bins,
    log_scale=(False, True),
    color="gainsboro",
    edgecolor="black",
    linewidth=.5,
    ax=ax)

ax.set_xlabel("# posts per user", fontsize=10)
ax.set_ylabel("# users", fontsize=10)
ax.set_ybound(upper=10000)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

LINE_ARGS = {
    "linewidth" : .5,
    "alpha"     : 1,
    "color"     : "black",
    "linestyle" : "dashed",
    "clip_on"   : False,
}
ax.axvline(c.MAX_POST_COUNT, **LINE_ARGS)
ax.text(c.MAX_POST_COUNT-10, 1, "max post cutoff",
    transform=ax.get_xaxis_transform(),
    ha="right", va="top", fontsize=10)

minor_tick_loc = np.diff(bins).mean()
ax.set_xlim(0, c.MAX_POST_COUNT)
ax.xaxis.set(major_locator=plt.MultipleLocator(100),
             minor_locator=plt.MultipleLocator(minor_tick_loc))
# ax.tick_params(axis="both", which="both", labelsize=10)
# ax.tick_params(axis="y", which="both", direction="in")


plt.savefig(export_fname)
plt.close()