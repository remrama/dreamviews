"""Visualize the number of dream reports per user, across whole corpus at once.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - visualization of post-per-user frequency, results/describe-usercount.png
"""
import os
import numpy as np
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


################################ I/O
export_fname = os.path.join(c.DATA_DIR, "results", "describe-usercount.png")
df = c.load_dreamviews_posts()

# get counts
counts = df["user_id"].value_counts(
    ).rename_axis("user_id").rename("n_posts")


################################ plot

FIGSIZE = (3, 1.8)
N_BINS = 50
HIST_ARGS = dict(lw=.5, color="gainsboro")
MAJOR_XTICK_LOC = 200

# generate bins
bins = np.linspace(0, c.MAX_POSTCOUNT, N_BINS+1)

# open figure and draw
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
ax.hist(counts.values, bins=bins, log=True,
    color="gainsboro", linewidth=.5, edgecolor="black")

# aesthetics
ax.set_xlabel(r"$n$ posts per user", labelpad=0)
ax.set_ylabel(r"$n$ users")
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
ax.axvline(c.MAX_POSTCOUNT, **LINE_ARGS)
ax.text(c.MAX_POSTCOUNT-10, 1, "max post cutoff",
    transform=ax.get_xaxis_transform(), ha="right", va="top")

minor_tick_loc = np.diff(bins).mean()
ax.set_xlim(0, c.MAX_POSTCOUNT)
ax.xaxis.set(major_locator=plt.MultipleLocator(MAJOR_XTICK_LOC),
             minor_locator=plt.MultipleLocator(minor_tick_loc))
# ax.tick_params(axis="both", which="both", labelsize=10)
# ax.tick_params(axis="y", which="both", direction="in")


# export
plt.savefig(export_fname)
c.save_hires_figs(export_fname)
plt.close()