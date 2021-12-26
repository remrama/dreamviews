"""
# of dream reports per user
"""
import os
import numpy as np
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


### handle i/o and load in data
export_fname = os.path.join(c.DATA_DIR, "results", "describe-usercount.png")

df, _ = c.load_dreamviews_data()

counts = df["user_id"].value_counts(
    ).rename_axis("user_id").rename("n_posts")

# generate bins
N_BINS = 50
bins = np.linspace(0, c.MAX_POSTCOUNT, N_BINS+1)

FIGSIZE = (3.5, 2.5)
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

HIST_ARGS = dict(lw=.5, color="gainsboro")

ax.hist(counts.values, bins=bins, log=True,
    color="gainsboro", linewidth=.5, edgecolor="black")

ax.set_xlabel("# posts per user", fontsize=10)
ax.set_ylabel("# users", fontsize=10)
ax.set_ybound(upper=10000)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# LINE_ARGS = {
#     "linewidth" : .5,
#     "alpha"     : 1,
#     "color"     : "black",
#     "linestyle" : "dashed",
#     "clip_on"   : False,
# }
# ax.axvline(c.MAX_POSTCOUNT, **LINE_ARGS)
# ax.text(c.MAX_POSTCOUNT-10, 1, "max post cutoff",
#     transform=ax.get_xaxis_transform(),
#     ha="right", va="top", fontsize=10)

minor_tick_loc = np.diff(bins).mean()
ax.set_xlim(0, c.MAX_POSTCOUNT)
ax.xaxis.set(major_locator=plt.MultipleLocator(100),
             minor_locator=plt.MultipleLocator(minor_tick_loc))
# ax.tick_params(axis="both", which="both", labelsize=10)
# ax.tick_params(axis="y", which="both", direction="in")


plt.savefig(export_fname)
c.save_hires_figs(export_fname, [".svg", ".pdf"])
plt.close()