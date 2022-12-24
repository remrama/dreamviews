"""
Visualize the number of dream reports per user, across whole corpus at once.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - visualization of post-per-user frequency, results/describe-usercount.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sea

import config as c

c.load_matplotlib_settings()


export_path = c.DATA_DIR / "results" / "describe-usercount.png"

# Data loading.
df = c.load_dreamviews_posts()
counts = df["user_id"].value_counts().rename_axis("user_id").rename("n_posts")

FIGSIZE = (3, 1.8)
N_BINS = 50
HIST_ARGS = dict(lw=.5, color="gainsboro")
MAJOR_XTICK_LOC = 200

# Generate bins.
bins = np.linspace(0, c.MAX_POSTCOUNT, N_BINS+1)

# Open figure.
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

# Draw.
ax.hist(counts.values, bins=bins, log=True, color="gainsboro", linewidth=.5, edgecolor="black")

# Aesthetics.
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
    transform=ax.get_xaxis_transform(), ha="right", va="top",
)
ax.set_xlim(0, c.MAX_POSTCOUNT)
minor_tick_loc = np.diff(bins).mean()
ax.xaxis.set(major_locator=plt.MultipleLocator(MAJOR_XTICK_LOC),
             minor_locator=plt.MultipleLocator(minor_tick_loc))

# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()
