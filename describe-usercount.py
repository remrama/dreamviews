"""Visualize the number of dream reports per user, across whole corpus at once.

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


################################################################################
# SETUP
################################################################################

# Load custom plotting aesthetics.
c.load_matplotlib_settings()

# Choose export location.
export_path = c.DATA_DIR / "results" / "describe-usercount.png"

# Load data.
df = c.load_dreamviews_posts()
counts = df["user_id"].value_counts().rename_axis("user_id").rename("n_posts")


################################################################################
# PLOTTING
################################################################################

# Generate bins.
n_bins = 50
bins = np.linspace(0, c.MAX_POSTCOUNT, n_bins+1)

# Open figure.
fig, ax = plt.subplots(figsize=(3, 1.8), constrained_layout=True)

# Draw.
ax.hist(counts.values, bins=bins, log=True, color="gainsboro", linewidth=.5, edgecolor="black")

# Indicate post limit.
ax.axvline(
    c.MAX_POSTCOUNT, color="black", ls="dashed", lw=0.5, alpha=1, clip_on=False
)
ax.text(
    c.MAX_POSTCOUNT - 10,
    1,
    "max post cutoff",
    transform=ax.get_xaxis_transform(),
    ha="right",
    va="top",
)

# Aesthetics.
ax.set_xlim(0, c.MAX_POSTCOUNT)
ax.set_xlabel(r"$n$ posts per user", labelpad=0)
ax.set_ylabel(r"$n$ users")
ax.set_ybound(upper=10000)
major_tick_loc = 200
minor_tick_loc = np.diff(bins).mean()
ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick_loc))
ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_loc))
ax.spines[["top", "right"]].set_visible(False)

# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()
