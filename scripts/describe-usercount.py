"""
Visualize the number of dream reports per user, across whole corpus at once.

IMPORTS
=======
    - posts, dreamviews-posts.tsv
EXPORTS
=======
    - visualization of post-per-user frequency, describe-usercount.png
"""

import matplotlib.pyplot as plt
import numpy as np

import config as c

########################################################################################
# SETUP
########################################################################################

# Load custom plotting aesthetics
c.load_matplotlib_settings()

# Choose export stem
EXPORT_STEM = "describe-usercount"

# Load data
df = c.load_dreamviews_posts()
counts = df["user_id"].value_counts().rename_axis("user_id").rename("n_posts")

########################################################################################
# PLOTTING
########################################################################################

N_BINS = 50
X_BOUNDS = (0, c.MAX_POSTCOUNT)
Y_MAX = 10000
MAJOR_TICK_LOC = 200
HIST_KWARGS = dict(log=True, color="gainsboro", linewidth=0.5, edgecolor="black")

# Generate bins
bins = np.linspace(0, c.MAX_POSTCOUNT, N_BINS + 1)
minor_tick_loc = np.diff(bins).mean()

assert counts.min() >= 1, "Post counts should be non-negative."
assert counts.max() <= c.MAX_POSTCOUNT, f"Post counts should be capped at {c.MAX_POSTCOUNT}."
values = counts.to_numpy()

# Open figure
fig, ax = plt.subplots(figsize=(3, 1.8), constrained_layout=True)

# Draw
yvals, _, _ = ax.hist(values, bins=bins, **HIST_KWARGS)
assert yvals.max() <= Y_MAX, f"Expected no more than {Y_MAX} users with the same post count."

# Indicate post limit
ax.axvline(c.MAX_POSTCOUNT, color="black", ls="dashed", lw=0.5, alpha=1, clip_on=False)
ax.text(
    c.MAX_POSTCOUNT - 10,
    1,
    "max post cutoff",
    transform=ax.get_xaxis_transform(),
    ha="right",
    va="top",
)

# Aesthetics
ax.set_xlim(*X_BOUNDS)
ax.set_xlabel(r"$n$ posts per user", labelpad=0)
ax.set_ylabel(r"$n$ users")
ax.set_ybound(upper=Y_MAX)
ax.xaxis.set_major_locator(plt.MultipleLocator(MAJOR_TICK_LOC))
ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick_loc))
ax.spines[["top", "right"]].set_visible(False)

# Export
c.export_fig(fig, EXPORT_STEM)
