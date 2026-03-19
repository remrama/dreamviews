"""
Visualize the amount of lucid and non-lucid posts per participant.

This was really annoying and I think there's more manual adjusting than normal.

IMPORTS
=======
    - posts, dreamviews-posts.tsv
EXPORTS
=======
    - raw counts,    describe-categorypairs.tsv
    - visualization, describe-categorypairs.png
"""

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import config as c

################################################################################
# SETUP
################################################################################

# Load custom matplotlib aesthetics
c.load_matplotlib_settings()

# Choose export stemp
EXPORT_STEM = "describe-categorypairs"

# Load data
df = c.load_dreamviews_posts()

# Generate a dataframe that has lucid and non-lucid post counts per user (for those with >= 1)
SORT_ORDER = ["nonlucid", "lucid"]
df_user = (
    df[df["lucidity"].str.contains("lucid")]
    .groupby(["user_id", "lucidity"])
    .size()
    .rename("count")
    .unstack(fill_value=0)
    .sort_values(SORT_ORDER, ascending=False)[SORT_ORDER]
)
df_user.columns = df_user.columns.map(lambda c: "n_" + c)
assert not df_user.gt(c.MAX_POSTCOUNT).any(axis=None), (
    f"Noone should have more than {c.MAX_POSTCOUNT} posts"
)

################################################################################
# PLOT MAIN AXIS
################################################################################

# Define bins
bin_sets = [np.arange(1, 11) * 10**i for i in range(3)]
bins = np.unique(np.concatenate(bin_sets))
bins = np.append(0, bins)
assert df_user.le(1000).all(axis=None), "Data exceeds upper bin range of 1000"

# Open the figure
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

# Set aspect of the main axes so it stays square
ax.set_aspect(1)

# Create axes above and to the right
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", 0.4, pad=0.15, sharex=ax)
ax_histy = divider.append_axes("right", 0.4, pad=0.15, sharey=ax)

# Pick colormap
COLOR_MAX = 1000  # See check for colormap limits below
color_norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=COLOR_MAX)
colormap = cc.cm.dimgray_r

X_VARIABLE = "lucid"
Y_VARIABLE = "nonlucid"
x_column = "n_" + X_VARIABLE
y_column = "n_" + Y_VARIABLE

# Draw 2D histogram on main axis
h, xedges, yedges, img = ax.hist2d(
    x_column,
    y_column,
    data=df_user,
    bins=bins,
    norm=color_norm,
    cmap=colormap,
)

# Make sure the max set for colors was appropriate
assert h.max() <= COLOR_MAX, f"Data exceeds upper colormap limit of {COLOR_MAX}"

# Set axis ticks
SYMLOG_THRESH = 10
ax.set_xscale("symlog", linthresh=SYMLOG_THRESH)
ax.set_yscale("symlog", linthresh=SYMLOG_THRESH)
major_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=SYMLOG_THRESH)
minor_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(
    base=10, linthresh=SYMLOG_THRESH, subs=np.linspace(0.1, 0.9, 9)
)
ax.xaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)
ax.yaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)

# Remove some edges
ax.spines[["top", "right"]].set_visible(False)

# Add colorbar
cax = inset_axes(ax, width="3%", height="30%", loc="upper right", borderpad=0.2)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=color_norm)
cbar = fig.colorbar(sm, cax=cax, orientation="vertical", ticklocation="left")
cbar.ax.tick_params(which="major", size=3, direction="in", color="white", pad=1)
cbar.ax.tick_params(which="minor", size=0, direction="in", color="white")
cbar.ax.set_axisbelow(False)
cbar.set_label(r"$n$ users", labelpad=0)

################################################################################
# PLOT MARGINAL AND TWIN AXES
################################################################################

# Create twin axes for cumulative distributions
ax_histx_twin = ax_histx.twinx()
ax_histy_twin = ax_histy.twiny()

hist_kwargs = dict(data=df_user, bins=bins)
barhist_kwargs = dict(linewidth=0.5, edgecolor="black", alpha=0.8)
linehist_kwargs = dict(linewidth=1, cumulative=True, histtype="step")
barhist_kwargs.update(hist_kwargs)
linehist_kwargs.update(hist_kwargs)

# Draw non-cumulative bar histograms on the regular axes
n, bins, patches = ax_histx.hist(x_column, color=c.COLORS[X_VARIABLE], **barhist_kwargs)
n, bins, patches = ax_histy.hist(
    y_column, orientation="horizontal", color=c.COLORS[Y_VARIABLE], **barhist_kwargs
)

# Draw cumulative line histograms on the twin/opposite axes
n, bins, patches = ax_histx_twin.hist(x_column, color=c.COLORS[X_VARIABLE], **linehist_kwargs)
n, bins, patches = ax_histy_twin.hist(
    y_column, orientation="horizontal", color=c.COLORS[Y_VARIABLE], **linehist_kwargs
)

# Remove some tick labels
ax_histx.tick_params(labelbottom=False)
ax_histy.tick_params(labelbottom=False, labelleft=False)
ax_histy_twin.tick_params(labeltop=False)

# Set marginal axes limits
MARGINAL_YMAX_TWIN = 4000
marginal_ymax = MARGINAL_YMAX_TWIN // 2
ax_histx.set_ybound(upper=marginal_ymax)
ax_histy.set_xbound(upper=marginal_ymax)
ax_histx_twin.set_ybound(upper=MARGINAL_YMAX_TWIN)
ax_histy_twin.set_xbound(upper=MARGINAL_YMAX_TWIN)

# Set marginal axes ticks
ax_histx.yaxis.set(
    major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5)
)
ax_histy.xaxis.set(
    major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5)
)
ax_histx_twin.yaxis.set(
    major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5)
)
ax_histy_twin.xaxis.set(
    major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5)
)

# Set marginal axes labels
xlabel = r"$n$ " + X_VARIABLE.replace("nl", "n-l") + " posts"
ylabel = r"$n$ " + Y_VARIABLE.replace("nl", "n-l") + " posts"
ax_histx.set_ylabel(r"$n$ users", labelpad=3)
ax.set_xlabel(xlabel, labelpad=0)
ax.set_ylabel(ylabel, labelpad=2)
ax_histx.set_ylabel(r"$n$ users", labelpad=1)
ax_histx_twin.set_ylabel(
    "cumulative\n" + r"$n$ users", rotation=0, labelpad=-17, y=0.75, ha="left", linespacing=1
)

# Draw marginal axes grids
marginal_grid_kwargs = dict(color="gainsboro", linewidth=0.5, alpha=1)
ax_histx.grid(axis="y", which="both", **marginal_grid_kwargs)
ax_histy.grid(axis="x", which="both", **marginal_grid_kwargs)
ax_histx.set_axisbelow(True)
ax_histy.set_axisbelow(True)

################################################################################
# DRAW HIGHLIGHTING LINES
################################################################################

# Put some lines on all axes to highlight sections
# Highlight users with >=1 of *both* lucid and nonlucid posts
# (see also g.refline)
LINE_KWARGS = dict(linewidth=1, color="black", alpha=1)
for cut in [1, 20]:
    linestyle = "dashed" if cut == 1 else "dotted"
    ax.hlines(cut, xmin=cut, xmax=bins[-1], linestyle=linestyle, **LINE_KWARGS)
    ax.vlines(cut, ymin=cut, ymax=bins[-1], linestyle=linestyle, **LINE_KWARGS)
    ax_histx_twin.axvline(cut, linestyle=linestyle, **LINE_KWARGS)
    ax_histy_twin.axhline(cut, linestyle=linestyle, **LINE_KWARGS)

# Add some explanatory text
n_paired = df_user.all(axis=1).sum()
text_total = (
    r"$n_{users}=$"
    + rf"${n_paired}$"
    + "\n with "
    + r"$\geq1$"
    + " lucid and "
    + r"$\geq1$"
    + " non-lucid"
)
ax.text(1, 0.06, text_total, transform=ax.transAxes, ha="right", va="bottom")

n_twenty = df_user.ge(20).all(axis=1).sum()
text_twenty = r"$n_{users}=$" + rf"${n_twenty}$" + ", " + r"$\geq20$" + " of each"
ax.text(1, 0.47, text_twenty, transform=ax.transAxes, ha="right", va="bottom")

################################################################################
# EXPORT
################################################################################

c.export_table(df_user, EXPORT_STEM)
c.export_fig(fig, EXPORT_STEM)
