"""
Visualize the amount of lucid and non-lucid posts per participant.

This was really annoying and I think there's more manual adjusting than normal.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - raw counts, results/describe-categorypairs.tsv
    - visualization, results/describe-categorypairs.png
"""
import colorcet as cc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

import config as c


################################################################################
# SETUP
################################################################################

# Load custom matplotlib aesthetics.
c.load_matplotlib_settings()

# Choose export paths.
export_path_table = c.DATA_DIR / "results" / "describe-categorypairs.tsv"
export_path_plot  = c.DATA_DIR / "results" / "describe-categorypairs.png"

# Load data.
df = c.load_dreamviews_posts()

# Generate a dataframe that has lucid and non-lucid post counts per user (for those with >= 1).
sort_order = ["nonlucid", "lucid"]
df_user = df[df["lucidity"].str.contains("lucid")
    ].groupby(["user_id", "lucidity"]
    ).size().rename("count"
    ).unstack(fill_value=0
    ).sort_values(sort_order, ascending=False
    )[sort_order]
df_user.columns = df_user.columns.map(lambda c: "n_"+c)
assert not df_user.gt(c.MAX_POSTCOUNT).any(axis=None), f"Noone should have more than {c.MAX_POSTCOUNT} posts"


################################################################################
# PLOT MAIN AXIS
################################################################################

# Define bins.
bin_sets = [ np.arange(1, 11) * 10**i for i in range(3) ]
bins = np.unique(np.concatenate(bin_sets))
bins = np.append(0, bins)
assert df_user.le(1000).all(axis=None), "Data exceeds upper bin range of 1000"

# Open the figure.
fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

# Set aspect of the main axes so it stays square.
ax.set_aspect(1)

# Create axes above and to the right.
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", 0.4, pad=.15, sharex=ax)
ax_histy = divider.append_axes("right", 0.4, pad=.15, sharey=ax)

# Pick colormap.
color_max = 1000  # See check for colormap limits below.
color_norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=color_max)
colormap = cc.cm.dimgray_r

x_variable = "lucid"
y_variable = "nonlucid"
x_column = "n_" + x_variable
y_column = "n_" + y_variable

# Draw 2D histogram on main axis.
h, xedges, yedges, img = ax.hist2d(
    x_column,
    y_column,
    data=df_user,
    bins=bins,
    norm=color_norm,
    cmap=colormap,
)

# Make sure the max set for colors was appropriate.
assert h.max() <= color_max, f"Data exceeds upper colormap limit of {color_max}"

# Set axis ticks.
symlog_thresh = 10
ax.set_xscale("symlog", linthresh=symlog_thresh)
ax.set_yscale("symlog", linthresh=symlog_thresh)
major_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=symlog_thresh)
minor_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(
    base=10, linthresh=symlog_thresh, subs=np.linspace(0.1, 0.9, 9)
)
ax.xaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)
ax.yaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)

# Remove some edges.
ax.spines[["top", "right"]].set_visible(False)

# Add colorbar.
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

# Create twin axes for cumulative distributions.
ax_histx_twin = ax_histx.twinx()
ax_histy_twin = ax_histy.twiny()

hist_kwargs = dict(data=df_user, bins=bins)
barhist_kwargs = dict(linewidth=.5, edgecolor="black", alpha=0.8)
linehist_kwargs = dict(linewidth=1, cumulative=True, histtype="step")
barhist_kwargs.update(hist_kwargs)
linehist_kwargs.update(hist_kwargs)

# Draw non-cumulative bar histograms on the regular axes.
n, bins, patches = ax_histx.hist(x_column, color=c.COLORS[x_variable], **barhist_kwargs)
n, bins, patches = ax_histy.hist(y_column,
    orientation="horizontal", color=c.COLORS[y_variable], **barhist_kwargs
)

# Draw cumulative line histograms on the twin/opposite axes.
n, bins, patches = ax_histx_twin.hist(x_column, color=c.COLORS[x_variable], **linehist_kwargs)
n, bins, patches = ax_histy_twin.hist(
    y_column, orientation="horizontal", color=c.COLORS[y_variable], **linehist_kwargs
)

# Remove some tick labels.
ax_histx.tick_params(labelbottom=False)
ax_histy.tick_params(labelbottom=False, labelleft=False)
ax_histy_twin.tick_params(labeltop=False)

# Set marginal axes limits.
marginal_ymax_twin = 4000
marginal_ymax = marginal_ymax_twin // 2
ax_histx.set_ybound(upper=marginal_ymax)
ax_histy.set_xbound(upper=marginal_ymax)
ax_histx_twin.set_ybound(upper=marginal_ymax_twin)
ax_histy_twin.set_xbound(upper=marginal_ymax_twin)

# Set marginal axes ticks.
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

# Set marginal axes labels.
xlabel = r"$n$ " + x_variable.replace("nl", "n-l") + " posts"
ylabel = r"$n$ " + y_variable.replace("nl", "n-l") + " posts"
ax_histx.set_ylabel(r"$n$ users", labelpad=3)
ax.set_xlabel(xlabel, labelpad=0)
ax.set_ylabel(ylabel, labelpad=2)
ax_histx.set_ylabel(r"$n$ users", labelpad=1)
ax_histx_twin.set_ylabel(
    "cumulative\n"+r"$n$ users", rotation=0, labelpad=-17, y=0.75, ha="left", linespacing=1
)

# Draw marginal axes grids.
marginal_grid_kwargs = dict(color="gainsboro", linewidth=0.5, alpha=1)
ax_histx.grid(axis="y", which="both", **marginal_grid_kwargs)
ax_histy.grid(axis="x", which="both", **marginal_grid_kwargs)
ax_histx.set_axisbelow(True)
ax_histy.set_axisbelow(True)


################################################################################
# DRAW HIGHLIGHTING LINES
################################################################################

# Put some lines on all axes to highlight sections.
# Highlight users with >=1 of *both* lucid and nonlucid posts.
# (see also g.refline)
line_kwargs = dict(linewidth=1, color="black")
for cut in [1, 20]:
    if cut == 1:
        more_line_kwargs = dict(linestyle="dashed", alpha=1)
    else:
        more_line_kwargs = dict(linestyle="dotted", alpha=1)
    line_kwargs.update(more_line_kwargs)
    ax.hlines(cut, xmin=cut, xmax=bins[-1], **line_kwargs)
    ax.vlines(cut, ymin=cut, ymax=bins[-1], **line_kwargs)
    ax_histx_twin.axvline(cut, **line_kwargs)
    ax_histy_twin.axhline(cut, **line_kwargs)

# Add some explanatory text.
n_paired = df_user.all(axis=1).sum()
text_total = (
    r"$n_{users}=$" + rf"${n_paired}$" + "\n with "+r"$\geq1$"+" lucid and "+r"$\geq1$"+" non-lucid"
)
ax.text(1, 0.06, text_total, transform=ax.transAxes, ha="right", va="bottom")

n_twenty = df_user.ge(20).all(axis=1).sum()
text_twenty = r"$n_{users}=$" + rf"${n_twenty}$" + ", " + r"$\geq20$"+" of each"
ax.text(1, 0.47, text_twenty, transform=ax.transAxes, ha="right", va="bottom")


################################################################################
# EXPORT
################################################################################

df_user.to_csv(export_path_table, sep="\t", index=True, encoding="utf-8")
plt.savefig(export_path_plot)
plt.savefig(export_path_plot.with_suffix(".pdf"))
plt.close()
