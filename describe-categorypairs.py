"""Count the number of participants with lucidity-labeled posts.

This was really annoying and I think there's more manual adjusting than normal.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - raw counts, results/describe-categorypairs.tsv
    - visualization, results/describe-categorypairs.png
"""
import os
import numpy as np
import pandas as pd
import config as c

import colorcet as cc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
c.load_matplotlib_settings()



################################# I/O

export_fname_table = os.path.join(c.DATA_DIR, "results", "describe-categorypairs.tsv")
export_fname_plot  = os.path.join(c.DATA_DIR, "results", "describe-categorypairs.png")

df = c.load_dreamviews_posts()

# generate dataframe that has the count of lucid and
# nonlucid dreams for each user that had at least 1
SORT_ORDER = ["nonlucid", "lucid"]
df_user = df[df["lucidity"].str.contains("lucid")
    ].groupby(["user_id", "lucidity"]
    ).size().rename("count"
    ).unstack(fill_value=0
    ).sort_values(SORT_ORDER, ascending=False
    )[SORT_ORDER]
df_user.columns = df_user.columns.map(lambda c: "n_"+c)

assert not df_user.gt(c.MAX_POSTCOUNT).any(axis=None), f"Noone should have more than {c.MAX_POSTCOUNT} posts"


################################# plotting

SYMLOG_THRESH = 10

# Define bins.

# This will set bins to end (inclusively) at 1000, so check that's okay.
assert df_user.le(1000).all(axis=None), "Axis limits will cutoff data."
bin_sets = [ np.arange(1, 11) * 10**i for i in range(3) ]
bins = np.unique(np.concatenate(bin_sets))
bins = np.append(0, bins)


# Get data for plot.
# H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
# # Histogram does not follow Cartesian convention (see Notes),
# # therefore transpose H for visualization purposes.
# H = H.T


X_VARIABLE = "lucid"
Y_VARIABLE = "nonlucid"
x_column = "n_" + X_VARIABLE
y_column = "n_" + Y_VARIABLE

fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

# Set aspect of the main axes so it stays square.
ax.set_aspect(1)

# Create axes above and to the right.
divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", .4, pad=.15, sharex=ax)
ax_histy = divider.append_axes("right", .4, pad=.15, sharey=ax)

COLOR_MAX = 1000
color_norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=COLOR_MAX)
colormap = cc.cm.dimgray_r # This still shows some gray on the limit.

# Draw 2d histogram on main axis.
h, xedges, yedges, img = ax.hist2d(x_column, y_column,
    data=df_user, bins=bins,
    norm=color_norm, cmap=colormap)
    # cmin=1, cmax=None,

# Make sure the max set for colors was appropriate
assert h.max() <= COLOR_MAX, f"I didn't expect more than {COLOR_MAX} users in a cell"

ax.set_xscale("symlog", linthresh=SYMLOG_THRESH)
ax.set_yscale("symlog", linthresh=SYMLOG_THRESH)
# major_ticks = [ b for b in bins if b in [0, 1, 10, 100, 1000] ]
# minor_ticks = [ b for b in bins if b not in [0, 1, 10, 100, 1000] ]
major_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(
    base=10, linthresh=SYMLOG_THRESH)
minor_ticks = plt.matplotlib.ticker.SymmetricalLogLocator(
    base=10, linthresh=SYMLOG_THRESH, subs=np.linspace(.1, .9, 9))
ax.xaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)
ax.yaxis.set(major_locator=major_ticks, minor_locator=minor_ticks)
# major_formatter=plt.LogFormatter(base=10, labelOnlyBase=True,
#     linthresh=symlog_lthresh),)

# ax.tick_params(which="both", axis="both", direction="inout")


# ###### Colorbar legend.

# Create inset axis for the colorbar legend.
# cax = fig.add_axes([.32, .79, .25, .02])
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax, width="3%", height="30%", loc="upper right", borderpad=.2) 

sm = plt.cm.ScalarMappable(cmap=colormap, norm=color_norm)
cbar = fig.colorbar(sm, cax=cax, orientation="vertical", ticklocation="left")
# cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
# cbar.outline.set_linewidth(.5)
# cbar.outline.set_visible(False)
cbar.ax.tick_params(which="major", size=3, direction="in", color="white", pad=1)#length=2
cbar.ax.tick_params(which="minor", size=0, direction="in", color="white")
cbar.ax.set_axisbelow(False)
# cbar.locator = plt.LogLocator(base=10)
# cbar.formatter = plt.ScalarFormatter()
# cbar.ax.yaxis.set(minor_locator=plt.NullLocator())
cbar.set_label(r"$n$ users", labelpad=0)
    # x=1.1,        # higher value moves label to the right
    # labelpad=-17, # higher value moves label down
    # va="center", ha="left")
# cbar.update_ticks()


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


############## Plot the marginal distributions.
### Normal bars on main and cumulative on twin.

ax_histx_twin = ax_histx.twinx()
ax_histy_twin = ax_histy.twiny()

BARHIST_KWARGS = dict(linewidth=.5, edgecolor="black", alpha=.8)
LINEHIST_KWARGS = dict(linewidth=1, cumulative=True, histtype="step")
HIST_KWARGS = dict(data=df_user, bins=bins)

# Normal bars.
n, bins, patches = ax_histx.hist(x_column, color=c.COLORS[X_VARIABLE], **HIST_KWARGS, **BARHIST_KWARGS)
n, bins, patches = ax_histy.hist(y_column, orientation="horizontal", color=c.COLORS[Y_VARIABLE], **HIST_KWARGS, **BARHIST_KWARGS)

# Cumulative lines on the opposite axes.
n, bins, patches = ax_histx_twin.hist(x_column, color=c.COLORS[X_VARIABLE], **HIST_KWARGS, **LINEHIST_KWARGS)
n, bins, patches = ax_histy_twin.hist(y_column, orientation="horizontal", color=c.COLORS[Y_VARIABLE], **HIST_KWARGS, **LINEHIST_KWARGS)


# Clear out some labels.
ax_histx.tick_params(labelbottom=False)
ax_histy.tick_params(labelbottom=False, labelleft=False)
ax_histy_twin.tick_params(labeltop=False)

# Set marginal limits and give them all same ticks.
MARGINAL_YMAX_TWIN = 4000
marginal_ymax = MARGINAL_YMAX_TWIN//2
ax_histx.set_ybound(upper=marginal_ymax)
ax_histy.set_xbound(upper=marginal_ymax)
ax_histx_twin.set_ybound(upper=MARGINAL_YMAX_TWIN)
ax_histy_twin.set_xbound(upper=MARGINAL_YMAX_TWIN)
ax_histx.yaxis.set(major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5))
ax_histy.xaxis.set(major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5))
ax_histx_twin.yaxis.set(major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5))
ax_histy_twin.xaxis.set(major_locator=plt.LinearLocator(numticks=2), minor_locator=plt.LinearLocator(numticks=5))

xlabel = r"$n$ " + X_VARIABLE.replace("nl", "n-l") + " posts"
ylabel = r"$n$ " + Y_VARIABLE.replace("nl", "n-l") + " posts"
ax_histx.set_ylabel(r"$n$ users", labelpad=3)
ax.set_xlabel(xlabel, labelpad=0)
ax.set_ylabel(ylabel, labelpad=2)
ax_histx.set_ylabel(r"$n$ users", labelpad=1)
# ax_histx_twin.set_ylabel("cumulative", rotation=0, labelpad=0)
ax_histx_twin.set_ylabel("cumulative\n"+r"$n$ users",
    rotation=0, labelpad=-17, y=.75, ha="left", linespacing=1)


MARGINAL_GRID_KWARGS = dict(color="gainsboro", linewidth=.5, alpha=1)
ax_histx.grid(axis="y", which="both", **MARGINAL_GRID_KWARGS)
ax_histy.grid(axis="x", which="both", **MARGINAL_GRID_KWARGS)
ax_histx.set_axisbelow(True)
ax_histy.set_axisbelow(True)

# mark the relevant (>=1) paired data on all axes
# (box on the joint and lines on the marginals)
# (see also g.refline)
LINE_KWARGS = dict(linewidth=1, color="black")
for cut in [1, 20]:
    if cut == 1:
        more_line_kwargs = dict(linestyle="dashed", alpha=1)
    else:
        more_line_kwargs = dict(linestyle="dotted", alpha=1)
    ax.hlines(cut, xmin=cut, xmax=bins[-1], **LINE_KWARGS, **more_line_kwargs)
    ax.vlines(cut, ymin=cut, ymax=bins[-1], **LINE_KWARGS, **more_line_kwargs)
    ax_histx_twin.axvline(cut, **LINE_KWARGS, **more_line_kwargs)
    ax_histy_twin.axhline(cut, **LINE_KWARGS, **more_line_kwargs)

# add some text also emphasizing the >= 1 part
n_paired = df_user.all(axis=1).sum()
txt = (r"$n_{users}=$" + rf"${n_paired}$"
    + "\n with "+r"$\geq1$"+" lucid and "+r"$\geq1$"+" non-lucid")
ax.text(1, .06, txt, transform=ax.transAxes, ha="right", va="bottom")

n_twenty = df_user.ge(20).all(axis=1).sum()
txt = r"$n_{users}=$" + rf"${n_twenty}$" + ", " + r"$\geq20$"+" of each"
ax.text(1, .47, txt, transform=ax.transAxes, ha="right", va="bottom")



# export plots and table
df_user.to_csv(export_fname_table, sep="\t", index=True, encoding="utf-8")
plt.savefig(export_fname_plot)
c.save_hires_figs(export_fname_plot, hires_extensions=[".pdf", ".svg"])
plt.close()

