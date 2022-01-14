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

import seaborn as sea
import matplotlib.pyplot as plt
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



################################# plotting

# define bins, this is weird with the symlog scale, i hated it
bin_sets = [
    np.linspace(0, 10, 11),
    np.linspace(10, 100, 10)[1:],
    np.linspace(100, 1000, 10)[1:],
    np.linspace(1000, 10000, 10)[1:],
]
bins = list(np.concatenate(bin_sets) - .5)

# intialize the facetgrid/jointgrid (this doesn't draw anything)
g = sea.JointGrid(data=df_user,
    x="n_nonlucid", y="n_lucid",
    marginal_ticks=True,
    height=3.5,
    ratio=4, # ratio of joint-to-marginal axis heights
    space=.4, # space between joint and marginal axes
)

# create in inset axis for the colorbar legend
cax = g.fig.add_axes([.28, .74, .25, .02])

# draw the joint (main) axis
g.plot_joint(sea.histplot,
    cmap=sea.light_palette(c.COLORS["ambiguous"], as_cmap=True),
    bins=bins, # ignored with discrete=True or (True, True)
    vmin=1, vmax=1000,
    norm=plt.matplotlib.colors.SymLogNorm(1, base=10),
    cbar=True, cbar_ax=cax,
    cbar_kws={"orientation" : "horizontal",
              "ticks" : [1, 10, 100, 1000],
              "format" : plt.ScalarFormatter()}
)

# colorbar aesthetics
cbar = g.ax_joint.get_children()[0].colorbar
cbar.set_label("# of users",
    x=1.1, fontsize=8, labelpad=-14,
    va="center", ha="left")
cax.tick_params(labelsize=8, length=2)
cbar.locator = plt.LogLocator(base=10)
cbar.formatter = plt.ScalarFormatter() # redundant???
cbar.update_ticks()
# cbar.ax.set_title
# cbar.set_ticks

# joint axis aesthetics
g.ax_joint.set(
    xlabel="# of non-lucid reports",
    ylabel="# of lucid reports",
    xscale="symlog",
    yscale="symlog", # linthreshy=1???
    xlim=(-.5, 1000),
    ylim=(-.5, 1000),
)
g.ax_joint.xaxis.set(major_formatter=plt.ScalarFormatter())
g.ax_joint.yaxis.set(major_formatter=plt.ScalarFormatter())
# g.ax_joint.yaxis.set(minor_locator=MinorSymLogLocator(1))

# plot marginal axis distributions
g.plot_marginals(sea.histplot,
    bins=bins, fill=False, discrete=False,
    cumulative=True, lw=1, color="black",
    clip_on=True, element="step")


# I set the colorbar max at 1000,
# so make sure there aren't any cells exceeding that!
mesh = g.ax_joint.get_children()[0]
max_mesh_val = mesh.get_array().data.max()
assert max_mesh_val <= 1000

# manually change colors bc I don't see a way at the initial stage
x_polys = g.ax_marg_x.get_children()[0] # gets poly collection
y_polys = g.ax_marg_y.get_children()[0]
x_polys.set_color(c.COLORS["nonlucid"])
y_polys.set_color(c.COLORS["lucid"])

# unlike when using JointPlot, setting
# marginal_ticks=True with JointGrid as
# done here also sets the axis labels as
# visible=False, so need to undo that
g.ax_marg_x.yaxis.label.set_visible(True)
g.ax_marg_y.xaxis.label.set_visible(True) # not using this one now but to avoid potential later confusion


# marginal axis aesthetics
MARGINAL_YMAX = 5000
MAJOR_TICK_LOC = 5000
MINOR_TICK_LOC = 1000
GRID_ARGS = {
    "linewidth" : .5,
    "color"     : "gainsboro",
    "alpha"     : 1,
}
g.ax_marg_x.set(ylim=(0, MARGINAL_YMAX), ylabel="# of users")
g.ax_marg_y.set(xlim=(0, MARGINAL_YMAX), xlabel="", xticklabels=[])
g.ax_marg_x.yaxis.set(major_locator=plt.MultipleLocator(MAJOR_TICK_LOC),
                      minor_locator=plt.MultipleLocator(MINOR_TICK_LOC))
g.ax_marg_y.xaxis.set(major_locator=plt.MultipleLocator(MAJOR_TICK_LOC),
                      minor_locator=plt.MultipleLocator(MINOR_TICK_LOC))
g.ax_marg_x.grid(axis="y", which="both", **GRID_ARGS)
g.ax_marg_y.grid(axis="x", which="both", **GRID_ARGS)
g.ax_joint.spines["top"].set_visible(True)
g.ax_marg_x.spines["top"].set_visible(True)
g.ax_marg_y.spines["top"].set_visible(True)
g.ax_joint.spines["right"].set_visible(True)
g.ax_marg_x.spines["right"].set_visible(True)
g.ax_marg_y.spines["right"].set_visible(True)


# mark the relevant (>=1) paired data on all axes
# (box on the joint and lines on the marginals)
LINE_ARGS = {
    "color" : "black",
    "linewidth" : 1,
    "linestyle" : "dashed",
}
g.ax_joint.hlines(.5, xmin=.5, xmax=1000, **LINE_ARGS)
g.ax_joint.vlines(.5, ymin=.5, ymax=1000, **LINE_ARGS)
g.ax_marg_x.axvline(.5, **LINE_ARGS)
g.ax_marg_y.axhline(.5, **LINE_ARGS)

# add some text also emphasizing the >= 1 part
n_paired = df_user.all(axis=1).sum()
txt = (r"$n_{users}=$" + rf"${n_paired}$"
    + "\n" + r"$with\ \geq1\ of\ each\ dream\ type$")
g.ax_joint.text(800, .6, txt,
    fontsize=8, ha="right", va="bottom")


# export plots and table
df_user.to_csv(export_fname_table, sep="\t", index=True, encoding="utf-8")
plt.savefig(export_fname_plot)
c.save_hires_figs(export_fname_plot)
plt.close()

