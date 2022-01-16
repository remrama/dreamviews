"""Visualize the amount of data for each label/category of interest
(ie, lucid, non-lucid, and nightmare labels).

Use a venn diagram to see how often the labels overlap.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - visualization, results/describe-categorycounts.png
"""
import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
c.load_matplotlib_settings()



######################## I/O

export_fname = os.path.join(c.DATA_DIR, "results", "describe-categorycounts.png")

df = c.load_dreamviews_posts()
df = df.set_index("post_id")

# make new columns that denote lucid/non-lucid, independent of overlap
df["lucid"]    = df["lucidity"].isin(["ambiguous", "lucid"])
df["nonlucid"] = df["lucidity"].isin(["ambiguous", "nonlucid"])
# and unspecified by itself for smaller/bottom axis
df["unspecified"] = df["lucidity"].eq("unspecified")



######################## define some stuff

VENN_ORDER_1 = ["nonlucid", "lucid", "nightmare"]
# VENN_ORDER_2 = ["unspecified", "nightmare"]

VENN_ARGS = {
    "alpha" : .4,
    "subset_label_formatter" : None,
    "normalize_to" : 1,
}

FIGSIZE = (3, 2.7)
GRIDSPEC_ARGS = dict(bottom=0, top=1, left=0, right=1)


######################## generate counts and venn info

## requires generating 7 (or 3) venn values in a specific order

#### open up the main figure and create both axes to be drawn on

fig, ax1 = plt.subplots(figsize=FIGSIZE, gridspec_kw=GRIDSPEC_ARGS)
# (using ax_top and bottom instead of 1/2 or a/b bc of possible confusion with Venn terms)
# ax2 = ax1.inset_axes([-.11, -.04, .46, .3])
# ax2.axis("off")


def draw_venn_plot(ax, columns):
    
    n_venns = len(columns)

    # generate boolean series' to calculate them
    if n_venns == 3:
        Abc = ( df[columns[0]] & ~df[columns[1]] & ~df[columns[2]])
        aBc = (~df[columns[0]] &  df[columns[1]] & ~df[columns[2]])
        ABc = ( df[columns[0]] &  df[columns[1]] & ~df[columns[2]])
        abC = (~df[columns[0]] & ~df[columns[1]] &  df[columns[2]])
        AbC = ( df[columns[0]] & ~df[columns[1]] &  df[columns[2]])
        aBC = (~df[columns[0]] &  df[columns[1]] &  df[columns[2]])
        ABC = ( df[columns[0]] &  df[columns[1]] &  df[columns[2]])
        series_list = [Abc, aBc, ABc, abC, AbC, aBC, ABC]

    elif n_venns == 2:
        Ab = ( df[columns[0]] & ~df[columns[1]])
        aB = (~df[columns[0]] &  df[columns[1]])
        AB = ( df[columns[0]] &  df[columns[1]])
        series_list = [Ab, aB, AB]


    # use the series to get the number of posts AND users at each venn location
    n_posts = [ s.sum() for s in series_list ]
    n_users = [ df.loc[s, "user_id"].nunique() for s in series_list ]

    # calculate the report:user fraction for each venn location
    venn_sizes = [ p/u for p, u in zip(n_posts, n_users) ]

    # generate new txt labels
    venn_labels = []
    for i, (p, u) in enumerate(zip(n_posts, n_users)):
        if i == 0 and n_venns == 3:
            txt = (r"$n_{posts}=$" + str(p)
                + "\n(" + r"$n_{users}=$" + str(u) + ")")
        else:
            txt = f"{p}\n({u})"
        venn_labels.append(txt)

    venn_colors = [ c.COLORS[x] for x in columns ]
    outer_labels = [ "non-lucid" if x == "nonlucid" else x for x in columns ]

    plot_venn_args = {
        "ax" : ax,
        "subsets" : venn_sizes,
        "set_labels" : outer_labels,
        "set_colors" : venn_colors
    }

    if n_venns == 3:
        ven = venn3(**plot_venn_args, **VENN_ARGS)
    elif n_venns == 2:
        ven = venn2(**plot_venn_args, **VENN_ARGS)

    # venn aesthetics

    # highlight the lucid circles in the main plot
    if n_venns == 3:
        for setid in ["100", "010"]:
            patch = ven.get_patch_by_id(setid)
            patch.set_lw(1)
            # to disentangle alpha of edge and face, need to 
            # first get the orig facecolor (which as alpha in rgba format)
            # then unset the main alpha and set edgecolor as rgba set.
            # (basically, rgba values don't work if alpha param is set, it overrides)
            facecolor_rgba = patch.get_facecolor()
            patch.set_alpha(None)
            patch.set_ec((0,0,0,1))
            patch.set_fc(facecolor_rgba)

    # change the font of text within circles/patches
    for setid in ["100", "010", "001", "10", "01"]:
        if n_venns == 2 and len(setid) == 3: continue
        ven.get_label_by_id(setid).set_fontsize(10)

    # change the font of the outer labels (e.g., non-lucid)
    for setid in ["A", "B", "C"]:
        if n_venns == 2 and setid == "C": continue
        ven.get_label_by_id(setid).set_fontsize(10)
        ven.get_label_by_id(setid).set_style("italic")

    # change the txt within each circle
    for vlabel, txt in zip(ven.subset_labels, venn_labels):
        vlabel.set_text(txt)
        vlabel.set_fontsize(8)



draw_venn_plot(ax=ax1, columns=VENN_ORDER_1)
# draw_venn_plot(ax=ax2, columns=VENN_ORDER_2)


plt.savefig(export_fname)
# leave out eps bc of transparency
c.save_hires_figs(export_fname, hires_extensions=[".svg", ".pdf"])
plt.close()