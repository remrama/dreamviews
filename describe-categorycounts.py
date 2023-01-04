"""
Use a Venn diagram to visualize the amount of lucid, non-lucid, and nightmare
data and how often they overlap.

IMPORTS
=======
    - posts, dreamviews-posts.tsv
EXPORTS
=======
    - visualization, describe-categorycounts.png
"""
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as c


################################################################################
# SETUP
################################################################################

# Load custom plotting aesthetics.
c.load_matplotlib_settings()

# Choose filename for exporting.
export_path = c.DATA_DIR / "derivatives" / "describe-categorycounts.png"

# Load data.
df = c.load_dreamviews_posts()
df = df.set_index("post_id")

# Make new columns that denote lucid/non-lucid, independent of overlap ...
df["lucid"] = df["lucidity"].isin(["ambiguous", "lucid"])
df["nonlucid"] = df["lucidity"].isin(["ambiguous", "nonlucid"])
# ... and unspecified by itself for smaller/bottom axis.
df["unspecified"] = df["lucidity"].eq("unspecified")

# Define some plotting variables.
figsize = (2.4, 2.3)
gridspec_kwargs = dict(bottom=0, top=1, left=0, right=1)
venn_order = ["nonlucid", "lucid", "nightmare"]
venn_kwargs = {
    "alpha": .4,
    "normalize_to": 1,
    "subset_label_formatter": None,
}


################################################################################
# MAIN PLOTTING FUNCTION
################################################################################

def draw_venn_plot(ax, columns):
    assert (n_venns := len(columns)) in [2, 3]

    # Generate boolean Series' to calculate frequencies with.
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

    # Get the number of posts AND number of users for each venn location.
    n_posts = [ s.sum() for s in series_list ]
    n_users = [ df.loc[s, "user_id"].nunique() for s in series_list ]

    # Calculate the report:user fraction for each venn location.
    venn_sizes = [ p/u for p, u in zip(n_posts, n_users) ]

    # Generate new text labels for each venn location.
    venn_labels = []
    for i, (p, u) in enumerate(zip(n_posts, n_users)):
        if i == 0 and n_venns == 3:
            txt = r"$n_{posts}=$" + str(p) + "\n(" + r"$n_{users}=$" + str(u) + ")"
        else:
            txt = f"{p}\n({u})"
        venn_labels.append(txt)

    # Choose colors and labels that appear outside the venn locations.
    venn_colors = [ c.COLORS[x] for x in columns ]
    outer_labels = [ "non-lucid" if x == "nonlucid" else x for x in columns ]

    # Draw the venn diagram.
    kwargs = dict(ax=ax, subsets=venn_sizes, set_labels=outer_labels, set_colors=venn_colors)
    venn_kwargs.update(kwargs)
    if n_venns == 3:
        ven = venn3(**venn_kwargs)
    elif n_venns == 2:
        ven = venn2(**venn_kwargs)

    ############################################################################
    # AESTHETICS
    ############################################################################

    # Highlight the lucidity-related circles in the main plot.
    if n_venns == 3:
        for setid in ["100", "010"]:
            patch = ven.get_patch_by_id(setid)
            patch.set_lw(1)
            # To disentangle alpha of edge and face, need to first get the
            # original facecolor (which as alpha in RGBA format) and then unset
            # the main alpha and set edgecolor as rgba set. (Basically, RGBA
            # values don't work if alpha param is set, it overrides.)
            facecolor_rgba = patch.get_facecolor()
            patch.set_alpha(None)
            patch.set_ec((0,0,0,1))
            patch.set_fc(facecolor_rgba)

    # Adjust the font of the outer labels.
    for setid in ["A", "B", "C"]:
        if n_venns == 2 and setid == "C":
            continue
        ven.get_label_by_id(setid).set_style("italic")

    # Adjust the inner text.
    for vlabel, txt in zip(ven.subset_labels, venn_labels):
        vlabel.set_text(txt)


################################################################################
# DRAW AND SAVE
################################################################################

# Open figure and apply function to draw.
fig, ax = plt.subplots(figsize=figsize, gridspec_kw=gridspec_kwargs)
draw_venn_plot(ax=ax1, columns=venn_order)

# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()
