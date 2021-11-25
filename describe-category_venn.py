"""
venn diagram showing how the category labels
that users apply to their posts overlap

Includes a main 3-circle venn about lucidity and nightmares
then a smaller 2-circle venn to show the posts without specificed lucidity
"""
import os
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt

from matplotlib_venn import venn2, venn3

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
# plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


#### i/o and load and manipulate data

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "data-category_venn.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8", index_col="post_id")

# make new columns that denote lucid/non-lucid, independent of overlap
df["lucid"]    = df["lucidity"].isin(["ambiguous", "lucid"])
df["nonlucid"] = df["lucidity"].isin(["ambiguous", "non-lucid"])
# and unspecified by itself for smaller/bottom axis
df["unspecified"] = df["lucidity"].eq("unspecified")


#################### generate venn info for both axes

## requires generating 7 (or 3) venn values in a specific order

VENN_ORDER_TOP = ["non-lucid", "lucid", "nightmare"]
VENN_ORDER_BOT = ["unspecified", "nightmare"]


##### main 3-circle venn

# generate boolean series' to calculate them
Abc = ( df.nonlucid & ~df.lucid & ~df.nightmare)
aBc = (~df.nonlucid &  df.lucid & ~df.nightmare)
ABc = ( df.nonlucid &  df.lucid & ~df.nightmare)
abC = (~df.nonlucid & ~df.lucid &  df.nightmare)
AbC = ( df.nonlucid & ~df.lucid &  df.nightmare)
aBC = (~df.nonlucid &  df.lucid &  df.nightmare)
ABC = ( df.nonlucid &  df.lucid &  df.nightmare)


# put the series in the necessary order
series_list_top = [Abc, aBc, ABc, abC, AbC, aBC, ABC]

# use the series to get the number of posts AND users at each venn location
n_posts_top = [ s.sum() for s in series_list_top ]
n_users_top = [ df.loc[s, "user_id"].nunique() for s in series_list_top ]

# calculate the report:user fraction for each venn location
venn_sizes_top = [ p/u for p, u in zip(n_posts_top, n_users_top) ]

# generate new txt labels too
venn_labels_top = []
for i, (p, u) in enumerate(zip(n_posts_top, n_users_top)):
    if i == 0:
        txt = (r"$n_{posts}=$" + str(p)
            + "\n(" + r"$n_{users}=$" + str(u) + ")")
    else:
        txt = f"{p}\n({u})"
    venn_labels_top.append(txt)

venn_colors_top = [ c.COLORS[x] for x in VENN_ORDER_TOP ]

top_venn_args = {
    "subsets"    : venn_sizes_top,
    "set_labels" : VENN_ORDER_TOP,
    "set_colors" : venn_colors_top,
}


##### same idea for 2-circle venn
### (should probably be a function or something)

Ab = ( df.unspecified & ~df.nightmare)
aB = (~df.unspecified &  df.nightmare)
AB = ( df.unspecified &  df.nightmare)

series_list_bot = [Ab, aB, AB]

n_posts_bot = [ s.sum() for s in series_list_bot ]
n_users_bot = [ df.loc[s, "user_id"].nunique() for s in series_list_bot ]

venn_sizes_bot = [ p/u for p, u in zip(n_posts_bot, n_users_bot) ]
venn_labels_bot = [ f"{p}\n({u})" for p, u in zip(n_posts_bot, n_users_bot) ]

venn_colors_bot = [ c.COLORS[x] for x in VENN_ORDER_BOT ]

bot_venn_args = {
    "subsets"    : venn_sizes_bot,
    "set_labels" : VENN_ORDER_BOT,
    "set_colors" : venn_colors_bot,
}




#### open up the main figure and create both axes to be drawn on

fig, ax_top = plt.subplots(figsize=(4, 3.5),
    gridspec_kw=dict(top=1, bottom=0, left=.12, right=1))
# (using ax_top and bottom instead of 1/2 or a/b bc of possible confusion with Venn terms)
ax_top.set_title("Individual dream report category overlap")

ax_bot = ax_top.inset_axes([-.11, -.04, .46, .3])
ax_bot.axis("off")


## draw both venns

VENN_ARGS = {
    "alpha" : .4,
    "subset_label_formatter" : None,
    "normalize_to" : 1,
}

venn_top = venn3(ax=ax_top, **top_venn_args, **VENN_ARGS)
venn_bot = venn2(ax=ax_bot, **bot_venn_args, **VENN_ARGS)

# venn aesthetics

# highlight the lucid circles in the main plot
for setid in ["100", "010"]:
    patch = venn_top.get_patch_by_id(setid)
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
    if len(setid) == 3:
        venn_top.get_label_by_id(setid).set_fontsize(10)
    else:
        venn_bot.get_label_by_id(setid).set_fontsize(10)

# change the font of the outer labels (e.g., non-lucid)
for setid in ["A", "B", "C"]:
    venn_top.get_label_by_id(setid).set_fontsize(10)
    venn_top.get_label_by_id(setid).set_style("italic")
    if setid != "C":
        venn_bot.get_label_by_id(setid).set_fontsize(10)
        venn_bot.get_label_by_id(setid).set_style("italic")

# change the txt within each circle
for vlabel, txt in zip(venn_top.subset_labels, venn_labels_top):
    vlabel.set_text(txt)
    vlabel.set_fontsize(8)

for vlabel, txt in zip(venn_bot.subset_labels, venn_labels_bot):
    vlabel.set_text(txt)
    vlabel.set_fontsize(8)



plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()