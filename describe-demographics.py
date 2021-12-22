"""
plot reported user gender and age
and how they interact
"""
import os
import numpy as np
import pandas as pd

import config as c

import seaborn as sea
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

import_fname = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-users.tsv")
export_fname_plot = os.path.join(c.DATA_DIR, "results", "describe-demographics.png")
export_fname_table = os.path.join(c.DATA_DIR, "results", "describe-demographics.tsv")


df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")


# replace gender NAs
GENDER_ORDER = ["male", "female", "trans", "unstated"]
df["gender"] = pd.Categorical(df["gender"].fillna("unstated"),
    categories=GENDER_ORDER, ordered=True)


# replace age NAs and bin age
max_age = df["age"].max()
CUT_BINS = [0, 20, 30, 40, max_age+1]
CUT_LABELS = ["<20", "<30", "<40", "  40+"]
CUT_LABELS_WITH_NA = ["<20", "<30", "<40", "  40+", "unstated"]
age_binned = pd.cut(df["age"],
    bins=CUT_BINS, labels=CUT_LABELS,
    right=False, include_lowest=True)
df["age"] = pd.Categorical(age_binned,
        categories=CUT_LABELS_WITH_NA,
        ordered=True
    ).fillna("unstated")


## export specific values
out = df.groupby(["gender","age"]).size().rename("count")
out.to_csv(export_fname_table, sep="\t", index=True, encoding="utf-8")


########## draw

# generate a custom color palette with one colormap
# for the binned ages and a blank/white for no info
cmap = plt.get_cmap("plasma", len(CUT_LABELS))
age_colors = cmap.colors
age_colors = np.vstack([age_colors, [1,1,1,1]]) # add a white
age_palette = { age: rgba for age, rgba in zip(CUT_LABELS_WITH_NA, age_colors) }

BAR_KWS = {
    "linewidth" : .5,
    "edgecolor" : "black",
    "alpha" : 1,
}

fig, ax = plt.subplots(figsize=(3, 3),
    constrained_layout=True)

sea.histplot(data=df, x="gender", hue="age",
    multiple="stack", stat="count", element="bars",
    palette=age_palette,
    hue_order=CUT_LABELS_WITH_NA[::-1],
    ax=ax, legend=True,
    **BAR_KWS)

# # can't change the width on histplot
# for ch in ax.get_children():
#     if isinstance(ch, plt.matplotlib.patches.Rectangle):
#         ch.set_width(.8)

ax.set_ylabel("# of users", fontsize=10)
ax.set_xlabel("reported gender", fontsize=10)#, labelpad=0)
# ax.tick_params(axis="x", which="major", pad=0)
ax.set_ylim(0, 2500)
ax.yaxis.set(
    major_locator=plt.MultipleLocator(500),
    minor_locator=plt.MultipleLocator(100))
ax.tick_params(axis="y", which="both",
    direction="in", right=True)
ax.set_axisbelow(True)
ax.yaxis.grid(which="major", color="gray", lw=.5)

legend = ax.get_legend()
handles = legend.legendHandles
legend.remove()
legend = ax.legend(title="reported age",
    handles=handles, labels=CUT_LABELS_WITH_NA[::-1],
    loc="upper left", bbox_to_anchor=(.3, .95),
    borderaxespad=0, frameon=False,
    labelspacing=0, # like rowspacing, vertical space between entries
    handletextpad=.2, # space between markers and labels
)
# legend._legend_box.sep = 1 # brings title up farther on top of handles/labels


plt.savefig(export_fname_plot)
plt.close()