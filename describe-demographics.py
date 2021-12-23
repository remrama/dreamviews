"""
plot reported user gender and age AND LOCATION NOW
and how they interact
"""
import os
import numpy as np
import pandas as pd
import geopandas

import config as c

import seaborn as sea
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


export_fname_plot = os.path.join(c.DATA_DIR, "results", "describe-demographics.png")
export_fname_table = os.path.join(c.DATA_DIR, "results", "describe-demographics.tsv")
export_fname_table_locs = os.path.join(c.DATA_DIR, "results", "describe-demographics_locations.tsv")


_, df = c.load_dreamviews_data()


####### country choropleth stuff

# load the world geopandas data to get country geometries
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# get a count per country
country_counts = df["country"].fillna("unstated"
    ).value_counts().rename("n_users").rename_axis("iso_a3")

## save before dropping the unstated and converting to log values
country_counts.to_csv(export_fname_table_locs, sep="\t", index=True, encoding="utf-8")

# generate string to show how many users are not included
unstated_n = country_counts.pop("unstated")
unstated_pct = unstated_n / (unstated_n+country_counts.sum()) * 100
unstated_txt = f"{unstated_n} ({unstated_pct:.0f}%) did not report location"

# # convert for plotting benefits
# country_counts = country_counts.apply(np.log10)


########## age/gender

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


###### open figure for both
# fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(6,3),
#     gridspec_kw=dict(width_ratios=[1,.03], wspace=0, left=0, bottom=.1, right=.9, top=1))
FIGSIZE = (7.5, 2.5)
# GRIDSPEC_KW = dct(left=.1, right=.99, bottom=.1, top=.99,
#     wspace=.1, width_ratios=[1, 4])
# fig, axes = plt.subplots(ncols=2, figsize=FIGSIZE, gridspec_kw=GRIDSPEC_KW)
fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
gs1 = fig.add_gridspec(1, 1, bottom=.25, top=.95, left=.09, right=.3)
gs2 = fig.add_gridspec(1, 1, bottom=0, top=1, left=.33, right=.92)
gs3 = fig.add_gridspec(1, 1, bottom=.1, top=.92, left=.91, right=.92)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs2[0])
ax3 = fig.add_subplot(gs3[0])

fig.text(0, 1, "A", fontsize=12, fontweight="bold", ha="left", va="top")
fig.text(.33, 1, "B", fontsize=12, fontweight="bold", ha="left", va="top")


sea.histplot(data=df, x="gender", hue="age",
    multiple="stack", stat="count", element="bars",
    palette=age_palette,
    hue_order=CUT_LABELS_WITH_NA[::-1],
    ax=ax1, legend=True,
    **BAR_KWS)

# # can't change the width on histplot
# for ch in ax1.get_children():
#     if isinstance(ch, plt.matplotlib.patches.Rectangle):
#         ch.set_width(.8)

ax1.set_ylabel("# of users", fontsize=10)
ax1.set_xlabel("reported gender", fontsize=10)#, labelpad=0)
ax1.tick_params(axis="x", which="major", labelrotation=25, pad=0)
ax1.set_xlim(-1, 4)
ax1.set_ylim(0, 2500)
ax1.yaxis.set(
    major_locator=plt.MultipleLocator(500),
    minor_locator=plt.MultipleLocator(100))
ax1.tick_params(axis="y", which="both",
    direction="in", right=True)
ax1.set_axisbelow(True)
ax1.yaxis.grid(which="major", color="gray", lw=.5)

legend = ax1.get_legend()
handles = legend.legendHandles
legend.remove()
legend = ax1.legend(title="reported age",
    handles=handles, labels=CUT_LABELS_WITH_NA[::-1],
    loc="upper left", bbox_to_anchor=(.3, .97),
    borderaxespad=0, frameon=False,
    labelspacing=0, # like rowspacing, vertical space between entries
    handletextpad=.2, # space between markers and labels
)
# legend._legend_box.sep = 1 # brings title up farther on top of handles/labels



########### choropleth

# add the user counts per country to the world geodataframe
myworld = world.merge(country_counts, left_on="iso_a3", right_index=True, how="left")
# myworld["n_users"] = myworld["n_users"].fillna(0)

# myworld.plot(ax=ax, color="white", edgecolor="black")
# norm = plt.matplotlib.colors.SymLogNorm(linthresh=20, vmin=1, vmax=2000, base=10)
norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=2000)
myworld.plot(
    ax=ax2, cax=ax3, column="n_users",
    edgecolor="black", linewidth=.3,
    cmap="viridis", norm=norm,
    legend=True, legend_kwds=dict(label="# of users", orientation="vertical"),
    missing_kwds=dict(facecolor="gainsboro", linewidth=.1),
)
ax2.axis("off")

ax2.text(.05, .2, unstated_txt, transform=ax2.transAxes,
    ha="left", va="top", fontsize=8)


# export with various extensions
plt.savefig(export_fname_plot)
c.save_hires_figs(export_fname_plot)
plt.close()