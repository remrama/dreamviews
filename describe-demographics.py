"""Describe demographics (age, gender, location) and how frequently they were provided.

IMPORTS
=======
    - user info, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - table of how many people provided info, results/describe-demographics_provided.tsv
    - table of reported ages and genders,     results/describe-demographics_agegender.tsv
    - table of reported ages and genders,     results/describe-demographics_location.tsv
    - visualization of reported age/gender,   results/describe-demographics_agegender.png
    - visualization of reported location,     results/describe-demographics_location.png
"""
import os
import numpy as np
import pandas as pd
import geopandas

import config as c

import colorcet as cc
import matplotlib.pyplot as plt
c.load_matplotlib_settings()


export_fname_agegender = os.path.join(c.DATA_DIR, "results", "describe-demographics_agegender.tsv")
export_fname_locations = os.path.join(c.DATA_DIR, "results", "describe-demographics_location.tsv")
export_fname_provided = os.path.join(c.DATA_DIR, "results", "describe-demographics_provided.tsv")
export_fname_agegender_plot = os.path.join(c.DATA_DIR, "results", "describe-demographics_agegender.png")
export_fname_locations_plot = os.path.join(c.DATA_DIR, "results", "describe-demographics_location.png")


df = c.load_dreamviews_users()


######################### count how many participants provided demographic info

reported_bool = df[["gender", "age", "country"]].notnull()
reported_sum = reported_bool.sum()
reported_sum.loc["gender+age"] = reported_bool[["gender","age"]].all(axis=1).sum()
reported_sum.loc["gender+country"] = reported_bool[["gender","country"]].all(axis=1).sum()
reported_sum.loc["age+country"] = reported_bool[["age","country"]].all(axis=1).sum()
reported_sum.loc["gender+age+country"] = reported_bool.all(axis=1).sum()
reported_pct = (reported_sum / len(df) * 100).round(0).astype(int)

reported = pd.concat([reported_sum, reported_pct], axis=1)
reported.columns = ["n_reported", "pct_reported"]

# export
reported.to_csv(export_fname_provided, index_label="demographic_variable", sep="\t", encoding="utf-8")



######################### get age and gender frequencies

# replace gender NAs
GENDER_ORDER = ["male", "female", "trans", "unstated"]
df["gender"] = pd.Categorical(df["gender"].fillna("unstated"),
    categories=GENDER_ORDER, ordered=True)

# replace age NAs and bin age
assert df["age"].dropna().ge(18).all(), "Didn't expect any reported ages under 18."
max_age = df["age"].dropna().astype(int).max()
CUT_BINS = [18, 25, 35, 45, 55, 65, max_age+1]
cut_labels = [ f"[{l}, {r})" for l, r in zip(CUT_BINS[:-1], CUT_BINS[1:]) ]
cut_labels[-1] = cut_labels[-1].replace(str(max_age+1), r"$\infty$") #r"$\geq$65"
cut_labels_with_na = cut_labels + ["unstated"]
age_binned = pd.cut(df["age"],
    bins=CUT_BINS, labels=cut_labels,
    right=False, include_lowest=True)
df["age"] = pd.Categorical(age_binned,
        categories=cut_labels_with_na,
        ordered=True
    ).fillna("unstated")

# export
out = df.groupby(["gender","age"]).size().rename("count")
out.to_csv(export_fname_agegender, index=True, sep="\t", encoding="utf-8")



######################### get location frequencies

# get a count per country
country_counts = df["country"].fillna("unstated"
    ).value_counts().rename("n_users").rename_axis("iso_a3")

# export (before dropping the unstated and converting to log values)
country_counts.to_csv(export_fname_locations, index=True, sep="\t", encoding="utf-8")



######################### plot age and gender

# # generate a custom color palette with one colormap
# # for the binned ages and a blank/white for no info
# cmap = plt.get_cmap("plasma", len(CUT_LABELS))
# age_colors = cmap.colors
# age_colors = np.vstack([age_colors, [1,1,1,1]]) # add a white
# age_palette = { age: rgba for age, rgba in zip(CUT_LABELS_WITH_NA, age_colors) }

# BAR_KWS = {
#     "linewidth" : .5,
#     "edgecolor" : "black",
#     "alpha" : 1,
# }

# ###### open figure for both
# # fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(6,3),
# #     gridspec_kw=dict(width_ratios=[1,.03], wspace=0, left=0, bottom=.1, right=.9, top=1))
# FIGSIZE = (6.9, 2.5)
# # GRIDSPEC_KW = dct(left=.1, right=.99, bottom=.1, top=.99,
# #     wspace=.1, width_ratios=[1, 4])
# # fig, axes = plt.subplots(ncols=2, figsize=FIGSIZE, gridspec_kw=GRIDSPEC_KW)
# fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
# gs1 = fig.add_gridspec(1, 1, bottom=.25, top=.95, left=.09, right=.3)
# gs2 = fig.add_gridspec(1, 1, bottom=0, top=1, left=.33, right=.92)
# gs3 = fig.add_gridspec(1, 1, bottom=.1, top=.92, left=.91, right=.92)
# ax1 = fig.add_subplot(gs1[0])
# ax2 = fig.add_subplot(gs2[0])
# ax3 = fig.add_subplot(gs3[0])
# fig.text(0, 1, "A", fontsize=12, fontweight="bold", ha="left", va="top")
# fig.text(.33, 1, "B", fontsize=12, fontweight="bold", ha="left", va="top")

fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True)


n_genders = df["gender"].nunique()
n_ages = df["age"].nunique()

bins = range(n_genders+1)

# Replace gender values with numeric so plot is in proper order.
df["gender_int"] = df["gender"].cat.codes

data = [ x for x in df.groupby("age")["gender_int"].apply(list) ]

# Use colormap for sequential age colors, but leave last one white for unstated.
age_colors = [ "white" if i==n_ages-1 else cc.cm.bgy(i/(n_ages-2)) for i in range(n_ages) ]

ax.hist(data, bins=bins, align="left", rwidth=.8,
    stacked=True, color=age_colors,
    edgecolor="black", linewidth=.5)

ax.set_xticks(bins[:-1])
ax.set_xticklabels(GENDER_ORDER)
ax.set_xlabel("Reported gender", labelpad=1)
ax.set_ylabel(r"$n$ users", labelpad=2)
ax.set_ybound(lower=0, upper=2500)
ax.yaxis.set(major_locator=plt.LinearLocator(5+1),
             minor_locator=plt.LinearLocator(5*5+1))
ax.tick_params(axis="y", which="both", direction="inout", pad=2)
ax.grid(axis="y", which="major", linewidth=1, color="gainsboro")
ax.grid(axis="y", which="minor", linewidth=.5, color="gainsboro")
ax.set_axisbelow(True)
legend_handles = [ plt.matplotlib.patches.Patch(
        edgecolor="black" if c == "white" else "none",
        linewidth=.3, facecolor=c, label=l)
    for l, c in zip(cut_labels_with_na, age_colors) ]
legend = ax.legend(handles=legend_handles[::-1], title="Reported age",
    loc="upper left", bbox_to_anchor=(.27, 1), borderaxespad=0,
    frameon=False, labelspacing=.1, handletextpad=.2)
legend._legend_box.sep = 2 # brings title up farther on top of handles/labels
legend._legend_box.align = "left"

# export
plt.savefig(export_fname_agegender_plot)
c.save_hires_figs(export_fname_agegender_plot)
plt.close()



################################# plot locations (choropleth)

# fig, ax = plt.subplots(figsize=(3.2,1.5), constrained_layout=False,
#     gridspec_kw=dict(left=0, bottom=0, top=1, right=1))
# # divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="2%", pad=0)
# cax = inset_axes(ax,
#                     width="2%",  
#                     height="100%",
#                     loc="right",
#                     borderpad=1
#                    )

# fig, (ax, cax) = plt.subplots(ncols=2, figsize=FIGSIZE,
#     gridspec_kw=dict(width_ratios=[1,.03], wspace=0, left=0, bottom=.1, right=.9, top=1))
FIGSIZE = (3.2, 1.4)
fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
gs1 = fig.add_gridspec(1, 1, bottom=0, top=1, left=0, right=.85)
gs2 = fig.add_gridspec(1, 1, bottom=.1, top=.92, left=.83, right=.85)
ax = fig.add_subplot(gs1[0])
cax = fig.add_subplot(gs2[0])

# load the world geopandas data to get country geometries
world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

# pop out the unstated bc it isn't for the plot
unstated_n = country_counts.pop("unstated")
# generate string to show how many users are not included
# unstated_pct = unstated_n / (unstated_n+country_counts.sum()) * 100
# unstated_txt = f"{unstated_n} ({unstated_pct:.0f}%) did not report location"

# # convert for plotting benefits
# country_counts = country_counts.apply(np.log10)

# add the user counts per country to the world geodataframe
myworld = world.merge(country_counts, left_on="iso_a3", right_index=True, how="left")
# myworld["n_users"] = myworld["n_users"].fillna(0)

# myworld.plot(ax=ax, color="white", edgecolor="black")
# norm = plt.matplotlib.colors.SymLogNorm(linthresh=20, vmin=1, vmax=2000, base=10)
CMAX = 2000
assert myworld["n_users"].max() <= CMAX, f"Colormap needs a higher max value than {CMAX}."
norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=CMAX)

# colormap = "viridis"
# jsd  --> bgy, bmy, CET_L8, kgy, dimgray
colormap = cc.cm.bgy
myworld.plot(
    ax=ax, cax=cax, column="n_users",
    edgecolor="black", linewidth=.3,
    cmap=colormap, norm=norm,
    legend=True, legend_kwds=dict(label=r"$n$ users", orientation="vertical"),
    missing_kwds=dict(facecolor="gainsboro", linewidth=.1),
)
ax.axis("off")

# ax2.text(.05, .2, unstated_txt, transform=ax2.transAxes,
#     ha="left", va="top", fontsize=8)


# export
plt.savefig(export_fname_locations_plot)
c.save_hires_figs(export_fname_locations_plot)
plt.close()