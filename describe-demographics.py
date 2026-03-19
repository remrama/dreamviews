"""Describe demographics (age, gender, location) and how frequently they were provided.

IMPORTS
=======
    - user info, dreamviews-posts.tsv
EXPORTS
=======
    - table of how many people provided info, describe-demographics_provided.tsv
    - table of reported ages and genders,     describe-demographics_agegender.tsv
    - table of reported ages and genders,     describe-demographics_location.tsv
    - visualization of reported age/gender,   describe-demographics_agegender.png
    - visualization of reported location,     describe-demographics_location.png
"""

import colorcet as cc
import geopandas
import matplotlib.pyplot as plt
import pandas as pd

import config as c

################################################################################
# SETUP
################################################################################

# Load custom matplotlib preferences
c.load_matplotlib_settings()

# Load data
df = c.load_dreamviews_users()

# Choose export stems
EXPORT_STEM_AGEGENDER = "describe-demographics_agegender"
EXPORT_STEM_LOCATION = "describe-demographics_location"
EXPORT_STEM_PROVIDED = "describe-demographics_provided"

################################################################################
# GET FREQUENCIES
################################################################################

# Count how many participants provided demographic info
reported_bool = df[["gender", "age", "country"]].notnull()
reported_sum = reported_bool.sum()
reported_sum.loc["gender+age"] = reported_bool[["gender", "age"]].all(axis=1).sum()
reported_sum.loc["gender+country"] = reported_bool[["gender", "country"]].all(axis=1).sum()
reported_sum.loc["age+country"] = reported_bool[["age", "country"]].all(axis=1).sum()
reported_sum.loc["gender+age+country"] = reported_bool.all(axis=1).sum()
reported_pct = (reported_sum / len(df) * 100).round(0).astype(int)
reported = pd.concat([reported_sum, reported_pct], axis=1)
reported.columns = ["n_reported", "pct_reported"]

# Export
c.export_table(reported, EXPORT_STEM_PROVIDED, index_label="demographic_variable")

# Get age and gender frequencies

# Replace gender NAs and make ordered categorical for plotting purposes
UNREPORTED_LABEL = "unreported"
GENDER_ORDER = ["male", "female", "trans", UNREPORTED_LABEL]
assert df["gender"].dropna().isin(GENDER_ORDER[:-1]).all(), (
    f"Expected gender values to be in {GENDER_ORDER[:-1]}."
)
df["gender"] = pd.Categorical(
    df["gender"].fillna(UNREPORTED_LABEL), categories=GENDER_ORDER, ordered=True
)

# Replace age NAs and make ordered categorical for plotting purposes
df["age"] = pd.Categorical(df["age"].fillna(UNREPORTED_LABEL), ordered=True)

# Export
df_out = df.groupby(["gender", "age"]).size().rename("count")
c.export_table(df_out, EXPORT_STEM_AGEGENDER)

# Get location frequencies
country_counts = (
    df["country"].fillna(UNREPORTED_LABEL).value_counts().rename("n_users").rename_axis("ISO_A3")
)

# Export (before dropping the unreported and converting to log values)
c.export_table(country_counts, EXPORT_STEM_LOCATION)

################################################################################
# PLOT AGE AND GENDER DATA
################################################################################

# Open figure
fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True)

# Add gender integer values for plotting order
df["gender_int"] = df["gender"].cat.codes

# Extract some relevant data
n_genders = df["gender"].nunique()
n_ages = df["age"].nunique()
bins = range(n_genders + 1)
data = [x for x in df.groupby("age")["gender_int"].apply(list)]

# Get age labels for legend
age_labels = df["age"].cat.categories.tolist()
age_labels = [label.replace("inf", r"$\infty$") for label in age_labels]

# Get sequential colormap colors for age, but leave last one white for unreported
AGE_CMAP = cc.cm.bgy
age_colors = ["white" if i == n_ages - 1 else AGE_CMAP(i / (n_ages - 2)) for i in range(n_ages)]

# Draw histogram
HIST_KWARGS = dict(align="left", rwidth=0.8, stacked=True, ec="black", lw=0.5)
ax.hist(data, bins=bins, color=age_colors, **HIST_KWARGS)

# Adjust aesthetics
ax.set_xticks(bins[:-1])
ax.set_xticklabels(GENDER_ORDER)
ax.set_xlabel("Reported gender", labelpad=1)
ax.set_ylabel(r"$n$ users", labelpad=2)
ax.set_ybound(lower=0, upper=2500)
ax.yaxis.set(major_locator=plt.LinearLocator(5 + 1), minor_locator=plt.LinearLocator(5 * 5 + 1))
ax.tick_params(axis="y", which="both", direction="inout", pad=2)
ax.grid(axis="y", which="major", linewidth=1, color="gainsboro")
ax.grid(axis="y", which="minor", linewidth=0.5, color="gainsboro")
ax.set_axisbelow(True)

# Add legend
PATCH_KWARGS = dict(linewidth=0.3)
legend_handles = [
    plt.matplotlib.patches.Patch(
        edgecolor="black" if color == "white" else "none",
        label=label,
        facecolor=color,
        **PATCH_KWARGS,
    )
    for label, color in zip(age_labels, age_colors, strict=True)
]
LEGEND_KWARGS = dict(
    title="Reported age",
    loc="upper left",
    bbox_to_anchor=(0.27, 1),
    borderaxespad=0,
    frameon=False,
    labelspacing=0.1,
    handletextpad=0.2,
)
legend = ax.legend(handles=legend_handles[::-1], **LEGEND_KWARGS)
legend._legend_box.sep = 2  # Brings title up farther on top of handles/labels
legend._legend_box.align = "left"

# Export
c.export_fig(fig, EXPORT_STEM_AGEGENDER)

################################################################################
# PLOT LOCATION DATA
################################################################################

fig = plt.figure(figsize=(3.2, 1.4), constrained_layout=False)
gs1 = fig.add_gridspec(1, 1, bottom=0, top=1, left=0, right=0.85)
gs2 = fig.add_gridspec(1, 1, bottom=0.1, top=0.92, left=0.83, right=0.85)
ax = fig.add_subplot(gs1[0])
cax = fig.add_subplot(gs2[0])

# Load the world geopandas data to get country geometries
map_filepath = c.fetch_file("ne_110m_admin_0_countries.zip")
world = geopandas.read_file(map_filepath)

# Pop out the "unreported" country bc it's meaningless here
unreported_n = country_counts.pop(UNREPORTED_LABEL)

# Add the country user counts to the world geodataframe
myworld = world.merge(country_counts, left_on="ISO_A3", right_index=True, how="left")

# Pick colormap info
LOCATION_CMAP = cc.cm.bgy
COLOR_MAX = 2000
assert myworld["n_users"].max() <= COLOR_MAX, f"Data exceeds colormap upper limit of {COLOR_MAX}."
color_norm = plt.matplotlib.colors.LogNorm(vmin=1, vmax=COLOR_MAX)

WORLD_PLOT_KWARGS = dict(
    column="n_users",
    edgecolor="black",
    linewidth=0.3,
    legend=True,
    legend_kwds=dict(label=r"$n$ users", orientation="vertical"),
    missing_kwds=dict(facecolor="gainsboro", linewidth=0.1),
)
myworld.plot(ax=ax, cax=cax, cmap=LOCATION_CMAP, norm=color_norm, **WORLD_PLOT_KWARGS)

ax.axis("off")

# Export
c.export_fig(fig, EXPORT_STEM_LOCATION)
