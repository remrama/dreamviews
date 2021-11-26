"""
A script to plot distribution of lucidity moments.
"""
import os
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-annotations.csv")
export_fname = os.path.join(c.DATA_DIR, "results", "annotations-lucidmoment.png")

df = pd.read_csv(import_fname, encoding="utf-8",
        index_col="report_id").rename_axis("post_id"
    ).rename(columns=dict(txt="post_txt"))

# only use hf's ratings for now
df = df[df["rater_id"].eq("hf24")]

# add a "length" column for each excerpt
# (hope to have this on the original output later)
df["length"] = df["end"] - df["start"]



########## find the lucidity moments

# find the FIRST lucid label for each post
# and the total length of each post
s1 = df.query("label=='lucid'").groupby("post_id"
    )["start"].min().rename("lumo")
s2 = df.groupby("post_id"
    )["end"].max().rename("post_length")

res = pd.concat([s1, s2], axis=1, join="inner")

# lucid moment proportion (of whole post)
res["lumo_prop"] = s1 / s2


########## find how long was TOTAL lucidity

#### could add this to an "agg" command with s1
#### it's just that s2 doesn't fit there bc not grouping
s3 = df.query("label=='lucid'").groupby("post_id"
    )["length"].sum().rename("lulen")

res = res.join(s3, how="left")
res["lulen_prop"] = s3 / s2


########## find how fragmented lucidity was

#### just messing around rn.
#### not sure the best way to do this.
#### but one way would be to count how many lucid
#### labels there are. NOT great bc it could be
#### that they are broken up by non-dream content.
#### but let's see.
####
#### this will likely also depend on rater
#### until that is all clear.
#### this is dumb. delete.
s4 = df.query("label!='nondream'"
    ).groupby("post_id"
    )["label"].value_counts(
    ).rename("lufrag"
    ).reset_index("label"
    ).query("label=='lucid'"
    ).drop(columns="label").squeeze()

res = res.join(s4, how="left")



########### draw

_, ax = plt.subplots(figsize=(3,3), constrained_layout=True)

sea.histplot(data=res, x="lumo_prop",
    stat="count", cumulative=False,
    element="bars", fill=True,
    binrange=(0, 1), binwidth=.05,
    color=c.COLORS["lucid"], alpha=1,
    edgecolor="black", linewidth=.5,
    kde=True, ax=ax)

# kde args are weird so have to adjust here
ax.lines[0].set(color="black", alpha=1,
    solid_capstyle="round")

ax.set_xlim(0, 1)
ax.set_ylim(0, 15)
ax.set_ylabel("# of dreams", fontsize=10)
xlabel = "Moment of lucidity\n(proportion of dream report)"
# xlabel = "start $\leftarrow$   Moment of lucidity   $\\rightarrow$ end\n(proportion of dream report)"
ax.set_xlabel(xlabel, fontsize=10)
ax.xaxis.set(major_locator=plt.MultipleLocator(.25),
             minor_locator=plt.MultipleLocator(.05),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))
ax.yaxis.set(major_locator=plt.MultipleLocator(5),
             minor_locator=plt.MultipleLocator(1))

for m in ["mean", "median"]:
    x = res["lumo_prop"].agg(m)
    linestyle = "solid" if m=="mean" else "dashed"
    halign = "left" if m=="mean" else "right"
    valign = "top" if m=="mean" else "bottom"
    ax.axvline(x, ls=linestyle, lw=1, c="k", alpha=1)
    ax.text(x, .9, f" {m} ",
        ha=halign, va=valign,
        transform=ax.get_xaxis_transform())


# export
plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()