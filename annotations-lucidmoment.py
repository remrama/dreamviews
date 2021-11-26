"""
A script to plot distribution of lucidity moments
and generate dataframe with text for pre/post lucidity.
Also save out summary moment stats.
"""
import os
import argparse
import pandas as pd
import config as c

import seaborn as sea
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

parser = argparse.ArgumentParser()
parser.add_argument("--dreamonly", action="store_true", help="remove nondream content")
args = parser.parse_args()

DREAM_ONLY = args.dreamonly

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-annotations.csv")
export_fname_txt = os.path.join(c.DATA_DIR, "derivatives", "posts-annotations_lucidprepost.tsv")
export_fname_stat = os.path.join(c.DATA_DIR, "results", "annotations-lucidmoment.tsv")
export_fname_plot = os.path.join(c.DATA_DIR, "results", "annotations-lucidmoment.png")

df = pd.read_csv(import_fname, encoding="utf-8",
        index_col="report_id").rename_axis("post_id"
    ).rename(columns=dict(text="post_txt"))

# only use hf's ratings for now
df = df[df["rater_id"].eq("hf24")]

# add a "length" column for each excerpt
# (hope to have this on the original output later)
df["length"] = df["end"] - df["start"]



########## find the lucidity moments

if not DREAM_ONLY:
    ###### if using the whole report (not dropping nondream content)
    # find the FIRST lucid label for each post
    # and the total length of each post
    s1 = df.query("label=='lucid'").groupby("post_id"
        )["start"].min().rename("lumo")
    # get dream length using either whole report or just dream content
    s2 = df.groupby("post_id"
        )["end"].max().rename("post_length")    # either of these work
    #     )["length"].sum().rename("post_length") # either of these work

else:
    ###### if using only dream content, slightly more complicated
    ## need a new "start" for the first lucidity since the existing
    ## character count includes non-dream

    # Sum up the length of nondream content prior to
    # the first lucid moment and subtract it out.
    # Then summing the other lengths should be fine.
    s1 = df.query("label=='lucid'").groupby("post_id"
        )["start"].min().rename("lumo")
    def find_nondream_beforelumo_length(_df):
        post_id = _df.index.unique()[0]
        if post_id in s1:
            lumo = s1.loc[post_id]
            x = _df.query("label=='nondream'"
                ).query(f"start<{lumo}"
                )["length"].sum()
        else:
            x = pd.NA
        return x
    s0 = df.groupby("post_id").apply(find_nondream_beforelumo_length
        ).dropna().astype(int).sort_index()
    s1 -= s0

    # get total length without nondream content
    s2 = df.query("label!='nondream'").groupby("post_id"
        )["length"].sum().rename("post_length")


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


#### save out csv with summary values
summary = res.describe().T.rename_axis("measure")
summary["count"] = summary["count"].astype(int)
summary.to_csv(export_fname_stat, sep="\t", encoding="utf-8",
    index=True, float_format="%.2f")


#####################
##################### generate dataframe with
##################### text for before and after lucidity

txt_results = {}
for post_id, lumo in s1.items():

    # get the full raw txt of report
    # (later might be loading in another file
    #  for this but for now text is here)
    post_txt =  df.loc[post_id].sort_values("start"
        )["post_txt"].str.cat()

    beforelucid_txt = post_txt[:lumo].strip()
    afterlucid_txt = post_txt[lumo:].strip()

    txt_results[post_id] = dict(before_ld=beforelucid_txt,
        after_ld=afterlucid_txt)

txt_out = pd.DataFrame.from_dict(txt_results, orient="index"
    ).unstack(
    ).rename_axis(["txt_loc", "post_id"]
    ).rename("post_txt"
    ).swaplevel().sort_index()

# remove rows with practically empty text
### (gotta be a better way)
### probably have to restrict in the liwc file ANYWAYS bc of tokenizing
nchars = txt_out.groupby("post_id"
    ).apply(lambda s: s.str.len()
    ).reset_index(
    ).query("post_txt>=50"
    ).groupby("post_id").size()
good_posts = nchars[ nchars==2 ].index.tolist()
txt_out = txt_out.loc[good_posts]

#### save
txt_out.to_csv(export_fname_txt, sep="\t", encoding="utf-8",
    index=True)



#####################
##################### draw
#####################
#####################

_, ax = plt.subplots(figsize=(3,3), constrained_layout=True)

binwidth = .1 if DREAM_ONLY else .05
ymax = 27 if DREAM_ONLY else 15
xlabel = "Moment of lucidity\n(proportion of full report)"
if DREAM_ONLY:
    xlabel = xlabel.replace("full report", "dream content")

sea.histplot(data=res, x="lumo_prop",
    stat="count", cumulative=False,
    element="bars", fill=True,
    binrange=(0, 1), binwidth=binwidth,
    color=c.COLORS["lucid"], alpha=1,
    edgecolor="black", linewidth=.5,
    kde=True, ax=ax)

# kde args are weird so have to adjust here
ax.lines[0].set(color="black", alpha=1,
    solid_capstyle="round")

ax.set_xlim(0, 1)
ax.set_ylim(0, ymax)
ax.set_ylabel("# of posts", fontsize=10)
# xlabel = "start $\leftarrow$   Moment of lucidity   $\\rightarrow$ end\n(proportion of dream report)"
ax.set_xlabel(xlabel, fontsize=10)
ax.xaxis.set(major_locator=plt.MultipleLocator(binwidth*5),
             minor_locator=plt.MultipleLocator(binwidth),
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
plt.savefig(export_fname_plot)
plt.savefig(export_fname_plot.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()