"""
run liwc on the pre/post lucid stuff
that comes out from tagtog annotations.

In previous stuff, there are like 4000 scripts
broken up to handle all the LIWC stuff.
Here, I'm gonna smash it all into one script.
This txt is small, and I don't want as detailed
output, and I'm not using lot of categories,
also no single word contributions.

So this will
1. run LIWC using that python pkg (generate category scores)
2. run stats comparing pre/post lucid moment (effect size rn)
3. a cheap visualization of the relevant effects
Exporting at the latter 2 stages.
"""
import os
import tqdm
import pandas as pd
import pingouin as pg
import config as c

import liwc
import nltk
from collections import Counter

tqdm.tqdm.pandas()


LIWC_CATEGORIES = ["insight", "agency", "negemo", "posemo"]

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-annotations_lucidprepost.tsv")
export_fname_stat = os.path.join(c.DATA_DIR, "results", "annotations-lucidmoment_liwc.tsv")
export_fname_plot = os.path.join(c.DATA_DIR, "results", "annotations-lucidmoment_liwc.png")
dict_fname = os.path.join(c.DATA_DIR, "dictionaries", "myliwc.dic")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    index_col="post_id")



############################
############################ get LIWC scores
############################

# load the dictionary file
parse, category_names = liwc.load_token_parser(dict_fname)

# build tokenizer
# see main liwc_scores file for reasoning
tknzr = nltk.tokenize.TweetTokenizer()
def tokenize4liwc(doc):
    # lowercase and break into tokens
    tokens = tknzr.tokenize(doc.lower())
    # remove isolated puncuation
    tokens = [ t for t in tokens if not (len(t)==1 and not t.isalpha()) ]
    return tokens


### run LIWC

def liwc_single_doc(doc):
    tokenized_doc = tokenize4liwc(doc)
    n_tokens = len(tokenized_doc)
    counts = Counter(category for token in tokenized_doc for category in parse(token))
    freqs = { category: n/n_tokens for category, n in counts.items() }
    return freqs

# generate a series of results, where each cell is a dict
liwc_ser = df["post_txt"].progress_apply(liwc_single_doc).apply(pd.Series)

# expand that series into a full dataframe of results
liwc_df = liwc_ser.apply(pd.Series).fillna(0)[LIWC_CATEGORIES]

# merge with the before/after labels
out = pd.concat([df, liwc_df], axis=1).drop(columns="post_txt")


# # export
# out.to_csv(export_fname, sep="\t", encoding="utf-8",
#     index=True, float_format="%.2f")



############################
############################ get LIWC effect sizes
############################

# i like to restructure and get columns for comparison
out_piv = out.pivot_table(index="post_id", columns="txt_loc")

stats_list = []
for cat in tqdm.tqdm(LIWC_CATEGORIES, desc="effsize"):
    aft, bef = out_piv[cat][["after_ld", "before_ld"]].T.values
    stats = pg.wilcoxon(aft, bef, alternative="two-sided").drop(columns="alternative")
    stats["cohen-d"] = pg.compute_effsize(aft, bef, paired=True, eftype="cohen")
    stats["cohen-d_lo"], stats["cohen-d_hi"] = pg.compute_bootci(aft, bef,
        paired=False, func="cohen", method="cper", confidence=.95, n_boot=2000, decimals=4)
    stats.index = [cat]
    stats.insert(0, "n", len(aft))
    stats_list.append(stats)

out_stats = pd.concat(stats_list).rename_axis("category")

## add descriptives to the stats
out2 = out.groupby("txt_loc").agg(["mean", "sem"]
    ).T.rename_axis(["category", "metric"]
    ).pivot_table(index="category", columns="metric"
    ).swaplevel(axis=1)
out2.columns = out2.columns.map("-".join)

out3 = out_stats.join(out2)


# export
out_stats.to_csv(export_fname_stat, sep="\t", encoding="utf-8",
    index=True, float_format="%.2f")



############################
############################ visualize
############################

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

n_cats = len(LIWC_CATEGORIES)

BAR_ARGS = dict(width=1, edgecolor="k", lw=.5, alpha=1)
ERROR_ARGS = dict(ecolor="k", elinewidth=.5, capsize=0)
GRIDSPEC_KWS = {
    "wspace" : .2,
    "top"    : .98,
    "right"  : .98,
    "bottom" : .15,
    "left"   : .1,
}

color_vals = [ "gainsboro", c.COLORS["lucid"] ]

_, axes = plt.subplots(ncols=n_cats, figsize=(5, 3),
    sharex=True, sharey=True, gridspec_kw=GRIDSPEC_KWS)

for ax, cat in zip(axes, LIWC_CATEGORIES):
    row = out3.loc[cat]
    yvals = row[["mean-before_ld", "mean-after_ld"]].values
    yerrs = row[["sem-before_ld", "sem-after_ld"]].values
    xvals = range(len(yvals))
    dval = row["cohen-d"]
    pval = row["p-val"]

    # convert to pct for interpretability
    yvals *= 100
    yerrs *= 100

    ax.bar(xvals, yvals, yerr=yerrs, color=color_vals,
        error_kw=ERROR_ARGS, **BAR_ARGS)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(xvals)
    ax.set_xticklabels(["before", "after"])
    ax.set_xlim(xvals[0]-.8, xvals[1]+.8)
    ax.set_ylim(0, 4.7)
    ax.yaxis.set(major_locator=plt.MultipleLocator(1),
                 minor_locator=plt.MultipleLocator(.2),
                 major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(r"% of dream content", fontsize=10)
        ax.set_xlabel("Before or after the moment of lucidity", ha="left", fontsize=10)
    ax.text(.05, 1, cat, transform=ax.transAxes, ha="left", va="top")

    if pval < .001:
        pval_str = r"$p<.001$"
    else:
        pval_str = f"{pval:.3f}".lstrip("0")
        pval_str = rf"$p={pval_str}$"
    effsize_str =  f"{abs(dval):.2f}".lstrip("0")
    effsize_str =  rf"$d={effsize_str}$"
    sigchars = "*" * sum([ pval<cutoff for cutoff in [.05, .01, .001] ])
    if not sigchars:
        sigchars = "^" if pval < .1 else "ns"
    SPACER = .2 # space between each stacked significance text
    y = max(yvals+yerrs) # highest of the two bars
    x = sum(xvals)/len(xvals) # middle of the two bars
    # draw a line a bit up from that
    ax.hlines(y=y+SPACER, xmin=xvals[0], xmax=xvals[1],
        linewidth=2, color="k", capstyle="round")
    # asterisks above that
    weight = "bold" if not sigchars.isalpha() else "normal"
    valign = "center" if not sigchars.isalpha() else "bottom"
    fsize = 14 if not sigchars.isalpha() else 10
    ax.text(x, y+SPACER, sigchars, weight=weight,
        fontsize=fsize, ha="center", va=valign)
    # effect size above that
    ax.text(x+BAR_ARGS["width"], y+SPACER*2, effsize_str,
        fontsize=10, ha="right", va="bottom")
    # pvalue above that
    ax.text(x+BAR_ARGS["width"], y+SPACER*3, pval_str,
        fontsize=10, ha="right", va="bottom")


# import seaborn as sea
# palette = dict(before_ld=c.COLORS["ambiguous"], after_ld=c.COLORS["lucid"])
# df_melt = out.melt(id_vars="txt_loc", var_name="category", value_name="liwc")
# g = sea.catplot(data=df_melt, x="txt_loc", y="liwc",
#     col="category", col_wrap=2, order=["before_ld", "after_ld"],
#     kind="bar", height=3, aspect=1, palette=palette,
#     sharey=False)

### draw the N and effect sizes to emphasize prelimary data
### maybe noisey low-counts in there. add WC for restriction?
###
### Eventually take the SAME timepoint from the SAME user's
### non-lucid dream and run ANOVA.
###
### Also could restrict on lucidity that lasts a while (lulength).

# export
plt.savefig(export_fname_plot)
plt.savefig(export_fname_plot.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()
