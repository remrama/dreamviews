"""
draw a single plot of all the LIWC effects.
"""
import os
import tqdm
import pandas as pd
import pingouin as pg
import config as c


import_fname_liwc = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc.tsv")
import_fname_attr = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "analysis-liwc.tsv")

# merge the clean data file and all its attributes with the liwc results
df_liwc = pd.read_csv(import_fname_liwc, sep="\t", encoding="utf-8", index_col="post_id")
df_attr = pd.read_csv(import_fname_attr, sep="\t", encoding="utf-8", index_col="post_id")
df = df_attr.join(df_liwc, how="inner")
assert len(df) == len(df_attr) == len(df_liwc)


##################### statistics

liwc_cats = df_liwc.columns.tolist()

######### wilcoxon

# Average the LD and NLD scores for each user.
# Some users might not have both dream types
# and they'll be removed.
avgs = df.groupby(["user_id", "lucidity"]
    )[liwc_cats].mean(
    ).drop(["ambiguous", "unspecified"], level="lucidity"
    ).rename_axis(columns="category"
    ).pivot_table(index="user_id", columns="lucidity"
    ).dropna()
# avgs.index.get_level_values("user_id").duplicated(keep=False)

# init the results with descriptives, then add stats
descrpt = avgs.agg(["mean", "std", "sem", "min", "max"]).T

wilcoxon_results = []
for cat in tqdm.tqdm(liwc_cats, desc="wilcoxon tests"):
    ld, nld = avgs[cat][["lucid", "non-lucid"]].T.values
    stats = pg.wilcoxon(ld, nld, alternative="two-sided")
    stats.index = [cat]
    stats["cohen-d"] = pg.compute_effsize(ld, nld, paired=True, eftype="cohen")
    stats["cohen-d_lo"], stats["cohen-d_hi"] = pg.compute_bootci(ld, nld,
        paired=True, func="cohen", method="cper",
        confidence=.95, n_boot=2000, decimals=4)
    stats["n"] = len(ld) # should be the same every time
    wilcoxon_results.append(stats)
wilc = pd.concat(wilcoxon_results
    ).rename_axis("category"
    ).drop(columns="alternative")

# merge descriptives and statistices
results = descrpt.join(wilc, how="inner")

# correct for multiple comparisons
_, fdr = pg.multicomp(results["p-val"].values, method="fdr_bh")
insert_indx = 1 + results.columns.tolist().index("p-val")
results.insert(insert_indx, "p-val_fdr", fdr)



######### lme with R

#### rpy2
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
base = importr("base")
stats = importr("stats")
# or setting R=ro.r 
# allows for access like R.summary and R.anova
# instead of base.summary and stats.anova
# (cleaner when next to python code)
lme4 = ro.packages.importr("lme4")

subdf = df.loc[df["lucidity"].str.contains("lucid")].copy().reset_index(drop=True)

ro.globalenv["lucidity"] = ro.FactorVector(subdf["lucidity"].values)
ro.globalenv["user_id"] = ro.FactorVector(subdf["user_id"].values)

me_results = []
for cat in tqdm.tqdm(liwc_cats, "mixed models"):
    ro.globalenv["iv"] = ro.FloatVector(subdf[cat].values)
    model = lme4.glmer("lucidity ~ iv + (1|user_id)", family="binomial")
    # model = lme4.glmer(f"lucidity ~ {cat} + (1|user_id)", data=subdf, family="binomial")
    # anova isn't the best, I want full summary,
    # but for now this is the only thing I can get as a dataframe
    anova_stats = stats.anova(model)
    anova_stats.index = [cat]
    me_results.append(anova_stats)
    # s = base.summary(model)
    # res = s.rx2("residuals")
    # print(base.summary(model))
    # https://stackoverflow.com/a/29152290
    # coeffs = R.summary(model).rx2('coefficients') # R = ro.r
    # coeffs = model.rx2("coefficients")
    # rx2 gives array but I want names :/
me_out = pd.concat(me_results
    ).rename_axis("category")
me_out.columns = me_out.columns.str.lower().str.replace(" ", "-"
    ).map(lambda x: f"lme4_{x}")


results = results.join(me_out, how="inner")



################### export
# sort values by the effect size
results = results.sort_values("cohen-d", ascending=False)

results.to_csv(export_fname, sep="\t", encoding="utf-8",
    index=True)
