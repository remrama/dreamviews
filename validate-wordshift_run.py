"""
Run resampled wordshifts on lemmatized dream reports.
Outputs tsv files that have different shift contribution
scores for lots of words at lots of resampled iterations.

Before resampling, save out a few wordshifts using whole dataset.
Useful for diagnostics, to make sure my custom plots and the
resampling don't stray too far off.


Output is used in other scripts to:
    0. first gets aggregated into summary table
    1. plot JSD shift for lucidity
    2. plot Fear shift for nightmares
    3. generate latex table of bigram contributions

Wordshifts are very easy to generate using shifterator.
Why am I making it so complicated?
Two related things: user-dependent contributions and significance testing.

(A) Account for biased user contributions, some have 1000 others have 1.
----
Solution: An ngram's score is no longer just count, it's now
    normalized by number of docs for that user and then
    that -- instead of raw counts -- is summed.

(B) Some imperfect way of determining whether a word's contribution is "meaningful".
----
Solution: Resample the shift scores to get 95% confidence intervals
    and compare to zero. (aggregated in another file)

(C) Include bigrams.
----
Solution: Run on posts transformed with gensim bigram model.

I'm also running (D) different wordshifts and (E) including nightmares
so this all leads to a lengthy if not overcomplicated script.
"""
import os
import tqdm
import argparse

import numpy as np
import pandas as pd
import config as c

import shifterator as sh
from gensim.models.phrases import Phrases, Phraser

import matplotlib.pyplot as plt # to close shift plots

parser = argparse.ArgumentParser()
parser.add_argument("--nobigrams", action="store_true",
    help="Don't include bigrams in the wordshift, just for development.")
parser.add_argument("--nonorm", action="store_true",
    help="Don't account for biased contributions, just for development.")
args = parser.parse_args()

NO_BIGRAMS = args.nobigrams
NO_NORMING = args.nonorm

TOP_N = 30 # save out top N ngrams to table

# ### most export filenames are below
# export_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-wordshift_scores.tsv")
# export_fname_stats = os.path.join(c.DATA_DIR, "results", "validate-wordshift_scores.tsv")
# if NIGHTMARES:
#     export_fname = export_fname.replace(".tsv", "-nightmares.tsv")
#     export_fname_stats = export_fname_stats.replace(".tsv", "-nightmares.tsv")
#     export_ext = "-nightmares.png"
# else:
#     export_ext = ".png"


df = c.load_dreamviews_posts()

TXT_COL = "post_lemmas"

# # remove stopwords from text column
# replace_regex = r"(?<=\b)(" + r"|".join(stops) + r")(?=\b)"
# df[TXT_COL] = df[TXT_COL].replace(replace_regex, "", regex=True).str.strip()

if not NO_BIGRAMS: # sorry for the double negative
    # build bigram model and convert text column to bigrams
    min_count = 1 # ignore all words and bigrams with total collected count lower than this value
    threshold = 1
    delim = "_" # for joining bigrams
    scoring = "default"
    sentences = df[TXT_COL].str.lower().str.split().tolist()
    # connector_words=ENGLISH_CONNECTOR_WORDS
    phrase_model = Phrases(sentences, delimiter=delim, min_count=min_count, threshold=threshold, scoring="default")
    # phrase_model = Phrases(phrase_model[sentences], delimiter=delim, min_count=3, threshold=threshold, scoring=scoring)
    phrase_model = Phraser(phrase_model) # memory benefits?
    unigram2ngram = lambda x: phrase_model[x]
    tqdm.tqdm.pandas()
    df[TXT_COL] = df[TXT_COL].str.lower().str.split().progress_apply(unigram2ngram).str.join(" ")
# def find_ngrams(input_list, n):
#     return zip(*[input_list[i:] for i in range(n)])
# def ngrams(x, n):
#     doc = x.split()
#     return " ".join(map(lambda x: "-".join(x), find_ngrams(doc, n)))


# reduce user bias
# df = df.groupby("user_id").sample(n=1, replace=False)

df["nightmare"] = df["nightmare"].map({True:"nightmare", False:"nonnightmare"})
nm_ser = df.set_index(["nightmare", "user_id"])[TXT_COL]

ld_ser = df.query("lucidity.str.contains('lucid')", engine="python"
    ).set_index(["lucidity", "user_id"])[TXT_COL]





def get_simple_freqs(series, group1, group2):
    ### simple way -- not controlling for user contributions
    # get ngram frequencies
    # explode to one ngram per row
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    ngram2freq_1 = ngrams_1.value_counts().to_dict()
    ngram2freq_2 = ngrams_2.value_counts().to_dict()
    return ngram2freq_1, ngram2freq_2


def get_normed_freqs(series, group1, group2):
    ### control for user counts
    # an ngram's score is no longer a count/int
    # but normalized by number of docs for that user
    # and then added across normalized frequencies

    # get number of documents per user
    # user2freq_1 = ngrams_1.index.value_counts().to_dict()
    # user2freq_2 = ngrams_2.index.value_counts().to_dict()
    user2freq_1 = series.loc[group1].groupby("user_id").size()
    user2freq_2 = series.loc[group2].groupby("user_id").size()

    # get ngrams frequencies for each user
    # get ngram frequencies
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    userngrams2freq_1 = ngrams_1.groupby("user_id").value_counts()
    userngrams2freq_2 = ngrams_2.groupby("user_id").value_counts()

    # get normalized ngram score for each ngram/user combo.
    # divide user's ngram frequency by number of documents
    userngrams2norm_1 = userngrams2freq_1 / user2freq_1
    userngrams2norm_2 = userngrams2freq_2 / user2freq_2

    # sum across normed scores for each ngram instead of frequency
    ngram2freq_1 = userngrams2norm_1.groupby(TXT_COL).sum().to_dict()
    ngram2freq_2 = userngrams2norm_2.groupby(TXT_COL).sum().to_dict()

    # # get n users that use a given ngram
    # ngram2userfreq_1 = ngrams_1.reset_index().groupby("post_lemmas")["user_id"].nunique()
    # ngram2userfreq_2 = ngrams_2.reset_index().groupby("post_lemmas")["user_id"].nunique()

    return ngram2freq_1, ngram2freq_2


# def get_normed_freqs2(series):
#     ngram_ser = series.str.lower().str.split().explode()
#     ngrams_1 = ngram_ser.loc[GROUP_1]
#     ngrams_2 = ngram_ser.loc[GROUP_2]
#     ngram2freq_1 = ngrams_1.value_counts().to_dict()
#     ngram2freq_2 = ngrams_2.value_counts().to_dict()

#     # get n users that use a given ngram
#     ngram2userfreq_1 = ngrams_1.reset_index().groupby("post_lemmas")["user_id"].nunique().to_dict()
#     ngram2userfreq_2 = ngrams_2.reset_index().groupby("post_lemmas")["user_id"].nunique().to_dict()

#     ngram2user = ngram_ser.reset_index().groupby("post_lemmas")["user_id"].nunique().to_dict()

#     ngram2norm_1 = { t: ngram2userfreq_1[t]*f for t, f in ngram2freq_1.items() }
#     ngram2norm_2 = { t: ngram2userfreq_2[t]*f for t, f in ngram2freq_2.items() }
#     # ngram2norm_1 = { t: f/(ngram2userfreq_1[t] for t, f in ngram2freq_1.items() }
#     # ngram2norm_2 = { t: f/(ngram2userfreq_2[t]/f) for t, f in ngram2freq_2.items() }

#     return ngram2norm_1, ngram2norm_2

# vocab = set(list(ngram2freq_1) + list(ngram2freq_2))
# weights1 = { t: 1/ngram2userfreq_1[t] if t in ngram2freq_1 else 0 for t in vocab }
# weights2 = { t: 1/ngram2userfreq_2[t] if t in ngram2freq_2 else 0 for t in vocab }
# shift = sh.WeightedAvgShift(
#     type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
#     type2score_1=weights1, type2score_2=weights2,
#     reference_value="average")
# shift.get_shift_graph(top_n=50, detailed=True, show_plot=True)
# shift = sh.ProportionShift(type2freq_1=ngram2norm_1, type2freq_2=ngram2norm_2)
# shift.get_shift_graph(top_n=50, detailed=False, show_plot=True)
# shift_scores = shift.get_shift_scores(details=True)
# shift_score_names = ["type2p_diff", "type2s_diff", "type2p_avg",
#     "type2s_ref_diff", "type2shift_score"]
# score_dicts = { n: s for n, s in zip(shift_score_names, shift_scores) }


#########################
######################### Plots using whole dataset.
######################### For sanity checking later.
######################### And curiosity.
######################### But too much or paper.
#########################


def shift2df(shift, detail_level=2):
    assert detail_level in [0, 1, 2]
    if detail_level == 0:
        shift_scores = shift.get_shift_scores(details=False)
        shift_score_names = ["type2shift_score"]
        score_dicts = { n: s for n, s in zip(shift_score_names, shift_scores) }
    elif detail_level == 1:
        shift_scores = shift.get_shift_scores(details=True)
        shift_score_names = ["type2p_diff", "type2s_diff", "type2p_avg",
            "type2s_ref_diff", "type2shift_score"]
        score_dicts = { n: s for n, s in zip(shift_score_names, shift_scores) }
    elif detail_level == 2:
        score_dicts = { k: shift.__getattribute__(k)
            for k in shift.__dict__.keys() if k.startswith("type2") }
    df_ = pd.DataFrame(score_dicts).sort_index().rename_axis("ngram")
    return df_

################### do the one-off nightmare one first
nm_group1, nm_group2 = "nonnightmare", "nightmare"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(nm_ser, nm_group1, nm_group2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(nm_ser, nm_group1, nm_group2)

### Fear shift for nightmares
shift = sh.WeightedAvgShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
    type2score_1="NRC-emotion_fear_English", normalization="variation",
    stop_lens=[(.3, .7)])
out_fname = os.path.join(c.DATA_DIR, "results", f"validate-wordshift-fear_nm.png")
shift.get_shift_graph(top_n=50, system_names=[nm_group1, nm_group2],
    detailed=True, show_plot=False, filename=out_fname)
plt.close()

out_df = shift2df(shift)
export_fname = os.path.join(c.DATA_DIR, "derivatives", f"validate-wordshift_scores-fear_nm.tsv")
out_df.to_csv(export_fname, index=True, na_rep="NA", sep="\t", encoding="utf-8")



###################################### LUCID STUFF

ld_group1, ld_group2 = "nonlucid", "lucid"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(ld_ser, ld_group1, ld_group2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(ld_ser, ld_group1, ld_group2)

### Basic proportion shift to understand if the the fancy stuff is worth it.
shift = sh.ProportionShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2)
out_fname = os.path.join(c.DATA_DIR, "results", f"validate-wordshift-proportion.png")
shift.get_shift_graph(top_n=50, system_names=[ld_group1, ld_group2],
    detailed=False, show_plot=False, filename=out_fname)
plt.close()

### JSD shift because this is the main one of interest.
shift = sh.JSDivergenceShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
    weight_1=0.5, weight_2=0.5, base=2,
    normalization="variation", alpha=1, reference_value="average")
out_fname = os.path.join(c.DATA_DIR, "results", f"validate-wordshift-jsd.png")
shift.get_shift_graph(top_n=50, system_names=[ld_group1, ld_group2],
    detailed=True, show_plot=False, filename=out_fname)
plt.close()

### Save out JSD word-level results, which has proportions in it too
# a = jsd_df.sort_values("type2p_diff", ascending=False, key=abs)
# a[a.index.str.contains("_")][:40]
out_df = shift2df(shift)
export_fname = os.path.join(c.DATA_DIR, "derivatives", f"validate-wordshift_scores-jsd.tsv")
out_df.to_csv(export_fname, index=True, na_rep="NA", sep="\t", encoding="utf-8")

# ### and these might be worth saving
# shift_component_sums = shift.get_shift_component_sums()
# jsd_score = shift.diff


# #### recreating shift score
# shift = sh.JSDivergenceShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
#     weight_1=0.5, weight_2=0.5, base=2,
#     normalization="variation", alpha=1, reference_value="average")
# df = shift2df(shift, detail_level=2).sort_values("type2shift_score", ascending=False, key=abs)

# S_true = df.type2shift_score

# A = df.type2p_diff * df.type2s_ref_diff
# B = df.type2p_avg * df.type2s_diff
# S = A + B
# abs_sum = S.abs().sum()
# S_pred = S / abs_sum

# # I don't see how Table 1 is true
# # (maybe something I don't understand about the derivation)
# # It doesn't take the reference into account, but maybe in the derivation.
# # And/or it doesn't have the addition part, but again I guess that's all in derivation.

# # So I should be able to also do this then???
# ## (It's off but seems relatively right)
# A = df.type2score_2 * df.type2p_2
# B = df.type2score_1 * df.type2p_1
# S = A - B
# abs_sum = S.abs().sum()
# S_pred2 = S / abs_sum

# S = df.type2p_diff * df.type2s_ref_diff
# abs_sum = S.abs().sum()
# S_pred3 = S / abs_sum





# #####################


# # Don't wanna save all words so each time keep anything
# # that was in the top 1000 word contributors before
# # or the top 1000 with just word freq shift.

# # top_pdiffs = jsd_df["type2p_diff"].sort_values(ascending=False, key=abs).index.tolist()[:100]
# # top_shifts = jsd_df["type2shift_score"].sort_values(ascending=False, key=abs).index.tolist()[:100]
# # keep_ngrams = set(top_pdiffs + top_shifts)

# def get_top_contributing_ngrams(shift, n=1000):
#     scores = shift.__getattribute__("type2shift_score")
#     descending = sorted(scores, key=lambda x: scores[x], reverse=True)
#     return set(descending[:n])


# N_ITERATIONS = 10

# results = {}

# for i in tqdm.trange(N_ITERATIONS, desc="wordshift resampling"):
#     subsample_ser = ser.sample(n=20000, replace=True, random_state=i)

#     if NO_NORMING:
#         ngram2freq_1, ngram2freq_2 = get_simple_freqs(subsample_ser)
#     else:
#         ngram2freq_1, ngram2freq_2 = get_normed_freqs(subsample_ser)

#     # metrics = ["type2p_diff", "type2s_diff", "type2s_ref_diff", "type2shift_score"]
#     score_dicts = {}

#     # shift = sh.ProportionShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2)
#     # keepers = get_top_contributing_ngrams(shift)
#     # # keep_scores = filter(lambda x: x[0] in keepers, shift.__getattribute__("type2p_diff").items())
#     # keep_scores = { k:v for k, v in shift.__getattribute__("type2p_diff").items() if k in keepers }
#     # score_dicts["type2p_diff"] = shift.__getattribute__("type2p_diff")
#     # # score_dicts["type2p_diff"] = shift.__getattribute__("type2p_diff")

#     shift = sh.JSDivergenceShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
#         weight_1=0.5, weight_2=0.5, base=2,
#         normalization="variation", alpha=1, reference_value=0)
#     keepers = get_top_contributing_ngrams(shift)
#     for m in ["type2s_diff", "type2s_ref_diff", "type2shift_score"]:
#         keep_scores = { k:v for k, v in shift.__getattribute__(m).items() if k in keepers }
#         score_dicts[f"jsd-{m}"] = shift.__getattribute__(m)

#     # # if NIGHTMARES:
#     # shift = sh.WeightedAvgShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
#     #     type2score_1="NRC-emotion_fear_English", normalization="variation",
#     #     stop_lens=[(.3, .7)])
#     # keepers = get_top_contributing_ngrams(shift)
#     # for m in ["type2s_diff", "type2s_ref_diff", "type2shift_score"]:
#     #     keep_scores = { k:v for k, v in shift.__getattribute__(m).items() if k in keepers }
#     #     score_dicts[f"fear-{m}"] = shift.__getattribute__(m)

#     results[i+1] = score_dicts
#     # # extract some relevant measures from results
#     # scores = shift.get_shift_scores(details=True)
#     # metrics = ["type2p_diff", "type2s_diff", "type2p_avg",
#     #     "type2s_ref_diff", "type2shift_score"]
#     # score_dicts = { m: s for m, s in zip(metrics, scores) }
#     # score_dicts = { m: s for m, s in zip(metrics, scores) }
#     # score_dicts = { attr: shift.__getattribute__(attr) for attr in shift.__dict__.keys() if attr.startswith("type2") }
#     # metrics = ["type2p_diff", "type2s_diff", "type2s_ref_diff", "type2shift_score"]
#     # score_dicts = { m: shift.__getattribute__(m) for m in metrics }


# df_list = []
# for iternum, iterdict in results.items():
#     df_ = pd.DataFrame(iterdict)
#     df_.insert(0, "iteration", iternum)
#     df_list.append(df_)

# res = pd.concat(df_list, axis=0)

# res = res.rename_axis("ngram").reset_index(drop=False
#     ).sort_values(["ngram", "iteration"]
#     ).reset_index(drop=True)


# # keep_ngrams = res.ngram.value_counts().loc[lambda x: x==10].index.tolist()
# # res = res[ res["ngram"].isin(keep_ngrams) ]

# # res.to_csv(export_fname, index=False, sep="\t", encoding="utf-8")


# ####### new dataframe with stats for each token
# ####### across all sampling iterations

# def ci_lo(x):
#     return np.quantile(x, .025)
# def ci_hi(x):
#     return np.quantile(x, .975)

# def pval(x):
#     fracs = [ np.mean(x<0), np.mean(x>0) ]
#     lowest = np.min(fracs)
#     return lowest / 2


# # add sign to the jsd shift score
# res["jsd-type2shift_score"] *= res["jsd-type2s_ref_diff"].transform(np.sign)

# # get mean, error, and pvalues for each score
# x = res.drop(columns="iteration").groupby("ngram")

# y = x.agg(["mean", "quantile"])

# column.describe(percentiles=[0.5, 0.95])
# df.groupby("AGGREGATE").quantile([0, 0.25, 0.5, 0.75, 0.95, 1])

# columns.agg('describe')[['25%', '50%', '75%', 'count']]
# f = lambda x: x.quantile(0.5)

# m = x.agg("mean")
# n = x.agg("quantile", q=.025)
# o = x.agg("quantile", q=.975)
# p = x.agg(pval)
# p = x.agg(["mean", "quantile", ""])

# stats_df = res.rename_axis("ngram").groupby("ngram").agg(["mean", ci_lo, ci_hi, pval]
#     ).drop(columns="iteration"
#     # sort by absolute mean of proportion
#     ).sort_values(("type2p_diff", "mean"), key=abs, ascending=False)


# stats_df.to_csv(export_fname_stats, index=True, sep="\t", encoding="utf-8")




# # reduce to only words that showed up in every resampling
# results_wide = pd.concat(df_list, axis=1, join="inner"
#     ).rename_axis("token").reset_index(drop=False)

# results = pd.wide_to_long(results_wide,
#         stubnames=score_dicts.keys(),
#         i="token", j="iteration", sep="-"
#     ).swaplevel(
#     ).sort_index()


# results.to_csv(export_fname1, index=True, na_rep="NA", sep="\t", encoding="utf-8")


