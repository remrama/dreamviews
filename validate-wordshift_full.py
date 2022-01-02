"""
"""
import os
import tqdm
import argparse
import numpy as np
import pandas as pd
import config as c

import shifterator as sh
# from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser


# try:
#     stops = set(stopwords.words("english"))
# except LookupError:
#     import nltk
#     nltk.download("stopwords")
#     stops = set(stopwords.words("english"))


parser = argparse.ArgumentParser()
parser.add_argument("--bigrams", action="store_true")
parser.add_argument("--nonorm", action="store_false")
parser.add_argument("--nightmares", action="store_true")
args = parser.parse_args()

INCLUDE_BIGRAMS = args.bigrams
NO_NORMING = args.nonorm
NIGHTMARES = args.nightmares

TOP_N = 30 # save out top N ngrams to table

### most export filenames are below
export_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-wordshift_scores.tsv")
export_fname_stats = os.path.join(c.DATA_DIR, "results", "validate-wordshift_scores.tsv")
if NIGHTMARES:
    export_fname = export_fname.replace(".tsv", "_nightmares.tsv")
    export_fname_stats = export_fname_stats.replace(".tsv", "_nightmares.tsv")
    export_ext = "-nightmares.png"
else:
    export_ext = ".png"


df, _ = c.load_dreamviews_data()

TXT_COL = "post_lemmas"


# # remove stopwords from text column
# replace_regex = r"(?<=\b)(" + r"|".join(stops) + r")(?=\b)"
# df[TXT_COL] = df[TXT_COL].replace(replace_regex, "", regex=True).str.strip()

if INCLUDE_BIGRAMS:
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

if NIGHTMARES:
    GROUP_1 = "nonnightmare"
    GROUP_2 = "nightmare"
    df["nightmare"] = df["nightmare"].map({True:"nightmare", False:"nonnightmare"})
    ser = df.set_index(["nightmare", "user_id"])[TXT_COL]
else:
    GROUP_1 = "nonlucid"
    GROUP_2 = "lucid"
    ser = df.query("lucidity.str.contains('lucid')"
        ).set_index(["lucidity", "user_id"])[TXT_COL]





def get_simple_freqs(series):
    ### simple way -- not controlling for user contributions
    # get ngram frequencies
    # explode to one ngram per row
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[GROUP_1]
    ngrams_2 = ngram_ser.loc[GROUP_2]
    ngram2freq_1 = ngrams_1.value_counts().to_dict()
    ngram2freq_2 = ngrams_2.value_counts().to_dict()
    return ngram2freq_1, ngram2freq_2


def get_normed_freqs(series):
    ### control for user counts
    # an ngram's score is no longer a count/int
    # but normalized by number of docs for that user
    # and then added across normalized frequencies

    # get number of documents per user
    # user2freq_1 = ngrams_1.index.value_counts().to_dict()
    # user2freq_2 = ngrams_2.index.value_counts().to_dict()
    user2freq_1 = ser.loc[GROUP_1].groupby("user_id").size()
    user2freq_2 = ser.loc[GROUP_2].groupby("user_id").size()

    # get ngrams frequencies for each user
    # get ngram frequencies
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[GROUP_1]
    ngrams_2 = ngram_ser.loc[GROUP_2]
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



def get_top_contributing_ngrams(shift, n=1000):
    scores = shift.__getattribute__("type2shift_score")
    descending = sorted(scores, key=lambda x: scores[x], reverse=True)
    return set(descending[:n])


N_ITERATIONS = 10

results = {}

for i in tqdm.trange(N_ITERATIONS, desc="wordshift resampling"):
    subsample_ser = ser.sample(n=20000, replace=True, random_state=i)

    if NO_NORMING:
        ngram2freq_1, ngram2freq_2 = get_simple_freqs(subsample_ser)
    else:
        ngram2freq_1, ngram2freq_2 = get_normed_freqs(subsample_ser)


    # metrics = ["type2p_diff", "type2s_diff", "type2s_ref_diff", "type2shift_score"]
    score_dicts = {}

    shift = sh.ProportionShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2)
    keepers = get_top_contributing_ngrams(shift)
    # keep_scores = filter(lambda x: x[0] in keepers, shift.__getattribute__("type2p_diff").items())
    keep_scores = { k:v for k, v in shift.__getattribute__("type2p_diff").items() if k in keepers }
    score_dicts["type2p_diff"] = shift.__getattribute__("type2p_diff")
    # score_dicts["type2p_diff"] = shift.__getattribute__("type2p_diff")
    if i == 0:
        ex_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_proportion"+export_ext)
        shift.get_shift_graph(top_n=100, detailed=False,
            system_names=["non", GROUP_2], show_plot=False, filename=ex_fname)

    shift = sh.EntropyShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
        normalization="variation", alpha=1, reference_value="average")
    keepers = get_top_contributing_ngrams(shift)
    for m in ["type2s_diff", "type2s_ref_diff", "type2shift_score"]:
        keep_scores = { k:v for k, v in shift.__getattribute__(m).items() if k in keepers }
        score_dicts[f"entropy-{m}"] = shift.__getattribute__(m)
    if i == 0:
        ex_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_entropy"+export_ext)
        shift.get_shift_graph(top_n=100, detailed=True,
            system_names=["non", GROUP_2], show_plot=False, filename=ex_fname)

    shift = sh.JSDivergenceShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
        weight_1=0.5, weight_2=0.5, base=2,
        normalization="variation", alpha=1, reference_value=0)
    keepers = get_top_contributing_ngrams(shift)
    for m in ["type2s_diff", "type2s_ref_diff", "type2shift_score"]:
        keep_scores = { k:v for k, v in shift.__getattribute__(m).items() if k in keepers }
        score_dicts[f"jsd-{m}"] = shift.__getattribute__(m)
    if i == 0:
        ex_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_jsd"+export_ext)
        shift.get_shift_graph(top_n=100, detailed=False,
            system_names=["non", GROUP_2], show_plot=False, filename=ex_fname)

    shift = sh.WeightedAvgShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
        type2score_1="NRC-emotion_fear_English", normalization="variation")
    keepers = get_top_contributing_ngrams(shift)
    for m in ["type2s_diff", "type2s_ref_diff", "type2shift_score"]:
        keep_scores = { k:v for k, v in shift.__getattribute__(m).items() if k in keepers }
        score_dicts[f"fear-{m}"] = shift.__getattribute__(m)
    if i == 0:
        ex_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_fear"+export_ext)
        shift.get_shift_graph(top_n=100, detailed=True,
            system_names=["non", GROUP_2], show_plot=False, filename=ex_fname)

    results[i+1] = score_dicts
    # # extract some relevant measures from results
    # scores = shift.get_shift_scores(details=True)
    # metrics = ["type2p_diff", "type2s_diff", "type2p_avg",
    #     "type2s_ref_diff", "type2shift_score"]
    # score_dicts = { m: s for m, s in zip(metrics, scores) }
    # score_dicts = { m: s for m, s in zip(metrics, scores) }
    # score_dicts = { attr: shift.__getattribute__(attr) for attr in shift.__dict__.keys() if attr.startswith("type2") }
    # metrics = ["type2p_diff", "type2s_diff", "type2s_ref_diff", "type2shift_score"]
    # score_dicts = { m: shift.__getattribute__(m) for m in metrics }


df_list = []
for iternum, iterdict in results.items():
    df_ = pd.DataFrame(iterdict)
    df_.insert(0, "iteration", iternum)
    df_list.append(df_)

res = pd.concat(df_list, axis=0)

res = res.rename_axis("ngram").reset_index(drop=False
    ).sort_values(["ngram", "iteration"]
    ).reset_index(drop=True)


keep_ngrams = res.ngram.value_counts().loc[lambda x: x==10].index.tolist()
res = res[ res["ngram"].isin(keep_ngrams) ]

res.to_csv(export_fname, index=False, sep="\t", encoding="utf-8")


####### new dataframe with stats for each token
####### across all sampling iterations

def ci_lo(x):
    return np.quantile(x, .025)
def ci_hi(x):
    return np.quantile(x, .975)

def pval(x):
    fracs = [ np.mean(x<0), np.mean(x>0) ]
    lowest = np.min(fracs)
    return lowest / 2


# add sign to the jsd shift score
res["jsd-type2shift_score"] *= res["jsd-type2s_ref_diff"].transform(np.sign)

# get mean, error, and pvalues for each score
x = res.drop(columns="iteration").groupby("ngram")

y = x.agg(["mean", "quantile"])

column.describe(percentiles=[0.5, 0.95])
df.groupby("AGGREGATE").quantile([0, 0.25, 0.5, 0.75, 0.95, 1])

columns.agg('describe')[['25%', '50%', '75%', 'count']]
f = lambda x: x.quantile(0.5)

m = x.agg("mean")
n = x.agg("quantile", q=.025)
o = x.agg("quantile", q=.975)
p = x.agg(pval)
p = x.agg(["mean", "quantile", ""])

stats_df = res.rename_axis("ngram").groupby("ngram").agg(["mean", ci_lo, ci_hi, pval]
    ).drop(columns="iteration"
    # sort by absolute mean of proportion
    ).sort_values(("type2p_diff", "mean"), key=abs, ascending=False)


stats_df.to_csv(export_fname_stats, index=True, sep="\t", encoding="utf-8")
