"""Use wordshift scores (and default visualizations) to separate texts.

Wordshifts are very easy to generate using shifterator.
https://github.com/ryanjgallagher/shifterator

This dataset is huge biases in how many users contribute posts.
Some users include 1000 posts, others 1. Word frequencies are
normalized to account for this, before being passed to shifterator.

Bigrams are included in the wordshift by first transforming text with gensim.

IMPORTS
=======
    - lemmatized posts, dreamviews-posts.tsv
EXPORTS
=======
    - raw JSD shift scores for lucidity,          validate-wordshift_jsd-scores.tsv
    - default JSD shift plot for lucidity,        validate-wordshift_jsd-plot.png
    - raw NRC-fear shift scores for nightmares,   validate-wordshift_fear-scores.tsv
    - default NRC-fear shift plot for nightmares, validate-wordshift_fear-plot.tsv
    - default proportion shift plot for lucidity, validate-wordshift_proportion-plot.tsv
    - table of top 1-grams higher in LDs,         validate-wordshift_proportion-top1grams.tsv
    - table of top 2-grams higher in LDs,         validate-wordshift_proportion-top2grams.tsv
"""
import argparse

from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shifterator as sh
import tqdm

import config as c


parser = argparse.ArgumentParser()
parser.add_argument("--nobigrams", action="store_true", help="Don't include bigrams in the wordshift, just for development.")
parser.add_argument("--nonorm", action="store_true", help="Don't account for biased contributions, just for development.")
args = parser.parse_args()

NO_BIGRAMS = args.nobigrams
NO_NORMING = args.nonorm


################################################################################
# SETUP
################################################################################

column_name = "post_lemmas"
top_n_plots = 50
top_n_tables = 100

# Choose export locations.
export_path_jsd_table = c.DATA_DIR / "derivatives" / f"validate-wordshift_jsd-scores.tsv"
export_path_jsd_plot = c.DATA_DIR / "derivatives" / f"validate-wordshift_jsd-plot.png"
export_path_fear_table = c.DATA_DIR / "derivatives" / f"validate-wordshift_fear-scores.tsv"
export_path_fear_plot = c.DATA_DIR / "derivatives" / f"validate-wordshift_fear-plot.png"
export_path_prop_plot = c.DATA_DIR / "derivatives" / f"validate-wordshift_proportion-plot.png"
export_path_top1grams = c.DATA_DIR / "derivatives" / f"validate-wordshift_proportion-ld1grams.tsv"
export_path_top2grams = c.DATA_DIR / "derivatives" / f"validate-wordshift_proportion-ld2grams.tsv"

# Load data.
df = c.load_dreamviews_posts()


################################################################################
# CONNECT BIGRAMS
################################################################################

# # remove stopwords from text column
# replace_regex = r"(?<=\b)(" + r"|".join(stops) + r")(?=\b)"
# df[TXT_COL] = df[TXT_COL].replace(replace_regex, "", regex=True).str.strip()

if not NO_BIGRAMS:  # Sorry for the double negative.
    # build bigram model and convert text column to bigrams
    min_count = 1 # ignore all words and bigrams with total collected count lower than this value
    threshold = 1
    delim = "_" # for joining bigrams
    scoring = "default"
    sentences = df[column_name].str.lower().str.split().tolist()
    # connector_words=ENGLISH_CONNECTOR_WORDS
    phrase_model = Phrases(sentences, delimiter=delim, min_count=min_count, threshold=threshold, scoring="default")
    # phrase_model = Phrases(phrase_model[sentences], delimiter=delim, min_count=3, threshold=threshold, scoring=scoring)
    phrase_model = Phraser(phrase_model) # memory benefits?
    unigram2ngram = lambda x: phrase_model[x]
    tqdm.tqdm.pandas(desc="Mixing bigrams into corpus")
    df[column_name] = (df[column_name]
        .str.lower()
        .str.split()
        .progress_apply(unigram2ngram)
        .str.join(" ")
    )
# def find_ngrams(input_list, n):
#     return zip(*[input_list[i:] for i in range(n)])
# def ngrams(x, n):
#     doc = x.split()
#     return " ".join(map(lambda x: "-".join(x), find_ngrams(doc, n)))

# reduce user bias
# df = df.groupby("user_id").sample(n=1, replace=False)


################################################################################
# EXTRACT DATA SUBSETS
################################################################################

# Extract lucidity Series for JSD and proportion shifts.
ld_ser = (df
    .query("lucidity.str.contains('lucid')", engine="python")
    .set_index(["lucidity", "user_id"])
    [column_name]
)

# Export nightmare Series for NRC-fear shift.
df["nightmare"] = df["nightmare"].map({True:"nightmare", False:"nonnightmare"})
nm_ser = df.set_index(["nightmare", "user_id"])[column_name]


################################################################################
# NORMALIZATION FUNCTIONS
################################################################################

def shift2df(shift, detail_level):
    """Convert shifterator object into pandas dataframe of scores.
    detail_level=0 : shift scores
    detail_level=1 : shift scores and difference measures
    detail_level=2 : shift scores and difference measures and scores to calculate differences
    """
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
    return pd.DataFrame(score_dicts).sort_index().rename_axis("ngram")

def get_simple_freqs(series, group1, group2):
    """Return raw word frequencies without any normalization.
    """
    # Split reports and explode to one ngram per row.
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    # Count n-gram frequencies.
    ngram2freq_1 = ngrams_1.value_counts().to_dict()
    ngram2freq_2 = ngrams_2.value_counts().to_dict()
    return ngram2freq_1, ngram2freq_2

def get_normed_freqs(series, group1, group2):
    """
    Return word frequencies normalized by user contributions.
    N-gram frequencies are counted within each user, and then
    divided by the amount of posts that user contributed.
    Then *those* are added across the corpus, instead of raw counts.
    """
    # Get post frequency per user, for each corpus.
    user2freq_1 = series.loc[group1].groupby("user_id").size()
    user2freq_2 = series.loc[group2].groupby("user_id").size()

    # Get n-gram frequencies per user, for each corpus.
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    userngrams2freq_1 = ngrams_1.groupby("user_id").value_counts()
    userngrams2freq_2 = ngrams_2.groupby("user_id").value_counts()

    # Normalize each n-gram frequency for each user, for each corpus.
    userngrams2norm_1 = userngrams2freq_1 / user2freq_1 # User's ngram frequency over document frequency.
    userngrams2norm_2 = userngrams2freq_2 / user2freq_2

    # Sum across normalized user/ngram frequencies, for each corpus.
    ngram2freq_1 = userngrams2norm_1.groupby(column_name).sum().to_dict()
    ngram2freq_2 = userngrams2norm_2.groupby(column_name).sum().to_dict()

    return ngram2freq_1, ngram2freq_2


################################################################################
# CALCULATE AND PLOT WORDSHIFTS
################################################################################

# NRC-fear shift comparing nightmares vs non-nightmares #
#########################################################

# Get frequencies.
nm_group1 = "nonnightmare"
nm_group2 = "nightmare"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(nm_ser, nm_group1, nm_group2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(nm_ser, nm_group1, nm_group2)

# Get shift scores.
shift = sh.WeightedAvgShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2,
    type2score_1="NRC-emotion_fear_English",
    normalization="variation", stop_lens=[(.3, .7)])

# Draw/export plot.
shift.get_shift_graph(
    top_n=top_n_plots,
    system_names=[nm_group1, nm_group2],
    detailed=True,
    show_plot=False,
    filename=export_path_fear_plot,
)
plt.close()

# Export scores.
out_df = shift2df(shift, detail_level=2)
out_df.to_csv(export_path_fear_table, index=True, na_rep="NA", sep="\t", encoding="utf-8")

# JSD shift comparing lucids vs non-lucids #
############################################

# Get frequencies.
ld_group1 = "nonlucid"
ld_group2 = "lucid"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(ld_ser, ld_group1, ld_group2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(ld_ser, ld_group1, ld_group2)

# Get shift scores.
shift = sh.JSDivergenceShift(
    type2freq_1=ngram2freq_1,
    type2freq_2=ngram2freq_2,
    weight_1=0.5,
    weight_2=0.5,
    base=2,
    normalization="variation",
    alpha=1,
    reference_value="average",
)

# Draw/export plot.
shift.get_shift_graph(
    top_n=top_n_plots,
    system_names=[ld_group1, ld_group2],
    detailed=True,
    show_plot=False,
    filename=export_path_jsd_plot,
)
plt.close()

# Export scores.
out_df = shift2df(shift, detail_level=2)
out_df.to_csv(export_path_jsd_table, index=True, na_rep="NA", sep="\t", encoding="utf-8")

# Proportion shift comparing lucids vs non-lucids #
###################################################

# Get shift scores.
shift = sh.ProportionShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2)

# Draw/export plot.
shift.get_shift_graph(
    top_n=top_n_plots,
    system_names=[ld_group1, ld_group2],
    detailed=False,
    show_plot=False,
    filename=export_path_prop_plot,
)
plt.close()

# Generate tables for top 1-gram and 2-gram differences.
prop_df = shift2df(shift, detail_level=1)

# Reduce to the top N differences.
# !! NOTE !!
#   For the proportion shifts, the final shift score is just the p_diff score is
#   just the p_diff but normalized. So just use p_diff, which could also come
#   from the JSD output, so this could happen in a few places.
grams1 = prop_df.loc[~prop_df.index.str.contains("_")]
grams2 = prop_df.loc[ prop_df.index.str.contains("_")]
## (add key=abs to sort_values to get highest absolute differences, not just higher in LD)
top_1grams = grams1["type2p_diff"].sort_values(ascending=False)[:top_n_tables]
top_2grams = grams2["type2p_diff"].sort_values(ascending=False)[:top_n_tables]

# Export.
top_1grams.to_csv(export_path_top1grams, index=True, sep="\t", encoding="utf-8")
top_2grams.to_csv(export_path_top2grams, index=True, sep="\t", encoding="utf-8")
