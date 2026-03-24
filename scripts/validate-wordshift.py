"""
Use wordshift scores (and default visualizations) to separate texts.

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
    - raw JSD shift scores for lucidity,          validate-wordshift_jsd.tsv
    - default JSD shift plot for lucidity,        validate-wordshift_jsd_src.png
    - raw NRC-fear shift scores for nightmares,   validate-wordshift_fear.tsv
    - default NRC-fear shift plot for nightmares, validate-wordshift_fear_src.png
    - default proportion shift plot for lucidity, validate-wordshift.tsv
    - table of top 1-grams higher in LDs,         validate-wordshift_top1grams.tsv
    - table of top 2-grams higher in LDs,         validate-wordshift_top2grams.tsv
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import shifterator as sh
from gensim.models.phrases import Phraser, Phrases
from tqdm import tqdm

import config as c

parser = argparse.ArgumentParser()
parser.add_argument(
    "--nobigrams",
    action="store_true",
    help="Don't include bigrams in the wordshift, just for development.",
)
parser.add_argument(
    "--nonorm",
    action="store_true",
    help="Don't account for biased contributions, just for development.",
)
args = parser.parse_args()

NO_BIGRAMS = args.nobigrams
NO_NORMING = args.nonorm

########################################################################################
# SETUP
########################################################################################

COLUMN_NAME = "post_lemmas"
TOP_N_PLOTS = 50
TOP_N_TABLES = 100
FLOAT_FORMAT = "%.5e"

# Choose export locations
EXPORT_STEM = "validate-wordshift"

export_stem_jsd = f"{EXPORT_STEM}_jsd"
export_stem_fear = f"{EXPORT_STEM}_fear"
export_stem_top1grams = f"{EXPORT_STEM}_ld1grams"
export_stem_top2grams = f"{EXPORT_STEM}_ld2grams"

# Need to export these with full paths for the shifterator plotting function
export_path_prop_plot = (c.figures_dir / f"{EXPORT_STEM}_src").with_suffix(".png")
export_path_jsd_plot = (c.figures_dir / f"{export_stem_jsd}_src").with_suffix(".png")
export_path_fear_plot = (c.figures_dir / f"{export_stem_fear}_src").with_suffix(".png")

# Load data
df = c.load_dreamviews_posts(lemmas=True)

########################################################################################
# CONNECT BIGRAMS
########################################################################################

# # remove stopwords from text column
# replace_regex = r"(?<=\b)(" + r"|".join(stops) + r")(?=\b)"
# df[TXT_COL] = df[TXT_COL].replace(replace_regex, "", regex=True).str.strip()

if not NO_BIGRAMS:  # Sorry for the double negative
    # build bigram model and convert text column to bigrams
    min_count = 1  # ignore all words and bigrams with total collected count lower than this value
    threshold = 1
    delim = "_"  # for joining bigrams
    scoring = "default"
    sentences = df[COLUMN_NAME].str.lower().str.split().tolist()
    # connector_words=ENGLISH_CONNECTOR_WORDS
    phrase_model = Phrases(
        sentences, delimiter=delim, min_count=min_count, threshold=threshold, scoring="default"
    )
    # phrase_model = Phrases(phrase_model[sentences], delimiter=delim, min_count=3, threshold=threshold, scoring=scoring)  # noqa: E501
    phrase_model = Phraser(phrase_model)  # memory benefits?

    def _unigram2ngram(x):
        return phrase_model[x]

    tqdm.pandas(desc="Mixing bigrams into corpus")
    df[COLUMN_NAME] = (
        df[COLUMN_NAME].str.lower().str.split().progress_apply(_unigram2ngram).str.join(" ")
    )
# def find_ngrams(input_list, n):
#     return zip(*[input_list[i:] for i in range(n)])
# def ngrams(x, n):
#     doc = x.split()
#     return " ".join(map(lambda x: "-".join(x), find_ngrams(doc, n)))

# reduce user bias
# df = df.groupby("user_id").sample(n=1, replace=False)

########################################################################################
# EXTRACT DATA SUBSETS
########################################################################################

# Extract lucidity Series for JSD and proportion shifts
ld_ser = df.query("lucidity.str.contains('lucid')", engine="python").set_index(
    ["lucidity", "user_id"]
)[COLUMN_NAME]

# Export nightmare Series for NRC-fear shift
df["nightmare"] = df["nightmare"].map({True: "nightmare", False: "nonnightmare"})
nm_ser = df.set_index(["nightmare", "user_id"])[COLUMN_NAME]

########################################################################################
# NORMALIZATION FUNCTIONS
########################################################################################


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
        score_dicts = {n: s for n, s in zip(shift_score_names, shift_scores, strict=True)}
    elif detail_level == 1:
        shift_scores = shift.get_shift_scores(details=True)
        shift_score_names = [
            "type2p_diff",
            "type2s_diff",
            "type2p_avg",
            "type2s_ref_diff",
            "type2shift_score",
        ]
        score_dicts = {n: s for n, s in zip(shift_score_names, shift_scores, strict=True)}
    elif detail_level == 2:
        score_dicts = {
            k: shift.__getattribute__(k) for k in shift.__dict__ if k.startswith("type2")
        }
    return pd.DataFrame(score_dicts).sort_index().rename_axis("ngram")


def get_simple_freqs(series, group1, group2):
    """Return raw word frequencies without any normalization."""
    # Split reports and explode to one ngram per row
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    # Count n-gram frequencies
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
    # Get post frequency per user, for each corpus
    user2freq_1 = series.loc[group1].groupby("user_id").size()
    user2freq_2 = series.loc[group2].groupby("user_id").size()

    # Get n-gram frequencies per user, for each corpus
    ngram_ser = series.str.lower().str.split().explode()
    ngrams_1 = ngram_ser.loc[group1]
    ngrams_2 = ngram_ser.loc[group2]
    userngrams2freq_1 = ngrams_1.groupby("user_id").value_counts()
    userngrams2freq_2 = ngrams_2.groupby("user_id").value_counts()

    # Normalize each n-gram frequency for each user, for each corpus
    userngrams2norm_1 = (
        userngrams2freq_1 / user2freq_1
    )  # User's ngram frequency over document frequency
    userngrams2norm_2 = userngrams2freq_2 / user2freq_2

    # Sum across normalized user/ngram frequencies, for each corpus
    ngram2freq_1 = userngrams2norm_1.groupby(COLUMN_NAME).sum().to_dict()
    ngram2freq_2 = userngrams2norm_2.groupby(COLUMN_NAME).sum().to_dict()

    return ngram2freq_1, ngram2freq_2


########################################################################################
# CALCULATE AND PLOT WORDSHIFTS
########################################################################################

# NRC-fear shift comparing nightmares vs non-nightmares #
#########################################################

# Get frequencies
NM_GROUP1 = "nonnightmare"
NM_GROUP2 = "nightmare"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(nm_ser, NM_GROUP1, NM_GROUP2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(nm_ser, NM_GROUP1, NM_GROUP2)

# Get shift scores
shift = sh.WeightedAvgShift(
    type2freq_1=ngram2freq_1,
    type2freq_2=ngram2freq_2,
    type2score_1="NRC-emotion_fear_English",
    normalization="variation",
    stop_lens=[c.NIGHTMARE_SHIFT_STOPS],
)

# Draw/export plot
shift.get_shift_graph(
    top_n=TOP_N_PLOTS,
    system_names=[NM_GROUP1, NM_GROUP2],
    detailed=True,
    show_plot=False,
    filename=export_path_fear_plot,
)
plt.close()

# Export scores
out_df = shift2df(shift, detail_level=2)
c.export_table(out_df, export_stem_fear, float_format=FLOAT_FORMAT)

# JSD shift comparing lucids vs non-lucids #
############################################

# Get frequencies
LD_GROUP1 = "nonlucid"
LD_GROUP2 = "lucid"
if NO_NORMING:
    ngram2freq_1, ngram2freq_2 = get_simple_freqs(ld_ser, LD_GROUP1, LD_GROUP2)
else:
    ngram2freq_1, ngram2freq_2 = get_normed_freqs(ld_ser, LD_GROUP1, LD_GROUP2)

# Get shift scores
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

# Draw/export plot
shift.get_shift_graph(
    top_n=TOP_N_PLOTS,
    system_names=[LD_GROUP1, LD_GROUP2],
    detailed=True,
    show_plot=False,
    filename=export_path_jsd_plot,
)
plt.close()

# Export scores
out_df = shift2df(shift, detail_level=2)
c.export_table(out_df, export_stem_jsd, float_format=FLOAT_FORMAT)

# Proportion shift comparing lucids vs non-lucids #
###################################################

# Get shift scores
shift = sh.ProportionShift(type2freq_1=ngram2freq_1, type2freq_2=ngram2freq_2)

# Draw/export plot
shift.get_shift_graph(
    top_n=TOP_N_PLOTS,
    system_names=[LD_GROUP1, LD_GROUP2],
    detailed=False,
    show_plot=False,
    filename=export_path_prop_plot,
)
plt.close()

# Generate tables for top 1-gram and 2-gram differences
prop_df = shift2df(shift, detail_level=1)

# Reduce to the top N differences
# !! NOTE !!
#   For the proportion shifts, the final shift score is just the p_diff score is
#   just the p_diff but normalized. So just use p_diff, which could also come
#   from the JSD output, so this could happen in a few places
grams1 = prop_df.loc[~prop_df.index.str.contains("_")]
grams2 = prop_df.loc[prop_df.index.str.contains("_")]
## (add key=abs to sort_values to get highest absolute differences, not just higher in LD)
top_1grams = grams1["type2p_diff"].sort_values(ascending=False)[:TOP_N_TABLES]
top_2grams = grams2["type2p_diff"].sort_values(ascending=False)[:TOP_N_TABLES]

# Export
c.export_table(top_1grams, export_stem_top1grams, float_format=FLOAT_FORMAT)
c.export_table(top_2grams, export_stem_top2grams, float_format=FLOAT_FORMAT)
