"""
"""
import os
import argparse
import pandas as pd
import config as c

import shifterator as sh

from nltk.corpus import stopwords

try:
    stops = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    stops = set(stopwords.words("english"))

TXT_COL = "post_lemmas"

parser = argparse.ArgumentParser()
parser.add_argument("--ngrams", type=int, default=0)
args = parser.parse_args()

N = args.ngrams

export_fname1 = os.path.join(c.DATA_DIR, "results", "validate-wordshift_entropy.png")
export_fname2 = os.path.join(c.DATA_DIR, "results", "validate-wordshift_jsd.png")
if N:
    export_fname1 = export_fname1.replace(".png", f"-{N}grams.png")
    export_fname2 = export_fname2.replace(".png", f"-{N}grams.png")

df, _ = c.load_dreamviews_data()
df = df[["user_id", "lucidity", TXT_COL]].set_index("lucidity")

# # reduce to only lucid and non-lucid
# df = df.loc[["lucid", "nonlucid"]]

# reduce user bias
# df = df.groupby("user_id").sample(n=1, replace=False)

ld_posts  = df.loc["lucid",    TXT_COL]
nld_posts = df.loc["nonlucid", TXT_COL]

if N:
    from collections import Counter
    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
    doc1 = nld_posts.str.lower().str.split().explode().tolist()
    doc2 = ld_posts.str.lower().str.split().explode().tolist()
    doc1 = [ w for w in doc1 if w not in stops ]
    doc2 = [ w for w in doc2 if w not in stops ]
    bigrams1 = map(lambda x: "-".join(x), find_ngrams(doc1, N))
    bigrams2 = map(lambda x: "-".join(x), find_ngrams(doc2, N))
    type2freq_1 = Counter(bigrams1)
    type2freq_2 = Counter(bigrams2)

else:
    # convert to frequencies for shifterator
    type2freq_1 = nld_posts.str.lower().str.split().explode().value_counts().to_dict()
    type2freq_2  = ld_posts.str.lower().str.split().explode().value_counts().to_dict()
    # remove stop words
    type2freq_1 = { k: v for k, v in type2freq_1.items() if k not in stops }
    type2freq_2 = { k: v for k, v in type2freq_2.items() if k not in stops }


shift = sh.EntropyShift(type2freq_1=type2freq_1, type2freq_2=type2freq_2,
    normalization="variation", alpha=1, reference_value="average")
shift.get_shift_graph(top_n=100, detailed=True, system_names=["non-lucid", "lucid"],
    show_plot=False, filename=export_fname1)

shift = sh.JSDivergenceShift(type2freq_1=type2freq_1, type2freq_2=type2freq_2,
    weight_1=0.5, weight_2=0.5, base=2,
    normalization="variation", alpha=1, reference_value=0)
shift.get_shift_graph(top_n=100, detailed=False, system_names=["non-lucid", "lucid"],
    show_plot=False, filename=export_fname2)
