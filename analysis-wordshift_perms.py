"""
Get the difference in word frequencies
between lucid and non-lucid dreams.

Use a resampling procedure too.
The resampling procedure is a bit convoluted
but it guarantess there is only 1 post per
user and there is an equal amount of LD and nonLD.

Shifterator has a lot of docs,
I found this one most useful for extracting stuff:
Accessing Word Contribution Components
https://github.com/ryanjgallagher/shifterator/blob/5fb9f6c54c9b995848244676deaaa043578237bb/docs/cookbook/weighted_avg_shifts.rst

The difference in relative frequencies (uparrow / downarrow) is given in the attribute type2p_diff
The difference in scores (bigtriangleup / bigtriangledown) is given in type2s_diff
The difference between the average score and the reference score (+ / -) is given in type2s_ref_diff
The average relative frequency is given in type2p_avg
The overall word shift contributions are stored in type2shift_score
shift.types             # the whole vocab
shift.type2freq_1       ## these are mostly the same as freq input,
shift.type2freq_2       ##  just in some cases it might merge vocabs
shift.type2p_avg
shift.type2p_diff       # main proportion shift of interest (can use type2freq_* to get this)
shift.type2s_diff       # will be all zeros if same scores for each corpus
shift.type2s_ref_diff   # main score relative to reference of interest
shift.type2score_1      ## these will be same mostly
shift.type2score_2      ##  (and u can get type2s_diff from these)
shift.type2shift_score  # normalized frequency times ref_diff
"""
import os
import tqdm
import pandas as pd
import config as c

import shifterator as sh


N_ITERATIONS = 100
TXT_COL = "post_lemmas"

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "validate-wordshift_perms.tsv")
df = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["user_id", "lucidity", TXT_COL], index_col="lucidity")


# reduce to only lucid and non-lucid
df = df.loc[["lucid", "non-lucid"]]


# initialize a list to hold all the results
df_list = []


for i in tqdm.trange(N_ITERATIONS, desc="wordshift resampling"):

    # sample one dream per person
    df_sample = df.groupby("user_id").sample(1, replace=False)

    # the number of lucid and non-lucid dreams will differ
    # so find the minimum amount and resample with replacement equally
    resample_size = df_sample.index.value_counts().min()

    # resample with replacement for each of LD and nonLD
    ld_posts  = df_sample.loc["lucid",     TXT_COL].sample(resample_size, replace=True)
    nld_posts = df_sample.loc["non-lucid", TXT_COL].sample(resample_size, replace=True)

    # convert to frequencies for shifterator
    ld_freqs  = ld_posts.str.lower().str.split().explode().value_counts().to_dict()
    nld_freqs = nld_posts.str.lower().str.split().explode().value_counts().to_dict()

    # get differences
    shift = sh.ProportionShift(type2freq_1=nld_freqs, type2freq_2=ld_freqs)
    # shift = sh.JSDivergenceShift(type2freq_1=nld_freqs, type2freq_2=ld_freqs,
    #     weight_1=0.5, weight_2=0.5, base=2, alpha=1)

    # extract some relevant measures from results
    metric_dicts = { attr: shift.__getattribute__(attr)
        for attr in shift.__dict__.keys() if attr.startswith("type2") }
    # type_scores = [ (k,v) for k, v in shift.type2shift_score.items() ]
    
    # convert to a dataframe and save to results list
    res = pd.DataFrame(metric_dicts)
    res.columns = res.columns.map(lambda c: f"{c}-{i+1}")
    df_list.append(res)


# reduce to only words that showed up in every resampling
results_wide = pd.concat(df_list, axis=1, join="inner"
    ).rename_axis("token").reset_index(drop=False)

results = pd.wide_to_long(results_wide,
        stubnames=metric_dicts.keys(),
        i="token", j="iteration", sep="-"
    ).swaplevel(
    ).sort_index()


results.to_csv(export_fname, index=True, na_rep="NA", sep="\t", encoding="utf-8")
