"""
save out 2 csvs
1 for top categories
1 for top tags
Categories are more useful bc they are constrained by the DreamViews site.
Tags are a free-for-all so lots of misspellings etc.
"""
import os
import ast
import pandas as pd
import config as c


import_fname1 = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
import_fname2 = os.path.join(c.DATA_DIR, "derivatives", "posts-raw.tsv")
# export filename changes over 2 loop iterations so see below

N_MIN_LABELS = 10
N_MIN_USERS = 10 # per label

# load in clean data for limited dataset
# but also need the raw file for tags and categories
df = pd.read_csv(import_fname1, sep="\t", encoding="utf-8", index_col="post_id")
ser = pd.read_csv(import_fname2, sep="\t", encoding="utf-8",
    usecols=["post_id", "tags", "categories"], index_col="post_id")

df = df.join(ser, how="left")

for col in ["tags", "categories"]:

    # tag and category columns are strings that look like lists.
    # convert them to actual lists
    df[col] = df[col].apply(ast.literal_eval)
    
    # generate a new axis name
    singular = "tag" if col == "tags" else "category"
    
    # explode each list so each row is one label
    labels = df.explode(col).dropna(subset=[col])

    # get a count for each label (tag or category)
    label_counts = labels[col].value_counts(
        ).rename("n_posts").rename_axis(singular)

    # get a count of how many unique users there are for each label
    user_counts = labels.groupby(col
        ).user_id.nunique(
        ).rename("n_users").rename_axis(singular)

    # merge into one dataframe that accounts for both total and user frequency
    res = pd.concat([label_counts, user_counts], axis=1, join="inner")

    # drop really low counts from the table entirely
    res = res[ res["n_posts"] >= N_MIN_LABELS ]
    res = res[ res["n_users"] >= N_MIN_LABELS ]

    # generate a weight thingy
    res["weight"] = res["n_posts"] * res["n_users"]/res.size
    res = res.sort_values("weight", ascending=False)

    # export
    export_fname = os.path.join(c.DATA_DIR, "results", f"describe-labels_{col}.tsv")
    res.to_csv(export_fname, sep="\t", encoding="utf-8",
        index=True, float_format="%.1f")