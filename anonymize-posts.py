"""
All data in the whole dataset, from the start, is anonymous.
But ethics in this situation are weird and for a variety
of reasons (described in paper) I am more comfortable
releasing this version with extra pre-cautionary steps
to make de-identification risk practically zero.

Reduce the dataset down to bare essentials,
disconnect from demographics,
dream reports are only lemmatized version,
which is already heavily preprocessed
but shuffle the words around anyways.
Include lucidity and nightmare labels
but not the full category lists.
"""
import os
import random
import config as c

random.seed(6)

export_fname = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-posts_superanon.tsv")

df = c.load_dreamviews_posts()

KEEP_COLUMNS = ["post_id", "user_id", "lucidity", "post_lemmas"]
df = df[KEEP_COLUMNS]

df["post_lemmas"] = df["post_lemmas"].str.split(
    ).apply(lambda x: random.sample(x, len(x))
    ).str.join(" ")

df.to_csv(export_fname, index=False, na_rep="NA", sep="\t", encoding="utf-8")
