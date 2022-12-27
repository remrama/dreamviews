"""Generate tables of labels (ie, categories and tags) ordered by frequency.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - ordered table of top categories, results/describe-topcategories.tsv
    - ordered table of top tags,       results/describe-toptags.tsv
"""
import pandas as pd

import config as c


n_min_labels = 10  # Only keep labels that show up >= <x> times ...
n_min_users = 10  # ... and across >= <y> unique users.

# Load data.
df = c.load_dreamviews_posts()

for col in ["tags", "categories"]:

    # Tag and category columns are strings with double colons separating labels.
    # Convert them to lists.
    df[col] = df[col].str.split("::")
    
    # Generate a new axis name.
    singular = "tag" if col == "tags" else "category"
    
    # Explode each list so each row is one label.
    labels = df.explode(col).dropna(subset=[col])

    # Get a count for each label (tag or category).
    label_counts = labels[col].value_counts(
        ).rename("n_posts").rename_axis(singular)

    # Get a count of how many unique users there are for each label.
    user_counts = labels.groupby(col
        ).user_id.nunique(
        ).rename("n_users").rename_axis(singular)

    # Merge into one dataframe that accounts for both total and user frequency.
    res = pd.concat([label_counts, user_counts], axis=1, join="inner")

    # Drop really low counts from the table entirely.
    res = res[ res["n_posts"] >= n_min_labels ]
    res = res[ res["n_users"] >= n_min_users ]

    res = res.sort_values("n_posts", ascending=False)
    res.index = res.index.str.replace("_", " ")

    # export
    export_path = c.DATA_DIR / "results" / f"describe-top{col}.tsv"
    res.to_csv(export_path, index=True, sep="\t", encoding="utf-8")
