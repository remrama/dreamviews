"""
Classify lucid vs non-lucid labels.

Simple "Bag of Words" (BoW) classifier, not trying to optimize.
The whole thing is kinda tricky because there are multiple levels of imbalance.
    1. Imbalanced classes (more non-lucid than lucid).
    2. Imbalanced user contributions (some users have more than 1 dream).
Simple if non-optimal solution is to take one dream per user, then to class
imbalance, take a random subset of each class that's the size of the size of the
lowest class. Ugh. It just involve some resampling but I'm just not messing with
all this rn.

IMPORTS
=======
    - posts, dreamviews-posts.tsv
EXPORTS
=======
    - numpy file with predictions and labels, validate-classifier.npz
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import tqdm

import config as c


export_path = c.DATA_DIR / "derivatives" / "validate-classifier.npz"

column_name = "post_lemmas"
n_splits = 5
train_size = 0.7
nonlucid_digit = 0
lucid_digit = 1

# Initialize the classification pipeline components.
vectorizer = CountVectorizer(
    max_df=.75,
    min_df=100,
    ngram_range=(1,1),
    max_features=5000,
    binary=False,
)
clf = SVC(kernel="linear", C=1.)
cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=2)

# Load data.
df = c.load_dreamviews_posts()
usecols = ["post_id", "user_id", "lucidity", column_name]
df = df[usecols].set_index("post_id")
# Drop non-lucid data.
df = df[ df["lucidity"].str.contains("lucid") ]

# Make an effort to balance training by...
# ...downsampling to one dream per user
df = df.groupby("user_id").sample(n=1, replace=False, random_state=0)
# ...getting the minimum number of either class (LD or NLD)
n_per_class = df["lucidity"].value_counts().min()
# ...and downsampling both classes to this minimum amount.
df = df.groupby("lucidity").sample(n=n_per_class, replace=False, random_state=1)

# Convert to vectors for training/testing.
corpus = df[column_name].tolist()
X = vectorizer.fit_transform(corpus)
y = df["lucidity"].map({"nonlucid": nonlucid_digit, "lucid": lucid_digit}).values

# Cross-validation.
true_labels_list = []
pred_labels_list = []
for train_index, test_index in tqdm.tqdm(cv.split(X, y), total=n_splits, desc="DreamViews clf"):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    true_labels_list.append(y_test)
    pred_labels_list.append(y_pred)
cv_true_labels = np.row_stack(true_labels_list)
cv_pred_labels = np.row_stack(pred_labels_list)

# Export.
np.savez(export_path, true_labels=cv_true_labels, predicted_labels=cv_pred_labels)
