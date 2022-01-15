"""Classify lucid vs non-lucid labels.

Simple "Bag of Words" (BoW) classifier, not trying to optimize.
The whole thing is kinda tricky because there are multiple levels of imbalance.
    1. Imbalanced classes (more non-lucid than lucid).
    2. Imbalanced user contributions (some users have more than 1 dream).
Simple if non-optimal solution is to take one dream per user, then
to class imbalance, take a random subset of each class that's the
size of the size of the lowest class. Ugh. It just involve some
resampling but I'm just not messing with all this rn.

IMPORTS
=======
    - posts, derivatives/dreamviews-posts.tsv
EXPORTS
=======
    - numpy file with predictions and labels, derivatives/validate-classifier.npz
"""
import os
import tqdm
import numpy as np
import pandas as pd
import config as c

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate


############################ set up classification stuff

# define important variables
TXT_COL = "post_lemmas" # raw txt or lemmas
N_SPLITS = 5
TRAIN_SIZE = .7 # proportion of data
NONLUCID_DIGIT = 0
LUCID_DIGIT = 1

# initialize the classification pipeline components
vectorizer = CountVectorizer(max_df=.75, min_df=100,
    ngram_range=(1,1), max_features=5000, binary=False)
clf = SVC(kernel="linear", C=1.)
cv = StratifiedShuffleSplit(n_splits=N_SPLITS, train_size=TRAIN_SIZE, random_state=2)



############################ I/O

export_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier.npz")

df = c.load_dreamviews_posts()
usecols = ["post_id", "user_id", "lucidity", TXT_COL]
df = df[usecols].set_index("post_id")
# drop non-lucid data
df = df[ df["lucidity"].str.contains("lucid") ]


# Make an effort to balance training by...
# ...downsampling to one dream per user
df = df.groupby("user_id").sample(n=1, replace=False, random_state=0)
# ...getting the minimum number of either class (LD or NLD)
n_per_class = df["lucidity"].value_counts().min()
# ...and downsampling both classes to this minimum amount.
df = df.groupby("lucidity").sample(n=n_per_class, replace=False, random_state=1)



############################ classification

# convert to vectors for training/testing in sklearn
corpus = df[TXT_COL].tolist()
X = vectorizer.fit_transform(corpus)
y = df["lucidity"].map({"nonlucid":NONLUCID_DIGIT, "lucid":LUCID_DIGIT}).values

# run cross-validation
true_labels_list, pred_labels_list = [], []
for train_index, test_index in tqdm.tqdm(cv.split(X, y), total=N_SPLITS, desc="lucidity clf cross-validation"):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    true_labels_list.append(y_test)
    pred_labels_list.append(y_pred)
cv_true_labels = np.row_stack(true_labels_list)
cv_pred_labels = np.row_stack(pred_labels_list)


# export
np.savez(export_fname, true_labels=cv_true_labels, predicted_labels=cv_pred_labels)
