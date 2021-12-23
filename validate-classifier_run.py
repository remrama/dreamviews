"""
Can LDs and non-LDs -- AS LABELED BY USERS --
be distinguished based on word frequencies.

Simple "Bag of Words" (BoW) classifier.
The goal here is NOT to *optimize* the classifier,
but instead just to show they're different.

Kinda tricky because there are multiple levels of imbalance.
Imbalanced classes (more non-lucid than lucid).
Imbalanced user contributes (some users have more than 1 dream).

Simple if non-optimal solution is to take one dream per user
ONCE, without repeating the process.
Then to handle class imbalance, take a random subset of
non-lucids ONCE and again don't iterate over this sampling.

Save out cross-validated predictions and labels
to have freedom for scoring and plotting later.
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


##### variable setup

TXT_COL = "post_lemmas" # raw txt or lemmas
N_SPLITS = 5
TRAIN_SIZE = .7 # proportion of data
NONLUCID_DIGIT = 0
LUCID_DIGIT = 1


export_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier.npz")


usecols = ["post_id", "user_id", "lucidity", TXT_COL]
df, _ = c.load_dreamviews_data()
df = df[usecols].set_index("post_id")

# drop non-lucid data
df = df[ df["lucidity"].str.contains("lucid") ]


# find minimum number of LD and nLD dreams
# to keep training balanced

############ classification

vectorizer = CountVectorizer(
    max_df=.75, min_df=100,
    ngram_range=(1,1),
    max_features=5000,
    binary=False)

clf = SVC()

# downsample to one dream per user
df = df.groupby("user_id").sample(n=1, replace=False, random_state=0)

# find minimum number of either LD or NLD
n_per_class = df["lucidity"].value_counts().min()

# downsample for class
df = df.groupby("lucidity").sample(n=n_per_class, replace=False, random_state=1)

corpus = df[TXT_COL].tolist()
X = vectorizer.fit_transform(corpus)
y = df["lucidity"].map({"nonlucid":NONLUCID_DIGIT, "lucid":LUCID_DIGIT}).values


######### main cross-validation results for true accuracy
# manually loop to save predictions to plot the ROC curve
true_labels_list = []
pred_labels_list = []
cv = StratifiedShuffleSplit(n_splits=N_SPLITS, train_size=TRAIN_SIZE, random_state=2)
for train_index, test_index in tqdm.tqdm(cv.split(X, y), total=N_SPLITS, desc="clf cross-validation"):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_true_labels_list.append(y_test)
    cv_pred_labels_list.append(y_pred)
cv_true_labels = np.row_stack(cv_true_labels_list)
cv_pred_labels = np.row_stack(cv_pred_labels_list)


# export
np.savez(export_fname,
    true_labels=cv_true_labels,
    predicted_labels=cv_pred_labels,
)
