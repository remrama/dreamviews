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
from sklearn.model_selection import ShuffleSplit


##### variable setup

TXT_COL = "post_lemmas" # raw txt or lemmas
N_PERMUTATIONS = 1000 # for shuffling
N_SPLITS = 5
NONLUCID_DIGIT = 0
LUCID_DIGIT = 1
TRAIN_SIZE = .7 # proportion of data


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
cv_true_labels_list = []
cv_pred_labels_list = []
cv = StratifiedShuffleSplit(n_splits=N_SPLITS,
    train_size=TRAIN_SIZE, random_state=2)
for train_index, test_index in tqdm.tqdm(cv.split(X, y), total=N_SPLITS, desc="clf cross-validation"):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_true_labels_list.append(y_test)
    cv_pred_labels_list.append(y_pred)
    # acc = metrics.accuracy_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred, average="binary", pos_label=LUCID_DIGIT)
cv_true_labels = np.row_stack(cv_true_labels_list)
cv_pred_labels = np.row_stack(cv_pred_labels_list)


############### shuffled version
permuted_true_labels_list = []
permuted_pred_labels_list = []
cv = ShuffleSplit(n_splits=N_PERMUTATIONS,
    train_size=TRAIN_SIZE, random_state=3)
for train_index, test_index in tqdm.tqdm(cv.split(X, y), total=N_PERMUTATIONS, desc="clf permutations"):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    np.random.shuffle(y_test)
    permuted_true_labels_list.append(y_test)
    permuted_pred_labels_list.append(y_pred)
    # acc = metrics.accuracy_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred, average="binary", pos_label=LUCID_DIGIT)
permuted_true_labels = np.row_stack(permuted_true_labels_list)
permuted_pred_labels = np.row_stack(permuted_pred_labels_list)



# export
np.savez(export_fname,
    cv_true_labels=cv_true_labels,
    cv_pred_labels=cv_pred_labels,
    permuted_true_labels=permuted_true_labels,
    permuted_pred_labels=permuted_pred_labels,
)



# # gonna create my own cross-validator because
# # stratification keeps the BALANCE ACROSS folds,
# # but it doesn't balance out the classes themselves within folds.
# for _ in range(N_SPLITS):
#     # pick 1200 random non-lucid dreams
#     # pick 1200 random lucid dreams
#     # pick 300  random non-lucid dreams
#     # pick 300  random lucid dreams
#     train_index = np.concatenate([
#         np.random.choice(np.where(y==LUCID_DIGIT)[0], size=TRAIN_SIZE, replace=True),
#         np.random.choice(np.where(y==NONLUCID_DIGIT)[0], size=TRAIN_SIZE, replace=True)
#     ])
#     # get an index of what is left to choose from for testing
#     # create a temporary array
#     y_masked = y.copy()
#     y_masked[train_index] = DUMMY_DIGIT
#     test_index = np.concatenate([
#         np.random.choice(np.where(y_masked==LUCID_DIGIT)[0], size=TEST_SIZE, replace=True),
#         np.random.choice(np.where(y_masked==NONLUCID_DIGIT)[0], size=TEST_SIZE, replace=True)
#     ])

#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     weight_train = df.iloc[train_index].groupby("user_id").user_id.transform("count").values
#     weight_test  = df.iloc[test_index].groupby("user_id").user_id.transform("count").values
#     weight_train = weight_train / y_train.size
#     weight_test  = weight_test / y_test.size
