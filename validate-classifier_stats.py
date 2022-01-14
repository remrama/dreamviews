"""Turn the classifier prediction output to a meaningful table of performance metrics.

IMPORTS
=======
    - numpy file with clf predictions and labels, derivatives/validate-classifier.npz
EXPORTS
=======
    - table of raw performance metrics at each cv fold,   derivatives/validate-classifier_cv.npz
    - table of performance metrics averaged across folds, derivatives/validate-classifier_avg.npz
"""
import os
import numpy as np
import pandas as pd
import config as c

from sklearn import metrics


########## I/O

import_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier.npz")
export_fname_cv = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier_cv.tsv")
export_fname_cv_avg = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier_avg.tsv")

data = np.load(import_fname)
cv_true = data["true_labels"]
cv_pred = data["predicted_labels"]


########## get some classification performance measures for a table

METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
results = { m: [] for m in METRICS }
for t, p in zip(cv_true, cv_pred):
    for m in METRICS:
        score = metrics.get_scorer(m)._score_func(t, p)
        results[m].append(score)

df = pd.DataFrame(results).rename_axis("cv")
df.index += 1

# average over cross-validations for results table
avg = df.agg(["mean","std"]).T.rename_axis("scorer")
avg.columns = avg.columns.map(lambda x: "CV "+x)


# export
df.to_csv(export_fname_cv, index=True, sep="\t", encoding="utf-8")
avg.to_csv(export_fname_cv_avg, index=True, sep="\t", encoding="utf-8")
