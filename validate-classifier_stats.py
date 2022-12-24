"""Turn the classifier prediction output to a meaningful table of performance metrics.

IMPORTS
=======
    - numpy file with clf predictions and labels, derivatives/validate-classifier.npz
EXPORTS
=======
    - table of raw performance metrics at each cv fold,   derivatives/validate-classifier_cv.npz
    - table of performance metrics averaged across folds, derivatives/validate-classifier_avg.npz
"""
import numpy as np
import pandas as pd
from sklearn import metrics

import config as c


import_path = c.DATA_DIR / "derivatives" / "validate-classifier.npz"
export_path_cv = c.DATA_DIR / "derivatives" / "validate-classifier_cv.tsv"
export_path_cv_avg = c.DATA_DIR / "results" / "validate-classifier_avg.tsv"

# Load data.
data = np.load(import_path)
cv_true = data["true_labels"]
cv_pred = data["predicted_labels"]


# Get some classification performance measures for a table.
METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
results = { m: [] for m in METRICS }
for t, p in zip(cv_true, cv_pred):
    for m in METRICS:
        score = metrics.get_scorer(m)._score_func(t, p)
        results[m].append(score)
df = pd.DataFrame(results).rename_axis("cv")
df.index += 1

# Average over cross-validations for results table.
avg = df.agg(["mean","std"]).T.rename_axis("scorer")
avg.columns = avg.columns.map(lambda x: "CV "+x)

# Export
df.to_csv(export_path_cv, index=True, sep="\t", encoding="utf-8")
avg.to_csv(export_path_cv_avg, index=True, sep="\t", encoding="utf-8")
