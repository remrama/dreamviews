"""Turn the classifier prediction output to a meaningful table of performance metrics.

IMPORTS
=======
    - numpy file with clf predictions and labels, validate-classifier.npz
EXPORTS
=======
    - table of raw performance metrics at each cv fold,   validate-classifier_cv.npz
    - table of performance metrics averaged across folds, validate-classifier_avg.npz
"""

import numpy as np
import pandas as pd
from sklearn import metrics

import config as c

import_path = c.derivatives_dir / "validate-classifier.npz"
EXPORT_STEM = "validate-classifier"

# Load data
data = np.load(import_path)
cv_true = data["true_labels"]
cv_pred = data["predicted_labels"]

# Get some classification performance measures for a table
METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
results = {m: [] for m in METRICS}
for t, p in zip(cv_true, cv_pred, strict=True):
    for m in METRICS:
        score = metrics.get_scorer(m)._score_func(t, p)
        results[m].append(score)
df = pd.DataFrame(results).rename_axis("cv")
df.index += 1

# Average over cross-validations for results table
avg = df.agg(["mean", "std"]).T.rename_axis("scorer")
avg.columns = avg.columns.map(lambda x: "cv_" + x)

# Export
export_stem_cv = f"{EXPORT_STEM}_cv"
export_stem_avg = f"{EXPORT_STEM}_avg"
c.export_table(df, export_stem_cv)
c.export_table(avg, export_stem_avg)
