"""
plot the classifier results and get some stats(?)
"""
import os
import numpy as np
import config as c

from sklearn import metrics

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"



import_fname = os.path.join(c.DATA_DIR, "derivatives", "validate-classifier.npz")
export_fname = os.path.join(c.DATA_DIR, "results", "validate-classifier.png")

data = np.load(import_fname)

cv_true = data["cv_true_labels"]
cv_pred = data["cv_pred_labels"]
permuted_true = data["permuted_true_labels"]
permuted_pred = data["permuted_pred_labels"]


# get all the roc curves and interpolate
# so they have the same amount of points

fpr_list = []
tpr_list = []
for fold_true, fold_pred in zip(cv_true, cv_pred):
    # acc = metrics.accuracy_score(fold_true, fold_pred)
    # f1 = metrics.f1_score(fold_true, fold_pred, average="binary")
    fpr, tpr, _ = metrics.roc_curve(fold_true, fold_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

interpolated_tprs = []
base_fpr = np.linspace(0, 1, 101)
for fpr, tpr in zip(fpr_list, tpr_list):
    interp_tpr = np.interp(base_fpr, fpr, tpr)
    interp_tpr[0] = 0
    interpolated_tprs.append(interp_tpr)

# get the mean and CI for plotting
true_positive_rates = np.row_stack(interpolated_tprs)

mean = np.mean(true_positive_rates, axis=0)
ci = np.quantile(true_positive_rates, [.025, .975], axis=0)


fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)

# plot the individual cv folds
for fpr, tpr in zip(fpr_list, tpr_list):
    ax.plot(fpr, tpr, color="g", lw=1, ls="-", alpha=.3)

# plot the aggregate
ax.fill_between(base_fpr, ci[0], ci[1], color="k", alpha=.1)
ax.plot(base_fpr, mean, color="k", lw=1, ls="-", alpha=.3)

# plot theoretical chance line
ax.plot([0,1], [0,1], color="k", lw=1, ls="--", alpha=1)

# aesthetics
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.tick_params(axis="both", direction="in", top=True, right=True)
ax.grid(color="gainsboro")
ax.xaxis.set(major_locator=plt.MultipleLocator(.2),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))
ax.yaxis.set(major_locator=plt.MultipleLocator(.2),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))


# export
plt.savefig(export_fname)
plt.close()
