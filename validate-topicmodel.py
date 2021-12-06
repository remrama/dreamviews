"""plot topic model output/topics
"""
import os
import pandas as pd
import config as c

import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


import_fname = os.path.join(c.DATA_DIR, "results", "validate-topicmodel.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", "validate-topicmodel.png")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

n_topics = df["topic_number"].nunique()
n_words = df.groupby("topic_number").size().unique()[0] # per topic

FIGSIZE = (3, 7)

BAR_ARGS = dict(width=.8, color=c.COLORS["lucid"],
    edgecolor="black", linewidth=.5, alpha=1)

fig, axes = plt.subplots(nrows=n_topics, figsize=FIGSIZE,
    sharex=False, sharey=False, constrained_layout=True)

xvals = range(n_words)

for i, ax in enumerate(axes):
    topic_num = i + 1
    topic_df = df.query(f"topic_number=={topic_num}")
    words = topic_df["word"].tolist()
    weights = topic_df["weight"].tolist()
    ax.bar(xvals, weights, **BAR_ARGS)

    ax.set_xticks(xvals)
    ax.set_xticklabels(words, fontsize=6,
        rotation=33, ha="right")
    ax.set_ylabel("weight")
    ax.yaxis.set(major_locator=plt.MultipleLocator(.01),
                 minor_locator=plt.MultipleLocator(.005),
                 major_formatter=plt.FuncFormatter(c.no_leading_zeros))
    ax.yaxis.grid(True, which="major", linewidth=1, clip_on=False, color="gainsboro")
    ax.yaxis.grid(True, which="minor", linewidth=.3, clip_on=False, color="gainsboro")
    ax.set_axisbelow(True)


plt.savefig(export_fname)
plt.close()
