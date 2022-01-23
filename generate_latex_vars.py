"""Generate a table to pass variables to LaTeX.

Comb through final results files,
extract the relevant values,
and apply manuscript-appropriate formatting.

Might need to include full strings, since
p-values are represented differently depending
on the p-values.
    - p-values use < if <.001 else =
    - p-values round to 2 decimals if > .05 else 3

Output file is a 2-column csv (no header) with
key, value pairs (var name, var value) in each row.
To make this useful in LaTeX, the variable names
have to be informative and useful (systematic).

Use underscore/dash setup like BIDS.

sec-<sectionLabel>_test-<testLabel>_stat-<statisticName>
Sec, test, and stat aren't always perfect,
but can be treated arbitrarily for 3 useful tags.

Section matches that of LaTeX file and is used to find file.
Test is just used to help identify on LaTeX end.
Stat is used to find value in csv here in Python.

e.g.,
    sec-liwc_test-wilcoxon_stat-p
    sec-liwc_test-wilcoxon_stat-w
    sec-liwc_test-wilcoxon_stat-n
    sec-liwc_test-word_stat-d
"""
import os
import csv
import numpy as np
import pandas as pd

import config as c

from decimal import Decimal, ROUND_HALF_UP


export_fname = os.path.join(c.DATA_DIR, "results", "variables4latex.csv")


def pysci_to_latexsci(x):
    """https://stackoverflow.com/a/13490601
    other good options to do it in latex
    https://tex.stackexchange.com/a/269849
    """
    assert "e-" in x
    return x.replace("e-", r"\text{e-}")

def format_float_sci(x, **kwargs):
    """Wrapper around numpy scientific float formatter to
    remove the periods when there is no decimal point.
    """
    s = np.format_float_scientific(x, **kwargs)
    if "precision" in kwargs:
        s = s.replace(".", "")
    # if "e-" in s:
    #     s = pysci_to_latexsci(s)
    return s

def round_int_half_up(x):
    """Only for integers to be rounded to nearest whole number.
    """
    d = Decimal(str(x)).quantize(Decimal("1"), ROUND_HALF_UP)
    s = "{0:f}".format(d)
    return s

def format_float(x, trim_leading=True, **kwargs):
    """Wrapper around numpy float formatter to trail leading zeros.
    https://stackoverflow.com/a/69118309
    """
    # set min_digits same as precision if being used, so trailing zeros show up after rounding
    if "precision" in kwargs:
        kwargs["min_digits"] = kwargs["precision"]
    s = np.format_float_positional(x, **kwargs)
    if trim_leading is True and int(x) == 0 and len(s) > 1:
        s = s.replace("0.", ".")
    return s





STUFF2GRAB = [ # basename, index_column, index_value, column

    ## LIWC category stats
    ("validate-liwc_scores-stats.tsv", "category", "insight", "p-val"),
    ("validate-liwc_scores-stats.tsv", "category", "insight", "W-val"),
    ("validate-liwc_scores-stats.tsv", "category", "insight", "CLES"),
    ("validate-liwc_scores-stats.tsv", "category", "insight", "n"),
    ("validate-liwc_scores-stats.tsv", "category", "agency", "p-val"),
    ("validate-liwc_scores-stats.tsv", "category", "agency", "W-val"),
    ("validate-liwc_scores-stats.tsv", "category", "agency", "CLES"),
    ("validate-liwc_scores-stats.tsv", "category", "agency", "n"),

    ## Classifier performance
    ("validate-classifier_avg.tsv", "scorer", "accuracy", "CV mean"),
    ("validate-classifier_avg.tsv", "scorer", "accuracy", "CV std"),
    ("validate-classifier_avg.tsv", "scorer", "f1", "CV mean"),
    ("validate-classifier_avg.tsv", "scorer", "f1", "CV std"),
    ("validate-classifier_avg.tsv", "scorer", "precision", "CV mean"),
    ("validate-classifier_avg.tsv", "scorer", "precision", "CV std"),
    ("validate-classifier_avg.tsv", "scorer", "recall", "CV mean"),
    ("validate-classifier_avg.tsv", "scorer", "recall", "CV std"),
    ("validate-classifier_avg.tsv", "scorer", "roc_auc", "CV mean"),
    ("validate-classifier_avg.tsv", "scorer", "roc_auc", "CV std"),

    ## Demographic reporting rates
    ("describe-demographics_provided.tsv", "demographic_variable", "gender", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "age", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "age", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "country", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "country", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+age", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+age", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+country", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+country", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "age+country", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "age+country", "pct_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+age+country", "n_reported"),
    ("describe-demographics_provided.tsv", "demographic_variable", "gender+age+country", "pct_reported"),

]


with open(export_fname, "w", newline="", encoding="utf-8") as csvfile:
    rowwriter = csv.writer(csvfile, delimiter=",")

    for basename, index_col, index_label, col_name in STUFF2GRAB:
        # print(basename, index_col, index_label, col_name)

        shortname = basename.split("-", 1)[1].split(".")[0].split("_", 1)[0]
        col_name_alpha = "".join([ x for x in col_name if x.isalpha() ])
        key = f"{shortname}-{index_label}-{col_name_alpha}"

        import_fname = os.path.join(c.DATA_DIR, "results", basename)

        df = pd.read_csv(import_fname, index_col=index_col, sep="\t")

        var = df.loc[index_label, col_name]

        if col_name == "p-val": # round appropriately

            if var <= .0001:
                var = format_float_sci(var, precision=0, exp_digits=1)
            elif var <= .001: # round to 4 decimal points and trim leading zero
                var = format_float(var, precision=4, trim_leading=True)
            elif var <= .05: # same as above but for 3 decimals
                var = format_float(var, precision=3, trim_leading=True)
            else:
                var = format_float(var, precision=2, trim_leading=True)

        elif col_name == "W-val":
            var = round_int_half_up(var)

        elif col_name == "CLES":
            var = format_float(var, precision=2, trim_leading=True)

        elif index_col == "scorer":
            # round before multiplying to avoid half-up stuff
            var = round(var, 2) * 100
            var = int(var) 

        row = [key, var]
        rowwriter.writerow(row)
