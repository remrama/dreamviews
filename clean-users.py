"""
Cleaning/reducing the raw user file.
Nothing really happens here, just reducing columns.
"""
import os
import pandas as pd
import config as c

import_fname = os.path.join(c.DATA_DIR, "derivatives", "users-raw.tsv")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "users-clean.tsv")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

KEEP_COLUMNS = [
    "gender",
    "age",
    "country",
]

df[KEEP_COLUMNS].to_csv(export_fname,
    sep="\t", encoding="utf-8",
    na_rep="NA", index=False)
