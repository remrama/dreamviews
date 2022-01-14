"""convert some tsv tables to latex for manuscript

copy a tsv from derivatives folder to
a tex file in results folder.
"""
import os
import argparse
import pandas as pd
import config as c


parser = argparse.ArgumentParser()
parser.add_argument("--basename", type=str, required=True)
args = parser.parse_args()

IMPORT_BASENAME = args.basename

import_fname = os.path.join(c.DATA_DIR, "derivatives", f"{IMPORT_BASENAME}.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", f"{IMPORT_BASENAME}.tex")

# load specific table
df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# some case-specific manipulations
if IMPORT_BASENAME in ["describe-toptags", "describe-topcategories"]:
    df = df.head(10) # only want top of the table

# elif IMPORT_BASENAME == "describe-wordcount":
#     df.columns.names = [None, None]
#     df.index.name = None
#     df.to_latex(buf=export_fname, index=True, encoding="utf-8",
#         float_format="%.0f")


# export to a latex table file
df.to_latex(buf=export_fname,
    index=False, encoding="utf-8",
    float_format="%.2f",
    column_format="rrr",
    # caption="", label="table:classification",
)