"""Convert some tsv tables to latex for manuscript.

IMPORTS
=======
    - results file specified with argument, results/<basename>.tsv
EXPORTS
=======
    - same thing* but as latex table,       results/<basename>.tex

*There are some file-specific changes made, see below.
"""
import os
import argparse
import pandas as pd
import config as c


parser = argparse.ArgumentParser()
parser.add_argument("--basename", type=str, required=True)
args = parser.parse_args()

IMPORT_BASENAME = args.basename

import_fname = os.path.join(c.DATA_DIR, "results", f"{IMPORT_BASENAME}.tsv")
export_fname = os.path.join(c.DATA_DIR, "results", f"{IMPORT_BASENAME}.tex")

# load specific table
df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

# init defaults and just overwrite when desired
float_format = "%.2f"
index = False


# some case-specific manipulations
if IMPORT_BASENAME in ["describe-toptags", "describe-topcategories"]:
    df = df.head(10) # only want top of the table
elif "wordshift_proportion-ld" in IMPORT_BASENAME:
    df = df.head(30) # only want top of the table
    float_format = "%.4f" # extend bc small numbers
elif IMPORT_BASENAME == "describe-wordcount":
    #### some reshaping and selection here.
    # I want a table with two header rows (token type and lucidity)
    # and metrics going down the rows.
    df = df.rename(columns={"50%": "median"}
        ).replace("nonlucid", "non-lucid"
        ).set_index(["token_type", "lucidity"]
        ).rename_axis("metric", axis=1
        ).T.loc[["mean", "std", "min", "median", "max"],
                [("word","non-lucid"), ("word","lucid"), ("lemma","non-lucid"), ("lemma","lucid")]
        ]
    df.columns.names = [None, None]
    df.index.name = None
    index = True
    float_format = "%.0f" # reduce bc who cares


# export to a latex table file
df.to_latex(buf=export_fname,
    index=index, float_format=float_format,
    column_format="rrr", encoding="utf-8",
    # caption="", label="table:classification",
)