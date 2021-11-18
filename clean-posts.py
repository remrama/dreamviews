"""
clean raw posts

lots of magic happening here
Takes a long time (10 mins) just bc of language detection.
which only gets rid of around 200 posts, but still worth it imo

1. clean text
2. remove some rows
3. extract tags/categories of each post
4. drop some columns
"""
import os
import re
import ast
import tqdm
import calendar

import contractions
from unidecode import unidecode
import langdetect

import numpy as np
import pandas as pd

import config as c

tqdm.tqdm.pandas()


import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-raw.tsv")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")


df = pd.read_csv(import_fname, index_col="post_id",
    sep="\t", encoding="utf-8",
    parse_dates=["timestamp"])


################# clean text

# get rid of ampersands/slang/contractions/extraaaa letters
# for language models later
df["post_txt"] = df.post_txt.str.replace("&", "and", regex=False)
df["post_txt"] = df.post_txt.progress_apply(unidecode)
df["post_txt"] = df.post_txt.progress_apply(contractions.fix, slang=True)
# ^^^ contractions/unidecode also handled annoying apostrophes (’)

# replace any sequence of 4+ characters with 1 of that character
# gets rid of stuff like whoaaaaaaaaaaaa and --------------------
# will lead to some errors because it replaces with 1 letter but sometimes will need 2
df["post_txt"] = df.post_txt.str.replace(r"(.)\1{3,}", r"\1",
    flags=re.IGNORECASE, regex=True)

# replace urls and email addresses
df["post_txt"] = df.post_txt.str.replace(r"\S*@\S*\s?", "[email]", flags=re.IGNORECASE, regex=True)
df["post_txt"] = df.post_txt.str.replace(r"https?://\S+", "[url]", flags=re.IGNORECASE, regex=True)

# get rid of this thing that occurs on 20% of posts
# indicating it was updated at some point
updated_regex = r" Updated [0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M by [0-9]{5}"
df["post_txt"] = df.post_txt.str.replace(updated_regex, "", regex=True)



#### clean the dream report
### nvrmind not doing this stuff, it's not a huge problem don't mess with it
# REPLACEMENT_LIST = [
#     #r"\[(/?INDENT|/?RIGHT|/?CENTER|/?B)\]"
#     # these are the ones the don't have an = ever
#     r"\[/?(INDENT|RIGHT|CENTER|B|I|U|HR|IMG|LINK_TO_ANCHOR|SARCASM|DREAM LOGIC)\]",
#     # these need some leway as to what comes after because sometimes there's stuff there
#     r"\[/?COLOR.*\]",
#     r"\[/?SIZE.*\]",
#     r"\[/?FONT.*\]",
#     r"\[/?QUOTE.*\]",
#     r"\[/?SPOILER.*\]",
#     r"\[/?URL.*\]",
#     r"\[ATTACH=CONFIG\][0-9]*\[/ATTACH\]" # this is always JUST a number between so take it all out
# ]
# for compiler in REPLACEMENT_LIST:
#     dream_txt = re.sub(compiler, "", dream_txt, re.IGNORECASE)


# dream_txt = dream_txt.strip()



########################################
#####   remove some rows
# ya would save time to do this before cleaning text,
# but like it all here shut up

# drop reports from before the site was up (??)
df = df[ df["timestamp"] >= "2010-01-01" ]

# a few strange duplicated reports (<1%)
df = df.drop_duplicates(subset="post_txt", keep="first")

# # wrapper around language detection to prevent errors from not having alphacharacters
# def detect_language(doc):
#     if re.search(r"[a-zA-Z]", doc) is None:
#         return "nonalpha"
#     else:
#         return langdetect.detect(doc)
# don't save as a column, not necessary bc they will all be english
language = df.post_txt.progress_apply(langdetect.detect)
df = df[ language.eq("en") ]

# drop really short and really long reports
df["n_tokens"] = df["post_txt"].str.split().str.len()
df = df[ df["n_tokens"].between(c.MIN_TOKEN_COUNT, c.MAX_TOKEN_COUNT, inclusive="both") ]

# add a character count column too, even though not using it for restrictions
df["n_chars"] = df.post_txt.str.len()



########################################
#### break up tags and categories

# categories good, limited by dreamviews
# tags bad, not limited and user can put whatever

def code_lucidity(tags_str):
    # go from weird string to list of strings
    tags = ast.literal_eval(tags_str)
    if "lucid" in tags and "non-lucid" in tags:
        lucidity = "ambiguous"
    elif "lucid" in tags:
        lucidity = "lucid"
    elif "non-lucid" in tags:
        lucidity = "non-lucid"
    else:
        lucidity = "unspecified"
    return lucidity

def code_nightmare(tags_str):
    return "nightmare" in ast.literal_eval(tags_str)

df["lucidity"] = df.categories.apply(code_lucidity)
df["nightmare"] = df.categories.apply(code_nightmare)



KEEP_COLUMNS = [ # post_id is the index
    "user_id",
    "timestamp",
    "lucidity",
    "nightmare",
    "n_tokens",
    "n_chars",
    "post_txt",
]

df = df[KEEP_COLUMNS]

df.to_csv(export_fname, sep="\t",
    encoding="utf-8", index=True)
