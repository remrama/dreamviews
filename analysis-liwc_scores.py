"""
run liwc using not liwc
https://github.com/chbrown/liwc-python

Strings need to be tokenized and lowercased.
Details inside.
"""
import os
# import re
import tqdm
import argparse
import pandas as pd
import config as c

from collections import Counter
import liwc
import nltk

tqdm.tqdm.pandas()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dic", type=str, default="myliwc")
args = parser.parse_args()

DICTIONARY = args.dic
TXT_COL = "post_txt"

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc.tsv")
dict_fname = os.path.join(c.DATA_DIR, "dictionaries", f"{DICTIONARY}.dic")

ser = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["post_id", TXT_COL], index_col="post_id", squeeze=True)

# load the dictionary file
parse, category_names = liwc.load_token_parser(dict_fname)
# parse is a function from a token of text (a string) to a list of matching LIWC categories (a list of strings)
# category_names is all LIWC categories in the lexicon (a list of strings)

# LIWC vocab includes lots of apostrophed and hyphenated words
# and also some emoji things like :)
# So after tokenizing, I just want to get rid of solo punctuation
# like . and , and :
tknzr = nltk.tokenize.TweetTokenizer() # good for apostrophes and emojis
def tokenize4liwc(doc):
    # lowercase and break into tokens
    tokens = tknzr.tokenize(doc.lower())
    # remove isolated puncuation
    tokens = [ t for t in tokens if not (len(t)==1 and not t.isalpha()) ]
    return tokens

def liwc_single_doc(doc):
    tokenized_doc = tokenize4liwc(doc)
    n_tokens = len(tokenized_doc)
    counts = Counter(category for token in tokenized_doc for category in parse(token))
    freqs = { category: n/n_tokens for category, n in counts.items() }
    return freqs

res = ser.progress_apply(liwc_single_doc)

df = res.apply(pd.Series).fillna(0)
df = df[category_names] # reorder according to the LIWC dic file
df = df.sort_index() # just bc it looks nice

df.to_csv(export_fname, sep="\t", encoding="utf-8",
    index=True, float_format="%.2f")

# def tokenizer(doc):
#     for match in re.finditer(r"\w+", doc, re.UNICODE):
#         yield match.group(0)
    # # want to count the length of this iterator while also using the elements
    # n_tokens = 0
    # counts = Counter()
    # for token in tokenized_doc:
    #     n_tokens += 1
    #     categories = parse(token)
    #     counts.update(categories)
    # freqs = { category: n/n_tokens for category, n in counts.items() }
    # return freqs
    # # tokenized_doc = tokenizer(doc.lower())
    # # tokenized_doc = tokenizer(doc.lower())
    # # n_tokens = sum(1 for _ in tokenized_doc)
    # # counts = Counter(category for token in tokenized_doc for category in parse(token))

