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
parser.add_argument("-t", "--tokens", action="store_true",
    help="Get individual token/word contributions too. Way slower, more memory, extra file.")
args = parser.parse_args()

DICTIONARY = args.dic
TOKEN_FREQS = args.tokens

import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
dict_fname = os.path.join(c.DATA_DIR, "dictionaries", f"{DICTIONARY}.dic")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc.tsv")
if TOKEN_FREQS:
    export_fname2 = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc_tokens-data.npz")
    export_fname3 = os.path.join(c.DATA_DIR, "derivatives", "posts-liwc_tokens-attr.npz")

ser = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["post_id", "post_txt"], index_col="post_id", squeeze=True)

# load the dictionary file
parse, category_names = liwc.load_token_parser(dict_fname)
# parse is a function from a token of text (a string) to a list of matching LIWC categories (a list of strings)
# category_names is all LIWC categories in the lexicon (a list of strings)

# need the lexicon for the token counting stuff (to catch asterisks).
# this whole thing is kinda tricky
if TOKEN_FREQS:
    lexicon, _ = liwc.read_dic(dict_fname)
    # should all be unique anyways but sets are apparently faster to search in
    vocab = set(lexicon.keys())
    vocab_fulls = [ t for t in vocab if not t.endswith("*") ]
    vocab_stems = [ t.rstrip("*") for t in vocab if t.endswith("*") ]

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

# there's an easier way, without token/word frequencies and more concise code.
# so for now I'll leave that as a separate code chunk in case I want it later.
# but otherwise this could be more concise. weird.
if not TOKEN_FREQS: # simple

    def liwc_single_doc(doc):
        tokenized_doc = tokenize4liwc(doc)
        n_tokens = len(tokenized_doc)
        counts = Counter(category for token in tokenized_doc for category in parse(token))
        freqs = { category: n/n_tokens for category, n in counts.items() }
        return freqs

    res = ser.progress_apply(liwc_single_doc).apply(pd.Series)

    df = res.apply(pd.Series).fillna(0)
    df = df[category_names] # reorder according to the LIWC dic file
    df = df.sort_index() # just bc it looks nice

    df.to_csv(export_fname, sep="\t", encoding="utf-8",
        index=True, float_format="%.2f")

else:
    # each doc gets a counter for category frequency
    # and a counter for token contributions
    # (it's like a general token counter, but limited to words that matter for LIWC)
    # So it needs a more explicit loop instead of list comprehension, for both counters.
    def liwc_single_doc(doc):
        category_counts = Counter()
        token_counts = Counter()
        tokenized_doc = tokenize4liwc(doc)
        n_tokens = len(tokenized_doc)
        for token in tokenized_doc:
            current_state = category_counts.copy()
            category_counts.update(parse(token))
            if current_state != category_counts:
                # can't just add the token, because then the globbed
                # tokens will be different (eg, we need zombie's
                # and zombies and zombie to all be zombie* (if that's a token))
                if token in vocab_fulls:
                    token_counts.update([token])
                elif token in vocab_stems: # must be
                    # find the appropriate stem
                    # keep removing the end letter until it finds it
                    # (probably stupid)
                    while token not in vocab_stems:
                        token = token[:-1]
                    token += "*" # put the asterisk back
                    token_counts.update([token])
        cat_freqs = { category: n/n_tokens for category, n in category_counts.items() }
        tok_freqs = { token: n/n_tokens for token, n in token_counts.items() }
        return cat_freqs, tok_freqs

    res = ser.progress_apply(liwc_single_doc).apply(pd.Series)
    (_, cats), (_, toks) = res.items()

    cats = cats.apply(pd.Series).fillna(0).sort_index(axis=0)[category_names]
    toks = toks.apply(pd.Series).fillna(0).sort_index(axis=0).sort_index(axis=1)
    ### ^^^ this step is pretty rough on the tokens (wrt memory usage)

    ######## I think - for the tokens - this is too much memory used at once
    ######## and the file is big so use sparse matrix instead.
    cats.to_csv(export_fname, sep="\t", encoding="utf-8", index=True, float_format="%.2f")
    # toks.to_csv(export_fname_toks, sep="\t", encoding="utf-8", index=True, float_format="%.2f")

    import numpy as np
    from scipy import sparse
    M = sparse.csr_matrix(toks.values)
    T = toks.columns.values
    P = toks.index.values
    sparse.save_npz(export_fname2, M, compressed=True)
    np.savez(export_fname3, token=T, post_id=P)
