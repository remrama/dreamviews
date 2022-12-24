"""
Run custom LIWC analysis.

IMPORTS
=======
    - non-lemmatized posts, derivatives/dreamviews-posts.tsv
    - LIWC dictionary,      dictionaries/custom.dic
EXPORTS
=======
    - traditional (ie, total) LIWC scores for each dream report, derivatives/validate-liwc_scores.tsv
    IF WORD-LEVEL ANALYSIS
    - numpy arrays for frequency of each LIWC word and the corresponding post ID,
        derivatives/validate-liwc_wordscores-[data.npz] and [attr.npz]

The LIWC application/gui and its dictionaries are proprietary (https://liwc.net/).
But if you have the dictionaries, the application is just a word search
that counts up frequencies of those words. So if you have all words of
a dictionary, you can count the words yourself.
The benefit -- among many others -- have being able to count it
yourself is the freedom to run other analyses with it.

This script uses a custom dictionary file with Agency and Insight words
to get frequencies of *each word* in each category. So in the end there
is a total/traditional LIWC score (ie, frequency of all category words)
and also the ability to count individual word contributions.

You can just get traditional LIWC scores, which is faster and uses less
memory, or you can add the --words flag to also export a sparse matrix
holding individual word frequencies.

The one mystery is how the proprietary LIWC app tokenizes text when
running the word search. Here, the nltk TweetTokenizer is used bc
it doesn't break up words with apostrophes and stuff like that, which
is important for some LIWC categories. Then the liwc-python package
(https://github.com/chbrown/liwc-python) is used to count words.
This combo might be slightly different than "official" LIWC, but any
differences overall are likely minimal, and with such great benefits!
"""
import argparse
from collections import Counter

import liwc
import nltk
import numpy as np
import pandas as pd
from scipy import sparse
import tqdm

import config as c


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--words", action="store_true", help="Get individual word contributions. Slower, more memory, extra files.")
args = parser.parse_args()

GET_WORD_CONTRIBUTIONS = args.words

# Turn on pandas progress bar.
tqdm.tqdm.pandas(desc="word-level LIWCing" if GET_WORD_CONTRIBUTIONS else "LIWCing")

############################ I/O

# identify filenames
dict_fname = c.DATA_DIR / "dictionaries" / "custom.dic"
export_fname = c.DATA_DIR / "derivatives" / "validate-liwc_scores.tsv"
if GET_WORD_CONTRIBUTIONS:
    export_fname2 = c.DATA_DIR / "derivatives" / "validate-liwc_wordscores-data.npz"
    export_fname3 = c.DATA_DIR / "derivatives" / "validate-liwc_wordscores-attr.npz"

# load data
df = c.load_dreamviews_posts()
ser = df.set_index("post_id")["post_clean"]

# load LIWC parser, which takes a single token and finds all LIWC categories it's a part of
parse, category_names = liwc.load_token_parser(dict_fname)

if GET_WORD_CONTRIBUTIONS:
    # load the full LIWC lexicon, used later to catch asterisks (this whole thing is kinda tricky)
    lexicon, _ = liwc.read_dic(dict_fname)
    vocab = set(lexicon.keys()) # already unique but sets are faster to search through
    vocab_fulls = set([ t for t in vocab if not t.endswith("*") ])
    vocab_stems = set([ t.rstrip("*") for t in vocab if t.endswith("*") ])


############################ create an appropriate tokenizer for liwc

# LIWC vocab includes lots of apostrophed and hyphenated words, and emojis.
# The nltk tweet tokenizer is good for this situation, but I also wanna get rid of punctuation.
tknzr = nltk.tokenize.TweetTokenizer()
def tokenize4liwc(doc):
    # lowercase and break into tokens
    tokens = tknzr.tokenize(doc.lower())
    # remove isolated puncuation
    tokens = [ t for t in tokens if not (len(t)==1 and not t.isalpha()) ]
    return tokens


############################ run LIWC

# there's an easier way, without token/word frequencies and more concise code.
# so for now I'll leave that as a separate code chunk in case I want it later.
# but otherwise this could be more concise. weird.

if not GET_WORD_CONTRIBUTIONS: # Not using this but leaving it to show the much simpler case.
    
    def liwc_single_doc(doc):
        """Return, for a single document, total frequencies of each LIWC category
        """
        tokenized_doc = tokenize4liwc(doc)
        n_tokens = len(tokenized_doc)
        # get the counts for each category and divide them by the number of tokens/words in the document
        counts = Counter(category for token in tokenized_doc for category in parse(token))
        freqs = { category: n/n_tokens for category, n in counts.items() }
        return freqs

    res = ser.progress_apply(liwc_single_doc).apply(pd.Series)

    df = res.apply(pd.Series).fillna(0) # replace empty cells with zero, bc it means they had none of those words
    df = df[category_names] # reorder according to the LIWC dic file, just for cleanliness
    df = df.sort_index()    # also just bc it looks nice

    # export
    df.to_csv(export_fname, float_format="%.2f", index=True, sep="\t", encoding="utf-8")

else: # The more complex case of wanting individual word frequencies.

    def liwc_single_doc(doc):
        """Return, for a single document
        the total frequency of each LIWC category
        and also the total frequency for each unique word in the relevant LIWC corpus.

        Each doc gets 2 counters, one for category freq and one for token freq.
        The token freq functions like a general token counter but limits itself
        to words that are in the LIWC corpus (combined across all categories).
        """
        # initialize counters
        category_counts = Counter()
        token_counts = Counter()
        # tokenize document
        tokenized_doc = tokenize4liwc(doc)
        n_tokens = len(tokenized_doc)
        # loop over all tokens in the doc
        for token in tokenized_doc:
            # For each token/word, I wanna see if it's in a category,
            # but can't just add it if it is, because of the globbed (*) words.
            # Eg, zombie's, zombies, and zombie need to all be zombie* (if that's in a vocab).
            # So this does a hacky thing to see if the token contributes to a category
            # and then compare it with other words in the vocab.
            current_state = category_counts.copy()
            # update the category counter (just one token at a time, dumb)
            category_counts.update(parse(token))
            if current_state != category_counts: # True if the token is in *any* category, bc counter changed
                if token in vocab_fulls: # it's in normal vocab of non-globbed/stemmed words, just update
                    token_counts.update([token])
                # elif token in vocab_stems:
                else:# it's in the vocab of globbed/stem words,
                    # or it's not, like "recalled" isn't in any but still counts for "recall*".
                    # find the appropriate stem by continuously removing
                    # the end letter until it's found (probably stupid)
                    while token not in vocab_stems:
                        token = token[:-1]
                    token += "*" # put the asterisk back
                    token_counts.update([token])
                # else: # it's an extension not found
                #     raise Warning("Should never reach here.")
        # normalize counts so they are relative to total word/token count
        cat_freqs = { category: n/n_tokens for category, n in category_counts.items() }
        tok_freqs = { token: n/n_tokens for token, n in token_counts.items() }
        return cat_freqs, tok_freqs

    # run it
    res = ser.progress_apply(liwc_single_doc).apply(pd.Series)
    (_, cats), (_, toks) = res.items()

    # compile into useful objects
    cats = cats.apply(pd.Series).fillna(0).sort_index(axis=0)[category_names]
    toks = toks.apply(pd.Series).fillna(0).sort_index(axis=0).sort_index(axis=1)
    ### ^^^ this step is pretty rough on the tokens (wrt memory usage)

    # export the traditional LIWC results (ie, total category counts)
    ######## I think - for the tokens - this is too much memory used at once
    ######## and the file is big so use sparse matrix instead.
    cats.to_csv(export_fname, sep="\t", encoding="utf-8", index=True, float_format="%.2f")
    # toks.to_csv(export_fname_toks, sep="\t", encoding="utf-8", index=True, float_format="%.2f")

    # Export the word-level results.
    M = sparse.csr_matrix(toks.values)
    T = toks.columns.values
    P = toks.index.values
    sparse.save_npz(export_fname2, M, compressed=True)
    np.savez(export_fname3, token=T, post_id=P)
