"""
clean raw posts

lots going on here

Takes a long time bc of
language detection (~10 mins)
and lemmatization (~20 mins).

1. clean text
2. lemmatize text
3. remove some rows
4. extract tags/categories of each post
5. drop some columns

Yes, it would save a lot of time to remove
rows first, before cleaning and lemmatizing.
But some of the row removal is dependent on
those steps, and I'd rather have it all in one place.
"""
import os
import re
import ast
import tqdm
import calendar
import pandas as pd

import contractions
from unidecode import unidecode
import langdetect

import nltk

import config as c

tqdm.tqdm.pandas()


# make sure relevant nltk tools are present and downloaded
for x in ["corpora/wordnet", "corpora/stopwords", "taggers/averaged_perceptron_tagger"]:
    try:
        nltk.data.find(x)
    except LookupError:
        nltk.download(x)


import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-raw.tsv")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")


df = pd.read_csv(import_fname, index_col="post_id",
    sep="\t", encoding="utf-8",
    parse_dates=["timestamp"])



###################################################
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


###################################################
###############  lemmatization

# some setup
lemmatizer = nltk.stem.WordNetLemmatizer()
stops = nltk.corpus.stopwords.words("english")
def tag_nltk2wordnet(nltk_tag):
    if nltk_tag.startswith("J"):   return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith("V"): return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith("N"): return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith("R"): return nltk.corpus.wordnet.ADV
    else:                          return None

# this catches only words and hyphenated words
PATTERN = r"^[a-zA-Z]+\-?[a-zA-Z]*$"

def lemmatize_plus(doc):
    """does slightly more than lemmatize,
    just a few more restrictions
    ## 1. tokenize
    ## 2. get POS tags
    ## 3. remove unwanted POS (proper nouns)
    ## 4. lemmatization and lowercasing
    ## 5. remove stopwords and super short words

    # dates with a dash get labeled as nouns
    # ('09-02-2013', 'NN'), those with / and : are fine
    # so remove those by making sure all characters are alpha
    """
    # tokens_and_tags = nltk.tokenize.sent_tokenize(doc)
    tokens_and_tags = nltk.tag.pos_tag(nltk.tokenize.word_tokenize(doc))

    lemmas = []
    for token, nltk_tag in tokens_and_tags:

        wordnet_tag = tag_nltk2wordnet(nltk_tag)
        
        # keep words and hyphenated words
        if wordnet_tag and nltk_tag!="NNP" and re.match(PATTERN, token):
            # lemmatize and lowercase it
            # (returns the same string if no lemma)
            lemma = lemmatizer.lemmatize(token, wordnet_tag).lower()
            # check for stopwords
            if lemma not in stops and len(lemma)>=c.MIN_LEMMA_CHARS:
                lemmas.append(lemma)
    return " ".join(lemmas) if lemmas else pd.NA


df["post_lemmas"] = df["post_txt"].progress_apply(lemmatize_plus)

df["n_lemmas"] = df["post_lemmas"].str.split().str.len()


########################################
#####   remove some rows

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

# drop with too few lemmas
df = df[ df["n_lemmas"] >= c.MIN_LEMMA_COUNT ]

# # add a character count column too, even though not using it for restrictions
# df["n_chars"] = df.post_txt.str.len()

# restrict based on the number of dream reports
df = df.sort_values(["user_id", "timestamp"]) # should be redundant but it's critical
df["post_number"] = df.groupby("user_id")["timestamp"].transform(lambda s: range(1, 1+len(s)))
df = df[ df["post_number"] <= c.MAX_POST_COUNT ]

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
    "post_lemmas",
    "post_txt",
]

df = df[KEEP_COLUMNS]

df.to_csv(export_fname, sep="\t",
    encoding="utf-8", index=True)
