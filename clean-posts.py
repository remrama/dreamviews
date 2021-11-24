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
UPDATED_PATTERN = r" Updated [0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M by [0-9]{5}"
df["post_txt"] = df.post_txt.str.replace(UPDATED_PATTERN, "", regex=True)


# ################# regex section
# ## I do some simpler regex stuff above,
# ## but here is the more questionable parts.
# ## Also yes it could be more efficient by searching
# ## for things at once, but I like it this way as
# ## it makes commenting/describing easiert and
# ## I'm not worried about efficiency.
# ##
# ## From manual inspection of posts, there are some
# ## very common patterns, mostly at the beginning of
# ## posts, that should be taken out. Some of these are
# ## a bit risky, but I think worth it. Most of them
# ## are only present at an <1% rate, up to 5%.
# ## Moreover, these are likely to impact results
# ## that properly account for repeated measurements
# ## within users, since they are specific to users.
# ##
# ## examples of shit
# ## (Non-lucid) NON-DREAM DREAM LUCID Spoiler for 18+:
# ## 7:20
# ## I took 100 Mg b6 and 10 Mg taurine prebed and fell asleep around 10pm At 12:13am
# ## Lucid Dream 197: Yoshi's House of Horrors Series: Mind of a Gamer, Episode 1 around 2:00pm
# ## Date:14th of june. Total sleep: 9 hours. Daytime Techniques: RC's, mantras. Lucid Techniques: Stabilizisation, ''increase vividness'' mantra. Recall Techniques: DJ, mantras, tags. Fell Asleep: 00.45 Dream Title: Satan teacher/shounen fight. Dream: Satan teacher;
# ## (Non-lucid) NON-DREAM DREAM LUCID
# ## Non-lucid Lucid I did as i always do, i DILD'ed.
# ## WBTB 8:10 - 8:30 am, GM4mg+Choline300mgs 8:27am WILD
# ## June 25-26, 2010.
# ## Supplements Taken: Lucidimine after 6 hours of sleep 1.
# ## Non-Dream Dream I recalled fragments from 4 dreams today (Ugh! - wish I would start remembering details of one dream instead of fragments of 4!). Dream 1:
# ## Aug 16 61/An old
# ## dream 1 -
# ## March, 27, 2012 1:15 am - , March , 27, 2012 10:40 am

# # make a list of patterns that will be search simultaneously
# PATTERN_LIST = [
#     r"had",
#     r"(long|I)",
# ]

# regex = re.compile("|".join(PATTERN_LIST))
# # regex = re.compile("|".join(map(re.escape, PATTERN_LIST)))

# ## first and probably most controversial, I'm gonna try
# ## and take out multiple dreams. This is really weird.
# # It's most commonly denoted with "Dream 2", ~4%, and <1% with "dream 2")
# # r"(D|d)ream 2.*"
# # r"Dream #?2.*"
# # r"Dream Two.*"
# # r"\s2\s.*" # also wipes out anything with the # 2 in it
# # r"\s2\W\s.*"

# # **note when iterating these, the string changes each time
# # these could ignore case
# REGEX_PATTERN_LIST = [
    
#     ### first get rid of lots of html stuff that doesn't get caught earlier
#     # r"\[(/?INDENT|/?RIGHT|/?CENTER|/?B)\]",
#     # these ones never have an = preceding them
#     r"\[/?(INDENT|RIGHT|CENTER|B|I|U|HR|IMG|LINK_TO_ANCHOR|SARCASM|DREAM LOGIC)\]",
#     # these need some leway as to what comes after because sometimes there's stuff there
#     r"\[/?COLOR.*?\]",
#     r"\[/?SIZE.*?\]",
#     r"\[/?FONT.*?\]",
#     r"\[/?QUOTE.*?\]",
#     r"\[/?SPOILER.*?\]",
#     r"\[/?URL.*?\]",
#     r"\[ATTACH=CONFIG\][0-9]*\[/ATTACH\]" # this is always JUST a number between so take it all out

#     # date stuff
#     # days of week (~5%)
#     r"\b(" + r"|".join(calendar.day_name) + r")\b",
#     r"[0-9]{1,2}(th|nd|rd|st)",
#     # finds 07.11.2010 or 8/11/18 or 8-11-2001
#     r"[0-9]{1,4}(-|/|\.)[0-9]{1,4}(-|/|\.)[0-9]{1,4}",
#     r"({months}),? [0-9]{{1,2}}(th|nd|rd|st)?, [0-9]{{4}}\.?".format(
#         months="|".join(calendar.month_name[1:]+calendar.month_abbr[1:])),
#     r"\b(" + r"|".join(calendar.month_name[1:]+calendar.month_abbr[1:]) + r")\b",
#     # times likes 3:32 AM or 12:12 or 6:30am or 10pm (~70%, but again not after the "Updated" replacement)
#     r"[0-9]{1,2}(:[0-9]{2})?\s?((A|P)M)?",

#     # many users have standard ways of starting their reports
#     # that include predictable phrases. Take them out here.
#     # important here that we're using \A to say only look at the start of the string
#     # ALSO pay attention to what is being removed BEFORE this,
#     # because it changes what starts the string
#     r"\ACommentary",
#     r"\ANon(\s|-)dream dream lucid",
#     r"\ANon(\s|-)lucid lucid",
#     r"\Anon-dream dream semi-lucid lucid false awakening",
#     r"\Anon-dream non-lucid lucid",
#     r"\AAwake\|Dreaming\|Lucid",
#     r"\AOriginally Posted (by)?",
#     r"\AOld LD from",
#     r"\AJournal Entry",
#     r"\AMorning of",
#     r"\ANight of",
#     r"\AEarly evening  of",
#     # r"\A\W*", # any nonword/number characters at the start
#     # r"\ALast night",
# ]

# # #CASE sensitive ones from one user
# # r"NON-DREAM"
# # r"DREAM"
# # r"LUCID"
# # r"NOTES"


df["report_clean"], df["regex_catches"] = zip(*df["post_txt"].progress_apply(replacements))

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
