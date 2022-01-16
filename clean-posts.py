"""
Convert raw DreamViews html dream report files
into individual text files and a corresponding
attributes file.

DONT CHANGE ANY OF THIS EXCEPT COMMENTS BECAUSE ONCE THE
DREAM REPORTS ARE UPLOADED TO TAGTOG AND MANUALLY ANNOTATED
THATS THAT. im talking to you you piece of shit.

Also generate random IDs for all posts and users,
exporting the user legend as a dictionary for later use.

Each dream report also ends up with a unique
identifier that is a filename for the text
file and a row in the attributes file
that has all the relevant corresponding info
(username, date, tags, categories, title)

Only real exclusion besides a few anomolous errors
is that there is a minimum amount of alphabetic
characters required for the dream report.

Extremely minimal cleaning here. The goal is not to clean
dream reports, only to make them all consistent with each
other and remove html decoding errors. Most of these
corrections fix minimal instances anyways, like just a couple,
so they probably coulda been left anyways :/
- convert to ASCII characters (what a fucking headache)
- replace all extra space characters with a single space.
  This is probably the biggest deal, but I think it makes
  sense for consistency and length and annotator neutrality.
  Might be wrong on this one I couldn't decide what was best.
Some notes about what I'm NOT removing
- remove some bracketed formatting content
- non-english posts

This is in NO WAY optimized for speed. It takes too long,
lots of text gets analyzed that is later tossed out. W/e.
Not a huge concern it's really only running once.
"""
import os
import re
import json
import tqdm
import zipfile
import datetime

import unidecode
import contractions
import langdetect
import spacy

import uuid
import random

import pandas as pd

from bs4 import BeautifulSoup

# from collections import Counter

import config as c


############ Make sure NLTK and spaCy tools are downloaded.

# don't use the small model -- bad at entity recognition. large and trf are good
SPACY_MODEL = "en_core_web_lg" # en_core_web_lg, en_core_web_trf (en_core_web_sm for testing only)

# check spaCy model
if not spacy.util.is_package(SPACY_MODEL):
    resp = input(f"{SPACY_MODEL} not found -- download now?? (Y/n)")
    if resp.lower() in ["", "y"]:
        spacy.cli.download(SPACY_MODEL)

####### Load spaCy model

# Only redacting here, which is just named entity recognition.
# For the lg model you can disable most other thing and this
# will speed up the spaCy/nlp stuff. You can't get lemmas but okay.
SPACY_PIPE_DISABLES = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
# nlp = spacy.load(SPACY_MODEL, disable=SPACY_PIPE_DISABLES)
nlp = spacy.load(SPACY_MODEL)
nlp.add_pipe("merge_entities") # so "john paul" gets treated as a single entity


# # restrict uber-short posts, token limit later will catch most
# # this is just to make sure there *is* text
# MINIMUM_ALPHA_CHARS = 10


def convert2ascii(txt, retain_whitespace_count=False):
    # Replace annoying unicode surrogates (??) that cause warnings in unidecode.
    ascii_txt = re.sub(surrogate_re, " ", txt)
    # Unidecode does the heavy-lifting on conversion to ASCII.
    ascii_txt = unidecode.unidecode(ascii_txt, errors="ignore", replace_str="")
    # Replace some non-printable whitespace characters that are technically ASCII but not printable.
    ascii_txt = re.sub(extra_ascii_chars, " ", ascii_txt)
    # Reduce to single whitespaces. (*after* ASCII conversion)
    whitespace_re = r"\s" if retain_whitespace_count else r"\s+"
    ascii_txt = re.sub(whitespace_re, " ", ascii_txt)
    # Final strip to be sure there aren't leading/trailing whitespaces after all the processing.
    if not retain_whitespace_count:
        ascii_txt = ascii_txt.strip()
    assert ascii_txt.isascii() and ascii_txt.isprintable()
    return ascii_txt


# initialize a separate random object so the seed is different from other one
rd4lemma = random.Random()
rd4lemma.seed(6)
def lemmatize(doc, shuffle=False, pos_remove_list=["PROPN", "SMY"]):
    """takes a spaCy doc.
    Not stressing too hard on this because it's likely
    that one will want to tokenize/lemmatize their own way.
    Also some thing are easy to do later and not always wanted (like removing stop words).
    """
    token_list = []
    for token in doc:
        if (token.is_alpha
            ) and (len(token) >= 3
            ) and (not token.like_email
            ) and (not token.like_url
            ) and (not token.like_num
            ) and (not token.is_stop
            ) and (not token.is_oov
            ) and (not token.pos_ in pos_remove_list):
            token_list.append( token.lemma_.lower() ) # *almost* always lowercase by default
    if shuffle:
        token_list = random.sample(token_list, len(token_list))
    return " ".join(token_list) if token_list else None


# set up randomizer stuff
rd = random.Random()
def generate_id(n_chars):

    """generates a random character sequence.
    Force to start with a letter so it's always interpreted as categorical in file readers.
    """
    _gen_str = lambda nchars: uuid.UUID(int=rd.getrandbits(128)).hex[:n_chars]
    id_string = _gen_str(n_chars)
    while not id_string[0].isalpha():
        id_string = _gen_str(n_chars)
    return id_string.upper()


import_fname = os.path.join(c.DATA_DIR, "source", "dreamviews-posts.zip")

export_fname_posts   = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-posts.tsv")
export_fname_userkey = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-users_key.json")

# create datetime objects for comparison later
start_datetime = datetime.datetime.strptime(c.START_DATE, "%Y-%m-%d")
end_datetime = datetime.datetime.strptime(c.END_DATE, "%Y-%m-%d")

# read in all the html files at once
with zipfile.ZipFile(import_fname, mode="r") as zf:
    html_files = [ zf.read(fn) for fn in zf.namelist() ]



# Select which of the following named entity labels
# in spaCy that should get removed/replaced.
# CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP,
# ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
entities_to_redact = ["PERSON"] #, "LOC", "GPE", "ORG", "DATE", "TIME"]


surrogate_re = r"[\ud83d\ud83c\udf37\udf38\udf39\udf3a\udc2c]+"
# control_chars_re = r"[^\x09\x0A\x0D\x20-\x7F]+"
# control_chars_re = r"[^\x00-\x7F]+"
# just need to catch \x1b and \x7f (unidecode does the rest)
extra_ascii_chars = r"[\x1b\x7f]+"


## Initialize empty dictionaries to store content that survives restrictions.
# Gets turned into dataframe for export at the end.
data = {} # ( post_id, post_data ) key, value pairs
user_raw2id_mapping = {} # ( raw_username, unique_username ) key, value pairs
# user_counts = Counter()  # to keep track of n posts per user

# Initialize a random state value.
# This will get incremented every post that gets *looked* at,
# including the posts that don't survive restrictions. The
# purpose of this approach is that even if restriction criteria
# get readjusted and this gets re-run, *as long as the raw html
# zip file stays the same* then the random IDs should be reproducible.
random_state = 0

for html_byt in tqdm.tqdm(html_files, desc="parsing html and processing text"):

    # # convert from bytes to string
    # html_str = html_byt.decode(encoding="windows-1252", errors="strict") # windows-1252 == cp1252

    # get all the blog posts from the current html
    # convert html to soup object thing
    soup = BeautifulSoup(html_byt, "html.parser", from_encoding="windows-1252")
    page_posts  = soup.find_all("div", class_="blogbody")
    page_users  = soup.find_all("div", class_="popupmenu memberaction")
    page_dates  = soup.find_all("div", class_="blog_date")
    page_titles = soup.find_all("a", class_="blogtitle")
    # Make sure each dream journal entry has a
    # corresponding username, date, and title
    assert len(page_posts) == len(page_users) == len(page_dates) == len(page_titles)


    #### Loop over each entry (and components) of the current html page
    #### and perform *minimal* cleaning and further parsing of the html.
    #### Clean up the dates a bit and strip strip away excess edge whitespace
    #### in some instances. The Tags and Categories need to be parsed out of the report.
    
    #### Note that there are some "continue" statements
    #### that will push into the next loop and prevent
    #### saving that data. It's exclusion criteria.
    for post, user, date, title in zip(page_posts,
                                       page_users,
                                       page_dates,
                                       page_titles):
        
        random_state += 1
        rd.seed(random_state)
        
        # extract text from soup/html object
        # (NOT using str.strip at this stage because
        #  in some cases the username is space-like and that collapses it.)
        # report_txt = report.get_text(separator=" ")
        # user_txt   = user.get_text(separator=" ")
        # date_txt   = date.get_text(separator=" ")
        # title_txt  = title.get_text(separator=" ")
        # ## ^^^ this separator thing is slightly annoying bc it adds
        ##     spaces even when they aren't necessary (like next to newlines)
        ## so try this as an extreme in the OTHER direction
        ## where all the newlines are gone and replaced with spaces
        # report_txt = " ".join([ x for x in report.stripped_strings ])
        # user_txt   = " ".join([ x for x in user.stripped_strings ])
        # date_txt   = " ".join([ x for x in date.stripped_strings ])
        # title_txt  = " ".join([ x for x in title.stripped_strings ])
        # post_txt  = post.get_text(separator=" ", strip=True)
        post_txt  = post.text
        user_txt  = user.text # def dont use strip=True on user bc of username with just spaces
        date_txt  = date.text
        title_txt = title.text



        ############################################################
        #################   CLEAN/ANONYMIZE USER   #################
        ############################################################
        #### Do this first so each user gets a unique ID
        #### even if they don't pass restrictions. This keeps IDs
        #### reproducible even if restrictions change.

        # Convert to printable ASCII.
        # Using a little more caution with replacing whitespace
        # here because the number of spaces can differentiate between users
        # (e.g., some usernames are a bunch of spaces). In practice
        # that means using \s instead of \s+ for regex to convert
        # *each* whitespace character to a single space, rather than reducing.
        user_txt = user_txt.lstrip("\n").rstrip("\n")
        # user_txt = unidecode.unidecode(user_txt, errors="ignore", replace_str="")
        # user_txt = re.sub(r"\s", " ", user_txt) 
        # assert user_txt.isascii() and user_txt.isprintable()
        # If there is an @ in the username, it is changed to [email\xa0protected]
        # These are NOT emails! Need to get the real username, or drop
        # them because it will mess with repeated measures stats (they aren't same user).
        # if user_txt == "[email protected]":
        if re.search(r"\[email\s+protected\]", user.text) is not None:
            # the real username is still in the user item somewhere
            user_txt = user.find("a").attrs["title"].split(" is offline")[0]
        user_txt = convert2ascii(user_txt, retain_whitespace_count=True)

        # Generate random user ID.
        try:
            # First look for an existing ID that was already made.
            unique_user_id = user_raw2id_mapping[user_txt]
        except KeyError:
            # If not found, generate a new one.
            unique_user_id = generate_id(n_chars=4)
            # Keep generating until it's one that hasn't been generated before.
            while unique_user_id in user_raw2id_mapping.values():
                unique_user_id = generate_id(n_chars=4)
            user_raw2id_mapping[user_txt] = unique_user_id



        ########################################################
        #################   CLEAN/PARSE DATE   #################
        ########################################################

        # user info is (redundantly) in date_txt so drop it
        date_txt = date_txt.strip().split(", ", 1)[1]

        # There is sometimes an extra date "modifier" we don't need.
        # It's always in parentheses so check for it and remove if it's there.
        if "(" in date_txt and ")" in date_txt:
            date_txt, date_descriptor = date_txt.rstrip(")").split(" (", 1)

            # -- RESTRICT --
            # There is a kind of "community" dream journal focused on shared dreaming.
            # https://www.dreamviews.com/blogs/iosdp/
            # Remove anything from here, since sometimes the usernames are innacurate
            # with respect to the dream content, and also they are for shared dreaming
            # attempts specifically.
            # They are sometimes identifiable with a username of "IOSDP" but not always.
            # but other times not. Most effective to look at the parenthetical
            # next to the date.
            # Take out posts from the shared shared dream journal.
            if date_descriptor == "International Oneironaut Shared Dreaming Journal":
                continue
            
            
        # Recent posts are marked as "today" or "yesterday"
        # just skip them not worth the clunky conversion and
        # restricting before then anyways.
        if "Today" in date_txt or "Yesterday" in date_txt:
            continue

        # Convert string for iso-format for standardization.
        blogdatetime = datetime.datetime.strptime(date_txt, c.BLOG_TIMESTAMP_FORMAT)
        date_txt_iso = blogdatetime.strftime("%Y-%m-%dT%H:%M")

        # restrict to the time window for cleanliness (and pre-2010 is weird)
        if blogdatetime < start_datetime or blogdatetime > end_datetime:
            continue

        #############################################################################
        #################   Extract/parse the Tags and Categories   #################
        #############################################################################
        # The post text has more than just the dream report.
        # At the end it will ALWAYS have a "Categories" section,
        # even if with just an "Uncategorized" label.
        # There is also an optional "Tags" section that will immediately
        # precede the Categories section if the user includes and Tags.
        # The options for Category labels are limited (and thus of primary
        # interest to us), while Tags are custom and the user can input anything.

        #### Do this prior to text cleaning, otherwise it messes with parsing.

        ## Break the post text into post, tags, and categories.

        # -- RESTRICT --
        # There are some (10-20) posts that have multiple instances of "Tags" and/or "Categories".
        # Sometimes they are garbage anyways to-be skipped, like when
        # the post is actually a copy/paste of multiple prior entries.
        # Other times they are salvageable but it's annoying and not worth
        # it to save the handful. They are generally places where the user
        # inserted tags or categories manually at the end as well, or
        # when categories is in the "Updated" section that is later removed.
        # Skip them all.
        if post_txt.count("Categories") > 1 or post_txt.count("Tags") > 1:
            continue

        # break blog into dream report, tags, and categories
        ## see check a few lines down that makes sure there are limited appearances of these things
        tags_are_present = "Tags:" in post_txt
        if tags_are_present:
            # split_rule_re = r"Tags:|(?<!Added )Categories")
            split_rule_re = r"Tags:|Categories"
            post_txt, tag_txt, cat_txt = re.split(split_rule_re, post_txt)
        else:
            split_rule_re = r"Categories"
            post_txt, cat_txt = re.split(split_rule_re, post_txt)
            tag_txt = None

        # get rid of sometimes where there is an "Attached Thumbnails" section at the end
        cat_txt = cat_txt.split("Attached Thumbnails")[0]
        
        # Strip excess whitespace off everything.
        post_txt = post_txt.strip()
        cat_txt = cat_txt.strip()
        if tags_are_present:
            tag_txt = tag_txt.strip()

        # Extract lists for each of tags and categories.
        # cat_txt = "::".join(re.split(r",\s+", cat_txt))
        # tag_txt = "::".join(re.split(r",\s+", tag_txt))
        tags = [] if tag_txt is None else \
               [ t.strip().lower().replace(" ", "_") for t in tag_txt.split(", ") ]
        cats = [ c.strip().lower().replace(" ", "_") for c in cat_txt.split(", ") ]
        # tags = [] if tag_txt is None else re.split(r",\s+", tag_txt.strip())
        # cats = re.split(r",\s+", cat_txt.strip())

        ## Use the category list to come up with useful (specific) labels.

        # Identify if the post was lucid.
        if "lucid" in cats and "non-lucid" in cats:
            post_lucidity = "ambiguous"
        elif "lucid" in cats:
            post_lucidity = "lucid"
        elif "non-lucid" in cats:
            post_lucidity = "nonlucid" # remove hyphen for future convenience
        else:
            post_lucidity = "unspecified"

        # Identify if the post was a nightmare.
        post_was_nightmare = "nightmare" in cats

        # # Convert to printable ASCII.
        # title_txt = re.sub(surrogate_re, " ", title_txt)
        # title_txt = unidecode.unidecode(title_txt, errors="ignore", replace_str="")
        # title_txt = re.sub(extra_ascii_chars, " ", title_txt)
        # title_txt = re.sub(r"\s+", " ", title_txt)
        # title_txt = title_txt.strip()
        # assert title_txt.isascii() and title_txt.isprintable()

        # Merge the tags and categories into strings for saving in dataframe.
        cats = "::".join(cats)
        if tags is not None:
            tags = "::".join(tags)

        # Convert to printable ASCII.
        tags = convert2ascii(tags)
        cats = convert2ascii(cats)

        # # skip some weird entries
        # # eg, one entry that is copy/pasted multiple entries
        # # which breaks this and shouldnt be counted anyways.
        # # this is a good way to ensure single entries
        # if (("Tags:" in post_txt and len(components) != 3)
        #     or ("Tags:" not in post_txt and len(components) != 2)):
        #     continue
        # # this is late to check, but want it after this continue section which will catch some of these assertion errors
        # assert post_txt.count("Categories") == 1
        # assert post_txt.count("Tags:") in [0, 1]
        
        # if "Tags:" in post_txt: # same as len(components) == 3
        #     post_txt, tags, cats = components
        # else:        
        #     post_txt, cats = components
        #     tags = None


        #################################################################
        #################   CLEAN POST (dream report)   #################
        #################################################################

        ## Convert to printable ASCII.
        post_txt = convert2ascii(post_txt)

        # -- RESTRICT --
        # A lot of posts start with Originally posted by ...
        # These should be pulled based on our stats
        # accounting for repeated measures within subjects
        # but these mess that up.
        if post_txt.startswith("Originally posted"):
            continue

        # Replace the few stupid apostrophe representations.
        post_txt = post_txt.replace("&#39;", "'")

        ## Minor text corrections to make later life easier.
        # replace ampersands
        post_txt = post_txt.replace("&", "and")
        # replace contractions with full words
        post_txt = contractions.fix(post_txt, slang=True)
        # replace any sequence of 4+ characters with 1 of that character
        # gets rid of stuff like whoaaaaaaaaaaaa and --------------------
        # will lead to some errors because it replaces with 1 letter but sometimes will need 2
        post_txt = re.sub(r"(.)\1{3,}", r"\1", post_txt, flags=re.IGNORECASE)


        ## DreamViews posts have some commons text patterns that can be removed.
        ## These are all regexes that are specific to the needs of cleaning DreamViews text.

        # There are some leftover block formatting tags.
        post_txt = re.sub(r"\[(/?INDENT|/?RIGHT|/?CENTER|/?B)\]", "", post_txt, flags=re.IGNORECASE)
        # These ones never have a = preceding them.
        post_txt = re.sub(r"\[/?(INDENT|RIGHT|CENTER|B|I|U|HR|IMG|LINK_TO_ANCHOR|SARCASM|DREAM LOGIC)\]", "", post_txt, flags=re.IGNORECASE)
        # These need some leway as to what comes after because sometimes there's stuff there.
        post_txt = re.sub(r"\[/?COLOR.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[/?SIZE.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[/?FONT.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[/?QUOTE.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[/?SPOILER.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[/?URL.*?\]", "", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"\[ATTACH=CONFIG\][0-9]*\[/ATTACH\]", "", post_txt, flags=re.IGNORECASE)

        # Posts can be updated and have this stereotyped amendment
        # at the end if they were (about 20% of posts have this).
        # Example: Updated 12-08-2021 at 10:28 PM by 34880
        # Example: Updated 08-05-2017 at 01:09 PM by 93119 (Added Categories)
        # Example: Updated 04-20-2014 at 12:36 PM by 68865 (remembered another fragment)
        updated_re = r" Updated [0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M by [0-9]{1,5}( \(.*?\))?"
        post_txt = re.sub(updated_re, "", post_txt)


        ## Redactions.
        # redact emails
        # The text already has some "@[email\xa0protected]" parts, basically anything after an @. Kinda dumb.
        # They aren't always emails so just replace with nothing.
        post_txt = re.sub(r"@\[email protected\]", "", post_txt) # first
        post_txt = re.sub(r"\S*@\S*\s?", "[[URL]]", post_txt) # just in case there are still any

        # redact URLs
        post_txt = re.sub(r"https?://\S+", "[[URL]]", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"www\.\S+", "[[URL]]", post_txt, flags=re.IGNORECASE)


        # -- RESTRICT -- based on written language (must be English)
        # -- RESTRICT -- based on absence of any alpha characters
        # Run at same time since language detection will error out without any alpha characteres anyways.
        # Make sure there is at least a few characters.
        # This is NOT a step restricting on word count per se.
        # Sometimes posts are just images or something, and so
        # the following processing steps won't even work because
        # there is literally no text. This is mostly just to prevent errors.
        if re.search(r"[a-zA-Z]", post_txt) is None:
            continue
        language = langdetect.detect(post_txt)
        # try:
        #     language = langdetect.detect(post_txt)
        # except langdetect.LangDetectException as err:
        #     if err.code == 5: # err.__str__()=="No features in text."
        #         continue
        if language != "en":
            continue

        # Use spaCy to recognize/identify named entities.
        doc = nlp(post_txt)

        # -- RESTRICT -- based on number of words.
        # Note the wordcount is placed prior to entity replacement for convenience.
        # This way don't have to re-"doc" the redacted text.
        # n_tokens = len(doc) # no distinction between punctuation and words
        n_words = sum( t.is_alpha for t in doc )
        if not c.MIN_WORDCOUNT <= n_words <= c.MAX_WORDCOUNT:
            continue

        # Replace the entities with the entity label in double square brackets.
        # Loop over the entities in reverse and modify the text with replacements
        # (loop in reverse so that indices still work after string modifications).
        # redacted_text = doc.text
        for ent in reversed(doc.ents):
            if ent.label_ in entities_to_redact:
                post_txt = (post_txt[:ent.start_char]
                    + "[["+ent.label_+"]]" + post_txt[ent.end_char:])


        # lemmatize while we're here and spaCy is running
        lemmatized_text = lemmatize(doc, shuffle=True)

        ###################################################
        #################   CLEAN TITLE   #################
        ###################################################

        # Convert to printable ASCII.
        title_txt = convert2ascii(title_txt)





        ##################################################################
        #################   Save to running dictionary   #################
        ##################################################################

        # # -- RESTRICT -- based on the number of posts per user.
        # user_counts.update([unique_user_id])
        # nposts_this_user = user_counts[unique_user_id]
        # if nposts_this_user > c.MAX_POSTCOUNT:
        #     continue

        single_post_data = {
            "user_id"     : unique_user_id,
            # "user_postn"  : nposts_this_user,
            "timestamp"   : date_txt_iso,
            "title"       : title_txt,
            "tags"        : tags,
            "categories"  : cats,
            "lucidity"    : post_lucidity,
            "nightmare"   : post_was_nightmare,
            "wordcount"   : n_words,
            "post_clean"  : post_txt,
            "post_lemmas" : lemmatized_text,
        }

        # Generate random ID for this specific post and
        # use it as an identifier in the data dictionary.
        unique_post_id = generate_id(n_chars=8)
        while unique_post_id in data:
            unique_post_id = generate_id(n_chars=8)
        data[unique_post_id] = single_post_data




############################################################
#################   Aggregate and Export   #################
############################################################

# Generate a dataframe from all the posts.
df = pd.DataFrame.from_dict(data, orient="index"
    ).sort_values(["user_id", "timestamp"])

# # a few strange duplicated reports (<1%)
# df = df.drop_duplicates(subset="post_txt", keep="first")

# Add a column that identifies the post # in sequence for a given user.
# df = df.sort_values(["user_id", "timestamp"]) # should be redundant but it's critical
df.insert(1, "nth_post",
    df.groupby("user_id")["timestamp"].transform(lambda s: range(1, 1+len(s)))
)

# -- RESTRICT -- based on the number of posts per user.
df = df[ df["nth_post"].le(c.MAX_POSTCOUNT) ]


# Drop the raw2id mapping dictionary to ONLY those users that survived restrictions
#### so that only "used" users go into the raw2id mapping key.
out_mapping_key = { username: userid for username, userid in user_raw2id_mapping.items()
    if userid in df["user_id"].unique() }


## Write two files.

# tsv with data
df.to_csv(export_fname_posts, encoding="ascii",
    index=True, index_label="post_id", sep="\t")

# json with usernames
with open(export_fname_userkey, "wt", encoding="ascii") as outfile:
    json.dump(out_mapping_key, outfile, indent=4, sort_keys=True, ensure_ascii=False)
