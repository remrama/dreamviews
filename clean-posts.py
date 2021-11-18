"""
Convert raw DreamViews html dream report files
into individual text files and a corresponding
attributes file. Also export a json/dictionary
with all the users and a map to unique anonymized IDs.

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
- fix an issue with decoding single apostrophe
- conform everything to be utf-8 compatible
- replace all extra space characters with a single space.
  This is probably the biggest deal, but I think it makes
  sense for consistency and length and annotator neutrality.
  Might be wrong on this one I couldn't decide what was best.
Some notes about what I'm NOT removing
- remove some bracketed formatting content
- non-english posts

DONT CHANGE ANY OF THIS EXCEPT COMMENTS BECAUSE ONCE THE
DREAM REPORTS ARE UPLOADED TO TAGTOG AND MANUALLY ANNOTATED
THATS THAT. im talking to you you piece of shit.
"""
import os
import re
import csv
import json
import tqdm
import zipfile
import datetime

import uuid
import random

import pandas as pd

from bs4 import BeautifulSoup
from unidecode import unidecode

import config as c


BLOG_TIMESTAMP_FORMAT = "%m-%d-%Y at %I:%M %p"

# set up randomizer stuff
rd = random.Random()
rd.seed(0)
def generate_id(n_chars):
    """generates a random character sequence.
    Force to start with a letter so it's always interpreted as categorical.
    """
    _gen_str = lambda nchars: uuid.UUID(int=rd.getrandbits(128)).hex[:n_chars]
    id_string = _gen_str(n_chars)
    while not id_string[0].isalpha():
        id_string = _gen_str(n_chars)
    return id_string.upper()


import_fname = os.path.join(c.DATA_DIR, "source", "dreamviews-posts.zip")

export_txt_directory = os.path.join(c.DATA_DIR, "derivatives", "posts")
export_post_attrs_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-attributes.tsv")
export_user_key_fname = os.path.join(c.DATA_DIR, "derivatives", "users-raw2anon_key.json")

# if the directory already exists, delete and remake it
if os.path.isdir(export_txt_directory):
    import shutil
    shutil.rmtree(export_txt_directory)
os.mkdir(export_txt_directory)


# Entries use "Today" and "Yesterday" strings for time sometime so
# need to convert those to actual dates for saving dates of blog posts.
# In practice, this means getting actual date of today and yesterday for modifying blogdates later.
dayonly_format   = BLOG_TIMESTAMP_FORMAT.split(" at ")[0]
today            = datetime.datetime.strptime(c.DATA_COLLECTION_DATE, "%Y-%m-%d").date()
yesterday        = today - datetime.timedelta(days=1)
today_string     = today.strftime(dayonly_format)
yesterday_string = yesterday.strftime(dayonly_format)


# read in all the html files at once
with zipfile.ZipFile(import_fname, mode="r") as zf:
    html_files = [ zf.read(fn) for fn in zf.namelist() ]


attributes = {}

user_mappings = {} # running dict of unique users and anonymized IDs

for html_byt in tqdm.tqdm(html_files, desc="parsing html dream journal pages"):

    # # convert from bytes to string
    # html_str = html_byt.decode(encoding="windows-1252", errors="strict") # windows-1252 == cp1252

    # get all the blog posts from the current html
    # convert html to soup object thing
    soup = BeautifulSoup(html_byt, "html.parser", from_encoding="windows-1252")
    page_reports = soup.find_all("div", class_="blogbody")
    page_users   = soup.find_all("div", class_="popupmenu memberaction")
    page_dates   = soup.find_all("div", class_="blog_date")
    page_titles  = soup.find_all("a", class_="blogtitle")
    # Make sure each dream journal entry has a
    # corresponding username, date, and title
    assert len(page_reports) == len(page_users) == len(page_dates) == len(page_titles)


    #### Loop over each entry (and components) of the current html page
    #### and perform *minimal* cleaning and further parsing of the html.
    #### Clean up the dates a bit and strip strip away excess edge whitespace
    #### in some instances. The Tags and Categories need to be parsed out of the report.
    for report, user, date, title in zip(page_reports,
                                         page_users,
                                         page_dates,
                                         page_titles):
        
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
        report_txt = " ".join([ x for x in report.stripped_strings ])
        user_txt   = " ".join([ x for x in user.stripped_strings ])
        date_txt   = " ".join([ x for x in date.stripped_strings ])
        title_txt  = " ".join([ x for x in title.stripped_strings ])
        # there are a few instances of icky html that doesn't get
        # decoded properly, resulting in stuff like "&#39;" representing quotes
        # in the text. it's just in a few examples this should take care of it
        report_txt = report_txt.replace("&#39;", "'")
        # also there are still a (very) few instances of
        # characters that won't translate to utf-8 (surrogate unicode errors).
        # only appears in report bodies and titles
        report_txt = report_txt.encode("utf-8", errors="ignore").decode("utf-8")
        title_txt  = title_txt.encode("utf-8", errors="ignore").decode("utf-8")

        ## The title is basically good as it is.
        ## The username needs to just have edges stripped.
        ## The date needs kind of a clunky fix to get cleaned up.
        ## The dream report needs to have the Tags and Categories
        ##    popped out but otherwise it's good (just strip).
        ## The resulting Tags and Categories need to be adjusted a bit.

        ### handle date ###

        # convert the date to a ISO string
        # dates sometimes have a custom modifier
        # so break it apart, resulting in two items
        # a date and a "date modifier" which is whatever text the user puts there

        # user info is (redundantly) in date_txt so drop it
        date_txt = date_txt.strip().split(", ", 1)[1]

        # the excess date "modifier" is always in parentheses
        # when present, so check for it and remove if it's there
        if "(" in date_txt and ")" in date_txt:
            date_txt, _ = date_txt.rstrip(")").split(" (", 1)
        # else:
        #     datemod = "NaN"
        # replace "today"/"yesterday" with proper date strings before datetime stripping
        if "Today" in date_txt:
            date_txt = date_txt.replace("Today", today_string)
        elif "Yesterday" in date_txt:
            date_txt = date_txt.replace("Yesterday", yesterday_string)
        # change date format to isoformatted string for output
        blogdatetime = datetime.datetime.strptime(date_txt, BLOG_TIMESTAMP_FORMAT)
        date_txt_iso = blogdatetime.strftime("%Y-%m-%dT%H:%M")
        # date_txt_iso = blogdatetime.isoformat()


        ##############


        ### handle dream report ###

        # The report/body has more than just the dream report
        # it often contains "Tags" and always contains "Categories".
        # Tags come from a user-generated pool while categories come from
        # a predetermined pool (so more consistent).

        # break blog into dream report, tags, and categories
        ## see check a few lines down that makes sure there are limited appearances of these things
        split_rule = "Tags:|Categories" if "Tags:" in report_txt else "Categories"
        components = re.split(split_rule, report_txt)

        # skip some weird entries
        # eg, one entry that is copy/pasted multiple entries
        # which breaks this and shouldnt be counted anyways.
        # this is a good way to ensure single entries
        if (("Tags:" in report_txt and len(components) != 3)
            or ("Tags:" not in report_txt and len(components) != 2)):
            continue
        # this is late to check, but want it after this continue section which will catch some of these assertion errors
        assert report_txt.count("Categories") == 1
        assert report_txt.count("Tags:") in [0, 1]
        
        if "Tags:" in report_txt:
            dream_txt, tags, cats = components
        else:        
            dream_txt, cats = components
            tags = None

        #### clean the dream report
        # dream_txt = dream_txt.strip()
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
        if sum([ c.isalpha() for c in dream_txt ]) < c.MIN_ALPHA_CHARS:
            continue

        #### clean the tags/categories
        tags = [] if tags is None else \
               [ t.strip().lower().replace(" ", "_") for t in tags.split(", ") ]
        cats = [ c.strip().lower().replace(" ", "_") for c in cats.split(", ") ]

        ##############


        ### handle username ###

        # username starts and end with return char, strip them off
        # assert user_txt.startswith("\n") and user_txt.endswith("\n")
        # user_txt = user_txt[1:-1]
        user_txt = user_txt.lstrip("\n").rstrip("\n")

        # add user to the "master" dictionary with an anonymous ID
        if user_txt not in user_mappings.keys():
            unique_user_id = generate_id(n_chars=4)
            while unique_user_id in user_mappings.values():
                unique_user_id = generate_id(n_chars=4)
            user_mappings[user_txt] = unique_user_id
        else: # need this for saving in the attributes file
            unique_user_id = user_mappings[user_txt]

        ##############


        ###### export/saving stuff

        # generate a random/unique ID for this dream report
        unique_id = generate_id(n_chars=8)
        while unique_id in attributes:
            unique_id = generate_id(n_chars=8)

        payload = {
            "user_id"    : unique_user_id,
            "timestamp"  : date_txt_iso,
            "title"      : title_txt,
            "tags"       : tags,
            "categories" : cats,
            # "length"     : char_length
        }

        attributes[unique_id] = payload

        # export individual text file and add attributes to ongoing json
        export_fname_txt = os.path.join(export_txt_directory, f"{unique_id}.txt")
        with open(export_fname_txt, "xt", encoding="utf-8") as outfile:
            outfile.write(dream_txt)

        ##############



# export the post attributes info
df = pd.DataFrame.from_dict(attributes, orient="index"
    ).sort_values(["user_id", "timestamp"])
df.to_csv(export_post_attrs_fname, sep="\t", encoding="utf-8",
    index=True, index_label="post_id",)
# with open(export_fname_json, "wt", encoding="utf-8") as outfile:
#     json.dump(attributes, outfile, indent=4, sort_keys=True, ensure_ascii=False)

# export the user mapping dictionary
with open(export_user_key_fname, "wt", encoding="utf-8") as outfile:
    json.dump(user_mappings, outfile, indent=4, sort_keys=True, ensure_ascii=False)
