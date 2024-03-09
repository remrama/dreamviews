"""
Convert raw DreamViews dream journal HTML pages into a tsv file holding
dream reports and their attributes.

Also generate random IDs for all posts and users, exporting the user legend as a json file.

Each dream report also ends up with a unique identifier that is a filename for
the text file and a row in the attributes file that has all the relevant
corresponding info (username, date, tags, categories, title, etc.).

Only real exclusion besides a few anomalous errors is that there is a minimum
amount of alphabetic characters required for the dream report.

Extremely minimal cleaning here. The goal is not to clean dream reports, only to
make them all consistent with each other and remove HTML decoding errors. Most
of these corrections fix minimal instances anyways, like just a couple, so they
probably coulda been left anyways :/
- convert to ASCII characters (what a fucking headache)
- replace all extra space characters with a single space.
  This is probably the most controversial step, but I think it makes sense for
  consistency and length. Might be wrong on this one I couldn't decide.
- remove some bracketed formatting content
- non-english posts

This is in NO WAY optimized for speed. It takes too long, lots of text gets
analyzed that is later tossed out. Not that worried about it.
"""
import datetime
import json
import random
import re
import uuid
import zipfile

from bs4 import BeautifulSoup
import contractions
import langdetect
import pandas as pd
import spacy
import tqdm
import unidecode

import config as c


# Identify filepaths.
import_path = c.DATA_DIR / "source" / "dreamviews-posts.zip"
export_path_posts = c.DATA_DIR / "derivatives" / "dreamviews-posts.tsv"
export_path_userkey = c.DATA_DIR / "derivatives" / "dreamviews-users_key.json"

# Load spaCy model (used for named entity recognition).
nlp = spacy.load("en_core_web_lg")
# # Speed up spaCy by disabling some unncessary stuff.
# SPACY_PIPE_DISABLES = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]
# nlp = spacy.load(SPACY_MODEL, disable=SPACY_PIPE_DISABLES)
nlp.add_pipe("merge_entities")  # So "John Paul" gets treated as a single entity.

# Initialize some randomizers.
rd4ids = random.Random()
rd4shuf = random.Random()

# Create datetime objects to restrict posts.
start_date, end_date = "2010-01-01", "2020-12-31"
start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d")
end_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d")

# read in all the html files at once
with zipfile.ZipFile(import_path, mode="r") as zf:
    html_files = [ zf.read(fn) for fn in zf.namelist() ]


################################################################################
# PREPROCESSING FUNCTIONS
################################################################################

def convert2ascii(text, retain_whitespace_count=False):
    """Return a printable ASCII string."""
    # Replace annoying unicode surrogates (??) that cause warnings in unidecode.
    text = re.sub(r"[\ud83d\ud83c\udf37\udf38\udf39\udf3a\udc2c]+", " ", text)
    # Unidecode does the heavy-lifting on conversion to ASCII.
    text = unidecode.unidecode(text, errors="ignore", replace_str="")
    # Replace some non-printable whitespace characters that are technically ASCII but not printable.
    text = re.sub(r"[\x1b\x7f]+", " ", text)
    # Reduce to single whitespaces (*after* ASCII conversion).
    whitespace_re = r"\s" if retain_whitespace_count else r"\s+"
    text = re.sub(whitespace_re, " ", text)
    # Final strip to be sure there aren't leading/trailing whitespaces after all the processing.
    if not retain_whitespace_count:
        text = text.strip()
    assert text.isascii() and text.isprintable()
    return text

def lemmatize(doc, shuffle=False, pos_remove_list=["PROPN", "SMY"]):
    """Convert a spaCy doc to space-separate string of shuffled lemmas."""
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
            token_list.append(token.lemma_.lower())
    if shuffle:
        token_list = random.sample(token_list, len(token_list))
    if token_list:
        return " ".join(token_list)

def generate_id(n_chars):
    """Return a random alpha-numeric character sequence (str) of length n_chars.
    Always starts with a capital letter, to ensure categorical interpretation later.
    """
    _gen_str = lambda nchars: uuid.UUID(int=rd4ids.getrandbits(128)).hex[:n_chars]
    id_string = _gen_str(n_chars)
    while not id_string[0].isalpha():
        id_string = _gen_str(n_chars)
    return id_string.upper()


################################################################################
# CLEANING LOOP
################################################################################

# Initialize empty dictionaries to store content that survives exclusion.
data = {}  # To hold key, value pairs of post_id, post_data
user_mapping = {}  # key, value pairs of raw_username, unique_username

# Initialize a random state value that will increment every post.
random_state = 0

# Loop over each HTML file
for html in tqdm.tqdm(html_files, desc="Cleaning DreamViews posts"):

    # Extract the post, user, date, and title, from each post of the current html file.
    soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
    page_posts = soup.find_all("div", class_="blogbody")
    page_users = soup.find_all("div", class_="popupmenu memberaction")
    page_dates = soup.find_all("div", class_="blog_date")
    page_titles = soup.find_all("a", class_="blogtitle")
    assert len(page_posts) == len(page_users) == len(page_dates) == len(page_titles)

    ## Loop over each entry (and components) of the current HTML page and
    ## perform *minimal* cleaning and further parsing of the HTML.
    ##
    ## There are some "continue" statements that will push into the next loop
    ## and prevent saving that data (in cases where the post fails inclusion).
    for post, user, date, title in zip(page_posts, page_users, page_dates, page_titles):
        
        random_state += 1
        rd4ids.seed(random_state)

        post_txt = post.text
        user_txt = user.text  # WARNING: Don't use strip here bc some usernames are just spaces.
        date_txt = date.text
        title_txt = title.text

        ########################################################################
        # CLEAN/ANONYMIZE USER
        ########################################################################
        # WARNING: Do this first so each user gets a unique ID even if they
        #          don't get included.

        ## Convert to printable ASCII.
        # Using a little more caution with replacing whitespace here because the
        # number of spaces can differentiate between users (e.g., some usernames
        # are a bunch of spaces). In practice that means using \s instead of \s+
        # for regex to convert *each* whitespace character to a single space.
        user_txt = user_txt.lstrip("\n").rstrip("\n")
        user_txt = convert2ascii(user_txt)
        # If there is an @ in the username, it got changed to [email\xa0protected]
        # even though it's NOT an email. Need to get the real username, or drop
        # them because it will mess with repeated measures analyses (bc they
        # aren't same user).
        if re.search(r"\[email\s+protected\]", user.text) is not None:
            # The real username is still in the user item somewhere.
            user_txt = user.find("a").attrs["title"].split(" is offline")[0]
        user_txt = convert2ascii(user_txt, retain_whitespace_count=True)

        # Generate random user ID.
        if user_txt not in user_mapping:
            unique_user_id = generate_id(n_chars=4)
            while unique_user_id in user_mapping.values():
                unique_user_id = generate_id(n_chars=4)
            user_mapping[user_txt] = unique_user_id

        ########################################################################
        # CLEAN/PARSE DATE
        ########################################################################

        # Remove user info from date_txt.
        date_txt = date_txt.strip().split(", ", 1)[1]

        # Remove occasional date "modifier" in parenthesis.
        if "(" in date_txt and ")" in date_txt:
            date_txt, date_descriptor = date_txt.rstrip(")").split(" (", 1)
            # Remove a community dream journal focused on shared dreaming.
            if date_descriptor == "International Oneironaut Shared Dreaming Journal":
                continue
            
        # Skip recent posts marked as "today" or "yesterday" bc not worth converting.
        if "Today" in date_txt or "Yesterday" in date_txt:
            continue

        # Convert string to iso-format for standardization.
        blogdatetime = datetime.datetime.strptime(date_txt, "%m-%d-%Y at %I:%M %p")
        date_txt_iso = blogdatetime.strftime("%Y-%m-%dT%H:%M")

        # Drop posts outside desired time window.
        if blogdatetime < start_datetime or blogdatetime > end_datetime:
            continue

        ########################################################################
        # EXTRACT TAGS AND CATEGORIES
        ########################################################################
        # WARNING: Do prior to text cleaning, otherwise it messes with parsing.
        #          The post text has more than just the dream report. At the end
        #          it will ALWAYS have a "Categories" section, even if just the
        #          "Uncategorized" label. There is also an optional "Tags"
        #          section that will immediately precede the Categories section
        #          if the user included Tags.

        # Skip a few posts (~10-20) that have mutliple Tag/Category instances
        # because they are mostly garbage (e.g., multiple posts within one).
        if post_txt.count("Categories") > 1 or post_txt.count("Tags") > 1:
            continue

        # Break the post text into post, tags, and categories.
        if "Tags:" in post_txt:
            post_txt, tag_txt, cat_txt = re.split(r"Tags:|Categories", post_txt)
        else:
            post_txt, cat_txt = re.split(r"Categories", post_txt)
            tag_txt = None

        # Remove occasional thumbnails sections.
        cat_txt = cat_txt.split("Attached Thumbnails")[0]
        
        # Strip excess whitespace off everything.
        post_txt = post_txt.strip()
        cat_txt = cat_txt.strip()
        if tag_txt is not None:
            tag_txt = tag_txt.strip()

        # Extract lists for each of tags and categories.
        cats = [ c.strip().lower().replace(" ", "_") for c in cat_txt.split(", ") ]
        if tag_txt is None:
            tags = []
        else:
            tags = [ t.strip().lower().replace(" ", "_") for t in tag_txt.split(", ") ]

        # Generate custom lucidity labels that are more specific.
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
        
        ########################################################################
        # CLEAN POSTS
        ########################################################################

        # Convert to printable ASCII.
        post_txt = convert2ascii(post_txt)

        # Remove a lot of posts that start with "Originally posted by..." and
        # thus probably aren't dreams from this actual user.
        if post_txt.startswith("Originally posted"):
            continue

        # Minor text cleaning.
        post_txt = post_txt.replace("&#39;", "'")  # Replace the few stupid apostrophes.
        post_txt = post_txt.replace("&", "and")  # Replace ampersands.
        post_txt = contractions.fix(post_txt, slang=True)  # Replace contractions.
        # Reduce any sequence of 4+ consecutive characters to just 1, getting
        # rid of stuff like whoaaaaaaaaaaaa and --------------------.
        post_txt = re.sub(r"(.)\1{3,}", r"\1", post_txt, flags=re.IGNORECASE)

        # Remove some character sequences idiosyncratic to DreamViews posts.
        # Leftover block formatting tags.
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

        # Remove any amendment timestamps (~20% of posts have updates/amendments).
        # Example: Updated 12-08-2021 at 10:28 PM by 34880
        # Example: Updated 08-05-2017 at 01:09 PM by 93119 (Added Categories)
        # Example: Updated 04-20-2014 at 12:36 PM by 68865 (remembered another fragment)
        updated_re = r" Updated [0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M by [0-9]{1,5}( \(.*?\))?"
        post_txt = re.sub(updated_re, "", post_txt)

        # Redact emails.
        # Anything after an "@" is already replaced with "@[email\xa0protected]".
        # They aren't always emails so just remove (rather than replace).
        post_txt = re.sub(r"@\[email protected\]", "", post_txt)
        post_txt = re.sub(r"\S*@\S*\s?", "", post_txt)  # Just in case.

        # Redact URLS.
        post_txt = re.sub(r"https?://\S+", "[[URL]]", post_txt, flags=re.IGNORECASE)
        post_txt = re.sub(r"www\.\S+", "[[URL]]", post_txt, flags=re.IGNORECASE)

        # Check for letters.
        if re.search(r"[a-zA-Z]", post_txt) is None:
            continue

        # Check for English language.
        language = langdetect.detect(post_txt)
        if language != "en":
            continue

        # Create spaCy doc for more advanced processing.
        doc = nlp(post_txt)

        # Remove short posts.
        # Note the wordcount is calcualted prior to entity replacement for convenience.
        # This way don't have to re-"doc" the redacted text.
        n_words = sum( t.is_alpha for t in doc )
        if not (c.MIN_WORDCOUNT <= n_words <= c.MAX_WORDCOUNT):
            continue

        # Redact names, replacing with [[PERSON]].
        # Loop over entities in reverse so indices still work after replacements.
        for ent in reversed(doc.ents):
            if ent.label_ == "PERSON":
                post_txt = (post_txt[:ent.start_char]
                    + "[["+ent.label_+"]]" + post_txt[ent.end_char:])

        # Lemmatize and shuffle.
        lemmatized_text = lemmatize(doc, shuffle=True)

        ########################################################################
        # CLEAN TITLE
        ########################################################################

        # Convert to printable ASCII.
        title_txt = convert2ascii(title_txt)

        ########################################################################
        # UPDATE DICTIONARIES
        ########################################################################

        # Generate random post ID.
        unique_post_id = generate_id(n_chars=8)
        while unique_post_id in data:
            unique_post_id = generate_id(n_chars=8)

        data[unique_post_id] = {
            "user_id"     : unique_user_id,
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


################################################################################
# EXPORTING
################################################################################

# Generate a dataframe from all the posts.
df = pd.DataFrame.from_dict(data, orient="index")

# Remove any duplicated reports.
df = df.drop_duplicates(subset="post_clean", keep="first")

# Add a column that identifies the post # in sequence for a given user.
df = df.sort_values(["user_id", "timestamp"])
df.insert(1, "nth_post",
    df.groupby("user_id")["timestamp"].transform(lambda s: range(1, 1+len(s)))
)

# Remove posts beyond predetermined amount.
df = df.query(f"nth_post <= {c.MAX_POSTCOUNT}")

# Remove users who didn't survive exclusion from the user legend.
user_mapping = { username: userid for username, userid
    in user_mapping.items() if userid in df["user_id"].unique() }

# Export posts as a tsv file.
df.to_csv(export_path_posts, encoding="ascii", index=True, index_label="post_id", sep="\t")

# Export username legend as a json file.
with open(export_path_userkey, "wt", encoding="ascii") as f:
    json.dump(user_mapping, f, indent=4, sort_keys=True, ensure_ascii=False)
