"""
Convert raw DreamViews dream journal and user HTML pages into a single zipfile containing two tsv files:
- clean.zip
    - posts.tsv holding dream reports and their attributes
    - users.tsv holding usernames and their attributes

Minimal cleaning is performed:
    - ensure universal utf-8 encoding
    - remove HTML formatting tags
    - remove non-english posts
    - remove spam (super short posts, super long posts, users with super high post count)
    - restrict date
    - extract post attributes (username, title, tags, categories)
    - generate a unique identifier for each user
    - generate a unique identifier for each post
    - generate convenience lucidity and nightmare labels based on post categories
"""

import argparse
import csv
import io
import random
import re
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pycountry
import tqdm
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory


VERSION = 1.0
RANDOM_SEED = 32
MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000  # limiting the number of posts a single user can have
MIN_DATE = "2010-01-01T00:00+01:00"  # ISO format or None
MAX_DATE = "2020-12-31T23:59+01:00"  # ISO format or None
HTML_ENCODING = "windows-1252"
USERS_COLUMNS = ["age", "gender", "country"]
POSTS_COLUMNS = [
    "post_id", "user", "date", "title", "modifier", "categories", "tags", "lucidity", "nightmare", "dream"
]
POSTS_ARCNAME = "posts.tsv"
USERS_ARCNAME = "users.tsv"

DetectorFactory.seed = 0
rd = random.Random()
rd.seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath", type=str, required=False, help="Filepath to raw HTML zip")
args = parser.parse_args()

filepath = args.filepath

if filepath is None:
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Choose file to clean")
    if not filepath:
        sys.exit()

import_filepath = Path(filepath).expanduser()
export_filepath = import_filepath.with_stem(f"dreamviews_clean-v{VERSION}")

utc = ZoneInfo("UTC")
cet = ZoneInfo("CET")

min_dt = MIN_DATE if MIN_DATE is None else datetime.fromisoformat(MIN_DATE).astimezone(cet)
max_dt = MAX_DATE if MAX_DATE is None else datetime.fromisoformat(MAX_DATE).astimezone(cet)

# Get string representations of "Today" and "Yesterday" dates, relative to date of scraping
# DreamViews timestamps are in GMT+1 timezone (aka UTC+1, aka CET)
scraped_timestamp = int(import_filepath.stem.split("-")[1])
scraped_today_utc = datetime.fromtimestamp(scraped_timestamp, tz=utc)
scraped_today_cet = scraped_today_utc.astimezone(cet)
scraped_yesterday_cet = scraped_today_cet - timedelta(days=1)
today_str = scraped_today_cet.strftime("%m-%d-%Y")
yesterday_str = scraped_yesterday_cet.strftime("%m-%d-%Y")




def generate_id(n_chars):
    """Return a random alpha-numeric character sequence (str) of length n_chars.
    Always starts with a capital letter, to ensure categorical interpretation later.
    """
    _gen_str = lambda nchars: uuid.UUID(int=rd.getrandbits(128)).hex[:n_chars]
    id_string = _gen_str(n_chars)
    while not id_string[0].isalpha():
        id_string = _gen_str(n_chars)
    return id_string.upper()


with zipfile.ZipFile(import_filepath, mode="r") as zf:
    arcnames = zf.namelist()
    post_arcnames = [an for an in arcnames if an.startswith("posts/")]
    user_arcnames = [an for an in arcnames if an.startswith("users/")]
    # First, generate unique IDs for each username
    user2id_map = {Path(an).stem: generate_id(n_chars=4) for an in user_arcnames}
    assert len(user2id_map) == len(set(user2id_map.values())), "Retry for unique username IDs"

    ################################################################################################
    # Posts
    ################################################################################################
    posts_data = []
    for an in tqdm.tqdm(post_arcnames, desc="DreamViews cleaning posts"):
        content = zf.read(an)
        # Extract the post, user, date, and title, from each post of the current HTML file
        soup = BeautifulSoup(content, "html.parser", from_encoding=HTML_ENCODING)
        entries = soup.find_all("div", class_="wrapper")
        for e in entries:

            # Extract title
            title = e.find("a", class_="blogtitle").get_text()

            # Extract user
            username = e.find("div", class_="popupmenu memberaction").find("a")["href"].split("/")[-2]
            # user = e.find("div", class_="popupmenu memberaction").get_text(strip=True)
            # blog_url = e.find("a", class_="blogtitle")["href"]
            # _, user_url, title_url = urlparse(blog_url).path.strip("/").split("/")
            # Convert username to custom identifier
            user = user2id_map[username]

            # Extract date and convert to ISO format
            # _, user_str, date_and_modifier = e.find("div", class_="blog_date").stripped_strings
            date_text = e.find("div", class_="blog_date").get_text().split(", ", 1)[1]
            # Replace Today and Yesterday strings with UTC+1 date ISO strings
            date_str = date_text.replace("Today", today_str).replace("Yesterday", yesterday_str)
            date_str = re.search(r"\d\d-\d\d-\d{4} at \d\d:\d\d [AP]M", date_str).group()
            # date_str = re.search(r"[0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M", date_text).group()
            if (modifier := re.search(r"\((.+)\)", date_text)) is not None:
                modifier = modifier.group(1)  # Occasional date "modifier" in parenthesis
                # if modifier == "International Oneironaut Shared Dreaming Journal":
                #     continue  # Skip a community dream journal focused on shared dreaming
            date_dt = datetime.strptime(date_str, "%m-%d-%Y at %I:%M %p").replace(tzinfo=cet)
            date = date_dt.isoformat(timespec="minutes")

            # Extract categories
            categories = e.find("dd").get_text(strip=True)

            # Extract tags
            meta = e.find("div", class_="blogmeta")
            if (tags := meta.find("div", class_="tags")) is not None:
                tags = tags.find("span").get_text(strip=True)

            # Extract post (i.e., dream report)
            dream = e.find("blockquote").get_text().strip()

            # ## This should no longer be necessary?
            # # Redact emails
            # # Anything after an "@" is already replaced with "@[email\xa0protected]".
            # # They aren't always emails so just remove (rather than replace).
            # dream = re.sub(r"@\[email protected\]", "", dream)
            # dream = re.sub(r"\S*@\S*\s?", "", dream)  # Just in case.
            # # ## This should no longer be necessary
            # # Redact URLS
            # dream = re.sub(r"https?://\S+", "[[URL]]", dream, flags=re.IGNORECASE)
            # dream = re.sub(r"www\.\S+", "[[URL]]", dream, flags=re.IGNORECASE)
            # ## This should no longer be necessary
            # # Remove any amendment timestamps (~20% of posts have updates/amendments).
            # # Example: Updated 12-08-2021 at 10:28 PM by 34880
            # # Example: Updated 08-05-2017 at 01:09 PM by 93119 (Added Categories)
            # # Example: Updated 04-20-2014 at 12:36 PM by 68865 (remembered another fragment)
            # updated_re = r" Updated [0-9]{2}-[0-9]{2}-[0-9]{4} at [0-9]{2}:[0-9]{2} [AP]M by [0-9]{1,5}( \(.*?\))?"
            # dream = re.sub(updated_re, "", dream)

            # Remove some character sequences idiosyncratic to DreamViews posts.
            # Leftover block formatting tags.
            dream = re.sub(r"\[(/?INDENT|/?RIGHT|/?CENTER|/?B)\]", "", dream, flags=re.IGNORECASE)
            # These ones never have a = preceding them.
            dream = re.sub(r"\[/?(INDENT|RIGHT|CENTER|B|I|U|HR|IMG|LINK_TO_ANCHOR|SARCASM|DREAM LOGIC)\]", "", dream, flags=re.IGNORECASE)
            # These need some leway as to what comes after because sometimes there's stuff there.
            dream = re.sub(r"\[/?COLOR.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[/?SIZE.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[/?FONT.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[/?QUOTE.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[/?SPOILER.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[/?URL.*?\]", "", dream, flags=re.IGNORECASE).rstrip()
            dream = re.sub(r"\[ATTACH=CONFIG\][0-9]*\[/ATTACH\]", "", dream, flags=re.IGNORECASE)

            # Generate random post ID
            post_id = generate_id(n_chars=8)
            # while unique_post_id in data:
            #     post_id = generate_id(n_chars=8)

            ####################################################################
            # Filtering
            ####################################################################

            if min_dt is not None and date_dt <= min_dt:
                continue  # Filter out old posts
            if max_dt is not None and date_dt >= max_dt:
                continue  # Filter out recent posts

            n_words = len(re.findall(r"\w+", dream))
            if not (MIN_WORDCOUNT <= n_words <= MAX_WORDCOUNT):
                continue  # Filter out short and long posts

            language = detect(dream)
            if language != "en":
                continue  # Filter out non-English posts

            if dream.startswith("Originally posted"):
                continue  # Filter out a lot of posts that start with "Originally posted by..."
                          # and thus probably aren't dreams from this actual user

            ## This should no longer be necessary
            # Remove occasional thumbnails sections
            assert not "Attached Thumbnails" in categories

            ####################################################################
            # Convenience labels
            ####################################################################
            # Generate custom lucidity labels that are more specific.
            # # Identify if the post was a nightmare.
            # if (nightmare := re.search(r"\bnightmare\b", categories)) is not None:
            #     assert nightmare.group() == "nightmare"
            nightmare = "nightmare" in categories.split(",")
            nonlucid = "non-lucid" in categories.split(",")
            lucid = "lucid" in categories.split(",")
            if lucid and nonlucid:
                lucidity = "ambiguous"
            elif lucid:
                lucidity = "lucid"
            elif nonlucid:
                lucidity = "non-lucid"
            else:
                lucidity = "unspecified"

            posts_data_row = [
                post_id, user, date, title, modifier, categories, tags, lucidity, nightmare, dream
            ]
            posts_data.append(posts_data_row)

    ############################################################################
    # Users
    ############################################################################
    unique_users = {row[1] for row in data}
    unique_usernames = {k for k, v in user2id_map.items() if v in unique_users}
    user_arcnames = {an for an in user_arcnames if an.split("/")[1].split(".")[0] in unique_usernames}
    users_data = []
    for an in tqdm.tqdm(user_arcnames, desc="DreamViews cleaning users"):
        content = zf.read(an)
        soup = BeautifulSoup(content, "html.parser", from_encoding=HTML_ENCODING)
        # Age
        age = soup.find("dt", string="Date of Birth")
        if age is not None:
            age = age.find_next("dd").get_text()
            age = re.search(r"\(([0-9]+)\)", age)
            if age is not None:
                age = int(age.group(1))
        # Gender
        gender = soup.find("dt", string="Gender:")
        if gender is not None:
            gender = gender.find_next("dd").get_text()
        # Country
        country = soup.find("dt", string="Country Flag:")
        if country is not None:
            country = country.find_next("dd").get_text()
            # Get ISO 3166-1 alpha-2 country code
            country_replacements = {
                "Vanutau" : "Vanuatu",
                "BosniaHerzegovina" : "Bosnia and Herzegovina",
                "Catalonia" : "Spain",
                "BasqueCountry" : "Spain",
                "TrinidadTobago" : "Trinidad and Tobago",
                "StKittsNevis" : "Saint Kitts and Nevis",
                "Indonezia" : "Indonesia",
                "HeardIslandandMcDonald" : "Heard Island and McDonald Islands",
            }
            country = country_replacements.get(country, country)
            # country = " ".join(re.split(r"(?<=[a-z])(?<!Mc)(?=[A-Z])", country))
            country = re.sub(r"([a-z])(?<!Mc)([A-Z])", r"\1 \2", country)
            country_ = pycountry.countries.get(name=country)
            if country_ is None:
                try:
                    country_ = pycountry.countries.lookup(country)
                except LookupError:
                    try:
                        fuzzy_countries = pycountry.countries.search_fuzzy(country)
                        country_ = fuzzy_countries[0]
                    except LookupError:
                        try:
                            subdivision = pycountry.subdivisions.lookup(country)
                            country_ = pycountry.countries.get(alpha_2=subdivision.country_code)
                        except LookupError:
                            continue
                finally:
                    if country_ is None:
                        raise ValueError(f"Country {country} could not be identified")
            country = country_.alpha_2
        # Save results to list
        users_data_row = [age, gender, country]
        users_data.append(users_data_row)


import pandas as pd
df = pd.DataFrame(data, columns=columns).set_index("post_id")
# df["title"].apply(lambda s: s.encode("utf-8"))
# df.replace({None: ""}).map(lambda s: s.encode("utf-8"))

# Remove any duplicated reports
df = df.drop_duplicates(subset="dream", keep="first")

# Add a column that identifies the post # in sequence for a given user
df = df.sort_values(["user", "date"])
df.insert(1, "nth_post",
    df.groupby("user")["date"].transform(lambda s: range(1, 1+len(s)))
)

# Remove posts beyond predetermined amount.
df = df.query(f"nth_post <= {MAX_POSTCOUNT}")


################################################################################
# Export
################################################################################

posts_info = {"data": posts_data, "columns": POSTS_COLUMNS, "arcname": POSTS_ARCNAME}
users_info = {"data": users_data, "columns": USERS_COLUMNS, "arcname": USERS_ARCNAME}

with zipfile.ZipFile(export_filepath, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fileinfo in [posts_info, users_info]:
        with tempfile.TemporaryFile(mode="w+t", newline="", encoding="utf-8") as tf:
            writer = csv.writer(tf, delimiter="\t", quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(fileinfo["columns"])
            writer.writerows(fileinfo["data"])
            tf.flush()
            tf.seek(0)
            zf.writestr(fileinfo["arcname"], tf.read())
