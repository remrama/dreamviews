"""
Clean/reduce the raw user file.

There is a lot of likely useless user info that won't be in the final output file.
Only takes a while because of the standardized country codes.
"""

import json
import re
import zipfile

import numpy as np
import pandas as pd
import pycountry
from bs4 import BeautifulSoup
from tqdm import tqdm

import config as c

import_path_html = c.sourcedata_dir / "dreamviews-users.zip"
import_path_posts = c.raw_dir / "dreamviews-posts.tsv"
import_path_userkey = c.raw_dir / "dreamviews-users_key.json"
export_path = c.raw_dir / "dreamviews-users.tsv"

# Select which columns will be included in the output file
# WARNING: Never include "biography" which sometimes has real names
KEEP_COLUMNS = [
    "gender",
    "age",
    "country",
]

USER_ATTRIBUTES = [  # these all get lowercased and cleaned when turned into columns
    "Join Date",
    "Last Activity",
    "Wiki Contributions",
    "DJ Entries",
    "Age",
    "Country Flag:",
    "Location:",
    "Gender:",
    "LD Count:",
    "Biography:",
    "Interests:",
    "Occupation:",
    "How you found us:",
    "Total Posts",
    "Posts Per Day",
    "Total Messages",
    "Most Recent Message",
    "Last Activity",
    "Join Date",
    "Referrals",
    "Points",
    "Level",
    "Level up completed",
    "Points required",
    "Activity",
    "Activity last 30 days",
    "Activity last 7 days",
    "Points for User",
    "Points for being the Arm of the Law",
    "Points for every day since registration",
    "Points for Friends",
    "Points for posting Visitormessages",
    "Points for Referrals",
    "Points for threads",
    "Points for Threads",
    "Points for tagging threads",
    "Points for using rating",
    "Points for replies",
    "Points for sticky threads",
    "Points for Misc",
    "Dream Journal",
    "Custom",
    "Points spend in Shop",
]

# Replacements for country names that don't match lookup in pycountry
COUNTRY_REPLACEMENTS = {
    "USA": "United States",
    "Vanutau": "Vanuatu",
    "BosniaHerzegovina": "Bosnia and Herzegovina",
    "UnitedStatesMinorOutlying": "United States Minor Outlying Islands",
    "Catalonia": "Spain",
    "BasqueCountry": "Spain",
    "SouthKorea": "Korea, Republic of",
    "NorthKorea": "Korea, Republic of",
    "Aland": "Åland Islands",
    "Reunion": "Réunion",
    "Wales": "United Kingdom",
    "TrinidadTobago": "Trinidad and Tobago",
    "Norfolk": "United Kingdom",
    "Taiwan": "Taiwan, Province of China",
    "StKittsNevis": "Saint Kitts and Nevis",
    "Indonezia": "Indonesia",
    "HeardIslandandMcDonald": "Heard Island and McDonald Islands",
    "Iran": "Iran, Islamic Republic of",
    "Turkey": "Türkiye",
}

# Load extracted posts data to get the list of users who survived filtering
df = c.load_dreamviews_posts()
surviving_user_ids = df["user_id"].unique()

# Load in key to get unique anonymous user IDs from the raw IDs
with open(import_path_userkey, "rt", encoding="utf-8") as f:
    user_mapping = json.load(f)

data = {}
# Loop over all the raw html files and get user info from each
with zipfile.ZipFile(import_path_html, mode="r") as zf:
    filenames = zf.namelist()
    for fn in tqdm(filenames, desc="Extracting users"):
        # get the original username (raw ID)
        username = fn[:-5]  # remove ".html" off the end
        if username not in user_mapping:
            # Skip users who aren't in the user mapping, which means they are also not
            # in the posts sourcedata. This shouldn't really happen, and only happens
            # once because of some weirdness in the data collection process.
            # The sourcedata users file was created with a few merges of prior scrapes.
            continue
        # get the anonymized username (user ID)
        user_id = user_mapping[username]
        if user_id not in surviving_user_ids:
            continue  # skip users whose posts didn't survive filtering
        html = zf.read(fn)  # read in the html file
        soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
        ## All good info is within <dt> tags. But not all users have all <dt> tags,
        ## and there are some unwanted <dt> tags. So grab all the <dt> tags and search
        ## for those desired. If a user doesn't have them, it just won't get added
        ## All <dt> tags are immediately followed by a <dd> tag that has the response info
        all_dt_tags = soup.find_all("dt")
        user_data = {}
        for dt_tag in all_dt_tags:
            header = dt_tag.get_text()
            if header in USER_ATTRIBUTES:
                response = dt_tag.find_next("dd").get_text(separator=" ", strip=True)
                # Replace any numerical commas
                if len(response.replace(",", "")) == sum([char.isdigit() for char in response]):
                    response = response.replace(",", "")
                # Clean up the attribute name before using it as a dictionary key
                attr_key = header.rstrip(":").replace(" ", "_").lower()
                user_data[attr_key] = response
        if user_data:
            data[user_id] = user_data

# Aggregate user data into a dataframe
df = pd.DataFrame.from_dict(data, orient="index")

# Convert join date to year-month-day other date columns (last_activity and
# most_recent_message) could also be converted but they aren't that useful and
# sometimes have they "Today/Yesterday" in them. Not keeping any of them anyways
# so don't worry about converting
df["join_date"] = pd.to_datetime(df["join_date"], format="%m-%d-%Y").dt.strftime("%Y-%m-%d")

df = df.sort_values(["join_date", "last_activity"])


# Get country codes
def get_country_code(x):
    # Make minor adjustments before looking up in pycountry
    if pd.isna(x):
        return pd.NA
    elif x in COUNTRY_REPLACEMENTS:
        x = COUNTRY_REPLACEMENTS[x]
    else:
        # Add spaces to multiword countries
        x = re.sub(r"(\w)([A-Z])", r"\1 \2", x)
    # Look up country code
    country = pycountry.countries.get(name=x)
    if country is None:
        # Try fuzzy search, which will raise error if nothing found
        possible_codes = pycountry.countries.search_fuzzy(x)
        if len(possible_codes) > 1:
            raise Warning(f"found >1 fuzzy codes for {x}. Inspect!")
        else:
            country = possible_codes[0]
    return country.alpha_3


df["country"] = df["country_flag"].apply(get_country_code)
df["gender"] = df["gender"].str.lower()
df["age"] = df["age"].astype("Int64")

# Bin age into categories for de-identification purposes
AGE_BINS = [18, 25, 35, 45, 55, 65, np.inf]
min_age = AGE_BINS[0]
assert df["age"].dropna().ge(min_age).all(), f"Didn't expect any reported ages under {min_age}."
age_labels = [f"[{left}, {right})" for left, right in zip(AGE_BINS[:-1], AGE_BINS[1:], strict=True)]
df["age"] = pd.cut(df["age"], bins=AGE_BINS, labels=age_labels, right=False, include_lowest=True)

# Add empty rows for users who survived filtering but had no extracted info
missing_user_ids = set(surviving_user_ids) - set(df.index)
if missing_user_ids:
    df.loc[missing_user_ids] = pd.NA

# Export
df[KEEP_COLUMNS].to_csv(
    export_path, encoding="ascii", index_label="user_id", na_rep="n/a", sep="\t"
)
