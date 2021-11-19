"""
Cleaning/reducing the raw user file.

1. convert country names to standardized country codes
2. reduce to only users that pass more stringent cleaning
3. reduce columns
"""
import os
import re
import tqdm

import pandas as pd

import pycountry

import config as c

tqdm.tqdm.pandas()

import_fname = os.path.join(c.DATA_DIR, "derivatives", "users-raw.tsv")
export_fname = os.path.join(c.DATA_DIR, "derivatives", "users-clean.tsv")

import_fname_posts = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")

df = pd.read_csv(import_fname, sep="\t", encoding="utf-8")

retained_users = pd.read_csv(import_fname_posts,
        sep="\t", encoding="utf-8",
        usecols=["user_id"], squeeze=True
    ).unique()

df = df[ df["user_id"].isin(retained_users) ].reset_index(drop=True)


# load in the clean posts file to get a list of all users
# that have a post remaining in the dataset after cleaning

COUNTRY_REPLACEMENTS = {
    "USA" : "United States",
    "Vanutau" : "Vanuatu",
    "BosniaHerzegovina" : "Bosnia and Herzegovina",
    "UnitedStatesMinorOutlying" : "United States Minor Outlying Islands",
    "Catalonia" : "Spain",
    "BasqueCountry" : "Spain",
    "SouthKorea" : "Korea, Republic of",
    "NorthKorea" : "Korea, Republic of",
    "Aland" : "Åland Islands",
    "Reunion" : "Réunion",
    "Wales" : "United Kingdom",
    "TrinidadTobago" : "Trinidad and Tobago",
    "Norfolk" : "United Kingdom",
    "Taiwan" : "Taiwan, Province of China",
    "StKittsNevis" : "Saint Kitts and Nevis",
    "Indonezia" : "Indonesia",
    "HeardIslandandMcDonald" : "Heard Island and McDonald Islands",
    "Iran" : "Iran, Islamic Republic of",
}


def get_country_code(x):

    # minor string adjustments before lookup in pycountry
    if pd.isna(x):
        return pd.NA
    elif x in COUNTRY_REPLACEMENTS:
        x = COUNTRY_REPLACEMENTS[x]
    else:
        # add spaces to multiword countries (for pycountry lookup
        x = re.sub(r"(\w)([A-Z])", r"\1 \2", x)

    # look up code
    country = pycountry.countries.get(name=x)
    if country is None:
        # try fuzzy search, which will raise error if nothing found
        possible_codes = pycountry.countries.search_fuzzy(x)
        if len(possible_codes) > 1:
            raise Warning(f"found >1 fuzzy codes for {x}. Inspect!")
        else:
            country = possible_codes[0]
    return country.alpha_3


df["country"] = df["country_flag"].progress_apply(get_country_code)
df["gender"] = df["gender"].str.lower()

KEEP_COLUMNS = [
    "user_id",
    "gender",
    "age",
    "country",
]

df[KEEP_COLUMNS].to_csv(export_fname,
    sep="\t", encoding="utf-8",
    na_rep="NA", index=False)
