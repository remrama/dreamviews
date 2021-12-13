"""
Cleaning/reducing the raw user file.

There is a lot of likely useless user info
that won't be in the final output file.

Main time consumption is getting standardized country codes.
"""
import os
import re
import tqdm
import json
import zipfile
import datetime
import pycountry
import pandas as pd
from bs4 import BeautifulSoup
import config as c


export_fname = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-users.tsv")

import_fname_html = os.path.join(c.DATA_DIR, "source", "dreamviews-users.zip")
import_fname_user_key = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-users_key.json")


# select which columns survive to final output file
KEEP_COLUMNS = [ ## DONT include "biography" which sometimes has real names
    "user_id",
    "gender",
    "age",
    "country",
]

USER_ATTRIBUTES = [ # these all get lowercased and cleaned when turned into columns
    "Join Date", "Last Activity", "Wiki Contributions",
    "DJ Entries", "Age", "Country Flag:", "Location:",
    "Gender:", "LD Count:", "Biography:", "Interests:",
    "Occupation:", "How you found us:", "Total Posts",
    "Posts Per Day", "Total Messages", "Most Recent Message",
    "Last Activity", "Join Date", "Referrals", "Points",
    "Level", "Level up completed", "Points required",
    "Activity", "Activity last 30 days", "Activity last 7 days",
    "Points for User", "Points for being the Arm of the Law",
    "Points for every day since registration",
    "Points for Friends", "Points for posting Visitormessages",
    "Points for Referrals", "Points for threads",
    "Points for Threads", "Points for tagging threads",
    "Points for using rating", "Points for replies",
    "Points for sticky threads", "Points for Misc",
    "Dream Journal", "Custom", "Points spend in Shop"
]





# Load in key to get unique anonymous user IDs from the raw IDs.
with open(import_fname_user_key, "rt", encoding="utf-8") as f:
    user_raw2id_key = json.load(f)


# Loop over all the raw html files and get user info from each.

with zipfile.ZipFile(import_fname_html, mode="r") as zf:
    fnames = zf.namelist()
    all_user_data = {}
    for fn in tqdm.tqdm(fnames, desc="parsing html user pages"):
        # get the anonymized username
        username = fn[:-5] # remove ".html" off the end
        user_id = user_raw2id_key[username]
        html = zf.read(fn) # read in the html file
        soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
        ### All good info is within <dt> tags.
        ### But not all users have all <dt> tags,
        ### and there are some unwanted <dt> tags.
        ### So grab all the <dt> tags and search
        ### for those desired. If a user doesn't
        ### have them, it just won't get added.
        ###
        ### All <dt> tags are immediately followed
        ### by a <dd> tag that has the response info.
        all_dt_tags = soup.find_all("dt")
        single_user_data = {}
        for dt_tag in all_dt_tags:
            header = dt_tag.get_text()
            if header in USER_ATTRIBUTES:
                response = dt_tag.find_next("dd").get_text(separator=" ", strip=True)
                # replace commas if it's in a number
                if len(response.replace(",", "")) == sum([ char.isdigit() for char in response ]):
                    response = response.replace(",", "")
                # minor cleanup for the attribute name before using it as a dict key
                attr_key = header.rstrip(":").replace(" ", "_").lower()
                single_user_data[attr_key] = response

        if single_user_data:
            all_user_data[user_id] = single_user_data



# Aggregate user data into a dataframe.
df = pd.DataFrame.from_dict(all_user_data, orient="index")

# Convert timestamps to iso format for consistency with other files.
dayonly_format   = c.BLOG_TIMESTAMP_FORMAT.split(" at ")[0]
today            = datetime.datetime.strptime(c.DATA_COLLECTION_DATE, "%Y-%m-%d").date()
yesterday        = today - datetime.timedelta(days=1)
today_string     = today.strftime(dayonly_format)
yesterday_string = yesterday.strftime(dayonly_format)
for col in ["join_date", "last_activity", "most_recent_message"]:
    df[col] = df[col].str.replace("Today", today_string
                    ).str.replace("Yesterday", yesterday_string)
df["join_date"] = pd.to_datetime(df["join_date"],
    format="%m-%d-%Y").dt.strftime("%Y-%m-%d")
df["last_activity"] = pd.to_datetime(df["last_activity"],
    format="%m-%d-%Y %H:%M %p").dt.strftime("%Y-%m-%dT%H:%M")
df["most_recent_message"] = pd.to_datetime(df["most_recent_message"],
    format="%m-%d-%Y %H:%M %p").dt.strftime("%Y-%m-%dT%H:%M")

df.sort_values(["join_date", "last_activity"], inplace=True)


###################################################
###############  Get country codes  ###############
###################################################

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
        # add spaces to multiword countries (for pycountry lookup)
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


tqdm.tqdm.pandas(desc="country code lookups")
df["country"] = df["country_flag"].progress_apply(get_country_code)
df["gender"] = df["gender"].str.lower()




# Export.

df[KEEP_COLUMNS].to_csv(export_fname, encoding="ascii",
    sep="\t", na_rep="NA", index=False)
