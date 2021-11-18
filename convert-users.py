"""
Handle the raw user html.

There is a ton of likely useless user info,
that I want to keep, but also don't want to clutter
up the rest of the more useable stuff.
So save out 2 files. A complete and restrictured tsv.

Each file has one row for each (anonymized) user,
but the FULL one has all the gunk in it.
"""
import os
import json
import tqdm
import zipfile

import pandas as pd

from bs4 import BeautifulSoup

import config as c


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

# take out the biography bc it sometimes contains real names
USER_ATTRIBUTES.remove("Biography:")


import_fname_html = os.path.join(c.DATA_DIR, "source", "dreamviews-users.zip")
import_fname_user_key = os.path.join(c.DATA_DIR, "derivatives", "users-anon_key.json")

export_fname = os.path.join(c.DATA_DIR, "derivatives", "users-raw.tsv")


with open(import_fname_user_key, "rt", encoding="utf-8") as f:
    user_key = json.load(f)



with zipfile.ZipFile(import_fname_html, mode="r") as zf:
    
    fnames = zf.namelist()
    
    user_attribute_data = {}

    for fn in tqdm.tqdm(fnames, desc="parsing html user pages"):
        
        # get the anonymized username
        username = fn[:-5] # remove ".html" off the end
        user_id = user_key[username]

        # read in the html file
        html = zf.read(fn)

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

        user_data = {}
        for dt_tag in all_dt_tags:
            header = dt_tag.get_text()
            if header in USER_ATTRIBUTES:
                response = dt_tag.find_next("dd").get_text(separator=" ", strip=True)
                # replace commas if it's in a number
                if len(response.replace(",", "")) == sum([ char.isdigit() for char in response ]):
                    response = response.replace(",", "")
                # minor cleanup for the attribute name before using it as a dict key
                attr_key = header.rstrip(":").replace(" ", "_").lower()
                user_data[attr_key] = response

        if user_data:
            user_attribute_data[user_id] = user_data


# aggregate into dataframe
df = pd.DataFrame.from_dict(user_attribute_data, orient="index"
    ).rename(columns={"country_flag":"country"})

# convert timestamps to iso format for consistency with other files
df["join_date"] = pd.to_datetime(df["join_date"], format="%m-%d-%Y").dt.strftime("%Y-%m-%d")
df["last_activity"] = pd.to_datetime(df["last_activity"], format="%m-%d-%Y %H:%M %p"
    ).dt.strftime("%Y-%m-%dT%H:%M")
df["most_recent_message"] = pd.to_datetime(df["most_recent_message"], format="%m-%d-%Y %H:%M %p"
    ).dt.strftime("%Y-%m-%dT%H:%M")

df.sort_values(["join_date", "last_activity"], inplace=True)

df.to_csv(export_fname, sep="\t", encoding="utf-8",
    na_rep="NA", index=True, index_label="user_id")
