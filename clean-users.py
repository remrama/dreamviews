"""
Clean/reduce the raw user file.

There is a lot of likely useless user info that won't be in the final output file.
Only takes a while because of the standardized country codes.
"""
import json
import re
import zipfile

from bs4 import BeautifulSoup
import pandas as pd
import pycountry
import tqdm

import config as c


export_path = c.DATA_DIR / "derivatives" / "dreamviews-users.tsv"
import_path_html = c.DATA_DIR / "source" / "dreamviews-users.zip"
import_path_userkey = c.DATA_DIR / "source" / "dreamviews-users_key.json"

# Select which columns will be included in the output file.
# WARNING: Never include "biography" which sometimes has real names.
keep_columns = [
    "gender",
    "age",
    "country",
]

user_attributes = [ # these all get lowercased and cleaned when turned into columns
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
with open(import_path_userkey, "rt", encoding="utf-8") as f:
    user_mapping = json.load(f)

data = {}
# Loop over all the raw html files and get user info from each.
with zipfile.ZipFile(import_path_html, mode="r") as zf:
    filenames = zf.namelist()
    for fn in tqdm.tqdm(filenames, desc="DreamViews user cleaner"):
        # get the anonymized username
        username = fn[:-5] # remove ".html" off the end
        user_id = user_mapping[username]
        html = zf.read(fn) # read in the html file
        soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
        ## All good info is within <dt> tags. But not all users have all <dt> tags,
        ## and there are some unwanted <dt> tags. So grab all the <dt> tags and search
        ## for those desired. If a user doesn't have them, it just won't get added.
        ## All <dt> tags are immediately followed by a <dd> tag that has the response info.
        all_dt_tags = soup.find_all("dt")
        user_data = {}
        for dt_tag in all_dt_tags:
            header = dt_tag.get_text()
            if header in user_attributes:
                response = dt_tag.find_next("dd").get_text(separator=" ", strip=True)
                # Replace any numerical commas.
                if len(response.replace(",", "")) == sum([ char.isdigit() for char in response ]):
                    response = response.replace(",", "")
                # Clean up the attribute name before using it as a dictionary key.
                attr_key = header.rstrip(":").replace(" ", "_").lower()
                user_data[attr_key] = response
        if user_data:
            data[user_id] = user_data

# Aggregate user data into a dataframe.
df = pd.DataFrame.from_dict(all_user_data, orient="index")

# Convert join date to year-month-day other date columns (last_activity and
# most_recent_message) could also be converted but they aren't that useful and
# sometimes have they "Today/Yesterday" in them. Not keeping any of them anyways
# so don't worry about converting.
df["join_date"] = pd.to_datetime(df["join_date"],
    format="%m-%d-%Y").dt.strftime("%Y-%m-%d")

df = df.sort_values(["join_date", "last_activity"])

# Get country codes.
country_replacements = {
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
    # Make minor adjustments before looking up in pycountry.
    if pd.isna(x):
        return pd.NA
    elif x in country_replacements:
        x = country_replacements[x]
    else:
        # Add spaces to multiword countries.
        x = re.sub(r"(\w)([A-Z])", r"\1 \2", x)
    # Look up country code.
    country = pycountry.countries.get(name=x)
    if country is None:
        # Try fuzzy search, which will raise error if nothing found.
        possible_codes = pycountry.countries.search_fuzzy(x)
        if len(possible_codes) > 1:
            raise Warning(f"found >1 fuzzy codes for {x}. Inspect!")
        else:
            country = possible_codes[0]
    return country.alpha_3

tqdm.tqdm.pandas(desc="DreamViews user cleaner - country code lookups")
df["country"] = df["country_flag"].progress_apply(get_country_code)
df["gender"] = df["gender"].str.lower()

# Export.
df[keep_columns].to_csv(export_path, encoding="ascii", index_label="user_id", na_rep="NA", sep="\t")
