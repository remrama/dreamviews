"""
Scrape user profiles.

Dream journal posts have to be scraped first because this searches for the
usernames of those who posted to dream journals.

A few users whose special characters were converted to ASCII during preprocessing
won't make the list, but that's okay. Many users don't report any info anyways.
"""

import json
import zipfile

import requests
from tqdm import tqdm

import config as c

import_path = c.derivatives_dir / "dreamviews-users.json"
export_path = c.sourcedata_dir / "dreamviews-users.zip"

BASE_USER_URL = "https://www.dreamviews.com/members"

# Get all unique usernames from dataset
with open(import_path, "rt", encoding="ascii") as f:
    user_mappings = json.load(f)
user_list = list(user_mappings)

with (
    requests.Session() as session,
    zipfile.ZipFile(export_path, mode="x", compression=zipfile.ZIP_DEFLATED) as zf,
):
    # Loop over all the users and try to grab each user profile, skipping failures
    for user in tqdm(user_list, desc="Scraping users"):
        url = f"{BASE_USER_URL}/{user}"
        export_name = f"{user}.html"
        response = session.get(url)
        if response.ok and "This user has not registered" not in response.text:
            zf.writestr(export_name, response.content)
