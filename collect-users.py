"""Get user info for all users in the dataset.

Usernames are first extracted from the
scraped dream reports, so that has to be done first.

A few users whose special characters were converted
to ASCII won't make the list, but that's okay.
Many users don't report an info anyways.
"""
import os
import json
import tqdm
import zipfile
import requests

import pandas as pd

from bs4 import BeautifulSoup

import config as c


BASE_USER_URL = "https://www.dreamviews.com/members"


import_fname = os.path.join(c.DATA_DIR, "derivatives", "dreamviews-users_key.json")

export_fname = os.path.join(c.DATA_DIR, "source", "dreamviews-users.zip")


# get all unique usernames from dataset
with open(import_fname, "rt", encoding="ascii") as f:
    user_mappings = json.load(f)
user_list = list(user_mappings)

with requests.Session() as session:

    with zipfile.ZipFile(export_fname, mode="x", compression=zipfile.ZIP_DEFLATED) as zf:

        for user in tqdm.tqdm(user_list, desc="DreamViews users crawl"):

            url = f"{BASE_USER_URL}/{user}"
            export_singlefile_fname = f"{user}.html"

            # get page info and export as part of zipfile
            response = session.get(url)
            if response.ok and "This user has not registered" not in response.text:
                zf.writestr(export_singlefile_fname, response.content)
