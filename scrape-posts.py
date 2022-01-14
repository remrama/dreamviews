"""Scrape all DreamViews dream journal entries,
saving the raw html files into a zipfile for later processing.
"""
import os
import tqdm
import zipfile
import requests

from bs4 import BeautifulSoup

import config as c


DREAMVIEWS_URL = "https://www.dreamviews.com/blogs/recent-entries"

export_fname = os.path.join(c.DATA_DIR, "source", "dreamviews-posts.zip")


with requests.Session() as session:

    # Get the total number of pages by loading the 
    # first (i.e., most recent) dream journal page,
    # then finding the "Last" page link and extracting
    # the number of pages.
    page = session.get(DREAMVIEWS_URL).text
    soup = BeautifulSoup(page, "html.parser")
    lasturl = soup.find("span", class_="first_last").find("a")["href"]
    lastnum = lasturl.rstrip(".html").split("/index")[1]
    assert lastnum.isdigit()
    n_pages = int(lastnum)

    # loop over all dream journal pages
    with zipfile.ZipFile(export_fname, mode="x", compression=zipfile.ZIP_DEFLATED) as zf:

        for i in tqdm.trange(1, n_pages+1, desc="DreamViews posts crawl"):

            url = f"{DREAMVIEWS_URL}/index{i}.html"
            export_singlefile_fname = f"index{i:04d}.html"

            # get page info and export as part of zipfile
            r = session.get(url)
            zf.writestr(export_singlefile_fname, r.content)