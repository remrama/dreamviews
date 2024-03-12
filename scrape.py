"""
Scrape raw HTML DreamViews dream journal posts and user profiles.
"""

import argparse
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from tkinter import Tk, filedialog
from urllib.parse import urlparse

import requests
import tqdm
from bs4 import BeautifulSoup


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sleep", type=float, default=0.2, help="Sleep after each request")
parser.add_argument("-d", "--directory", type=str, required=False, help="File output location")
parser.add_argument("-t", "--test", type=int, required=False, help="Scrape a subset of post pages")
args = parser.parse_args()

directory = args.directory
sleep_ = args.sleep
test = args.test

if directory is None:
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Choose directory for file output")
    if not directory:
        sys.exit()

timestamp = int(datetime.now(timezone.utc).timestamp())
parent = Path(directory).expanduser()
filepath = parent / f"dreamviews_raw-{timestamp}.zip"
assert parent.is_dir(), f"{parent} must be an existing directory"
assert filepath.suffix == ".zip", f"{filepath} must end with .zip"

POSTS_BASE_URL = "https://www.dreamviews.com/blogs/recent-entries"
USERS_BASE_URL = "https://www.dreamviews.com/members"
HTML_ENCODING = "windows-1252"  # requests incorrectly identifies it as ISO-8859-1
users_list_url = f"{USERS_BASE_URL}/list/"
href_filter = re.compile(USERS_BASE_URL)
scraped_user_urls = set()

with requests.Session() as session:

    # Get the total number of dream journal pages to be scraped
    response = session.get(POSTS_BASE_URL)
    soup = BeautifulSoup(response.content, "html.parser", from_encoding=HTML_ENCODING)
    last_url = soup.find("a", title=re.compile("Last Page - Results"))["href"]
    last_index = urlparse(last_url).path.split("/")[-1].lstrip("index").rstrip(".html")
    n_total_pages = int(last_index) if test is None else test

    # Loop over all dream journal pages and save each one as an HTML file
    with zipfile.ZipFile(filepath, mode="x", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in tqdm.trange(n_total_pages, desc="DreamViews scraper"):

            # Save this dream journal page
            url = f"{POSTS_BASE_URL}/index{i+1}.html"
            arcname = urlparse(url).path.replace("/blogs/recent-entries", "posts")
            response = session.get(url)
            sleep(sleep_)
            zf.writestr(arcname, response.content)

            # Save any unsaved user profile pages from unique users on this page
            soup = BeautifulSoup(response.content, "html.parser", from_encoding=HTML_ENCODING)
            page_user_urls = {tag["href"] for tag in soup.find_all("a", href=href_filter)}
            for url in page_user_urls:
                if url not in scraped_user_urls and url != users_list_url:
                    # Save this user profile page
                    response = session.get(url)
                    arcname = urlparse(url).path.replace("/members", "users").rstrip("/") + ".html"
                    sleep(sleep_)
                    zf.writestr(arcname, response.content)
                    scraped_user_urls.add(url)
