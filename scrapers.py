"""
Functions for scraping DreamViews dream journal posts and user profiles.
"""

import re
import zipfile
from time import sleep
from urllib.parse import urlparse

import requests
import tqdm
from bs4 import BeautifulSoup


def scrape_posts(export_path: str, delay: float, test: bool = False) -> None:
    """
    Scrape all DreamViews dream journal entries and save the raw HTML files into a zipfile.

    export_path : str or Path object
        Full filepath where results will be exported. Must have .zip extension and not exist.
    delay : float
        Time in seconds to wait between requests.
    test : bool
        If True, scrape the first 5 pages of posts. If False (default), scrape all pages.
    """
    dreamviews_posts_url = "https://www.dreamviews.com/blogs/recent-entries"
    with requests.Session() as session:
        if test:
            n_total_pages = 5
        else:
            # Get the total number of pages by loading the first (i.e., most recent) dream
            # journal page and then finding the "Last" page link to extract the number of pages
            response = session.get(dreamviews_posts_url)
            soup = BeautifulSoup(response.text, "html.parser")
            last_url = soup.find("span", class_="first_last").find("a")["href"]
            last_index = urlparse(last_url).path.split("/")[-1].lstrip("index").rstrip(".html")
            n_total_pages = int(last_index)
        # Loop over all dream journal pages and save each one as an HTML file in the zip
        with zipfile.ZipFile(export_path, mode="x", compression=zipfile.ZIP_DEFLATED) as zf:
            for i in tqdm.trange(n_total_pages, desc="Scraping DreamViews posts"):
                url_slug = f"index{i+1}.html"
                url = f"{dreamviews_posts_url}/{url_slug}"
                response = session.get(url)
                zf.writestr(url_slug, response.content)
                sleep(delay)


def scrape_users(export_path: str, import_path: str, delay: float) -> None:
    """
    Scrape DreamViews user profiles and save the raw HTML files into a zipfile.
    Only include users who have posted one or more dream journal posts.

    export_path : str or Path object
        Full filepath where results will be exported. Must have .zip extension and not exist.
    import_path : str or Path object
        Full filepath where results from scrape_posts exist.
    delay : float
        Time in seconds to wait between requests.
    """
    href_filter = re.compile("https://www.dreamviews.com/members")
    scraped_urls = set()
    with requests.Session() as session:
        with zipfile.ZipFile(export_path, mode="x", compression=zipfile.ZIP_DEFLATED) as export_zf:
            with zipfile.ZipFile(import_path, mode="r") as import_zf:
                for fn in tqdm.tqdm(import_zf.namelist(), desc="Scraping DreamViews users"):
                    html = import_zf.read(fn)
                    soup = BeautifulSoup(html, "html.parser", from_encoding="windows-1252")
                    urls = {tag["href"] for tag in soup.find_all("a", href=href_filter)}
                    for url in urls:
                        if url not in scraped_urls and not url.endswith("list/"):
                            response = session.get(url)
                            username = urlparse(url).path.rsplit("/", 2)[-2]
                            export_zf.writestr(f"{username}.html", response.content)
                            scraped_urls.add(url)
                            sleep(delay)


if __name__ == "__main__":

    import argparse
    import sys
    import time
    from pathlib import Path
    from tkinter import filedialog
    from tkinter import Tk

    unix_time = int(time.time())
    basename_posts = f"dreamviews_raw_posts-{unix_time}.zip"
    basename_users = f"dreamviews_raw_users-{unix_time}.zip"

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sleep", type=float, default=0.2, help="Delay between requests")
    parser.add_argument("-d", "--directory", type=str, required=False, help="File output location")
    parser.add_argument("-t", "--test", action="store_true", help="Scrape a subset of post pages")
    args = parser.parse_args()

    if args.directory is None:
        root = Tk()
        root.withdraw()
        parent = filedialog.askdirectory(title="Choose directory where results will be saved")
        if not parent:
            sys.exit()
    else:
        parent = Path(args.directory).expanduser()

    path_posts = parent / basename_posts
    path_users = parent / basename_users
    assert not path_posts.exists(), f"{path_posts} already exists"
    assert not path_users.exists(), f"{path_users} already exists"
    assert path_posts.suffix == ".zip", f"{path_posts} must end with .zip"
    assert path_users.suffix == ".zip", f"{path_users} must end with .zip"

    parent.mkdir(exist_ok=True)

    scrape_posts(export_path=path_posts, delay=args.delay, test=args.test)
    scrape_users(export_path=path_users, import_path=path_posts, delay=args.delay)
