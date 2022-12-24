"""
Scrape all DreamViews dream journal entries saving the raw html files into a zipfile.
"""
import requests
import zipfile

from bs4 import BeautifulSoup
import tqdm

import config as c


export_path = c.DATA_DIR / "source" / "dreamviews-posts.zip"

dreamviews_url = "https://www.dreamviews.com/blogs/recent-entries"


with requests.Session() as session:

    # Get the total number of pages by loading the first (i.e., most recent) dream
    # journal page and then finding the "Last" page link to extract the number of pages.
    page = session.get(dreamviews_url).text
    soup = BeautifulSoup(page, "html.parser")
    lasturl = soup.find("span", class_="first_last").find("a")["href"]
    lastnum = lasturl.rstrip(".html").split("/index")[1]
    assert lastnum.isdigit()
    n_pages = int(lastnum)

    # Loop over all dream journal pages and save each one as an html file in the zip.
    with zipfile.ZipFile(export_path, mode="x", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in tqdm.trange(1, n_pages+1, desc="DreamViews posts crawl"):
            export_name = f"index{i:04d}.html"
            url = f"{dreamviews_url}/index{i}.html"
            r = session.get(url)
            zf.writestr(export_name, r.content)
