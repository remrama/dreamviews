import os
from pathlib import Path

import pandas as pd
import pooch
from dotenv import load_dotenv
from matplotlib.pyplot import rcParams

load_dotenv()

OUTPUT_DIR = "../output"
MANUSCRIPT_DIR = "../manuscript"

output_dir = Path(OUTPUT_DIR).expanduser()
manuscript_dir = Path(MANUSCRIPT_DIR).expanduser()

sourcedata_dir = output_dir / "sourcedata"
raw_dir = output_dir / "raw"
derivatives_dir = output_dir / "derivatives"
tables_dir = output_dir / "tables"
figures_dir = output_dir / "figures"

sourcedata_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)
derivatives_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

SPACY_MODEL = "en_core_web_lg"

MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000  # limiting the number of posts a single user can have

NIGHTMARE_SHIFT_STOPS = (0.3, 0.7)

COLORS = {
    "lucid": "#3a90fe",
    "nonlucid": "#a89008",
    "ambiguous": "#719083",  # lucid/nonlucid blend
    "unspecified": "darkgray",
    "nightmare": "red",
    "novel-user": "gold",
    "repeat-user": "goldenrod",
}

DERIVATIVES_REGISTRY = {
    "ne_110m_admin_0_countries.zip": {
        "url": "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        "known_hash": "md5:374f5381a2ff702d3d79d345a9e5f65c",
    },
}

RAW_REGISTRY = {
    "v1": {
        "doi": "10.5281/zenodo.19161757",
        "files": {
            "dreamviews-posts.tsv": "md5:d78a499937e063add677a3ed28393d36",
            "dreamviews-users.tsv": "md5:2dd38e4ba29d910ffafb718072dec8ba",
        },
    },
}

SOURCE_REGISTRY = {
    "v1": {
        "doi": "10.5281/zenodo.19040637",
        "files": {
            "dreamviews-posts.zip": "md5:84c2f6e6454a879d22deea1c5c227c1e",
            "dreamviews-users.zip": "md5:3824a274b927165d20be4c79cf0f905a",
            "InsightAgency.dic": "md5:8ed19dfe5ee57db694b97e7b2eacdf66",
        },
    },
}


def fetch_deriv_file(filename):
    fetcher = pooch.create(
        path=derivatives_dir,
        base_url="",
        registry={k: v["known_hash"] for k, v in DERIVATIVES_REGISTRY.items()},
        urls={k: v["url"] for k, v in DERIVATIVES_REGISTRY.items()},
        allow_updates=False,
    )
    return Path(fetcher.fetch(filename))


def _zenodo_doi_to_pooch_url(doi, as_doi_url=False):
    if as_doi_url:
        return f"doi:{doi}"
    record_id = doi[-8:]
    return f"https://zenodo.org/api/records/{record_id}/files"


def fetch_raw_file(filename, version):
    assert version in RAW_REGISTRY, f"Version {version} not found in RAW_REGISTRY"
    registry = RAW_REGISTRY[version]["files"]
    doi = RAW_REGISTRY[version]["doi"]
    base_url = _zenodo_doi_to_pooch_url(doi, as_doi_url=True)
    fetcher = pooch.create(path=raw_dir, base_url=base_url, registry=registry, allow_updates=False)
    return Path(fetcher.fetch(filename))


def fetch_source_file(filename, version):
    assert version in SOURCE_REGISTRY, f"Version {version} not found in SOURCE_REGISTRY"
    registry = SOURCE_REGISTRY[version]["files"]
    doi = SOURCE_REGISTRY[version]["doi"]
    # Accessing restricted files requires using the API with an access token.
    # The URL for fetching a specific file is different and has a suffix after the filename
    # so we can't use the standard pooch base_url since it appends filenames at the end.
    # Construct each URL manually instead.
    record_id = doi[-8:]
    api_url = f"https://zenodo.org/api/records/{record_id}/files/{{filename}}/content"
    urls = {filename: api_url.format(filename=filename) for filename in registry}
    fetcher = pooch.create(
        path=sourcedata_dir, base_url="", registry=registry, urls=urls, allow_updates=False
    )
    # Create authorized downloader
    token = os.environ.get("ZENODO_TOKEN")
    downloader = pooch.HTTPDownloader(headers={"Authorization": f"Bearer {token}"})
    return Path(fetcher.fetch(filename, downloader=downloader))


def load_dreamviews_users(version="v1"):
    filepath = fetch_raw_file("dreamviews-users.tsv", version)
    users = pd.read_csv(filepath, sep="\t", encoding="ascii")
    return users


def load_dreamviews_posts(lemmas=False, version="v1"):
    filepath = fetch_raw_file("dreamviews-posts.tsv", version)
    posts = pd.read_csv(filepath, sep="\t", encoding="ascii", parse_dates=["timestamp"])
    if lemmas:
        lemmas_fpath = derivatives_dir / "lemmas.tsv"
        lemmas = pd.read_csv(lemmas_fpath, sep="\t", encoding="ascii")
        posts = posts.merge(lemmas, on="post_id", how="inner", validate="one_to_one")
    return posts


def export_table(dataframe, filestem, **kwargs):
    default_kwargs = {
        "sep": "\t",
        "index": True,
        "encoding": "utf-8",
        "float_format": "%.5f",
        "na_rep": "n/a",
    }
    kwargs = {**default_kwargs, **kwargs}
    if kwargs["sep"] == "\t":
        suffix = ".tsv"
    elif kwargs["sep"] == ",":
        suffix = ".csv"
    else:
        raise ValueError("Unsupported separator")
    export_path = (tables_dir / filestem).with_suffix(suffix)
    dataframe.to_csv(export_path, **kwargs)
    return


def export_fig(fig, filestem, close=True, **kwargs):
    assert "format" not in kwargs, "format should not be specified in kwargs"
    default_kwargs = {"dpi": 600, "metadata": dict(CreationDate=None)}
    kwargs = {**default_kwargs, **kwargs}
    formats = ["png", "pdf"]
    for fmt in formats:
        export_path = (figures_dir / filestem).with_suffix(f".{fmt}")
        fig.savefig(export_path, **kwargs)
    if close:
        fig.clf()
    return


def load_matplotlib_settings():
    rcParams["font.family"] = "Times New Roman"
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "Times New Roman"
    rcParams["mathtext.cal"] = "Times New Roman"
    rcParams["mathtext.it"] = "Times New Roman:italic"
    rcParams["mathtext.bf"] = "Times New Roman:bold"
    rcParams["font.size"] = 8
    rcParams["axes.titlesize"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["axes.labelsize"] = 8
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8
    rcParams["legend.fontsize"] = 8
    rcParams["legend.title_fontsize"] = 8


def no_leading_zeros(x, pos):
    # a custom tick formatter for matplotlib to show decimals without a leading zero
    val_str = "{:g}".format(x)
    if abs(x) > 0 and abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str
