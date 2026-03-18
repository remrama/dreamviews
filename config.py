from pathlib import Path

import pandas as pd
import pooch
from matplotlib.pyplot import rcParams

DATA_DIR = "../data"

data_dir = Path(DATA_DIR).expanduser()

sourcedata_dir = data_dir / "sourcedata"
raw_dir = data_dir / "raw"
derivatives_dir = data_dir / "derivatives"

sourcedata_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)
derivatives_dir.mkdir(parents=True, exist_ok=True)

MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000  # limiting the number of posts a single user can have

COLORS = {
    "lucid": "#3a90fe",
    "nonlucid": "#a89008",
    "ambiguous": "#719083",  # lucid/nonlucid blend
    "unspecified": "darkgray",
    "nightmare": "red",
    "novel-user": "gold",
    "repeat-user": "goldenrod",
}

REGISTRY = {
    "a_AgencyCommunion.dic": {
        "url": "https://osf.io/62txv/download",
        "known_hash": "md5:d2240a5eb36568d9eefaa428130a0577",
    },
}

fetcher = pooch.create(
    path=raw_dir,
    base_url="",
    registry={k: v["known_hash"] for k, v in REGISTRY.items()},
    urls={k: v["url"] for k, v in REGISTRY.items()},
)


def fetch_file(filename):
    return Path(fetcher.fetch(filename))


def load_dreamviews_users():
    users_fname = raw_dir / "dreamviews-users.tsv"
    users = pd.read_csv(users_fname, sep="\t", encoding="ascii")
    return users


def load_dreamviews_posts():
    posts_fname = raw_dir / "dreamviews-posts.tsv"
    posts = pd.read_csv(posts_fname, sep="\t", encoding="ascii", parse_dates=["timestamp"])
    return posts


def export_table(dataframe, filestem, **kwargs):
    default_kwargs = {"sep": "\t", "index": True, "encoding": "utf-8", "float_format": "%.5f", "na_rep": "n/a"}
    kwargs = {**default_kwargs, **kwargs}
    if kwargs["sep"] == "\t":
        suffix = ".tsv"
    elif kwargs["sep"] == ",":
        suffix = ".csv"
    else:
        raise ValueError("Unsupported separator")
    export_path = (derivatives_dir / filestem).with_suffix(suffix)
    dataframe.to_csv(export_path, **kwargs)
    return


def save_and_close_fig(fig, filestem, **kwargs):
    assert "format" not in kwargs, "format should not be specified in kwargs"
    default_kwargs = {"dpi": 600}
    kwargs = {**default_kwargs, **kwargs}
    formats = ["png", "pdf"]
    for fmt in formats:
        export_path = (derivatives_dir / filestem).with_suffix(f".{fmt}")
        fig.savefig(export_path, **kwargs)
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
