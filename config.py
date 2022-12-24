from pathlib import Path

from matplotlib.pyplot import rcParams
import pandas as pd


DATA_DIR = "../data"
DATA_DIR = Path(DATA_DIR).expanduser()

MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000  # limiting the number of posts a single user can have

COLORS = {
    "lucid"        : "#3a90fe",
    "nonlucid"     : "#a89008",
    "unspecified"  : "darkgray",
    "ambiguous"    : "#719083",  # lucid/nonlucid blend
    "nightmare"    : "red",
    "novel-user"   : "gold",
    "repeat-user"  : "goldenrod",
}

def load_dreamviews_users():
    users_fname = DATA_DIR / "derivatives" / "dreamviews-users.tsv"
    users = pd.read_csv(users_fname, sep="\t", encoding="ascii")
    return users

def load_dreamviews_posts():
    posts_fname = DATA_DIR / "derivatives" / "dreamviews-posts.tsv"
    posts = pd.read_csv(posts_fname, sep="\t", encoding="ascii", parse_dates=["timestamp"])
    return posts

def strip_doublebracket_content(txt):
    """match anything in double square brackets (including the brackets)
    Beware -- will leave extra space if there was a space on both sides.
    """
    return re.sub(r"\[\[.*?\]\]", "", redacted_text)

def load_matplotlib_settings():
    rcParams["savefig.dpi"] = 600
    rcParams["interactive"] = True
    rcParams["font.family"] = "Times New Roman"
    # rcParams["font.sans-serif"] = "Arial"
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
    # a custom tick formatter for matplotlib
    # to show decimals without a leading zero
    val_str = "{:g}".format(x)
    if abs(x) > 0 and abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str
