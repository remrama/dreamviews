import os
DATA_DIR = "~/PROJECTS/dreamviews_ds/data"
DATA_DIR = os.path.expanduser(DATA_DIR)

BLOG_TIMESTAMP_FORMAT = "%m-%d-%Y at %I:%M %p"

START_DATE = "2010-01-01"
END_DATE = "2020-12-31"


MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000 # limiting the number of posts a single user can have

COLORS = {
    "lucid"        : "royalblue",
    "nonlucid"     : "darkorange",
    "unspecified"  : "bisque",
    "ambiguous"    : "#c69c6d",
    "nightmare"    : "red",
    "novel-user"   : "gold",
    "repeat-user"  : "goldenrod",
}


def load_dreamviews_users():
    import os; import pandas as pd
    users_fname = os.path.join(DATA_DIR, "derivatives", "dreamviews-users.tsv")
    users = pd.read_csv(users_fname, sep="\t", encoding="ascii")
    return users

def load_dreamviews_posts():
    import os; import pandas as pd
    posts_fname = os.path.join(DATA_DIR, "derivatives", "dreamviews-posts.tsv")
    posts = pd.read_csv(posts_fname, sep="\t", encoding="ascii", parse_dates=["timestamp"])
    return posts

def strip_doublebracket_content(txt):
    """match anything in double square brackets (including the brackets)
    Beware -- will leave extra space if there was a space on both sides.
    """
    return re.sub(r"\[\[.*?\]\]", "", redacted_text)


def load_matplotlib_settings():
    from matplotlib.pyplot import rcParams
    rcParams["savefig.dpi"] = 600
    rcParams["interactive"] = True
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = "Arial"
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "Arial"
    rcParams["mathtext.cal"] = "Arial"
    rcParams["mathtext.it"] = "Arial:italic"
    rcParams["mathtext.bf"] = "Arial:bold"


def no_leading_zeros(x, pos):
    # a custom tick formatter for matplotlib
    # to show decimals without a leading zero
    val_str = "{:g}".format(x)
    if abs(x) > 0 and abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def save_hires_figs(png_fname, hires_extensions=[".svg", ".eps", ".pdf"]):
    # replace the extension and go down into a "hires" folder which should be there
    import os
    from matplotlib.pyplot import savefig
    png_dir, png_bname = os.path.split(png_fname)
    hires_dir = os.path.join(png_dir, "hires")
    for ext in hires_extensions:
        hires_bname = png_bname.replace(".png", ext)
        hires_fname = os.path.join(hires_dir, hires_bname)
        savefig(hires_fname)
