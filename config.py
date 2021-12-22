DATA_DIR = r"C:\Users\malle\PROJECTS\dreamviews_ds\data"

BLOG_TIMESTAMP_FORMAT = "%m-%d-%Y at %I:%M %p"

START_DATE = "2010-01-01"
END_DATE = "2020-12-31"

HIRES_IMAGE_EXTENSION = ".eps" # what to save in addition to png

MIN_WORDCOUNT = 50
MAX_WORDCOUNT = 1000
MAX_POSTCOUNT = 1000 # limiting the number of posts a single user can have

COLORS = {
    "lucid"        : "royalblue",
    "non-lucid"    : "darkorange",
    "unspecified"  : "bisque",
    "ambiguous"    : "#c69c6d",
    "nightmare"    : "red",
    "novel-user"   : "gold",
    "repeat-user"  : "goldenrod",
}


def strip_doublebracket_content(txt):
    """match anything in double square brackets (including the brackets)
    Beware -- will leave extra space if there was a space on both sides.
    """
    return re.sub(r"\[\[.*?\]\]", "", redacted_text)


def no_leading_zeros(x, pos):
    # a custom tick formatter for matplotlib
    # to show decimals without a leading zero
    val_str = "{:g}".format(x)
    if abs(x) > 0 and abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str