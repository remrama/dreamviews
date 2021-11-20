DATA_DIR = r"C:\Users\rrm2308\data\dreamviews_ds"

DATA_COLLECTION_DATE = "2021-10-09" # needed in convert-posts.py for defining today/yesterday

BLOG_TIMESTAMP_FORMAT = "%m-%d-%Y at %I:%M %p"

HIRES_IMAGE_EXTENSION = ".eps" # what to save in addition to png

MIN_ALPHA_CHARS = 100 # needed in convert-posts.py to restrict uber-short posts, token limit will catch the most
MIN_TOKEN_COUNT = 50
MAX_TOKEN_COUNT = 1000
MAX_POST_COUNT  = 1000 # limiting the number of posts a single user can have
MIN_LEMMA_COUNT = 3
MIN_LEMMA_CHARS = 3

COLORS = {
    "lucid"        : "royalblue",
    "non-lucid"    : "darkorange",
    "unspecified"  : "bisque",
    "ambiguous"    : "#c69c6d",
    "nightmare"    : "red",
    "novel-user"   : "gold",
    "repeat-user"  : "goldenrod",
}


def no_leading_zeros(x, pos):
    # a custom tick formatter for matplotlib
    # to show decimals without a leading zero
    val_str = "{:g}".format(x)
    if abs(x) > 0 and abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str