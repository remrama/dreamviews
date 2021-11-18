DATA_DIR = r"C:\Users\rrm2308\data\dreamviews_ds"

DATA_COLLECTION_DATE = "2021-10-09" # needed in convert-posts.py for defining today/yesterday

MIN_ALPHA_CHARS = 100 # needed in convert-posts.py to restrict uber-short posts, token limit will catch the most
MIN_TOKEN_COUNT = 50
MAX_TOKEN_COUNT = 1000

COLORS = {
    "lucid"        : "royalblue",
    "non-lucid"    : "darkorange",
    "unspecified"  : "bisque",
    "ambiguous"    : "#c69c6d",
    "nightmare"    : "red",
    "novel-user"   : "gold",
    "repeat-user"  : "goldenrod",
}