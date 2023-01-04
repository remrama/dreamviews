"""
Initialize the data directory structure used throughout the rest of the scripts.
Puts some new directories inside the <DATA_DIR> specified in the <config.py> configuration file.
"""
import config as c


DATA_SUBDIRECTORIES = [
    "source",         # for the RAW data -- no touchey
    "derivatives",    # for mid-stage, between source and results
]

for subdir in DATA_SUBDIRECTORIES:
    path = c.DATA_DIR / subdir
    path.mkdir(exist_ok=True)
