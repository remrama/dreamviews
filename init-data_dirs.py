"""Initialize the data directory structure used throughout the rest of the scripts.

Puts some new directories inside the "data directory"
specified in the <config.py> configuration file.

Subdirectory motivations are described briefly below.
"""
import os
import config as c

DATA_SUBDIRECTORIES = [
    "source",           # for the RAW data -- no touchey.
    "derivatives",      # for mid-stage, between source and results
    "results",          # for final output (plots, stats tables, etc.)
    "results/hires",    # for high resolution plots (vector graphics)
]

if not os.path.isdir(c.DATA_DIR):
    os.mkdir(c.DATA_DIR)

for subdir in DATA_SUBDIRECTORIES:
    subdir_path = os.path.join(c.DATA_DIR, subdir)
    if not os.path.isdir(subdir_path):
        os.mkdir(subdir_path)
