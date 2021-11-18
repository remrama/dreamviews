"""
sometimes it is better to have a folder with raw txt files
"""
import os
import tqdm
import pandas as pd
import config as c


import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")

export_txt_directory = os.path.join(c.DATA_DIR, "derivatives", "posts")

# # if the directory already exists, delete and remake it
# if os.path.isdir(export_txt_directory):
#     import shutil
#     shutil.rmtree(export_txt_directory)
os.mkdir(export_txt_directory)

ser = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["post_id", "post_txt"], index_col="post_id", squeeze=True)

n_posts = ser.size
for post_id, post_txt in tqdm.tqdm(ser.items(), total=n_posts, desc="writing posts as txt files"):
    export_fname = os.path.join(export_txt_directory, f"{post_id}.txt")
    with open(export_fname, "xt", encoding="utf-8") as f:
        f.write(post_txt)
