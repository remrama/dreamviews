"""
wordcloud, just bc.

This does NOT account for repeated reports per user.
It shouldn't be take seriously, but might have appeal at times.
"""
import os
import numpy as np
import pandas as pd
import config as c

from PIL import Image
from wordcloud import WordCloud

import seaborn as sea
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


import_fname = os.path.join(c.DATA_DIR, "derivatives", "posts-clean.tsv")
export_fname  = os.path.join(c.DATA_DIR, "results", "validate-wordcloud.png")

mask_img_fname = os.path.join(".", "img", "bubble.jpg")
icon_img_fname = os.path.join(".", "img", "sleeper.png")

TXT_COLUMN = "post_lemmas"

ser = pd.read_csv(import_fname, sep="\t", encoding="utf-8",
    usecols=["lucidity", TXT_COLUMN],
    index_col="lucidity", squeeze=True)


if os.path.exists(mask_img_fname):
    mask = np.array(Image.open(mask_img_fname))

if os.path.exists(mask_img_fname):
    icon = plt.matplotlib.image.imread(icon_img_fname)


# get pixel dimension things
ncol, nrow, _ = mask.shape # in pixels
# get all channels where color is shown
alphamask = np.where(icon[:,:,3] > 0)
ICON_HEIGHT = 75 # in pixels
ICON_WIDTH  = 75 # in pixels
extent = (0, ICON_WIDTH, nrow-ICON_HEIGHT, nrow)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

LUCIDITY_ORDER = ["non-lucid", "lucid"]

# need to find the arial font path for the WordCloud package.
# it's /Library/Fonts/Arial.ttf on mac
# and C:\\Windows\\Fonts\\arial.ttf on windows.
# use this to figure it out
all_ttf_files = plt.matplotlib.font_manager.findSystemFonts()
arial_path = [ f for f in all_ttf_files
    if "arial.ttf" in f.lower() ][0]


for ax, label in zip(axes, LUCIDITY_ORDER):
    
    # color the little icon image
    color = c.COLORS[label]
    rgba = plt.matplotlib.colors.to_rgba_array(color)
    icon[alphamask] = rgba

    # draw the little icon image
    ax.imshow(icon, origin="lower", extent=extent)

    # generate the wordcloud
    txt = ser.loc[label].str.cat(sep=" ")
    wc = WordCloud(font_path=arial_path,
        mask=mask,
        background_color=None,
        relative_scaling=.5,
        colormap="viridis",
        mode="RGBA",
        random_state=32,
    )
    wc_img = wc.generate(txt)

    # draw the wordcloud
    ax.imshow(wc_img, origin="upper",
        aspect="equal",
        interpolation="bilinear")

    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow, 0)
    ax.axis("off")


plt.tight_layout()

plt.savefig(export_fname)
plt.savefig(export_fname.replace(".png", c.HIRES_IMAGE_EXTENSION))
plt.close()
