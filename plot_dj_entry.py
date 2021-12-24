import os
import config as c

import matplotlib.pyplot as plt
import matplotlib.offsetbox as moffsetbox
c.load_matplotlib_settings()


import_fname = "./img/djentry.png"

img = plt.matplotlib.image.imread(import_fname)

FIGSIZE = (4, 2)
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.imshow(img)

ax.axis("off")


offsetbox = moffsetbox.TextArea("Test sentence.")

xy_arrowhead = (.5, 1)
xy_box = (.7, 1)

ab = moffsetbox.AnnotationBbox(offsetbox,
    xy=xy_arrowhead, xybox=xy_box, #(1.02, xy[1]),
    xycoords=("axes fraction", "axes fraction"),
    boxcoords=("axes fraction", "axes fraction"),
    box_alignment=(0., 0.5), # A tuple of two floats for a vertical and horizontal alignment
                             # of the offset box w.r.t. the boxcoords.
                             # The lower-left corner is (0, 0) and upper-right corner is (1, 1).
    frameon=True,
    pad=.4, # padding around the offset box
    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)
