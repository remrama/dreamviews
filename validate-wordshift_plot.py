"""Visualize word shift scores from shifterator.

IMPORTS
=======
    - raw JSD shift scores for lucidity,        validate-wordshift_jsd-scores.tsv
    - raw NRC-fear shift scores for nightmares, validate-wordshift_fear-scores.tsv
EXPORTS
=======
    - visualization of a shift, validate-wordshift_<shift>-myplot.png
"""
import argparse

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as c


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shift", choices=["jsd", "fear"], type=str, required=True)
args = parser.parse_args()

SHIFT_ID = args.shift


################################################################################
# SETUP
################################################################################

top_n = 30  # Number of top ranking words to plot.

# Load custom plotting preferences.
c.load_matplotlib_settings()

# Pick filepaths.
import_path = c.DATA_DIR / "derivatives" / f"validate-wordshift_{SHIFT_ID}-scores.tsv"
export_path = c.DATA_DIR / "derivatives" / f"validate-wordshift_{SHIFT_ID}-myplot.png"

# Load shift scores.
df = pd.read_csv(import_path, index_col="ngram", sep="\t", encoding="utf-8")


################################################################################
# PLOTTING SETUP
################################################################################

# Setup some arguments common to both panels.
bar_kwargs = dict(linewidth=0.5, height=0.8, edgecolor="black")

xbounds = dict(jsd=0.02, fear=.15)
colormap_names = dict(jsd="bgy", fear="coolwarm")

xlabels = {
    "jsd"  : r"non-lucid$\leftarrow$ $\Delta$ frequency $\rightarrow$lucid       ",
    "fear" : r"non-nightmare$\leftarrow$ $\Delta$ frequency $\rightarrow$nightmare       ",
}
ylabels = {
    "jsd"  : "JSD word shift contribution rank",
    "fear" : "NRC fear word shift contribution rank",
}
clabels = {
    #"jsd"  : r"common $\leftarrow$ $\Delta$ entropy $\rightarrow$ rare         ",
    "jsd"  : r"corpus $\leftarrow$ $\Delta$ entropy $\rightarrow$ lucid      ",
    "fear" : r"low  $\leftarrow$ fear intensity $\rightarrow$ high",
}

# !!!! IMPORTANT that these match up with what was passed to stop_lens in the wordshift script.
nrc_fear_stops = (0.3, 0.7)  # Like inner bounds, needs to match what was used
                             # for fear shift in validate-wordshift_run.py
nrc_fear_bounds = (0, 1)  # Minimum and maximum scores for fear ratings in shifterator.
                          # Also assumes later that the ref differences scores are this range
                          # but centered around 0.

# Reduce to the top n shift scores
top_df = df.sort_values("type2shift_score", ascending=False, key=abs)[:top_n]

# Extract relevant data for plotting.
labels = top_df.index.tolist()  # N-gram text to label each bar.
ylocs = np.arange(len(labels)) + 1  # Y-locations for each bar.
xlocs = top_df["type2p_diff"].values  # X-locations for each bar.
colors = top_df["type2s_ref_diff"].values  # Color values for each bar (applied to colormap).
alphas = top_df["type2s_diff"].values  # Alpha values for each bar.

# Select colormap.
cmap = cc.__dict__["m_" + colormap_names[SHIFT_ID]]  # m_ specifies matplotlib cmap in colorcet.
# Select normalization for the colormaps and mask out fear scale.
if SHIFT_ID == "jsd":
    cmin, cmax = colors.min(), colors.max()
    cmin =  np.floor(cmin * 10) / 10  # Round down to nearest 10th ...
    cmax =  np.ceil(cmax * 10) / 10  # ... and up (bc that's the ticks).
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
elif SHIFT_ID == "fear":
    # NRC emotion scale in shifterator is 0-1,
    # but the colormaps values are between -0.5 and 0.5 because they are a difference measure.
    fear_halfrange = np.mean(NRC_FEAR_BOUNDS)
    norm = plt.matplotlib.colors.CenteredNorm(halfrange=fear_halfrange)
    # Mask out the center of the colormap for fear, bc of restriction during analysis.
    cmask_min, cmask_max = [ x-fear_halfrange for x in NRC_FEAR_STOPS ]
    maskedcolors = cmap(np.linspace(0, 1, 256))
    rgba_mask = np.array([1, 1, 1, 1])  # White
    maskedcolors[int(round(norm(cmask_min) * 256)):int(round(norm(cmask_max) * 256)) + 1] = rgba_mask
    cmap = plt.matplotlib.colors.ListedColormap(maskedcolors)

# Normalize color and alpha values,
# not changing their values but just so that they can be passed meaningfully to matplotlib.
alpha_normer = plt.Normalize(alphas.min(), alphas.max())
normed_colors = norm(colors)  # Puts colors between 0 and 1 for cmap.
normed_alphas = alpha_normer(alphas)  # Puts alphas between 0 and 1 for set_alpha.

# Get RGBA color values from colormap and manipulate them a bit.
rgba_colors = cmap(normed_colors)
# rgba_colors[:, 3] = normed_alphas  # Adjust alphas.

################################################################################
# PLOTTING
################################################################################

# Open a figure.
figsize = (2.2, top_n / 7)
fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
ax.invert_yaxis()  # Flip so 1 is high on the y-axis.

# Draw the data.
bars = ax.barh(ylocs, xlocs, color=rgba_colors, **BAR_ARGS)

# Draw a midline.
ax.axvline(0, linewidth=1, color="black")

# Write words next to each bar.
for barx, bary, text in zip(xlocs, ylocs, labels):
    # If bar is to the right, place txt to the right (flush left),
    # otherwise reverse (left side flush right).
    horizontal_alignment = "left" if barx > 0 else "right"
    text = text.replace("_", " ")  # Spaces instead of underscores.
    # Add a space to both sides of the text to pad against bar edges.
    text = f" {text} "
    ax.text(barx, bary, text, ha=horizontal_alignment, va="center")

# Adjust axis limits and other tick-related aesthetics
ax.set_xlim(-xbounds[SHIFT_ID], xbounds[SHIFT_ID])
ax.set_ylim(top_n + 1, 0)
ax.set_xlabel(xlabels[SHIFT_ID], labelpad=1)
ax.set_ylabel(ylabels[SHIFT_ID], labelpad=-3)
yticklocs = [1, top_n]
ax.yaxis.set_major_locator(plt.FixedLocator(yticklocs))
ax.xaxis.set_major_locator(plt.MultipleLocator(xbounds[SHIFT_ID]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(c.no_leading_zeros))

# Add colorbar
cax = ax.inset_axes([0, 1.01, 1, .02])  # posx, posy, width, height
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticklocation="top")
cbar.ax.set_title(clabels[SHIFT_ID])
cbar.outline.set_linewidth(.5)
cbar.ax.tick_params(which="major", size=3, direction="in")
cbar.ax.tick_params(which="minor", size=1, direction="in")
cbar.formatter = plt.FuncFormatter(c.no_leading_zeros)
if SHIFT_ID == "jsd":
    major_cticks = plt.MultipleLocator(0.5)
    minor_cticks = plt.MultipleLocator(0.1)
elif SHIFT_ID == "fear":
    major_cticks = plt.FixedLocator([-fear_halfrange, cmask_min, 0, cmask_max, fear_halfrange])
    minor_cticks = plt.MultipleLocator(0.1)
cbar.locator = major_cticks
cbar.ax.xaxis.set_minor_locator(minor_cticks)
cbar.update_ticks()


# Export.
plt.savefig(export_path)
plt.savefig(export_path.with_suffix(".pdf"))
plt.close()
