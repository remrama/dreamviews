"""
Visualize word shift scores from shifterator.

IMPORTS
=======
    - raw JSD shift scores for lucidity,        validate-wordshift_jsd.tsv
    - raw NRC-fear shift scores for nightmares, validate-wordshift_fear.tsv
EXPORTS
=======
    - visualization of a shift, validate-wordshift_<shift>.png
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

########################################################################################
# SETUP
########################################################################################

TOP_N = 30  # Number of top ranking words to plot

# Load custom plotting preferences
c.load_matplotlib_settings()

# Pick filepaths
export_stem = f"validate-wordshift_{SHIFT_ID}"
import_path = c.tables_dir / f"{export_stem}.tsv"

# Load shift scores
df = pd.read_csv(import_path, index_col="ngram", sep="\t", encoding="utf-8")

########################################################################################
# PLOTTING SETUP
########################################################################################

# Setup some arguments common to both panels
BAR_KWARGS = dict(linewidth=0.5, height=0.8, edgecolor="black")

XBOUNDS = dict(jsd=0.02, fear=0.15)
COLORMAP_NAMES = dict(jsd="bgy", fear="coolwarm")

XLABELS = {
    "jsd": r"non-lucid$\leftarrow$ $\Delta$ frequency $\rightarrow$lucid       ",
    "fear": r"non-nightmare$\leftarrow$ $\Delta$ frequency $\rightarrow$nightmare       ",
}
YLABELS = {
    "jsd": "JSD word shift contribution rank",
    "fear": "NRC fear word shift contribution rank",
}
CLABELS = {
    "jsd": "Relative distinctiveness",
    "fear": r"low  $\leftarrow$ fear intensity $\rightarrow$ high",
}

NRC_FEAR_STOPS = c.NIGHTMARE_SHIFT_STOPS  # Like inner bounds, see validate-wordshift.py
NRC_FEAR_BOUNDS = (0, 1)  # Minimum and maximum scores for fear ratings in shifterator
# Also assumes later that the ref differences scores are this range, but centered around 0

# Reduce to the top n shift scores
top_df = df.sort_values("type2shift_score", ascending=False, key=abs)[:TOP_N]

# Extract relevant data for plotting
labels = top_df.index.tolist()  # N-gram text to label each bar
ylocs = np.arange(len(labels)) + 1  # Y-locations for each bar
xlocs = top_df["type2p_diff"].values  # X-locations for each bar
colors = top_df["type2s_ref_diff"].values  # Color values for each bar (applied to colormap)
alphas = top_df["type2s_diff"].values  # Alpha values for each bar
assert alphas.max() <= 0, "type2s_diff values are expected to all be negative"

# Select colormap
cmap = cc.__dict__["m_" + COLORMAP_NAMES[SHIFT_ID]]  # m_ specifies matplotlib cmap in colorcet
# Select normalization for the colormaps and mask out fear scale
if SHIFT_ID == "jsd":
    cmin, cmax = 0, 2
    assert colors.min() >= cmin, "Color values exceed expected minimum."
    assert colors.max() <= cmax, "Color values exceed expected maximum."
    cmin = np.floor(cmin * 10) / 10  # Round down to nearest 10th ..
    cmax = np.ceil(cmax * 10) / 10  # ... and up (bc that's the ticks)
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
elif SHIFT_ID == "fear":
    # NRC emotion scale in shifterator is 0-1,
    # but the colormap values are between -0.5 and 0.5 because they are a difference measure
    fear_halfrange = np.mean(NRC_FEAR_BOUNDS)
    norm = plt.matplotlib.colors.CenteredNorm(halfrange=fear_halfrange)
    # Mask out the center of the colormap for fear, bc of restriction during analysis
    cmask_min, cmask_max = [x - fear_halfrange for x in NRC_FEAR_STOPS]
    maskedcolors = cmap(np.linspace(0, 1, 256))
    rgba_mask = np.array([1, 1, 1, 1])  # White
    maskedcolors[int(round(norm(cmask_min) * 256)) : int(round(norm(cmask_max) * 256)) + 1] = (
        rgba_mask
    )
    cmap = plt.matplotlib.colors.ListedColormap(maskedcolors)

# Normalize color and alpha values,
# not changing their values but just so that they can be passed meaningfully to matplotlib
normed_colors = norm(colors)  # Puts colors between 0 and 1 for cmap
rgba_colors = cmap(normed_colors)  # Get RGBA color values from colormap
if SHIFT_ID == "jsd":
    alphas = np.abs(alphas)
    alpha_min = 0.01
    alpha_max = 3
    assert alphas.min() >= alpha_min, "Alpha values exceed expected minimum."
    assert alphas.max() <= alpha_max, "Alpha values exceed expected maximum."
    alpha_normer = plt.matplotlib.colors.LogNorm(alpha_min, alpha_max)
    normed_alphas = alpha_normer(alphas)  # Puts alphas between 0 and 1 for set_alpha
    rgba_colors[:, 3] = normed_alphas  # Adjust alphas

########################################################################################
# PLOTTING
########################################################################################

# Open a figure
figsize = (2.2, TOP_N / 7)
fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
ax.invert_yaxis()  # Flip so 1 is high on the y-axis

# Draw the data
bars = ax.barh(ylocs, xlocs, color=rgba_colors, **BAR_KWARGS)

# Draw a midline
ax.axvline(0, linewidth=1, color="black")

# Write words next to each bar
for barx, bary, text in zip(xlocs, ylocs, labels, strict=True):
    # If bar is to the right, place txt to the right (flush left),
    # otherwise reverse (left side flush right)
    horizontal_alignment = "left" if barx > 0 else "right"
    text = text.replace("_", " ")  # Spaces instead of underscores
    # Add a space to both sides of the text to pad against bar edges
    text = f" {text} "
    ax.text(barx, bary, text, ha=horizontal_alignment, va="center")

# Adjust axis limits and other tick-related aesthetics
ax.set_xlim(-XBOUNDS[SHIFT_ID], XBOUNDS[SHIFT_ID])
ax.set_ylim(TOP_N + 1, 0)
ax.set_xlabel(XLABELS[SHIFT_ID], labelpad=1)
ax.set_ylabel(YLABELS[SHIFT_ID], labelpad=-3)
yticklocs = [1, TOP_N]
ax.yaxis.set_major_locator(plt.FixedLocator(yticklocs))
ax.xaxis.set_major_locator(plt.MultipleLocator(XBOUNDS[SHIFT_ID]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(c.no_leading_zeros))

# Add colorbar
cax = ax.inset_axes([0, 1.01, 1, 0.02])  # posx, posy, width, height
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticklocation="top")
cbar.ax.set_title(CLABELS[SHIFT_ID])
cbar.outline.set_linewidth(0.5)
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

# Add a second colorbar for grayscale to show alphas, if jsd
if SHIFT_ID == "jsd":
    cax2 = ax.inset_axes([0.25, 0.55, 0.04, 0.4])  # posx, posy, width, height
    # Create a colormap that is pure black but with alpha values that go from 0 to 1
    vals = np.zeros((256, 4))  # 256-row array of pure black (0, 0, 0)
    vals[:, 3] = np.linspace(0, 1, 256)  # Set the alpha channel to go from 0 to 1
    pure_alpha_cmap = plt.matplotlib.colors.ListedColormap(vals)
    sm2 = plt.cm.ScalarMappable(cmap=pure_alpha_cmap, norm=alpha_normer)
    cbar2 = fig.colorbar(sm2, cax=cax2, orientation="vertical", ticklocation="left")
    clabel = "Asymmetry"
    cbar2.ax.set_ylabel("Asymmetry", labelpad=1)
    cbar2.ax.set_ylim(alpha_min, alpha_max)
    cbar2.outline.set_linewidth(0.5)
    cbar2.ax.tick_params(which="major", color="white", size=3, direction="in", pad=1)
    cbar2.ax.tick_params(which="minor", color="white", size=1, direction="in")
    cbar2.ax.set_axisbelow(False)

# Export
c.export_fig(fig, export_stem)
