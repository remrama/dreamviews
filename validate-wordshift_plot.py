"""Visualize word shift scores from shifterator.

IMPORTS
=======
    - raw JSD shift scores for lucidity,           results/validate-wordshift_jsd-scores.tsv
    - raw NRC-fear shift scores for nightmares,    results/validate-wordshift_fear-scores.tsv
EXPORTS
=======
    - visualization of a shift, results/validate-wordshift_<shift>-myplot.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import config as c

import matplotlib.pyplot as plt

c.load_matplotlib_settings()


# handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--shift", choices=["jsd", "fear"], type=str, required=True)
args = parser.parse_args()

SHIFT_ID = args.shift

TOP_N = 30 # number of top ranking words to plot

basename_in = f"validate-wordshift_{SHIFT_ID}-scores.tsv"
basename_out = f"validate-wordshift_{SHIFT_ID}-myplot.png"
import_fname = os.path.join(c.DATA_DIR, "results", basename_in)
export_fname = os.path.join(c.DATA_DIR, "results", basename_out)

# load relevant shift scores
df = pd.read_csv(import_fname, index_col="ngram", sep="\t", encoding="utf-8")



# #### recreating shift score
# ############## don't need this anymore #######################
# #### For JSD

# # Shift score is made of 3 components
# proportion_diffs = df.type2p_diff.values
# refscore_diffs = df.type2s_ref_diff.values
# # There's just one part of the shift scores that
# # is not output from get_scores. It's the last
# # summed component and is easy to get
# weighted_score_diffs = df.type2p_avg * df.type2s_diff

# ####### compare with original to check
# # Assuming I used "variation" as the normalization method,
# # recreate that here to get back the scores shifterator provides.
# raw_shift_scores = proportion_diffs*refscore_diffs + weighted_score_diffs
# norm_value = raw_shift_scores.abs().sum()
# regenerated_shift_scores = raw_shift_scores / norm_value
# # now compare
# orig_top = df.type2shift_score.sort_values(ascending=False, key=abs).head(10)
# custom_top = regenerated_shift_scores.sort_values(ascending=False, key=abs).head(10)
# assert orig_top.round(5).subtract(custom_top.round(5)).eq(0).all()

# # now I can still use the original shift scores so
# # no need to put that back into dataframe.
# # But I do need to put the score_diffs in there so
# # that when I sort everything I can pull that out.
# df["type2s_diff_weighted"] = weighted_score_diffs
# ############################################################################################


# # Open a figure with 2 panels (one for lucid JDS and other for nightmare fear)
# GRIDSPEC_ARGS = dict(wspace=.5, left=.08, right=.98, bottom=.15, top=.98)
# fig, axes = plt.subplots(ncols=2, figsize=(6, TOP_N/6),
#     gridspec_kw=GRIDSPEC_ARGS)

# Open a figure with 2 panels (one for lucid JDS and other for nightmare fear)
# GRIDSPEC_ARGS = dict(left=.08, right=.98, bottom=.15, top=.98)
figsize = (2.2, TOP_N/7)
fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # gridspec_kw=GRIDSPEC_ARGS)

# # Draw the A/B letters for manuscript
# fig.text(0, 1, "A", fontsize=12, fontweight="bold", ha="left", va="top")
# fig.text(.5, 1, "B", fontsize=12, fontweight="bold", ha="left", va="top")

# Setup some arguments common to both panels.
BAR_ARGS = dict(linewidth=.5, height=.8, edgecolor="black")

# limits on the x-axis (balanced, so left side will be same but negative)
# Need to make them kinda big to fit the text on the top bar.
# It's in data units so might need to be played with, and different for each panel.
XBOUNDS = dict(jsd=.02, fear=.15)

CMAP_NAMES = dict(jsd="viridis", fear="coolwarm")

XLABELS = {
    "jsd"  : r"non-lucid$\leftarrow$ $\Delta$ frequency $\rightarrow$lucid       ",
    "fear" : r"non-nightmare$\leftarrow$ $\Delta$ frequency $\rightarrow$nightmare       ",
}
YLABELS = {
    "jsd"  : "JSD word shift contribution rank",
    "fear" : "NRC fear word shift contribution rank",
}
CLABELS = {
    "jsd"  : r"common $\leftarrow$ entropy $\rightarrow$ surprising",
    "fear" : r"low  $\leftarrow$ fear intensity $\rightarrow$ high",
} #r"$\leftarrow$ corpus avg $\rightarrow$"

### !!!! IMPORTANT that these match up with what was passed to stop_lens in the wordshift script
NRC_FEAR_STOPS = (.3, .7) # like inner bounds, needs to match what was used for fear shift in validate-wordshift_run.py
NRC_FEAR_BOUNDS = (0, 1) # minimum and maximum scores for fear ratings in shifterator
                         # also assumes later that the ref differences scores are this range but centered around 0

# # Iterate over the two axes/panels and draw each plot.
# for shift_id, ax in zip(["jsd", "fear"], axes):
    # assert shift_id in ["jsd", "fear"], f"Not prepared to handle shift id: {shift_id}"

    # # load relevant shift scores
    # basename = f"validate-wordshift_{shift_id}-scores.tsv"
    # import_fname = os.path.join(c.DATA_DIR, "results", basename)
    # df = pd.read_csv(import_fname, index_col="ngram", sep="\t", encoding="utf-8")

# reduce to the top N shift scores
top_df = df.sort_values("type2shift_score", ascending=False, key=abs)[:TOP_N]

# extract the relevant data for plotting
labels = top_df.index.tolist()              # ngram text to label each bar
ylocs  = np.arange(len(labels)) + 1         # y-locations for each bar
xlocs  = top_df["type2p_diff"].values       # x-locations for each bar
colors = top_df["type2s_ref_diff"].values   # color values for each bar (applied to colormap)
alphas = top_df["type2s_diff"].values       # alpha values for each bar
# cmap_vals = top_df["type2s_diff_weighted"].values

# Select colormap
cmap = plt.get_cmap(CMAP_NAMES[SHIFT_ID])
# Select normalization for the colormaps and mask out fear scale
if SHIFT_ID == "jsd":
    # clist = [c.COLORS["nonlucid"], "white", c.COLORS["lucid"]]
    # cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("", clist)
    # cmax = np.abs(colors).max()
    # norm = plt.Normalize(-cmax, cmax)
    # norm = plt.matplotlib.colors.CenteredNorm()
    cmin, cmax = colors.min(), colors.max()
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
elif SHIFT_ID == "fear":
    # NRC emotion scale in shifterator is 0-1
    # but the colormaps values are between -.5 and 5
    # because they are a difference measure.
    fear_halfrange = np.mean(NRC_FEAR_BOUNDS)
    norm = plt.matplotlib.colors.CenteredNorm(halfrange=fear_halfrange)
    # mask out the center of the colormap for fear, bc of restriction during analysis
    cmask_min, cmask_max = [ x-fear_halfrange for x in NRC_FEAR_STOPS ]
    maskedcolors = cmap(np.linspace(0, 1, 256))
    rgba_mask = np.array([1, 1, 1, 1]) # white
    maskedcolors[int(round(norm(cmask_min)*256)):int(round(norm(cmask_max)*256))+1] = rgba_mask
    cmap = plt.matplotlib.colors.ListedColormap(maskedcolors)

# Normalize color and alpha values, not changing their values
# but just so that they can be passed meaningfully to matplotlib.
# def alpha_normer(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))
alpha_normer = plt.Normalize(alphas.min(), alphas.max())
normed_colors = norm(colors) # puts colors between 0 and 1 for cmap
normed_alphas = alpha_normer(alphas) # puts alphas between 0 and 1 for set_alpha

# Get RGBA color values from colormap and manipulate them a bit
rgba_colors = cmap(normed_colors) # extract RGBA color values from colormap
# rgba_colors[:, 3] = normed_alphas # adjust alphas

# Draw the data
ax.invert_yaxis() # flip to 1 is high on the y-axis
bars = ax.barh(ylocs, xlocs, color=rgba_colors, **BAR_ARGS)
ax.axvline(0, linewidth=1, color="black")
# for alpha, bar in zip(normed_alphas, bars.patches):
#     bar.set_alpha(alpha) # setting alpha manually bc can't pass multiple to ax.bar

# Write the text next to each bar
for barx, bary, text in zip(xlocs, ylocs, labels):
    # if bar is to the right, place txt to the right (flush left), otherwise reverse (left side flush right)
    horizontal_alignment = "left" if barx > 0 else "right"
    # spaces instead of underscores
    text = text.replace("_", " ")
    # add a space to both sides of the text to pad against bar edges
    text = f" {text} " # (for one side it will be meaningless, but this way catches all cases)
    ax.text(barx, bary, text, ha=horizontal_alignment, va="center")

# Adjust axis limits and other tick-related aesthetics
ax.set_xlim(-XBOUNDS[SHIFT_ID], XBOUNDS[SHIFT_ID])
ax.set_ylim(TOP_N+1, 0)
ax.set_xlabel(XLABELS[SHIFT_ID], labelpad=1)
ax.set_ylabel(YLABELS[SHIFT_ID], labelpad=-3)
# yticklocs = np.linspace(0, TOP_N, int(TOP_N/10+1))
# yticklocs[0] = 1
yticklocs = [1, TOP_N]
ax.yaxis.set(major_locator=plt.FixedLocator(yticklocs))
ax.xaxis.set(major_locator=plt.MultipleLocator(XBOUNDS[SHIFT_ID]),
             # minor_locator=plt.MultipleLocator(.01),
             major_formatter=plt.FuncFormatter(c.no_leading_zeros))

## Add a colorbar
# generate a new axis to draw it on directly
# cax = ax.inset_axes([.1, .9, .3, .05]) # posx, posy, width, height
cax = ax.inset_axes([0, 1.01, 1, .02]) # posx, posy, width, height
# draw the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticklocation="top")
# ### saving a few alternate methods in case
# cbax = inset_axes(ax, width="30%", height="3%", loc="upper center") 
# cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
# colorbar with manual axis inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# add label(s)
cbar.ax.set_title(CLABELS[SHIFT_ID])
# cbar.set_label(label, fontsize=8, labelpad=0)
# mess with edges/outline
cbar.outline.set_linewidth(.5)
# cbar.outline.set_visible(False)
# adjust ticks
cbar.ax.tick_params(which="major", size=3, direction="in")
cbar.ax.tick_params(which="minor", size=1, direction="in")
cbar.formatter = plt.FuncFormatter(c.no_leading_zeros)
if SHIFT_ID == "jsd":
    major_cticks = plt.MultipleLocator(.5)
    minor_cticks = plt.MultipleLocator(.1)
elif SHIFT_ID == "fear":
    major_cticks = plt.FixedLocator([-fear_halfrange, cmask_min, 0, cmask_max, fear_halfrange])
    minor_cticks = plt.MultipleLocator(.1)
cbar.locator = major_cticks
cbar.ax.xaxis.set_minor_locator(minor_cticks)
cbar.update_ticks()




########## export
plt.savefig(export_fname)
c.save_hires_figs(export_fname)
plt.close()
