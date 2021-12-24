import os
import config as c
import svgutils.transform as sg

export_fname = os.path.join(c.DATA_DIR, "results", "hires", "describe-categories.svg")
import_fname1 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorycounts.svg")
import_fname2 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorypairs.svg")


#create new SVG figure
fig = sg.SVGFigure("17cm", "8cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(import_fname1)
fig2 = sg.fromfile(import_fname2)

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot1.moveto(5, 20)
plot2.moveto(350, 0)

# add text labels
txt1 = sg.TextElement(5, 10, "A", size=12, weight="bold")
txt2 = sg.TextElement(320, 10, "B", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2])
fig.append([txt1, txt2])

# save generated SVG files
fig.save(export_fname)
