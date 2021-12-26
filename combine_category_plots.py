import os
import config as c
import svgutils.transform as sg

export_fname = os.path.join(c.DATA_DIR, "results", "hires", "describe-categories.svg")
import_fname1 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorycounts.svg")
import_fname2 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorypairs.svg")


#create new SVG figure
fig = sg.SVGFigure()

# load matpotlib-generated figures
fig1 = sg.fromfile(import_fname1)
fig2 = sg.fromfile(import_fname2)

# get the plot objects
plot1 = fig1.getroot()
plot2 = fig2.getroot()
plot1.moveto(5, 20)
plot2.moveto(350, 0)

# add text labels
txt1 = sg.TextElement(5, 10, "A", font="Arial", size=12, weight="bold")
txt2 = sg.TextElement(320, 10, "B", font="Arial", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2])
fig.append([txt1, txt2])

fig.set_size(("17cm", "8cm"))

# save generated SVG files
fig.save(export_fname)


# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPDF, renderPM

# export_fname_png = export_fname.replace(".svg", ".png")
# export_fname_pdf = export_fname.replace(".svg", ".pdf")
# drawing = svg2rlg(export_fname)
# renderPDF.drawToFile(drawing, export_fname_pdf)
# renderPM.drawToFile(drawing, export_fname_png, fmt="PNG")
