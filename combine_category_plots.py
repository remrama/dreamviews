"""Combine 2 panels into one figure.
Struggled with this, not a lot of maintained
python software to load/manipulate vector graphics.

svg-stack worked best, with limitations that
it can't add letters, so the A and B need to
be added when the original panel is made.

In other scripts I just make all panels at once,
but I used seaborn's JointGrid to make panel B here
which makes it's own figure and requires significant
effort to have it coexist with something else.
"""
## pip install svg-stack

import os
import config as c
import svg_stack as ss

export_fname = os.path.join(c.DATA_DIR, "results", "hires", "describe-categories.svg")
import_fname1 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorycounts.svg")
import_fname2 = os.path.join(c.DATA_DIR, "results", "hires", "describe-categorypairs.svg")

doc = ss.Document()
layout1 = ss.HBoxLayout()
layout1.setSpacing(100) # n pixels between images
layout1.addSVG(import_fname1, alignment=ss.AlignTop|ss.AlignHCenter)
layout1.addSVG(import_fname2, alignment=ss.AlignCenter)

# layout2 = ss.VBoxLayout()
# layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
# layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
# layout2.addSVG('red_ball.svg',alignment=ss.AlignCenter)
# layout1.addLayout(layout2)

doc.setLayout(layout1)

doc.save(export_fname)


## svg to other formats
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

export_fname_png = export_fname.replace(".svg", ".png")
export_fname_pdf = export_fname.replace(".svg", ".pdf")
drawing = svg2rlg(export_fname)
renderPDF.drawToFile(drawing, export_fname_pdf)
renderPM.drawToFile(drawing, export_fname_png, fmt="PNG")



###### other route w/ svg utils, had issues

# import svgutils.transform as sg

# #create new SVG figure
# fig = sg.SVGFigure()

# # load matpotlib-generated figures
# fig1 = sg.fromfile(import_fname1)
# fig2 = sg.fromfile(import_fname2)

# # get the plot objects
# plot1 = fig1.getroot()
# plot2 = fig2.getroot()
# plot1.moveto(5, 20)
# plot2.moveto(350, 0)

# # add text labels
# txt1 = sg.TextElement(5, 10, "A", font="Arial", size=12, weight="bold")
# txt2 = sg.TextElement(320, 10, "B", font="Arial", size=12, weight="bold")

# # append plots and labels to figure
# fig.append([plot1, plot2])
# fig.append([txt1, txt2])

# fig.set_size(("17cm", "8cm"))

# # save generated SVG files
# fig.save(export_fname)



