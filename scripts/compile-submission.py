"""
Compile a single flat zip for submitting LaTeX manuscript to arXiv.
"""

import argparse
import re
import shutil
import subprocess
import zipfile

import config as c

MANUSCRIPT_FNAME = "main.tex"
MAKEFILE_FNAME = "Makefile"
IMAGES_DIRNAME = "images"
OUT_DIRNAME = "out"  # matches Makefile output directory
SUBMISSION_DIRNAME = "submission"  # relative to manuscript dir
SUBMISSION_FSTEM = "submission"  # relative to submission dir
TEMP_DIRNAME = "TEMP"  # relative to submission dir

# Define paths
submission_dir = c.manuscript_dir / SUBMISSION_DIRNAME
temp_dir = submission_dir / TEMP_DIRNAME
images_dir = c.manuscript_dir / IMAGES_DIRNAME
original_makefile_fpath = c.manuscript_dir / MAKEFILE_FNAME
original_tex_fpath = c.manuscript_dir / MANUSCRIPT_FNAME
original_pdf_fpath = (c.manuscript_dir / OUT_DIRNAME / MANUSCRIPT_FNAME).with_suffix(".pdf")
export_zip_fpath = submission_dir / f"{SUBMISSION_FSTEM}.zip"
export_pdf_fpath = export_zip_fpath.with_suffix(".pdf")

parser = argparse.ArgumentParser(description="Compile LaTeX manuscript for submission.")
parser.add_argument(
    "--compile-original",
    action="store_true",
    help="Compile the original manuscript prior to compiling the submission.",
)
args = parser.parse_args()

COMPILE_ORIGINAL = args.compile_original

if COMPILE_ORIGINAL:
    # Compile the original manuscript to ensure that all files are up to date
    subprocess.run(["make", "clean"], cwd=c.manuscript_dir, check=True)
    subprocess.run(["make"], cwd=c.manuscript_dir, check=True)

# Verify latest build is up to date (asset files are checked below before copying over)
tex_mtime = original_tex_fpath.stat().st_mtime
pdf_mtime = original_pdf_fpath.stat().st_mtime
assert pdf_mtime > tex_mtime, "PDF is older than source. Needs rebuild."

# Create submission and temporary directories, deleting if it exists for a clean slate
if submission_dir.exists():
    shutil.rmtree(submission_dir)
submission_dir.mkdir()
temp_dir.mkdir()

# Track all files that need to be copied over and included in the submission zip file
included_files = set()

# Add makefile to included files, to be copied over for build and later removed before zipping
included_files.add(original_makefile_fpath)

# Load manuscript
with original_tex_fpath.open("r", encoding="utf-8") as f:
    manuscript = f.read()

# Identify bib file and add to included files, also to be copied and later removed before zipping
bib_fname = re.search(r"\\addbibresource\{([a-z]+?\.bib)\}", manuscript).group(1)
original_bib_fpath = c.manuscript_dir / bib_fname
included_files.add(original_bib_fpath)

# Get all image and figure files from the manuscripts
img_fnames = re.findall(r"\\includegraphics(?:\[.*?\])?\{(.*?)\}", manuscript)
figext = re.search(r"\\newcommand\{\\figext\}\{(.*?)\}", manuscript).group(1)
img_fnames = [fn.replace("\\figext", figext) for fn in img_fnames]
# Get all potential image and figure paths from source
potential_image_paths = set(images_dir.glob("*"))
potential_figure_paths = set(c.figures_dir.glob("*"))
potential_graphic_paths = potential_image_paths | potential_figure_paths
potential_graphics = {fp.name: fp for fp in potential_graphic_paths}
# Identify which images and figures should be copied over
for img_fname in img_fnames:
    img_fpath = potential_graphics[img_fname]
    assert img_fpath.stat().st_mtime < pdf_mtime, f"Image {img_fname} is newer than PDF."
    included_files.add(img_fpath)

# Identify all data files (tables) that are loaded by datatool in the manuscript
tables_dir_str = f"{c.tables_dir.as_posix()}/"
table_pattern = rf"\\DTLloaddb\{{.*?\}}\{{{tables_dir_str}(.*?)\}}"
table_fnames = re.findall(table_pattern, manuscript)
# Get all potential table paths from source
potential_table_paths = set(c.tables_dir.glob("*"))
potential_tables = {fp.name: fp for fp in potential_table_paths}
# Identify which tables should be copied over
for table_fname in table_fnames:
    table_fpath = potential_tables[table_fname]
    assert table_fpath.stat().st_mtime < pdf_mtime, f"Table {table_fname} is newer than PDF."
    included_files.add(table_fpath)

# Copy included files over to temporary directory
for fpath in included_files:
    shutil.copy2(fpath, temp_dir / fpath.name)

# Modify manuscript text to remove graphics path
manuscript = re.sub(r"\\graphicspath{.*}\n", "", manuscript)
# Modify manuscript text to remove tables directory from table paths
manuscript = re.sub(tables_dir_str, "", manuscript)
# Write modified manuscript to temporary directory
with open(temp_dir / MANUSCRIPT_FNAME, "w", encoding="utf-8") as f:
    f.write(manuscript)

# Run make to compile manuscript within temporary directory
subprocess.run(["make"], cwd=temp_dir, check=True)

# Remove temporary bib file
temp_bib_fpath = temp_dir / bib_fname
temp_bib_fpath.unlink()
# Move .bbl file from temporary directory output to temporary directory root
temp_out_bbl_fpath = (temp_dir / OUT_DIRNAME / MANUSCRIPT_FNAME).with_suffix(".bbl")
temp_bbl_fpath = (temp_dir / MANUSCRIPT_FNAME).with_suffix(".bbl")
temp_out_bbl_fpath.rename(temp_bbl_fpath)

# Rerun for final compilation using .bbl file instead of .bib
subprocess.run(["make", "clean"], cwd=temp_dir, check=True)
subprocess.run(["make", "quick"], cwd=temp_dir, check=True)

# Remove temporary makefile
temp_makefile_fpath = temp_dir / MAKEFILE_FNAME
temp_makefile_fpath.unlink()

# Create zip file for submission
with zipfile.ZipFile(export_zip_fpath, "w") as zf:
    for fpath in temp_dir.glob("*"):
        if fpath.is_file():
            zf.write(fpath, arcname=fpath.name)

# Move the compiled submission PDF out of the temporary directory to keep it
temp_pdf_fpath = (temp_dir / OUT_DIRNAME / MANUSCRIPT_FNAME).with_suffix(".pdf")
shutil.move(temp_pdf_fpath, export_pdf_fpath)

# Delete the temporary directory
shutil.rmtree(temp_dir)
