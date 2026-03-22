"""See the README.md for details about each script."""

import argparse
import subprocess
import sys

import spacy

from config import SPACY_MODEL, manuscript_dir


def run(script, *args, **kwargs):
    result = subprocess.run([sys.executable, script, *args], **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


parser = argparse.ArgumentParser(description="Run all steps")
parser.add_argument("--scrape", action="store_true", help="Scrape data.")
parser.add_argument("--extract", action="store_true", help="Extract data.")
parser.add_argument("--compile", action="store_true", help="Compile manuscript.")
args = parser.parse_args()

# Setup
try:
    spacy.load(SPACY_MODEL)
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)

# Scrape
if args.scrape:
    run("scrape-posts.py")
    run("extract-posts.py")
    run("scrape-users.py")

# Extract
if args.extract:
    if not args.scrape:
        run("extract-posts.py")
    run("extract-users.py")

# Describe
run("describe-totalcounts.py")
run("describe-usercount.py")
run("describe-toplabels.py")
run("describe-categorycounts.py")
run("describe-categorypairs.py")
run("describe-demographics.py")
run("describe-wordcount.py")

# Validate
run("validate-lemmas.py")
run("validate-classifier.py")
run("validate-classifier_stats.py")
run("validate-liwc.py", "--words")
run("validate-liwc_stats.py")
run("validate-liwc_word_stats.py")
run("validate-liwc_word_plot.py", "--category", "insight")
run("validate-liwc_word_plot.py", "--category", "agency")
run("validate-wordshift.py")
run("validate-wordshift_plot.py", "--shift", "jsd")
run("validate-wordshift_plot.py", "--shift", "fear")

# Compile
if args.compile:
    subprocess.run(["make"], check=True, cwd=manuscript_dir)
