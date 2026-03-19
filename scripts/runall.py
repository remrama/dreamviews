"""See the README.md for details about each script."""

import argparse
import subprocess
import sys

import spacy

from config import SPACY_MODEL


def run(script, *args):
    result = subprocess.run([sys.executable, script, *args])
    if result.returncode != 0:
        sys.exit(result.returncode)


parser = argparse.ArgumentParser(description="Run all steps")
parser.add_argument("--scrape", action="store_true", help="Scrape data")
parser.add_argument("--extract", action="store_true", help="Extract data")
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
