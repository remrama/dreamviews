"""See the README.md for details about each script."""

import argparse
import subprocess
import sys


def run(script, *args):
    result = subprocess.run([sys.executable, script, *args])
    if result.returncode != 0:
        sys.exit(result.returncode)


parser = argparse.ArgumentParser(description="Run all steps")
parser.add_argument("--scrape", action="store_true", help="Scrape data")
parser.add_argument("--extract", action="store_true", help="Extract data")
args = parser.parse_args()

# Setup
run("setup-data_dirs.py")
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], check=True)

# Scrape
if args.scrape:
    print("Scraping and extracting data will take hours...")
    run("scrape-posts.py")
    run("extract-posts.py")
    run("scrape-users.py")

# Extract
if args.extract:
    if not args.scrape:
        print("Extracting data will take hours...")
        run("extract-posts.py")
    run("extract-users.py")

# Describe
print("Descriptive analyses take just a minute altogether...")
run("describe-timecourse.py")
run("describe-usercount.py")
run("describe-toplabels.py")
run("describe-categorycounts.py")
run("describe-categorypairs.py")
run("describe-demographics.py")
run("describe-wordcount.py")

# Validate
print("Validation analyses are quick unless LIWCing...")
run("validate-classifier.py")
run("validate-classifier_stats.py")
run("validate-wordshift.py")
run("validate-wordshift_plot.py")
run("validate-liwc.py", "--words")
run("validate-liwc_stats.py")
run("validate-liwc_word_stats.py")
run("validate-liwc_word_plot.py")
