## See the README.md for details about each script.


# Set script to exit if any command fails.
set -e

# Handle command line argument.
if [[ $# -gt 1 ]]; then
  echo "!! noliwc is the only allowed argument !!"; exit;
elif [[ $# -eq 1 ]]; then
  if [[ $1 != "noliwc" ]]; then
    echo "!! noliwc is the only allowed argument !!"; exit;
  fi
fi

# Setup.
python setup-data_dirs.py
python -m spacy download en_core_web_lg

# Scrape and clean.
echo "Scraping and cleaning all data will take hours..."
python scrape-posts.py
python clean-posts.py
python scrape-users.py
python clean-users.py
python anonymize-posts.py

# Describe.
echo "Description analyses take just a minute altogether..."
python describe-timecourse.py
python describe-usercount.py
python describe-toplabels.py
python describe-categorycounts.py
python describe-categorypairs.py
python describe-demographics.py
python describe-wordcount.py

# Validate.
echo "Validation analyses are quick unless LIWCing..."
python validate-classifier.py
python validate-classifier_stats.py
python validate-wordshift.py
python validate-wordshift_plot.py
if [[ -z "$1" ]]; then  # no argument supplied (i.e., run liwc)
  python validate-liwc.py --words
  python validate-liwc_stats.py
  python validate-liwc_word_stats.py
  python validate-liwc_word_plot.py
fi
