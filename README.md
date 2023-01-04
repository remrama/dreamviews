# dreamviews_ds

Turning the public [DreamViews dream journal](https://www.dreamviews.com/blogs) into a usable dataset. This repo is for scraping, cleaning, describing, and validating the dataset.

## Code layout

### Miscellaneous files

* `config.py` houses directory info, default constants, and utility functions
* `environment.yaml` file to set up a conda environment
* `runall.sh` to run everything (`runall.sh noliwc` if you don't have a LIWC dictionary file)

### Setup

```shell
# Create the relevant subfolders of the data directory (specified in config.py).
python setup-data_dirs.py                   #=> <DATA_DIR>/source/
                                            #=> <DATA_DIR>/derivatives/

# Download spaCy model.
python -m spacy download en_core_web_lg
```

### Scrape raw data and clean it up

```shell
# Collect raw dream journal posts as html files.
python scrape-posts.py                      #=> source/dreamviews-posts.zip

# Convert raw html posts into a cleaned tsv file (exclusion criteria applied).
python clean-posts.py                       #=> derivatives/dreamviews-posts.tsv
                                            #=> derivatives/dreamviews-users_key.json

# Collect the relevant user profiles and clean them.
python scrape-users.py                      #=> source/dreamviews-users.zip
python clean-users.py                       #=> derivatives/dreamviews-users.tsv
```

### Describe the dataset with visualizations and summary statistics

```shell
# Visualize the amount of data over time.
python describe-timecourse.py               #=> derivatives/describe-timecourse.png
                                            #=> derivatives/describe-totalcounts.tsv
python describe-usercount.py                #=> derivatives/describe-usercount.png

# Identify the top labels (categories and tags).
python describe-toplabels.py                #=> derivatives/describe-topcategories.tsv
                                            #=> derivatives/describe-toptags.tsv

# Count the number of posts with each label of interest and how often they overlap.
python describe-categorycounts.py           #=> derivatives/describe-categorycounts.png

# Count the number of participants that have posts labeled as lucid and non-lucid.
python describe-categorypairs.py            #=> derivatives/describe-categorypairs.tsv
                                            #=> derivatives/describe-categorypairs.png

# Count the frequencies of reports age, gender, and location clusters.
python describe-demographics.py             #=> derivatives/describe-demographics_provided.tsv
                                            #=> derivatives/describe-demographics_agegender.tsv
                                            #=> derivatives/describe-demographics_location.tsv
                                            #=> derivatives/describe-demographics_agegender.png
                                            #=> derivatives/describe-demographics_location.png

# Count how many words are in each post.
python describe-wordcount.py                #=> derivatives/describe-wordcount.tsv
                                            #=> derivatives/describe-wordcount_perpost.png
                                            #=> derivatives/describe-wordcount_peruser.png
                                            #=> derivatives/describe-wordcount_lucidity.png
```

### Validate certain aspects of the dataset with statistical tests

```shell
# Train/test a classifier on the lucidity of a post.
python validate-classifier.py               #=> derivatives/validate-classifier.npz
python validate-classifier_stats.py         #=> derivatives/validate-classifier_cv.tsv
                                            #=> derivatives/validate-classifier_avg.tsv

# Identify words that distinguish lucid/non-lucid posts (and nightmares/non-nightmares).
python validate-wordshift.py                #=> derivatives/validate-wordshift_jsd-scores.tsv
                                            #=> derivatives/validate-wordshift_jsd-plot.png
                                            #=> derivatives/validate-wordshift_fear-scores.tsv
                                            #=> derivatives/validate-wordshift_fear-plot.png
                                            #=> derivatives/validate-wordshift_proportion-plot.png
                                            #=> derivatives/validate-wordshift_proportion-ld1grams.tsv
                                            #=> derivatives/validate-wordshift_proportion-ld2grams.tsv
python validate-wordshift_plot.py -s jsd    #=> derivatives/validate-wordshift_jsd-myplot.png
python validate-wordshift_plot.py -s fear   #=> derivatives/validate-wordshift_fear-myplot.png

# Compare lucid and non-lucid reports using LIWC categories Insight and Agency.
python validate-liwc.py --words             #=> derivatives/validate-liwc_scores.tsv
                                            #=> derivatives/validate-liwc_wordscores-data.npz
                                            #=> derivatives/validate-liwc_wordscores-attr.npz
python validate-liwc_stats.py               #=> derivatives/validate-liwc_scores-descr.tsv
                                            #=> derivatives/validate-liwc_scores-stats.tsv
                                            #=> derivatives/validate-liwc_scores-plot.png
python validate-liwc_word_stats.py          #=> derivatives/validate-liwc_wordscores-stats.tsv
python validate-liwc_word_plot.py -c agency #=> derivatives/validate-liwc_wordscores_agency-plot.png
python validate-liwc_word_plot.py -c insight #=> derivatives/validate-liwc_wordscores_insight-plot.png
```
