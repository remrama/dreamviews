# dreamviews_ds

Turning the public [DreamViews dream journal](https://www.dreamviews.com/blogs/) into a usable dataset. This repo is for scraping, cleaning, describing, and validating the dataset.

See [preprint]() for details and motivation.

---

See `config.py` for directory info and other configuration stuff.

You can `runall.sh`, but first make sure you adjust the `DATA_DIR` in `config.py` first to be wherever you want things output. Also use the `environment.yml` file to set up a conda environment that should be able to run smoothly and reproduce my exact output. Also add the argument `runall.sh noliwc` unless you have a LIWC file lying around.


### Setup

```shell
# Create the relevant subfolders of the data directory (specified in config.py).
python setup-data_dirs.py                   # ==> DATA_DIR/source/
                                            # ==> DATA_DIR/derivatives/
                                            # ==> DATA_DIR/results/
                                            # ==> DATA_DIR/results/hires/
```


### Scrape raw data and clean it up

```shell
# Collect raw dream journal posts as html files.
python scrape-posts.py                      # ==> DATA_DIR/source/dreamviews-posts.zip

# Convert raw html posts into a cleaned tsv file.
# This cleans the text and implements exclusion criteria.
# All posts and users get unique randomized IDs (also save from this).
python clean-posts.py                       # ==> DATA_DIR/derivatives/dreamviews-posts.tsv
                                            # ==> DATA_DIR/derivatives/dreamviews-users_key.json

# Collect the relevant user profiles and clean them.
python scrape-users.py                      # ==> DATA_DIR/source/dreamviews-users.zip
python clean-users.py                       # ==> DATA_DIR/derivatives/dreamviews-users.tsv
```


### Describe the dataset with visualizations and summary statistics

```shell
# Visualize the amount of data over time.
python describe-timecourse.py               # ==> DATA_DIR/results/describe-timecourse.png
python describe-usercount.py                # ==> DATA_DIR/results/describe-usercount.png

# Identify the top labels (categories and tags).
python describe-toplabels.py                # ==> DATA_DIR/results/describe-topcategories.tsv
                                            # ==> DATA_DIR/results/describe-toptags.tsv

# Count the number of posts with each label of interest and how often they overlap.
python describe-categorycounts.py           # ==> DATA_DIR/results/describe-categorycounts.png

# Count the number of participants that have posts labeled as lucid and non-lucid.
python describe-categorypairs.py            # ==> DATA_DIR/results/describe-categorypairs.tsv
                                            # ==> DATA_DIR/results/describe-categorypairs.png

# Count the frequencies of reports age, gender, and location clusters.
python describe-demographics.py             # ==> DATA_DIR/results/describe-demographics_provided.tsv
                                            # ==> DATA_DIR/results/describe-demographics_agegender.tsv
                                            # ==> DATA_DIR/results/describe-demographics_location.tsv
                                            # ==> DATA_DIR/results/describe-demographics_agegender.png
                                            # ==> DATA_DIR/results/describe-demographics_location.png

# Count how many words are in each post.
python describe-wordcount.py                # ==> DATA_DIR/results/describe-wordcount.tsv
                                            # ==> DATA_DIR/results/describe-wordcount_perpost.png
                                            # ==> DATA_DIR/results/describe-wordcount_peruser.png
                                            # ==> DATA_DIR/results/describe-wordcount_lucidity.png
```


### Validate certain aspects of the dataset 

```shell
# Train/test a classifier on the lucidity of a post.
python validate-classifier.py               # ==> DATA_DIR/derivatives/validate-classifier.npz
python validate-classifier_stats.py         # ==> DATA_DIR/derivatives/validate-classifier_cv.tsv
                                            # ==> DATA_DIR/results/validate-classifier_avg.tsv

# Identify words that distinguish lucid and non-lucid posts (and nightmares from non-nightmares).
python validate-wordshift.py                # ==> DATA_DIR/results/validate-wordshift_jsd-scores.tsv
                                            # ==> DATA_DIR/results/validate-wordshift_jsd-plot.png
                                            # ==> DATA_DIR/results/validate-wordshift_fear-scores.tsv
                                            # ==> DATA_DIR/results/validate-wordshift_fear-plot.png
                                            # ==> DATA_DIR/results/validate-wordshift_proportion-plot.png
                                            # ==> DATA_DIR/results/validate-wordshift_proportion-ld1grams.tsv
                                            # ==> DATA_DIR/results/validate-wordshift_proportion-ld2grams.tsv
python validate-wordshift_plot.py -s jsd    # ==> DATA_DIR/results/validate-wordshift_jsd-myplot.png
python validate-wordshift_plot.py -s fear   # ==> DATA_DIR/results/validate-wordshift_fear-myplot.png

# Compare lucid and non-lucid reports using LIWC categories Insight and Agency.
python validate-liwc.py --words             # ==> DATA_DIR/derivatives/validate-liwc_scores.tsv
                                            # ==> DATA_DIR/derivatives/validate-liwc_wordscores-data.npz
                                            # ==> DATA_DIR/derivatives/validate-liwc_wordscores-attr.npz
python validate-liwc_stats.py               # ==> DATA_DIR/results/validate-liwc_scores-descr.tsv
                                            # ==> DATA_DIR/results/validate-liwc_scores-stats.tsv
                                            # ==> DATA_DIR/results/validate-liwc_scores-plot.png
python validate-liwc_word_stats.py          # ==> DATA_DIR/results/validate-liwc_wordscores-stats.tsv
python validate-liwc_word_plot.py -c agency # ==> DATA_DIR/results/validate-liwc_wordscores_agency-plot.png
python validate-liwc_word_plot.py -c insight # ==> DATA_DIR/results/validate-liwc_wordscores_insight-plot.png
```


### Cleanup

```shell
# Generate some latex tables from the tsv output, for manuscript.
declare -a files2convert=(
    "describe-topcategories"
    "describe-toptags"
    "describe-wordcount"
    "validate-wordshift_proportion-ld1grams"
    "validate-wordshift_proportion-ld2grams"
)
for bn in "${arr[@]}"
do
    python tsv2latex.py --basename "${bn}"  # ==> DATA_DIR/results/<bn>.tex
done
```