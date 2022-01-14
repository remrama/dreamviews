# dreamviews_ds

Turning the public [DreamViews Dream Journal](https://www.dreamviews.com/blogs/) into a usable dataset. This repo is for collecting, cleaning, describing, and validating the dataset.

See [preprint]() for details and motivation.

---

See `config.py` for directory info and other configuration stuff.

You can `runall.sh`, but first make sure you adjust the `DATA_DIR` in `config.py` first to be wherever you want things output. Also use the `environment.yml` file to set up a conda environment that should be able to run smoothly and reproduce my exact output.

The code filenames start with one of `init`, `scrape`, `clean`, `describe`, `validate`, where the prefix explains which of the stages the file is used for.

Raw data from the web scraping is in `DATA_DIR/source`, "middle-ground" output, like the cleaned posts and aggregated users or raw LIWC output are in `DATA_DIR/derivatives`, and any final numbers (eg, descriptive stats and significance tests) and plots are in `DATA_DIR/results`.

Everything's in Python, and _almost_ everything runs with freely available tools. LIWC is the only hiccup, the dictionary file is proprietary so it's not included in the repo. You can run all the code at once using `runall.sh`, but without the LIWC file use `runall.sh noliwc`.


### A preliminary script

```shell
# create the relevant subfolders of the data directory (specified in config.py)
python init-data_dirs.py                    # ==> DATA_DIR/source/
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

# Generate a *super*-anonymized dataset.
python anonymize-posts.py                   # ==> DATA_DIR/derivatives/dreamviews-posts_superanon.tsv
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

# Visualize user demographics.
python describe-demographics.py             # ==> DATA_DIR/results/describe-demographics_provided.tsv
                                            # ==> DATA_DIR/results/describe-demographics_agegender.tsv
                                            # ==> DATA_DIR/results/describe-demographics_location.tsv
                                            # ==> DATA_DIR/results/describe-demographics.png

# Count how many words are in each post.
python describe-wordcount.py                # ==> DATA_DIR/derivatives/describe-wordcount.tsv
                                            # ==> DATA_DIR/results/describe-wordcount.png
```


### Validate certain aspects of the dataset 

```shell
# Train/test a classifier to identify the lucidity of a post.
python validate-classifier.py               # ==> DATA_DIR/derivatives/validate-classifier.npz
python validate-classifier_stats.py         # ==> DATA_DIR/derivatives/validate-classifier_cv.tsv
                                            # ==> DATA_DIR/derivatives/validate-classifier_avg.tsv

# Identify words that distinguish lucid and non-lucid posts.
# (Also words that distinguish nightmares from non-nightmares.)
python validate-wordshift.py                # ==> DATA_DIR/results/validate-wordshift_scores-jsd.tsv
                                            # ==> DATA_DIR/results/validate-wordshift_scores-fear_nm.tsv
                                            # ==> DATA_DIR/results/validate-wordshift-jsd.png
                                            # ==> DATA_DIR/results/validate-wordshift-proportion.png
                                            # ==> DATA_DIR/results/validate-wordshift-fear_nm.png
python validate-wordshift_plot.py           # ==> DATA_DIR/results/validate-wordshift.png

# Compare lucid and non-lucid reports using LIWC categories of interest (Agency and Insight).
python validate-liwc.py --words             # ==> DATA_DIR/derivatives/validate-liwc_scores.tsv
                                            # ==> DATA_DIR/derivatives/validate-liwc_wordscores-data.npz
                                            # ==> DATA_DIR/derivatives/validate-liwc_wordscores-attr.npz
python validate-liwc_stats.py               # ==> DATA_DIR/results/validate-liwc_scores-descr.tsv
                                            # ==> DATA_DIR/results/validate-liwc_scores-stats.tsv
                                            # ==> DATA_DIR/results/validate-liwc_scores-plot.png
python validate-liwc_word_stats.py          # ==> DATA_DIR/results/validate-liwc_wordscores-stats.tsv
python validate-liwc_word_plot.py           # ==> DATA_DIR/results/validate-liwc_wordscores-plot.png

# Generate some latex tables from the tsv output, for manuscript.
for bn in describe-wordcount describe-topcategories describe-toptags validate-classifier_avg
do
    python tsv2latex.py --basename ${bn}    # ==> DATA_DIR/results/<bn>.tex
done
```