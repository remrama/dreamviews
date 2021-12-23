# dreamviews_ds

Turning the public [DreamViews Dream Journal](https://www.dreamviews.com/blogs/) into a usable dataset. This repo is for collecting, cleaning, describing, and validating the dataset.

---

See `config.py` for directory info and other configuration stuff. The `DATA_DIR` in `config.py` needs to be set first to match the desired data output location.

The code filenames start with one of `init`, `collect`, `clean`, `extract`, `describe`, `validate`, where the prefix explains which of the stages the file is used for.

Raw data from the web scraping is in `DATA_DIR/source`, "middle-ground" output, like the cleaned posts and aggregated users or raw LIWC output are in `DATA_DIR/derivatives`, and any plots or statistics are output to `DATA_DIR/results`.


### A few preliminary scripts.

```shell
# create the relevant subfolders of the data directory (specified in config.py)
python init-generate_data_dirs.py   # ==> DATA_DIR/source/
                                    # ==> DATA_DIR/derivatives/
                                    # ==> DATA_DIR/results/
python init-generate_liwc_dict.py   # ==> DATA_DIR/dictionaries/myliwc.dic
```


### COLLECT and CLEAN data

Grab all the data and tidy it up a bit along with minimal text preprocessing.

```shell
# Collect raw dream journal posts as html files.
python collect-posts.py             # ==> DATA_DIR/source/dreamviews-posts.zip

# Convert raw html posts into a cleaned tsv file.
# This cleans the text and implemnts exclusion criteria.
# All posts and users get unique randomized IDs (also save from this).
python clean-posts.py               # ==> DATA_DIR/derivatives/dreamviews-posts.tsv
                                    # ==> DATA_DIR/derivatives/dreamviews-users_key.json

# Collect the relevant user profiles and clean them.
python collect-users.py             # ==> DATA_DIR/source/dreamviews-users.zip
python clean-users.py               # ==> DATA_DIR/derivatives/dreamviews-users.tsv
```


### DESCRIBE data

Export plots and summary statistics describing the dataset.

```shell
##### Each post can have "category" or "tag" labels.
##### Visualize the amount of posts from each relevant category.

# generate top categories and labels
python describe-toplabels.py        # ==> DATA_DIR/results/describe-topcategories.tsv
                                    # ==> DATA_DIR/results/describe-toptags.tsv

# lucid/non-lucid/nightmare overlap
python describe-categorycounts.py   # ==> DATA_DIR/results/describe-categorycounts.png

# number of participants with both lucid and non-lucid posts
python describe-categorypairs.py    # ==> DATA_DIR/results/describe-categorypairs.tsv
                                    # ==> DATA_DIR/results/describe-categorypairs.png


##### Visualize user demographics.
# reported gender, age, and location
python describe-demographics.py     # ==> DATA_DIR/results/describe-demographics.png
                                    # ==> DATA_DIR/results/describe-demographics.tsv
                                    # ==> DATA_DIR/results/describe-demographics_locations.tsv


##### Visualize the amount of data there is.
# frequency of posts over time, posts per user, and word counts
python describe-timecourse.py       # ==> DATA_DIR/results/describe-timecourse.png
python describe-usercount.py         # ==> DATA_DIR/results/describe-usercount.png
python describe-wordcount.py        # ==> DATA_DIR/results/describe-wordcount.png
```


### VALIDATE data

Show that the lucid and non-lucid posts are differentiable in predictable ways based on previous literature.

1. Show that LDs and non-LDs can be distinguished with language.
2. Show that LDs have more insight and agency in LIWC (lucidity and control, respectively).

```shell
# classifier
python validate-classifier_run.py   # ==> DATA_DIR/derivatives/validate-classifier.npz
python validate-classifier_stat.py  # ==> DATA_DIR/results/validate-classifier.png

# words that differentiate lucid and non-lucid
python validate-wordshift_run.py    # ==> DATA_DIR/derivatives/validate-wordshift.tsv
python validate-wordshift_stat.py   # ==> DATA_DIR/results/validate-wordshift.tsv
                                    # ==> DATA_DIR/results/validate-wordshift.png

# run LIWC to get effects at the category and word levels
python validate-liwc_run.py --words # ==> DATA_DIR/derivatives/posts-liwc.tsv
                                    # ==> DATA_DIR/derivatives/posts-liwc_words-data.npz
                                    # ==> DATA_DIR/derivatives/posts-liwc_words-attr.npz

# plot total insight and agency effects LD vs non-LD
python validate-liwc_stat.py        # ==> DATA_DIR/derivatives/validate-liwc.tsv
                                    # ==> DATA_DIR/results/validate-liwc.tsv
                                    # ==> DATA_DIR/results/validate-liwc.png

# plot individual word contributions for insight and agency effects LD vs non-LD
python validate-liwcwords_perm.py   # ==> DATA_DIR/results/validate-liwcwords.tsv
python validate-liwcwords_stat.py   # ==> DATA_DIR/results/validate-liwcwords.png
```
